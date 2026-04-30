"""
KAGGLE GPU-ENABLED TRAINING: RNN AND CNN MULTICLASS CLASSIFICATION
Assignment #3 - Neural Networks (ANN, CNN, RNN)

This file trains RNN (LSTM) and CNN models on medical text classification (credible/misleading/false)
Optimized for Kaggle GPU environment with CUDA acceleration.

Features:
- GPU-enabled training (detects and uses CUDA automatically)
- Implements CNN with multiple filter sizes (3, 4, 5)
- Implements RNN architecture
- Multiclass classification (3 classes)
- Early stopping to prevent overfitting
- Comprehensive evaluation metrics
"""

import pandas as pd
import numpy as np
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import subprocess
import shutil
import glob
import time
import tempfile

# Set to "train" or "predict".
RUN_MODE = "train"
PREDICTION_SINGLE_TEXT = ""
PREDICTION_OUTPUT_PATH = "predictions_output.csv"

# ============================================================================
# GPU SETUP & RANDOM SEEDS FOR REPRODUCIBILITY
# ============================================================================
# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================================
# NEURAL NETWORK ARCHITECTURES
# ============================================================================

class CNNClassifier(nn.Module):
    """
    CNN TEXT CLASSIFIER FOR MULTICLASS CLASSIFICATION
    
    Architecture Overview:
    - Embedding Layer: Converts word IDs to dense vectors (vocabulary → embedding_dim)
    - Conv1d Layers: Multiple filters with different kernel sizes (3, 4, 5)
      * Each filter captures patterns of different lengths (trigrams, 4-grams, 5-grams)
    - Max Pooling: Extracts the most important feature from each filter
    - Fully Connected: Combines all filter outputs → class predictions
    
    Why CNN for text?
    - Learns local patterns efficiently
    - Parallelizable (faster training than RNN)
    - Good for phrase-level features
    """
    
    def __init__(self, vocab_size, embedding_dim=128, num_filters=100, 
                 filter_sizes=[3, 4, 5], num_classes=3, dropout=0.5):
        super(CNNClassifier, self).__init__()
        
        # Embedding layer: word_id → dense vector
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Multiple convolutional filters
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                     out_channels=num_filters,
                     kernel_size=fs)
            for fs in filter_sizes
        ])
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Output layer: combine all filters → num_classes
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
    def forward(self, x):
        """
        Forward pass:
        x: (batch_size, seq_len) - sequences of word IDs
        
        Returns:
        output: (batch_size, num_classes) - class scores
        """
        # Embedding: (batch_size, seq_len) → (batch_size, seq_len, embedding_dim)
        x = self.embedding(x)
        
        # Reshape for Conv1d: (batch_size, embedding_dim, seq_len)
        x = x.permute(0, 2, 1)
        
        # Apply each convolutional filter
        conv_outputs = []
        for conv in self.convs:
            # Apply convolution and activation
            conv_out = torch.relu(conv(x))  # (batch_size, num_filters, conv_len)
            
            # Max pooling over time dimension
            pooled = torch.max_pool1d(conv_out, kernel_size=conv_out.size(2))
            # Remove time dimension: (batch_size, num_filters, 1) → (batch_size, num_filters)
            conv_outputs.append(pooled.squeeze(2))
        
        # Concatenate all filter outputs
        x = torch.cat(conv_outputs, dim=1)  # (batch_size, num_filters * len(filter_sizes))
        
        # Dropout + Output layer
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class RNNClassifier(nn.Module):
    """
    VANILLA RECURRENT NEURAL NETWORK (RNN) FOR MULTICLASS CLASSIFICATION
    
    Architecture Overview:
    - Embedding Layer: Converts word IDs to dense vectors
    - RNN Layer: Bidirectional vanilla RNN processes sequences
      * Reads left-to-right AND right-to-left
      * Captures both preceding and following context
      * Multiple layers (stacked) for deeper understanding
    - Output Layer: Combined bidirectional hidden state → predictions
    
    Why Vanilla RNN?
    - Simple recurrent computation: h(t) = tanh(W_h * h(t-1) + W_x * x(t))
    - Direct sequence processing with feedback connections
    - Bidirectional: understands context from both directions
    
    RNN Internal Mechanism:
    - Hidden state: h(t) maintains temporal information
    - Recurrent connections: previous hidden state influences current computation
    - Activation: tanh introduces non-linearity
    """
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, 
                 num_layers=2, num_classes=3, dropout=0.5):
        super(RNNClassifier, self).__init__()
        
        # Embedding layer: word_id → dense vector
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Vanilla RNN layer with bidirectional processing
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # Input format: (batch, seq_len, features)
            dropout=dropout if num_layers > 1 else 0,  # Dropout between RNN layers
            bidirectional=True,  # Process forward AND backward
            nonlinearity='tanh'  # Activation function
        )
        
        # Dropout after RNN
        self.dropout = nn.Dropout(dropout)
        
        # Output layer: bidirectional hidden (hidden_dim * 2) → num_classes
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        """
        Forward pass:
        x: (batch_size, seq_len) - sequences of word IDs
        
        Returns:
        output: (batch_size, num_classes) - class scores
        """
        # Embedding: (batch_size, seq_len) → (batch_size, seq_len, embedding_dim)
        x = self.embedding(x)
        
        # RNN: (batch_size, seq_len, embedding_dim) → output, h_n
        rnn_out, hidden = self.rnn(x)
        
        # Extract bidirectional hidden states
        # hidden shape: (num_layers * num_directions, batch_size, hidden_dim)
        # For bidirectional: last forward is hidden[-2], last backward is hidden[-1]
        forward_hidden = hidden[-2]  # Last forward layer
        backward_hidden = hidden[-1]  # Last backward layer
        
        # Concatenate forward and backward: (batch_size, hidden_dim * 2)
        hidden_combined = torch.cat((forward_hidden, backward_hidden), dim=1)
        
        # Dropout + Output layer
        x = self.dropout(hidden_combined)
        x = self.fc(x)
        
        return x


# ============================================================================
# DATASET CLASS
# ============================================================================

class TextDataset(Dataset):
    """Convert text sequences and labels to PyTorch Dataset"""
    
    def __init__(self, sequences, labels):
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_and_prepare_data(data_path='data/processed/medical_dataset.csv'):
    """
    Load medical dataset and perform basic preprocessing
    
    Requirements:
    - CSV file with 'text' and 'label' columns
    - Labels: 'credible', 'misleading', 'false'
    """
    print("="*70)
    print("LOADING AND PREPARING DATA")
    print("="*70)
    
    df = pd.read_csv(data_path)
    
    # Remove empty rows
    df = df.dropna(subset=['text', 'label'])
    df['text'] = df['text'].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
    df['label'] = df['label'].astype(str).str.lower().str.strip()
    
    # Keep only three classes
    valid_labels = ['credible', 'misleading', 'false']
    df = df[df['label'].isin(valid_labels)]
    
    # Remove short texts
    df = df[df['text'].str.len() > 20]
    
    print(f"\n✓ Total samples: {len(df)}")
    print(f"\n✓ Label distribution:")
    print(df['label'].value_counts())
    print(f"\n✓ Average text length: {df['text'].str.len().mean():.1f} characters")
    
    return df


def build_vocabulary(texts, max_vocab_size=10000):
    """
    Build vocabulary from texts
    
    Returns:
    - vocab: dict mapping word → index
    - max_vocab_size: number of unique words
    """
    from collections import Counter
    
    print("\nBuilding vocabulary...")
    
    # Count word frequencies
    word_counts = Counter()
    for text in texts:
        words = str(text).lower().split()
        word_counts.update(words)
    
    # Create vocab with most common words
    vocab = {word: idx + 2 for idx, (word, _) in 
             enumerate(word_counts.most_common(max_vocab_size))}
    vocab['<PAD>'] = 0      # Padding token
    vocab['<UNK>'] = 1      # Unknown word token
    
    print(f"✓ Vocabulary size: {len(vocab)}")
    return vocab


def text_to_sequence(text, vocab, max_length=512):
    """Convert text to sequence of word IDs"""
    words = str(text).lower().split()
    sequence = [vocab.get(word, vocab['<UNK>']) for word in words[:max_length]]
    # Pad to max_length
    sequence = sequence + [vocab['<PAD>']] * (max_length - len(sequence))
    return sequence[:max_length]


def prepare_sequences(texts, labels, vocab, max_length=512):
    """Convert all texts to sequences"""
    X = np.array([text_to_sequence(text, vocab, max_length) for text in texts])
    y = np.array(labels)
    return X, y


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def calculate_metrics(y_true, y_pred, y_proba=None, class_names=None):
    """
    Calculate comprehensive evaluation metrics
    
    Metrics:
    - Accuracy: fraction correct
    - Precision: true positives / (true positives + false positives)
    - Recall: true positives / (true positives + false negatives)
    - F1: harmonic mean of precision and recall
    - AUC: area under ROC curve
    - Confusion Matrix
    """
    if class_names is None:
        class_names = ['credible', 'misleading', 'false']
    
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(class_names)), zero_division=0
    )
    
    metrics['precision_macro'] = np.mean(precision)
    metrics['recall_macro'] = np.mean(recall)
    metrics['f1_macro'] = np.mean(f1)
    
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', labels=range(len(class_names)), zero_division=0
    )
    metrics['precision_weighted'] = p_w
    metrics['recall_weighted'] = r_w
    metrics['f1_weighted'] = f1_w
    
    # Per-class metrics
    for i, class_name in enumerate(class_names):
        metrics[f'precision_{class_name}'] = float(precision[i])
        metrics[f'recall_{class_name}'] = float(recall[i])
        metrics[f'f1_{class_name}'] = float(f1[i])
    
    # AUC (if probabilities provided)
    if y_proba is not None:
        try:
            if len(np.unique(y_true)) == len(class_names):
                metrics['auc_macro'] = roc_auc_score(
                    y_true, y_proba, average='macro', multi_class='ovr'
                )
                metrics['auc_weighted'] = roc_auc_score(
                    y_true, y_proba, average='weighted', multi_class='ovr'
                )
        except Exception as e:
            print(f"Warning: AUC calculation failed: {e}")
            metrics['auc_macro'] = 0.0
            metrics['auc_weighted'] = 0.0
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model(model, train_loader, val_loader, device, model_name, 
                num_epochs=20, patience=5, learning_rate=0.0005):
    """
    Train a PyTorch model with early stopping
    
    Parameters:
    - model: Neural network model
    - train_loader: DataLoader for training data
    - val_loader: DataLoader for validation data
    - device: 'cuda' or 'cpu'
    - num_epochs: Maximum number of training epochs
    - patience: Stop if validation loss doesn't improve for N epochs
    - learning_rate: Optimizer learning rate
    """
    
    print(f"\n{'='*70}")
    print(f"TRAINING {model_name.upper()}")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Initial Learning Rate: {learning_rate}")
    print(f"Batch Size: {train_loader.batch_size}")
    print(f"Total Epochs: {num_epochs} (with early stopping patience={patience})")
    
    model = model.to(device)
    
    # Loss function with label smoothing to prevent overconfidence
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimizer with weight decay (L2 regularization)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler (reduce LR if loss plateaus)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    for epoch in range(num_epochs):
        # ===== TRAINING PHASE =====
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Tracking
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # ===== VALIDATION PHASE =====
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f'models/{model_name.lower()}_best.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n✓ Early stopping triggered (no improvement for {patience} epochs)")
                break
    
    print(f"✓ Training completed!")
    return model, history


def evaluate_model(model, test_loader, device, class_names):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_labels = []
    all_proba = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            proba = torch.softmax(outputs, dim=1)
            
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())
            all_proba.extend(proba.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_proba = np.array(all_proba)
    
    metrics = calculate_metrics(all_labels, all_preds, all_proba, class_names)
    return all_preds, all_labels, all_proba, metrics


def plot_training_history(history, model_name):
    """Plot training and validation curves"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} - Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'{model_name} - Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'models/{model_name.lower()}_training_history.png', dpi=150)
    print(f"✓ Saved training plot to models/{model_name.lower()}_training_history.png")


def plot_confusion_matrix(confusion_matrix, class_names, model_name):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'models/{model_name.lower()}_confusion_matrix.png', dpi=150)
    print(f"✓ Saved confusion matrix to models/{model_name.lower()}_confusion_matrix.png")


# ============================================================================
# USER INPUT & FILE MANAGEMENT
# ============================================================================

def prompt_for_csv_path():
    """
    Find a CSV file automatically from Kaggle input or the current workspace.
    Validates that the file has required columns.
    """
    print("\n" + "="*70)
    print("STEP 1: UPLOAD DATA")
    print("="*70)
    print("\nPlease upload your CSV through Kaggle's Add Data / Upload interface.")
    print("The script will automatically look for the CSV in /kaggle/input or the current folder.")
    print("Required CSV format:")
    print("  - Column 1: 'text' (medical text/claim)")
    print("  - Column 2: 'label' (credible, misleading, or false)")
    print("\nExample:")
    print("  text,label")
    print("  \"This vaccine is safe\",credible")
    print("  \"5G causes COVID\",false\n")

    candidate_dirs = [
        "/kaggle/input/medical_dataset",
        "/kaggle/input",
        "/kaggle/working",
        ".",
        os.getcwd(),
    ]
    csv_candidates = []

    for base_dir in candidate_dirs:
        if os.path.exists(base_dir):
            csv_candidates.extend(glob.glob(os.path.join(base_dir, "**", "*.csv"), recursive=True))

    # Remove duplicates while preserving order.
    seen = set()
    csv_candidates = [path for path in csv_candidates if not (path in seen or seen.add(path))]

    valid_candidates = []
    for csv_path in csv_candidates:
        try:
            preview = pd.read_csv(csv_path, nrows=5)
            if 'text' in preview.columns and 'label' in preview.columns:
                valid_candidates.append(csv_path)
        except Exception:
            continue

    if not valid_candidates:
        print("❌ Error: No valid CSV file found.")
        print("Upload the dataset with Kaggle's Add Data panel, then rerun the notebook.")
        print("Required columns: text, label")
        print("Program will exit.")
        exit(1)

    # Prefer the medical_dataset Kaggle folder first, then other Kaggle input paths, then newest file.
    valid_candidates.sort(
        key=lambda path: (
            0 if path.startswith('/kaggle/input/medical_dataset') else 1 if path.startswith('/kaggle/input') else 2 if path.startswith('/kaggle/working') else 3,
            -os.path.getmtime(path),
        )
    )

    csv_path = valid_candidates[0]

    if len(valid_candidates) > 1:
        print("\nMultiple valid CSV files found. Using the best match automatically:")
        for candidate in valid_candidates[:10]:
            print(f"  - {candidate}")
    
    # Try to load and validate CSV
    try:
        df_test = pd.read_csv(csv_path)
        
        # Check required columns
        if 'text' not in df_test.columns or 'label' not in df_test.columns:
            print("❌ Error: CSV must have 'text' and 'label' columns")
            print(f"Found columns: {list(df_test.columns)}")
            print("Program will exit.")
            exit(1)
        
        # Check if data exists
        if len(df_test) == 0:
            print("❌ Error: CSV file is empty")
            print("Program will exit.")
            exit(1)
        
        print(f"✓ CSV loaded successfully!")
        print(f"  File: {os.path.basename(csv_path)}")
        print(f"  Total rows: {len(df_test)}")
        print(f"  Columns: {list(df_test.columns)}")
        print(f"  Labels found: {df_test['label'].unique().tolist()}\n")
        
        return csv_path
    
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        print("Program will exit.")
        exit(1)


def create_results_archive(archive_name='training_results'):
    """
    Create a RAR archive with all training results
    Falls back to ZIP if RAR is not available
    
    Includes:
    - Trained models (.pt files)
    - Metrics (JSON)
    - Visualizations (PNG)
    - Vocabulary (JSON)
    """
    print("\n" + "="*70)
    print("CREATING RESULTS ARCHIVE")
    print("="*70)
    
    models_dir = 'models'
    if not os.path.exists(models_dir):
        print("❌ Error: 'models' directory not found")
        return None

    # Always place final archive in Kaggle working if available.
    output_root = '/kaggle/working' if os.path.exists('/kaggle/working') else os.getcwd()
    archive_base = os.path.join(output_root, archive_name)

    include_paths = [
        'models',
        'predictions_output.csv',
        'train_kaggle_rnn_cnn.py',
        'streamlit_app.py',
        'requirements.txt',
    ]

    # Stage all artifacts into one folder so the final archive is complete.
    temp_dir = tempfile.mkdtemp(prefix='results_bundle_')
    bundle_root = os.path.join(temp_dir, 'bundle')
    os.makedirs(bundle_root, exist_ok=True)

    copied_items = []
    for path in include_paths:
        if not os.path.exists(path):
            continue

        target_path = os.path.join(bundle_root, os.path.basename(path))
        if os.path.isdir(path):
            shutil.copytree(path, target_path)
            copied_items.append(path + '/')
        else:
            shutil.copy2(path, target_path)
            copied_items.append(path)

    if not copied_items:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("❌ Error: No output files found to archive")
        return None

    print(f"\nBundling {len(copied_items)} items...")
    for item in copied_items:
        print(f"  ✓ {item}")

    rar_executable = shutil.which('rar')
    if rar_executable:
        rar_path = archive_base + '.rar'
        try:
            # Archive the entire staged bundle directory into one RAR.
            subprocess.run(
                [rar_executable, 'a', '-r', rar_path, 'bundle'],
                cwd=temp_dir,
                check=True,
                capture_output=True,
            )
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"\n✓ RAR archive created: {rar_path}")
            print(f"  Size: {os.path.getsize(rar_path) / (1024 * 1024):.2f} MB")
            return rar_path
        except Exception as rar_error:
            print(f"\n⚠ RAR creation failed: {rar_error}")
            print("Creating ZIP archive instead...\n")

    zip_path = archive_base + '.zip'
    try:
        created_zip = shutil.make_archive(archive_base, 'zip', root_dir=temp_dir, base_dir='bundle')
        shutil.rmtree(temp_dir, ignore_errors=True)
        if os.path.exists(created_zip):
            print(f"✓ ZIP archive created: {zip_path}")
            print(f"  Size: {os.path.getsize(created_zip) / (1024 * 1024):.2f} MB")
            return created_zip
        print("❌ Failed to create ZIP archive")
        return None
    except Exception as zip_error:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"❌ Error creating ZIP archive: {zip_error}")
        return None


# ============================================================================
# PREDICTION SUPPORT
# ============================================================================

def find_file(filename):
    """Find the first matching file in common Kaggle and local folders."""
    search_roots = [".", os.getcwd(), "/kaggle/working", "/kaggle/input"]

    candidates = []
    for root in search_roots:
        if os.path.exists(root):
            candidates.extend(glob.glob(os.path.join(root, "**", filename), recursive=True))

    seen = set()
    candidates = [path for path in candidates if not (path in seen or seen.add(path))]

    if not candidates:
        raise FileNotFoundError(f"Could not find {filename}.")

    candidates.sort(key=lambda path: (0 if os.path.dirname(path).endswith("models") else 1, len(path)))
    return candidates[0]


def load_prediction_metadata():
    """Load saved vocabulary and label order for inference."""
    metrics_path = find_file("metrics_summary.json")
    with open(metrics_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    vocab_path = find_file("vocabulary.json")
    with open(vocab_path, "r", encoding="utf-8") as handle:
        vocab = json.load(handle)

    label_classes = metadata.get("label_encoder_classes")
    if not label_classes:
        label_classes = ["credible", "false", "misleading"]

    max_seq_length = int(metadata.get("max_seq_length", 512))
    return vocab, label_classes, max_seq_length


def build_prediction_model(model_type, vocab_size, num_classes):
    """Create the matching network architecture for inference."""
    model_type = model_type.upper().strip()
    if model_type == "CNN":
        return CNNClassifier(
            vocab_size=vocab_size,
            embedding_dim=128,
            num_filters=100,
            filter_sizes=[3, 4, 5],
            num_classes=num_classes,
            dropout=0.5,
        )
    if model_type == "RNN":
        return RNNClassifier(
            vocab_size=vocab_size,
            embedding_dim=128,
            hidden_dim=256,
            num_layers=2,
            num_classes=num_classes,
            dropout=0.5,
        )
    raise ValueError("model_type must be 'CNN' or 'RNN'")


def load_prediction_model(model_type, vocab_size, num_classes):
    model = build_prediction_model(model_type, vocab_size, num_classes).to(device)
    model_path = find_file(f"{model_type.lower()}_final.pt")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model: {model_path}")
    return model


def discover_prediction_csv():
    """Find a CSV with a text column."""
    candidates = []
    for root in [".", os.getcwd(), "/kaggle/working", "/kaggle/input"]:
        if os.path.exists(root):
            candidates.extend(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))

    seen = set()
    candidates = [path for path in candidates if not (path in seen or seen.add(path))]

    for csv_path in candidates:
        try:
            preview = pd.read_csv(csv_path, nrows=5)
            if "text" in preview.columns:
                return csv_path
        except Exception:
            continue

    raise FileNotFoundError("No CSV with a 'text' column was found for prediction.")


def read_prediction_inputs():
    """Read either a single text or a CSV file."""
    if PREDICTION_SINGLE_TEXT.strip():
        return pd.DataFrame({"text": [PREDICTION_SINGLE_TEXT.strip()]})

    try:
        csv_path = discover_prediction_csv()
        print(f"Using input file: {csv_path}")
        df = pd.read_csv(csv_path)
        if "text" not in df.columns:
            raise ValueError("Input CSV must contain a 'text' column.")
        df = df.dropna(subset=["text"]).copy()
        df["text"] = df["text"].astype(str)
        return df
    except FileNotFoundError:
        statement = input("Enter a statement to classify: ").strip()
        if not statement:
            raise ValueError("No statement provided for prediction.")
        return pd.DataFrame({"text": [statement]})


def predict_text_probabilities(model, texts, vocab, class_labels, max_length):
    """Predict class probabilities and labels for a model."""
    sequences = np.array([text_to_sequence(text, vocab, max_length) for text in texts])
    loader = DataLoader(torch.LongTensor(sequences), batch_size=64, shuffle=False)

    all_probs = []
    with torch.no_grad():
        for batch_x in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())

    probabilities = np.vstack(all_probs)
    predicted_indices = probabilities.argmax(axis=1)
    predicted_labels = [class_labels[index] for index in predicted_indices]
    confidences = probabilities.max(axis=1)

    return predicted_labels, confidences, probabilities


def print_model_confidences(model_name, text, class_labels, predicted_label, confidence, probabilities):
    """Print a readable confidence table for one model."""
    print(f"\n{model_name} prediction")
    print("-" * 40)
    print(f"Predicted label: {predicted_label}")
    print(f"Confidence: {confidence:.4f}")
    for index, label in enumerate(class_labels):
        print(f"  {label}: {probabilities[index]:.4f}")


def combine_model_predictions(class_labels, cnn_probs, rnn_probs):
    """Combine CNN and RNN probabilities by averaging them."""
    combined_probs = (cnn_probs + rnn_probs) / 2.0
    final_index = int(np.argmax(combined_probs))
    final_label = class_labels[final_index]
    final_confidence = float(combined_probs[final_index])
    return final_label, final_confidence, combined_probs


def run_prediction_pipeline():
    print("\n" + "="*70)
    print("PREDICTION MODE")
    print("="*70 + "\n")

    vocab, label_classes, max_seq_length = load_prediction_metadata()
    input_df = read_prediction_inputs()

    cnn_model = load_prediction_model("CNN", vocab_size=len(vocab), num_classes=len(label_classes))
    rnn_model = load_prediction_model("RNN", vocab_size=len(vocab), num_classes=len(label_classes))

    output_rows = []
    for text in input_df["text"].tolist():
        cnn_pred, cnn_conf, cnn_probs = predict_text_probabilities(
            cnn_model, [text], vocab, label_classes, max_seq_length
        )
        rnn_pred, rnn_conf, rnn_probs = predict_text_probabilities(
            rnn_model, [text], vocab, label_classes, max_seq_length
        )

        final_label, final_confidence, combined_probs = combine_model_predictions(
            label_classes, cnn_probs[0], rnn_probs[0]
        )

        print(f"\nStatement: {text}")
        print_model_confidences("CNN", text, label_classes, cnn_pred[0], float(cnn_conf[0]), cnn_probs[0])
        print_model_confidences("RNN", text, label_classes, rnn_pred[0], float(rnn_conf[0]), rnn_probs[0])
        print("\nFinal combined result")
        print("-" * 40)
        print(f"Predicted label: {final_label}")
        print(f"Confidence: {final_confidence:.4f}")
        for index, label in enumerate(label_classes):
            print(f"  {label}: {combined_probs[index]:.4f}")

        output_row = {
            "text": text,
            "cnn_predicted_label": cnn_pred[0],
            "cnn_confidence": float(cnn_conf[0]),
            "rnn_predicted_label": rnn_pred[0],
            "rnn_confidence": float(rnn_conf[0]),
            "final_predicted_label": final_label,
            "final_confidence": final_confidence,
        }
        for index, label in enumerate(label_classes):
            output_row[f"cnn_prob_{label}"] = float(cnn_probs[0][index])
            output_row[f"rnn_prob_{label}"] = float(rnn_probs[0][index])
            output_row[f"final_prob_{label}"] = float(combined_probs[index])
        output_rows.append(output_row)

    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(PREDICTION_OUTPUT_PATH, index=False)
    print(f"\nSaved predictions to: {PREDICTION_OUTPUT_PATH}")
    print(output_df.head())


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_training_pipeline(csv_path):
    print("\n" + "="*70)
    print("KAGGLE GPU-ENABLED: RNN & CNN MULTICLASS CLASSIFICATION")
    print("Assignment #3 - Neural Networks")
    print("="*70 + "\n")
    
    # Create output directory
    os.makedirs('models', exist_ok=True)
    
    class_names = ['credible', 'misleading', 'false']
    max_seq_length = 512
    batch_size = 32
    
    # ===== LOAD DATA =====
    df = load_and_prepare_data(csv_path)
    
    # ===== BUILD VOCABULARY =====
    vocab = build_vocabulary(df['text'].values, max_vocab_size=10000)
    vocab_size = len(vocab)
    
    # ===== PREPARE SEQUENCES =====
    print("\nPreparing sequences...")
    X, y = prepare_sequences(df['text'].values, df['label'].values, vocab, max_seq_length)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Train/Val/Test Split (70/15/15)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=0.30, random_state=42, stratify=y_encoded
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    print(f"✓ Train set: {len(X_train)}")
    print(f"✓ Val set: {len(X_val)}")
    print(f"✓ Test set: {len(X_test)}")
    
    # Create DataLoaders
    train_dataset = TextDataset(X_train, y_train)
    val_dataset = TextDataset(X_val, y_val)
    test_dataset = TextDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    results = {}
    
    # ===== TRAIN CNN =====
    print(f"\n{'#'*70}")
    print("# CNN - CONVOLUTIONAL NEURAL NETWORK")
    print(f"{'#'*70}")
    
    cnn_model = CNNClassifier(vocab_size, embedding_dim=128, num_filters=100,
                             filter_sizes=[3, 4, 5], num_classes=len(class_names),
                             dropout=0.5)
    cnn_model, cnn_history = train_model(
        cnn_model, train_loader, val_loader, device, 'CNN',
        num_epochs=20, patience=5, learning_rate=0.0005
    )
    
    # Evaluate CNN
    cnn_preds, cnn_labels, cnn_proba, cnn_metrics = evaluate_model(
        cnn_model, test_loader, device, class_names
    )
    
    results['CNN'] = {
        'model': cnn_model,
        'metrics': cnn_metrics,
        'vocab': vocab
    }
    
    print("\n✓ CNN Evaluation Results:")
    print(f"  - Accuracy: {cnn_metrics['accuracy']:.4f}")
    print(f"  - F1-Macro: {cnn_metrics['f1_macro']:.4f}")
    print(f"  - F1-Weighted: {cnn_metrics['f1_weighted']:.4f}")
    print(f"  - Precision-Macro: {cnn_metrics['precision_macro']:.4f}")
    print(f"  - Recall-Macro: {cnn_metrics['recall_macro']:.4f}")
    
    plot_training_history(cnn_history, 'CNN')
    plot_confusion_matrix(np.array(cnn_metrics['confusion_matrix']), class_names, 'CNN')
    
    # ===== TRAIN RNN =====
    print(f"\n{'#'*70}")
    print("# RNN - RECURRENT NEURAL NETWORK")
    print(f"{'#'*70}")
    
    rnn_model = RNNClassifier(vocab_size, embedding_dim=128, hidden_dim=256,
                               num_layers=2, num_classes=len(class_names),
                               dropout=0.5)
    rnn_model, rnn_history = train_model(
        rnn_model, train_loader, val_loader, device, 'RNN',
        num_epochs=20, patience=5, learning_rate=0.0005
    )
    
    # Evaluate RNN
    rnn_preds, rnn_labels, rnn_proba, rnn_metrics = evaluate_model(
        rnn_model, test_loader, device, class_names
    )
    
    results['RNN'] = {
        'model': rnn_model,
        'metrics': rnn_metrics,
        'vocab': vocab
    }
    
    print("\n✓ RNN Evaluation Results:")
    print(f"  - Accuracy: {rnn_metrics['accuracy']:.4f}")
    print(f"  - F1-Macro: {rnn_metrics['f1_macro']:.4f}")
    print(f"  - F1-Weighted: {rnn_metrics['f1_weighted']:.4f}")
    print(f"  - Precision-Macro: {rnn_metrics['precision_macro']:.4f}")
    print(f"  - Recall-Macro: {rnn_metrics['recall_macro']:.4f}")
    
    plot_training_history(rnn_history, 'RNN')
    plot_confusion_matrix(np.array(rnn_metrics['confusion_matrix']), class_names, 'RNN')
    
    # ===== SAVE RESULTS =====
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")
    
    # Save models
    torch.save(cnn_model.state_dict(), 'models/cnn_final.pt')
    torch.save(rnn_model.state_dict(), 'models/rnn_final.pt')
    print("✓ Saved models to models/")
    
    # Save metrics
    metrics_summary = {
        'CNN': cnn_metrics,
        'RNN': rnn_metrics,
        'vocab_size': vocab_size,
        'max_seq_length': max_seq_length,
        'class_names': class_names,
        'label_encoder_classes': label_encoder.classes_.tolist(),
        'timestamp': datetime.now().isoformat()
    }
    
    with open('models/metrics_summary.json', 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print("✓ Saved metrics to models/metrics_summary.json")
    
    # Save vocabulary
    with open('models/vocabulary.json', 'w') as f:
        json.dump(vocab, f)
    print("✓ Saved vocabulary to models/vocabulary.json")
    
    # ===== SUMMARY =====
    print(f"\n{'='*70}")
    print("TRAINING SUMMARY")
    print(f"{'='*70}\n")
    
    print("CNN Results:")
    print(f"  Accuracy:  {cnn_metrics['accuracy']:.4f}")
    print(f"  F1-Score:  {cnn_metrics['f1_macro']:.4f} (macro) / {cnn_metrics['f1_weighted']:.4f} (weighted)")
    print(f"  AUC:       {cnn_metrics.get('auc_macro', 'N/A')}")
    
    print("\nRNN Results:")
    print(f"  Accuracy:  {rnn_metrics['accuracy']:.4f}")
    print(f"  F1-Score:  {rnn_metrics['f1_macro']:.4f} (macro) / {rnn_metrics['f1_weighted']:.4f} (weighted)")
    print(f"  AUC:       {rnn_metrics.get('auc_macro', 'N/A')}")
    
    # Determine winner
    if cnn_metrics['f1_macro'] > rnn_metrics['f1_macro']:
        print("\n✓ WINNER: CNN achieved higher F1-Score")
    else:
        print("\n✓ WINNER: RNN achieved higher F1-Score")
    
    print(f"\n{'='*70}\n")
    
    # ===== CREATE RESULTS ARCHIVE FOR DOWNLOAD =====
    archive_file = create_results_archive('training_results')
    
    if archive_file:
        print(f"\n{'='*70}")
        print("✓ TRAINING COMPLETE!")
        print(f"{'='*70}")
        print(f"\n📥 Download your results: {archive_file}")
        print(f"\nContains:")
        print(f"  - cnn_final.pt (trained CNN model)")
        print(f"  - rnn_final.pt (trained RNN model)")
        print(f"  - metrics_summary.json (all evaluation metrics)")
        print(f"  - vocabulary.json (word vocabulary)")
        print(f"  - Training plots and confusion matrices (PNG)\n")
    else:
        print("\n⚠ Warning: Could not create archive, but models are saved in 'models/' directory")
        print("Please manually download the 'models' folder\n")


def main():
    if RUN_MODE.strip().lower() == "predict":
        run_prediction_pipeline()
        return

    csv_path = prompt_for_csv_path()
    if csv_path:
        run_training_pipeline(csv_path)


if __name__ == '__main__':
    main()
