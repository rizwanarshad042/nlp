# Medical Misinformation Detection App
# Core imports
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import warnings
import numpy as np
np.warnings = warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='.*InconsistentVersionWarning.*')
warnings.filterwarnings('ignore', message='.*node array.*')


os.environ['SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL'] = 'True'

# Project utilities
from utils.labels import LABELS, LABEL_TO_ID, ID_TO_LABEL
from utils.disease_integration import extract_disease_from_text
from utils.disease_symptoms import get_symptoms
from utils.gemini_integration import check_gemini_api_key, gemini_explain_classification
from utils.disease_myths_facts import get_disease_myths_and_facts

def save_to_dataset(*args, **kwargs):
    return True


def upsert_disease(*args, **kwargs):
    return None


def gemini_list_symptoms(*args, **kwargs):
    return ""


def generate_and_save_disease_content(*args, **kwargs):
    return False, "Disease content generation has been removed."

try:
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TORCH_AVAILABLE = True
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    TRANSFORMER_AVAILABLE = False

try:
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


def check_ai_availability():
    groq_available, groq_message = check_gemini_api_key()
    
    if groq_available:
        return True, "Groq Integration Available", "groq"
    else:
        return False, f"No AI services available. Groq: {groq_message}", "none"


@st.cache_resource
def load_transformer_model():
    if not TRANSFORMER_AVAILABLE:
        return None, None
    
    # Try multiple possible paths (check biobert_final first as it's the trained model)
    possible_paths = [
        "models/transformer/biobert_final",
        "models/transformer/biobert_base_cased_v1_1_final",
        "models/transformer/biobert_training/checkpoint-6825"
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        return None, None
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        return tokenizer, model
    except Exception as e:
        print(f"Error loading transformer model from {model_path}: {e}")
        return None, None


@st.cache_resource
def load_ml_models():
    if not ML_AVAILABLE:
        return None, None, None
    
    models_dir = "models/ml"
    
    # Check if models directory exists
    if not os.path.exists(models_dir):
        st.error(f"Models directory not found: {models_dir}")
        st.info("Please train the models first by running: python train_all_models.py")
        return None, None, None
    
    try:
        # Load vectorizer with encoding compatibility
        vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
        if not os.path.exists(vectorizer_path):
            st.error(f"Vectorizer not found at {vectorizer_path}")
            st.info("Models need to be trained. Run: python train_all_models.py")
            return None, None, None
        
        try:
            vectorizer = joblib.load(vectorizer_path)
        except Exception as vec_error:
            # Try with explicit encoding
            try:
                import pickle
                with open(vectorizer_path, 'rb') as f:
                    vectorizer = pickle.load(f, encoding='latin1')
            except:
                st.error(f"Vectorizer version incompatibility: {vec_error}")
                st.info("Please retrain models with current Python/sklearn version")
                return None, None, None
        
        # Load Logistic Regression with version handling
        comprehensive_lr_path = os.path.join(models_dir, "logistic_regression.pkl")
        original_lr_path = os.path.join(models_dir, "logreg.pkl")
        
        logreg = None
        if os.path.exists(comprehensive_lr_path):
            try:
                logreg = joblib.load(comprehensive_lr_path)
            except Exception as lr_error:
                try:
                    import pickle
                    with open(comprehensive_lr_path, 'rb') as f:
                        logreg = pickle.load(f, encoding='latin1')
                except:
                    st.error(f"Logistic Regression version incompatibility: {lr_error}")
                    st.info("Please retrain models with: python train_all_models.py")
                    return None, None, None
        elif os.path.exists(original_lr_path):
            try:
                logreg = joblib.load(original_lr_path)
            except Exception as lr_error:
                try:
                    import pickle
                    with open(original_lr_path, 'rb') as f:
                        logreg = pickle.load(f, encoding='latin1')
                except:
                    st.error(f"Logistic Regression version incompatibility: {lr_error}")
                    st.info("Please retrain models with: python train_all_models.py")
                    return None, None, None
        else:
            st.error("No Logistic Regression model found")
            st.info("Train models by running: python train_all_models.py")
            return None, None, None
        
        # Load Random Forest with version handling and numpy dtype fix
        rf_path = os.path.join(models_dir, "random_forest.pkl")
        if not os.path.exists(rf_path):
            st.error(f"Random Forest model not found at {rf_path}")
            st.info("Train models by running: python train_all_models.py")
            return None, None, None
        
        try:
            # First attempt: standard loading
            rf = joblib.load(rf_path)
        except Exception as rf_error:
            # Check if it's a numpy dtype issue
            if "dtype" in str(rf_error).lower() or "node array" in str(rf_error).lower():
                st.warning("Random Forest has numpy dtype issue. Attempting compatibility fix...")
                try:
                    # Try loading with sklearn's old format
                    import pickle
                    with open(rf_path, 'rb') as f:
                        rf = pickle.load(f)
                except:
                    st.error(f"Random Forest dtype incompatibility: {rf_error}")
                    st.error("This is a numpy 2.x vs 1.x compatibility issue")
                    st.info("Solution: Retrain models in your current environment:")
                    st.code("python train_all_models.py", language="bash")
                    return None, None, None
            else:
                try:
                    import pickle
                    with open(rf_path, 'rb') as f:
                        rf = pickle.load(f, encoding='latin1')
                except:
                    st.error(f"Random Forest version incompatibility: {rf_error}")
                    st.info("Please retrain models with: python train_all_models.py")
                    return None, None, None
        
        return vectorizer, logreg, rf
        
    except Exception as e:
        st.error(f"Error loading ML models: {e}")
        st.info("Solution: Retrain models with your current environment by running: python train_all_models.py")
        return None, None, None


class CNNClassifier(nn.Module):

    def __init__(self, vocab_size, embedding_dim=128, num_filters=100, filter_sizes=(3, 4, 5), num_classes=3, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embedding_dim, num_filters, kernel_size=fs) for fs in filter_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        conv_outs = []
        for conv in self.convs:
            c = torch.relu(conv(x))
            p = torch.max_pool1d(c, kernel_size=c.size(2))
            conv_outs.append(p.squeeze(2))
        x = torch.cat(conv_outs, dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class LSTMClassifier(nn.Module):

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, num_classes=3, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        output, (hidden, _) = self.lstm(x)
        h = torch.cat((hidden[-2], hidden[-1]), dim=1)
        h = self.dropout(h)
        out = self.fc(h)
        return out


@st.cache_resource
def load_dl_models():
    if not TORCH_AVAILABLE:
        return None

    models_dir = "models/dl"
    vocab_path = os.path.join(models_dir, "vocab.json")
    cnn_path = os.path.join(models_dir, "cnn_best.pt")
    lstm_path = os.path.join(models_dir, "lstm_best.pt")

    if not (os.path.exists(cnn_path) and os.path.exists(lstm_path)):
        print(f"DL model files not found. CNN: {os.path.exists(cnn_path)}, LSTM: {os.path.exists(lstm_path)}")
        return None

    try:
        # Load or create vocabulary
        if os.path.exists(vocab_path):
            with open(vocab_path, "r") as f:
                data = json.load(f)
            vocab = data.get("vocab", {})
            max_len = data.get("max_len", 512)
            labels = data.get("labels", LABELS)
        else:
            # Create vocabulary from dataset if vocab.json is missing
            print("vocab.json not found. Creating vocabulary from dataset...")
            try:
                df = pd.read_csv("data/processed/medical_dataset.csv")
                from collections import Counter
                word_counts = Counter()
                for text in df['text'].dropna().astype(str):
                    words = text.lower().split()
                    word_counts.update(words)
                
                vocab = {word: idx + 2 for idx, (word, _) in enumerate(word_counts.most_common(10000))}
                vocab['<PAD>'] = 0
                vocab['<UNK>'] = 1
                max_len = 512
                labels = LABELS
                
                # Save vocab for future use
                vocab_data = {
                    "vocab": vocab,
                    "max_len": max_len,
                    "labels": labels
                }
                with open(vocab_path, "w") as f:
                    json.dump(vocab_data, f)
                print(f"Created and saved vocab.json with {len(vocab)} words")
            except Exception as e:
                print(f"Error creating vocabulary: {e}")
                return None

        vocab_size = len(vocab)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cnn_model = CNNClassifier(vocab_size=vocab_size, num_classes=len(labels))
        lstm_model = LSTMClassifier(vocab_size=vocab_size, num_classes=len(labels))

        cnn_model.load_state_dict(torch.load(cnn_path, map_location=device))
        lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))

        cnn_model.to(device).eval()
        lstm_model.to(device).eval()

        return {
            "vocab": vocab,
            "max_len": max_len,
            "labels": labels,
            "device": device,
            "cnn": cnn_model,
            "lstm": lstm_model,
        }
    except Exception as e:
        print(f"Error loading DL models: {e}")
        import traceback
        traceback.print_exc()
        return None


def _encode_text_dl(text: str, vocab: dict, max_len: int) -> torch.Tensor:
    """Convert text into a tensor of token ids using the saved vocabulary."""
    tokens = str(text).strip().lower().split()
    unk_id = vocab.get("<UNK>", 1)
    pad_id = vocab.get("<PAD>", 0)

    ids = [vocab.get(tok, unk_id) for tok in tokens[:max_len]]
    if len(ids) < max_len:
        ids += [pad_id] * (max_len - len(ids))

    return torch.LongTensor([ids[:max_len]])


def predict_dl_models(text: str, dl_bundle) -> dict:
    """
    Run CNN and LSTM models on a single text and return per-model probability dicts.
    Returns: {"CNN": {...}, "LSTM": {...}, "DL Ensemble": {...}}
    """
    if not dl_bundle or not text.strip():
        return {}

    vocab = dl_bundle["vocab"]
    max_len = dl_bundle["max_len"]
    labels = dl_bundle["labels"]
    device = dl_bundle["device"]
    cnn_model = dl_bundle["cnn"]
    lstm_model = dl_bundle["lstm"]

    x = _encode_text_dl(text, vocab, max_len).to(device)

    with torch.no_grad():
        cnn_logits = cnn_model(x)
        lstm_logits = lstm_model(x)

        cnn_probs = torch.softmax(cnn_logits, dim=1).cpu().numpy()[0]
        lstm_probs = torch.softmax(lstm_logits, dim=1).cpu().numpy()[0]
        ensemble_probs = (cnn_probs + lstm_probs) / 2.0

    results = {}
    for name, probs in [("CNN", cnn_probs), ("LSTM", lstm_probs), ("DL Ensemble", ensemble_probs)]:
        results[name] = {labels[i]: float(probs[i]) for i in range(len(labels))}

    return results


def predict_transformer(text: str, tokenizer, model) -> dict:
    """Make prediction using transformer model."""
    if not TRANSFORMER_AVAILABLE or tokenizer is None or model is None:
        return {}
    
    try:
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
        
        predictions = {}
        for i, label in enumerate(LABELS):
            predictions[label] = float(probabilities[0][i])
        
        return predictions
    except Exception as e:
        print(f"Error in transformer prediction: {e}")
        return {}


def predict_ml(text: str, vectorizer, logreg, rf) -> dict:
    """Make predictions using ML models."""
    if vectorizer is None or logreg is None or rf is None:
        return {}
    
    # Transform text
    X = vectorizer.transform([text])
    
    # Get predictions
    logreg_proba = logreg.predict_proba(X)[0]
    rf_proba = rf.predict_proba(X)[0]
    
    # Average predictions
    avg_proba = (logreg_proba + rf_proba) / 2
    
    predictions = {}
    for i, label in enumerate(LABELS):
        predictions[label] = float(avg_proba[i])
    
    return predictions


def main():
    st.set_page_config(
        page_title="Medical Misinformation Detection",
        page_icon="🏥",
        layout="wide"
    )
    
    st.markdown("""
    <style>
    /* Navigation buttons */
    .nav-button {
        width: 100%;
        padding: 12px 16px;
        margin: 4px 0;
        border: none;
        border-radius: 8px;
        background: #f8f9fa;
        color: #495057;
        font-size: 15px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
        text-align: left;
        border-left: 3px solid transparent;
    }
    
    .nav-button:hover {
        background: #e9ecef;
        border-left: 3px solid #6c757d;
        transform: translateX(2px);
    }
    
    .nav-button.active {
        background: #007bff;
        color: white;
        border-left: 3px solid #0056b3;
    }
    
    .nav-button.active:hover {
        background: #0056b3;
        border-left: 3px solid #004085;
    }
    
    /* Sidebar title */
    .sidebar-title {
        color: #495057;
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 20px;
        text-align: center;
        padding: 12px;
        background: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🏥 Medical Misinformation Detection System")
    st.markdown("---")
    
    with st.sidebar:
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'Classification'
        
        nav_options = {
            'Classification': '🔍',
            'Model Performance': '📈'
        }
        
        for page_name, icon in nav_options.items():
            button_text = f"{icon} {page_name}"
            
            if st.button(
                button_text,
                key=f'nav_{page_name}',
                use_container_width=True
            ):
                st.session_state.current_page = page_name
                st.rerun()
    
    page = st.session_state.current_page
    
    if page == "Classification":
        classification_page()
    elif page == "Model Performance":
        model_performance_page()


def classification_page():
    st.header("🔍 Medical Statement Classification")
    
    ai_available, ai_message, ai_provider = check_ai_availability()
    
    if not ai_available:
        st.warning(f"AI Integration Unavailable: {ai_message}")
        st.info("💡 AI explanations are disabled. Set up Groq API key in api_keys.py to enable AI-powered explanations.")
    
    st.markdown("---")
    
    with st.spinner("Loading models..."):
        tokenizer, transformer_model = load_transformer_model()
        vectorizer, logreg, rf = load_ml_models()
        dl_bundle = load_dl_models()
    
    st.subheader("Enter Medical Statement")
    statement = st.text_area(
        "Paste or type your medical statement here:",
        placeholder="e.g., 'Garlic cures COVID-19' or 'Vaccines prevent severe illness'",
        height=100
    )
    return



def disease_symptoms_page():
    st.header("🦠 Disease Symptoms Database")
    
    disease_csv_path = "disease_symptoms.csv"
    if os.path.exists(disease_csv_path):
        df = pd.read_csv(disease_csv_path)
        st.subheader(f"Current Database ({len(df)} diseases)")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No disease symptoms database found.")
        df = pd.DataFrame(columns=['disease_name', 'symptoms'])
    
    st.markdown("---")
    
    st.subheader("Add New Disease")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        new_disease = st.text_input("Disease Name")
        add_method = st.radio(
            "How to get symptoms?",
            ["Manual Entry", "AI Query"]
        )
        
        if 'previous_add_method' not in st.session_state:
            st.session_state.previous_add_method = add_method
        elif st.session_state.previous_add_method != add_method:
            if 'ai_symptoms_generated' in st.session_state:
                st.session_state.ai_symptoms_generated = False
            if 'generated_symptoms' in st.session_state:
                st.session_state.generated_symptoms = ""
            if 'generated_disease' in st.session_state:
                st.session_state.generated_disease = ""
            st.session_state.previous_add_method = add_method
    
    with col2:
        if add_method == "Manual Entry":
            new_symptoms = st.text_area("Symptoms (comma-separated)")
        else:
            ai_available, ai_message, ai_provider = check_ai_availability()
            if ai_available:
                if 'ai_symptoms_generated' not in st.session_state:
                    st.session_state.ai_symptoms_generated = False
                if 'generated_symptoms' not in st.session_state:
                    st.session_state.generated_symptoms = ""
                if 'generated_disease' not in st.session_state:
                    st.session_state.generated_disease = ""
                
                if not st.session_state.ai_symptoms_generated:
                    st.info("💡 AI will automatically generate symptoms when you click 'Query AI for Symptoms'")
                    if st.button("Query AI for Symptoms"):
                        if new_disease:
                            with st.spinner("Querying AI..."):
                                try:
                                    symptoms = gemini_list_symptoms(new_disease)
                                    
                                    if symptoms and symptoms != "AI not available":
                                        # Store generated data in session state
                                        st.session_state.generated_symptoms = symptoms
                                        st.session_state.generated_disease = new_disease
                                        st.session_state.ai_symptoms_generated = True
                                        
                                        st.success(f"✅ Found symptoms: {symptoms}")
                                        st.success(f"✅ Ready to save {new_disease} to disease database!")
                                        st.rerun()
                                    else:
                                        st.error("Could not retrieve symptoms from AI")
                                        
                                except Exception as e:
                                    st.error(f"Error querying AI: {e}")
                        else:
                            st.warning("Please enter a disease name first")
                else:
                    st.success(f"✅ Generated symptoms for {st.session_state.generated_disease}:")
                    st.write(f"**{st.session_state.generated_symptoms}**")
                    
                    if st.button("Submit Disease with AI Generated Symptoms", type="primary"):
                        try:
                            upsert_disease(st.session_state.generated_disease, st.session_state.generated_symptoms)
                            st.success(f"✅ Successfully saved {st.session_state.generated_disease} to disease database!")
                            
                            from utils.disease_myths_facts import save_to_dataset
                            fact_statement = f"{st.session_state.generated_disease} symptoms include: {st.session_state.generated_symptoms}"
                            save_to_dataset(fact_statement, "credible", "ai_generated", "medical_fact", st.session_state.generated_disease)
                            st.info("💡 Also saved as credible medical fact in main dataset")
                            
                            st.info("🚀 Generating comprehensive content for the new disease...")
                            with st.spinner("Generating 300 statements (100 credible, 100 misinformation, 100 facts)..."):
                                success, message = generate_and_save_disease_content(
                                    st.session_state.generated_disease, 
                                    ai_available, 
                                    ai_provider
                                )
                                
                                if success:
                                    st.success(f"✅ {message}")
                                    st.info("🎉 Your dataset now has comprehensive content for this disease!")
                                else:
                                    st.warning(f"Bulk content generation failed: {message}")
                                    st.info("💡 Disease symptoms were still saved successfully")
                            
                            st.session_state.ai_symptoms_generated = False
                            st.session_state.generated_symptoms = ""
                            st.session_state.generated_disease = ""
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error saving to database: {e}")
                    
                    if st.button("🔄 Generate New Symptoms"):
                        st.session_state.ai_symptoms_generated = False
                        st.rerun()
                
                # No symptoms input field needed for AI query
                new_symptoms = ""  # Set empty since AI will handle it
            else:
                st.warning("AI not available: " + ai_message)
                st.info("💡 Set GROQ_API_KEY (free) or OPENAI_API_KEY to enable AI symptom queries")
                new_symptoms = st.text_area("Symptoms (comma-separated)", placeholder="Enter symptoms manually")
    
    if add_method == "Manual Entry":
        if st.button("Add Disease to Database"):
            if new_disease and new_symptoms:
                try:
                    upsert_disease(new_disease, new_symptoms)
                    st.success(f"Added {new_disease} to database!")
                    
                    from utils.disease_myths_facts import save_to_dataset
                    fact_statement = f"{new_disease} symptoms include: {new_symptoms}"
                    save_to_dataset(fact_statement, "credible", "manual_entry", "medical_fact", new_disease)
                    st.info("💡 Also saved as credible medical fact in main dataset")
                    
                    # Generate bulk content for the new disease
                    ai_available, ai_message, ai_provider = check_ai_availability()
                    if ai_available:
                        st.info("🚀 Generating comprehensive content for the new disease...")
                        with st.spinner("Generating 300 statements (100 credible, 100 misinformation, 100 facts)..."):
                            success, message = generate_and_save_disease_content(
                                new_disease, 
                                ai_available, 
                                ai_provider
                            )
                            
                            if success:
                                st.success(f"✅ {message}")
                                st.info("🎉 Your dataset now has comprehensive content for this disease!")
                            else:
                                st.warning(f"Bulk content generation failed: {message}")
                                st.info("💡 Disease symptoms were still saved successfully")
                    else:
                        st.info("💡 AI not available for bulk content generation, but disease was saved successfully")
                    
                    st.rerun()
                except Exception as e:
                    st.error(f"Error adding disease: {e}")
            else:
                st.warning("Please provide both disease name and symptoms.")


def classification_page():
    st.header("🔍 Medical Statement Analysis")

    ai_available, ai_message, ai_provider = check_ai_availability()
    if not ai_available:
        st.warning(f"AI Integration Unavailable: {ai_message}")

    with st.spinner("Loading models..."):
        tokenizer, transformer_model = load_transformer_model()
        vectorizer, logreg, rf = load_ml_models()
        dl_bundle = load_dl_models()

    statement = st.text_area(
        "Paste or type your medical statement here:",
        placeholder="e.g., 'Garlic cures COVID-19' or 'Vaccines prevent severe illness'",
        height=120,
    )

    def merge_predictions(prediction_sets):
        totals = {label: 0.0 for label in LABELS}
        total_weight = 0.0
        weights = {
            'ML': 0.35,
            'DL': 0.30,
            'Transformer': 0.35,
        }

        for model_name, scores in prediction_sets.items():
            weight = weights.get(model_name, 0.2)
            total_weight += weight
            for label, score in scores.items():
                if label in totals:
                    totals[label] += score * weight

        if total_weight > 0:
            for label in totals:
                totals[label] /= total_weight

        best_label = max(totals, key=totals.get)
        return best_label, float(totals[best_label]), totals

    if st.button("Analyze Statement", type="primary"):
        if not statement.strip():
            st.warning("Please enter a statement to analyze.")
            return

        prediction_sets = {}

        if vectorizer and logreg and rf:
            try:
                prediction_sets['ML'] = predict_ml(statement, vectorizer, logreg, rf)
            except Exception as exc:
                st.warning(f"ML prediction failed: {exc}")

        if dl_bundle:
            try:
                dl_predictions = predict_dl_models(statement, dl_bundle)
                if dl_predictions:
                    averaged = {label: 0.0 for label in LABELS}
                    count = 0
                    for probs in dl_predictions.values():
                        count += 1
                        for label, score in probs.items():
                            if label in averaged:
                                averaged[label] += score
                    if count > 0:
                        for label in averaged:
                            averaged[label] /= count
                        prediction_sets['DL'] = averaged
            except Exception as exc:
                st.warning(f"DL prediction failed: {exc}")

        if tokenizer and transformer_model:
            try:
                transformer_probs = predict_transformer(statement, tokenizer, transformer_model)
                if transformer_probs:
                    prediction_sets['Transformer'] = transformer_probs
            except Exception as exc:
                st.warning(f"Transformer prediction failed: {exc}")

        st.subheader("1. Classification")
        if prediction_sets:
            final_label, final_confidence, combined_scores = merge_predictions(prediction_sets)
            st.success(f"**{final_label.title()}** ({final_confidence:.2%})")
            st.caption("Combined from available trained models")
            st.bar_chart(pd.DataFrame([combined_scores]))
        else:
            final_label = 'credible'
            final_confidence = 0.0
            st.info("No trained models were available. Falling back to symptom lookup only.")

        diseases = extract_disease_from_text(statement)
        st.subheader("2. Disease Symptoms")
        if diseases:
            for disease in diseases:
                symptoms = get_symptoms(disease)
                if symptoms and str(symptoms).strip() and not str(symptoms).lower().startswith("symptoms not"):
                    st.write(f"**{disease.title()}** symptoms: {symptoms}")
                else:
                    st.write(f"**{disease.title()}**: Symptoms not available")
        else:
            st.info("No specific disease was detected in the statement.")

        st.subheader("3. AI Explanation")
        if ai_available:
            try:
                explanation = gemini_explain_classification(statement, final_label, final_confidence)
                st.write(explanation)
            except Exception as exc:
                st.warning(f"AI explanation could not be generated: {exc}")
        else:
            st.info("AI explanation unavailable.")

        st.subheader("4. Myths and Facts")
        if diseases:
            for disease in diseases[:1]:
                myths_facts = get_disease_myths_and_facts(disease, ai_available=ai_available, ai_provider=ai_provider)
                myths = (myths_facts.get('myths') or [])[:5]
                facts = (myths_facts.get('facts') or [])[:5]

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Myths about {disease.title()}**")
                    if myths:
                        for idx, myth in enumerate(myths, 1):
                            st.write(f"{idx}. {myth}")
                    else:
                        st.info("No myths found.")
                with col2:
                    st.markdown(f"**Facts about {disease.title()}**")
                    if facts:
                        for idx, fact in enumerate(facts, 1):
                            st.write(f"{idx}. {fact}")
                    else:
                        st.info("No facts found.")
        else:
            st.info("Myths and facts are shown after a disease is detected in the statement.")


def rag_comparison_page():
    st.info("RAG comparison has been removed from the app.")


def model_performance_page():
    st.header("📈 Model Performance")
    
    results_dirs = ["results/ml", "results/dl", "results/transformer"]
    
    for results_dir in results_dirs:
        if os.path.exists(results_dir):
            st.subheader(f"{results_dir.split('/')[-1].upper()} Model Results")
            
            metrics_files = [f for f in os.listdir(results_dir) if f.endswith('_metrics.json')]
            
            if metrics_files:
                for metrics_file in metrics_files:
                    model_name = metrics_file.replace('_metrics.json', '')
                    
                    with open(os.path.join(results_dir, metrics_file), 'r') as f:
                        metrics = json.load(f)
                    
                    st.markdown(f"**{model_name.upper().replace('_', ' ')}**")
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
                    
                    with col2:
                        # Try f1_macro first, then f1_weighted, then f1
                        f1_score = metrics.get('f1_macro', metrics.get('f1_weighted', metrics.get('f1', 0)))
                        st.metric("F1 Score", f"{f1_score:.4f}")
                    
                    with col3:
                        # Try precision_macro first, then precision_weighted, then precision
                        precision = metrics.get('precision_macro', metrics.get('precision_weighted', metrics.get('precision', 0)))
                        st.metric("Precision", f"{precision:.4f}")
                    
                    with col4:
                        # Try recall_macro first, then recall_weighted, then recall
                        recall = metrics.get('recall_macro', metrics.get('recall_weighted', metrics.get('recall', 0)))
                        st.metric("Recall", f"{recall:.4f}")
                    
                    with col5:
                        # Try auc_macro first, then auc_weighted, then auc
                        auc = metrics.get('auc_macro', metrics.get('auc_weighted', metrics.get('auc', 0)))
                        st.metric("AUC", f"{auc:.4f}")
                    
                    # Show additional metrics in expandable section
                    with st.expander("View All Metrics"):
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.write("**Macro Metrics:**")
                            st.write(f"- Precision (Macro): {metrics.get('precision_macro', 0):.4f}")
                            st.write(f"- Recall (Macro): {metrics.get('recall_macro', 0):.4f}")
                            st.write(f"- F1 (Macro): {metrics.get('f1_macro', 0):.4f}")
                            st.write(f"- AUC (Macro): {metrics.get('auc_macro', 0):.4f}")
                        
                        with col_b:
                            st.write("**Other Metrics:**")
                            st.write(f"- Exact Match: {metrics.get('exact_match', 0):.4f}")
                            st.write(f"- Top-2 Accuracy: {metrics.get('top2_accuracy', 0):.4f}")
                            st.write(f"- Top-3 Accuracy: {metrics.get('top3_accuracy', 0):.4f}")
                        
                        # Per-class metrics
                        if any('precision_credible' in str(k) for k in metrics.keys()):
                            st.write("**Per-Class Metrics:**")
                            for label in ['credible', 'false', 'misleading']:
                                if f'precision_{label}' in metrics:
                                    st.write(f"**{label.title()}:**")
                                    st.write(f"  - Precision: {metrics.get(f'precision_{label}', 0):.4f}")
                                    st.write(f"  - Recall: {metrics.get(f'recall_{label}', 0):.4f}")
                                    st.write(f"  - F1: {metrics.get(f'f1_{label}', 0):.4f}")
                    
                    cm_path = os.path.join(results_dir, f"{model_name}_confusion_matrix.png")
                    if os.path.exists(cm_path):
                        st.image(cm_path, caption=f"{model_name} Confusion Matrix")
                    
                    st.markdown("---")
            else:
                st.info(f"No metrics found in {results_dir}")
        else:
            st.info(f"No results found for {results_dir}")


def rag_comparison_page():
    st.info("RAG comparison has been removed from the app.")


def classification_page():
    st.header("🦠 Disease Symptoms Lookup")
    st.write("Enter a medical statement and only the matched disease symptoms will be shown.")

    statement = st.text_area(
        "Paste or type your medical statement here:",
        placeholder="e.g., 'Garlic cures COVID-19' or 'Vaccines prevent severe illness'",
        height=120,
    )

    if st.button("Show Symptoms", type="primary"):
        if not statement.strip():
            st.warning("Please enter a statement first.")
            return

        diseases = extract_disease_from_text(statement)

        if not diseases:
            st.info("No specific diseases were detected in the statement.")
            return

        st.write("**Diseases detected:**", ", ".join(diseases))

        for disease in diseases:
            symptoms = get_symptoms(disease)
            if symptoms and str(symptoms).strip() and not str(symptoms).lower().startswith("symptoms not"):
                st.write(f"**{disease.title()}** symptoms: {symptoms}")
            else:
                st.write(f"**{disease.title()}**: Symptoms not available")


def disease_symptoms_page():
    st.info("This page has been removed. Use Classification to look up symptoms from a statement.")


if __name__ == "__main__":
    main()
