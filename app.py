import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import warnings

# Suppress sklearn version warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='.*InconsistentVersionWarning.*')

from utils.labels import LABELS, LABEL_TO_ID, ID_TO_LABEL
from utils.disease_integration import classify_with_disease_context, get_top_myths_and_facts, extract_disease_from_text
from utils.disease_myths_facts import display_disease_myths_and_facts, generate_and_save_disease_content
from utils.disease_myths_facts import save_to_dataset
from advanced_data_augmentation import MedicalDataAugmenter
from utils.disease_symptoms import get_symptoms, upsert_disease
from utils.gemini_integration import gemini_explain_classification, check_gemini_api_key, gemini_list_symptoms, gemini_classify_statement
from utils.feedback_storage import get_stored_label, store_feedback, store_feedback_with_tags
from utils.symptom_disease_matcher import get_matcher
from utils.unknown_disease_handler import handle_unknown_disease, get_unknown_disease_handler

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
    """Check if Groq AI explanation service is available"""
    groq_available, groq_message = check_gemini_api_key()
    
    if groq_available:
        return True, "Groq Integration Available", "groq"
    else:
        return False, f"No AI services available. Groq: {groq_message}", "none"


@st.cache_resource
def load_transformer_model():
    """Load the fine-tuned BioBERT model."""
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
    """Load comprehensive ML models (Logistic Regression and Random Forest) trained on all datasets."""
    if not ML_AVAILABLE:
        return None, None, None
    
    models_dir = "models/ml"
    try:
        vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
        if not os.path.exists(vectorizer_path):
            st.error(f"Vectorizer not found at {vectorizer_path}")
            return None, None, None
        vectorizer = joblib.load(vectorizer_path)
        
        comprehensive_lr_path = os.path.join(models_dir, "logistic_regression.pkl")
        original_lr_path = os.path.join(models_dir, "logreg.pkl")
        
        if os.path.exists(comprehensive_lr_path):
            logreg = joblib.load(comprehensive_lr_path)
        elif os.path.exists(original_lr_path):
            logreg = joblib.load(original_lr_path)
        else:
            st.error("No Logistic Regression model found")
            return None, None, None
        
        rf_path = os.path.join(models_dir, "random_forest.pkl")
        if not os.path.exists(rf_path):
            st.error(f"Random Forest model not found at {rf_path}")
            return None, None, None
        rf = joblib.load(rf_path)
        
        return vectorizer, logreg, rf
    except Exception as e:
        st.error(f"Error loading ML models: {e}")
        return None, None, None


class CNNClassifier(nn.Module):
    """CNN-based Deep Learning Model (must match training architecture)."""

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
    """LSTM-based Deep Learning Model (must match training architecture)."""

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
    """
    Load CNN and LSTM deep learning models plus vocabulary for live inference.
    Requires `models/dl/vocab.json`, `cnn_best.pt`, and `lstm_best.pt`.
    If vocab.json is missing, creates a default vocabulary from the dataset.
    """
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
            'Disease Symptoms': '🦠',
            'Model Performance': '📈',
            'RAG vs Non-RAG': '🤖'
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
    elif page == "Disease Symptoms":
        disease_symptoms_page()
    elif page == "Model Performance":
        model_performance_page()
    elif page == "RAG vs Non-RAG":
        rag_comparison_page()


def classification_page():
    st.header("🔍 Medical Statement Classification")
    
    ai_available, ai_message, ai_provider = check_ai_availability()
    
    if not ai_available:
        st.warning(f"⚠️ AI Integration Unavailable: {ai_message}")
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
    auto_save = True
    do_augment = False
    augment_max = 5
    
    if st.button("Classify Statement", type="primary"):
        if not statement.strip():
            st.warning("Please enter a statement to classify.")
            return
        
        with st.spinner("Analyzing statement..."):
            stored_result = get_stored_label(statement, use_similarity=True)
            stored_label = None
            stored_similarity = 0.0
            if stored_result:
                stored_label, stored_similarity = stored_result
            
            # Always load all models (don't skip even if stored feedback exists)
            all_predictions = {}
            ordered_predictions = {}  # For ordered display

            # 1. Load BioBERT first (Transformer)
            if transformer_model is not None and tokenizer is not None:
                transformer_pred = predict_transformer(statement, tokenizer, transformer_model)
                if transformer_pred:
                    all_predictions["BioBERT"] = transformer_pred
                    ordered_predictions["BioBERT"] = transformer_pred
            else:
                st.warning("⚠️ BioBERT model not available")
            
            # 2. Load DL models (CNN, LSTM)
            if dl_bundle is not None:
                dl_preds = predict_dl_models(statement, dl_bundle)
                for model_name, preds in dl_preds.items():
                    if model_name != "DL Ensemble":  # Show individual models, not ensemble
                        all_predictions[model_name] = preds
                        ordered_predictions[model_name] = preds
            else:
                st.warning("⚠️ DL models not available")
            
            # 3. Load ML models last
            if vectorizer is not None and logreg is not None and rf is not None:
                ml_pred = predict_ml(statement, vectorizer, logreg, rf)
                if ml_pred:
                    all_predictions["ML Models"] = ml_pred
                    ordered_predictions["ML Models"] = ml_pred
            else:
                st.warning("⚠️ ML models not available")
            
            if not all_predictions:
                st.error("No trained models available. Please train models first.")
                return
            
            # Show stored feedback info if available, but still show all models
            if stored_label:
                if stored_similarity >= 1.0:
                    st.info(f"📝 Note: Stored label available: **{stored_label.title()}** (exact match from previous feedback) - Showing all model predictions below")
                else:
                    st.info(f"📝 Note: Stored label available: **{stored_label.title()}** (similar statement, {stored_similarity:.1%} similarity) - Showing all model predictions below")
            
            st.subheader("Classification Results")
            
            # Display models in order: BioBERT, DL models, ML models
            cols = st.columns(len(ordered_predictions))
            
            for i, (model_name, predictions) in enumerate(ordered_predictions.items()):
                with cols[i]:
                    st.markdown(f"**{model_name}**")
                    
                    best_label = max(predictions.keys(), key=lambda k: predictions[k])
                    best_confidence = predictions[best_label]
                    
                    for label, prob in predictions.items():
                        color = "green" if label == "credible" else "orange" if label == "misleading" else "red"
                        st.progress(prob, text=f"{label.title()}: {prob:.2%}")
                    
                    if best_confidence > 0.6:
                        st.success(f"Prediction: **{best_label.title()}** ({best_confidence:.2%})")
                    elif best_confidence > 0.45:
                        st.warning(f"Prediction: **{best_label.title()}** ({best_confidence:.2%})")
                    else:
                        st.info(f"Prediction: **{best_label.title()}** ({best_confidence:.2%}) - Low confidence")

            # Final prediction: Use BioBERT if available, otherwise use weighted ensemble
            st.markdown("---")
            st.subheader("Final Prediction")
            
            # Priority: BioBERT > Weighted Ensemble > Single Model
            if "BioBERT" in all_predictions:
                # Use BioBERT as final prediction (highest priority)
                biobert_preds = all_predictions["BioBERT"]
                final_label = max(biobert_preds.keys(), key=lambda k: biobert_preds[k])
                final_confidence = biobert_preds[final_label]
                st.caption("Using BioBERT (Transformer) model prediction")
            elif len(all_predictions) > 1:
                # Fallback to weighted ensemble
                model_weights = {
                    "BioBERT": 0.55,
                    "LSTM": 0.20,
                    "CNN": 0.10,
                    "ML Models": 0.15
                }
            
                weighted_scores = {"credible": 0.0, "misleading": 0.0, "false": 0.0}
                total_weight = 0.0
                
                for model_name, predictions in all_predictions.items():
                    weight = model_weights.get(model_name, 0.0)
                    if weight > 0:
                        for label in ["credible", "misleading", "false"]:
                            weighted_scores[label] += predictions.get(label, 0.0) * weight
                        total_weight += weight
                
                if total_weight > 0:
                    if total_weight < 1.0:
                        for label in weighted_scores:
                            weighted_scores[label] /= total_weight
                
                final_label = max(weighted_scores.keys(), key=lambda k: weighted_scores[k])
                final_confidence = weighted_scores[final_label]
                st.caption("Using weighted ensemble (BioBERT not available)")
            elif len(all_predictions) == 1:
                # Only one model available
                single_model_preds = list(all_predictions.values())[0]
                final_label = max(single_model_preds.keys(), key=lambda k: single_model_preds[k])
                final_confidence = single_model_preds[final_label]
                st.caption(f"Using {list(all_predictions.keys())[0]} model")
            else:
                final_label = None
                final_confidence = 0.0

            if final_label is not None:
                if final_confidence > 0.6:
                    st.success(f"**{final_label.title()}** ({final_confidence:.2%})")
                elif final_confidence > 0.45:
                    st.warning(f"**{final_label.title()}** ({final_confidence:.2%})")
                else:
                    st.info(f"**{final_label.title()}** ({final_confidence:.2%}) - Low confidence")
            else:
                st.info("No final prediction available.")
            
            st.subheader("Disease Analysis")
            base_predictions = {}
            if "ML Models" in all_predictions:
                base_predictions = all_predictions["ML Models"]
            elif all_predictions:
                first_model_name = list(all_predictions.keys())[0]
                base_predictions = all_predictions[first_model_name]

            context_result = classify_with_disease_context(statement, base_predictions) if base_predictions else None

            new_diseases_processed = {}
            
            try:
                candidate_list = []
                if context_result and context_result.get("diseases_found"):
                    candidate_list = context_result["diseases_found"]
                else:
                    extracted_candidates = extract_disease_from_text(statement)
                    if isinstance(extracted_candidates, str):
                        candidate_list = [extracted_candidates]
                    elif extracted_candidates:
                        candidate_list = extracted_candidates

                for disease in candidate_list:
                    if not disease or len(disease.split()) > 5:
                        continue
                    # Try to get symptoms from database
                    existing = get_symptoms(disease)
                    # Also check if disease exists in disease_symptoms.csv with case-insensitive matching
                    if not existing or str(existing).strip() == "" or str(existing).lower().startswith("symptoms not"):
                        # Try case-insensitive lookup
                        try:
                            import pandas as pd
                            if os.path.exists("disease_symptoms.csv"):
                                df = pd.read_csv("disease_symptoms.csv")
                                matched = df[df["disease_name"].str.lower() == disease.strip().lower()]
                                if not matched.empty and "symptoms" in matched.columns:
                                    existing = str(matched.iloc[0]["symptoms"])
                                    if existing and existing.strip() and not existing.lower().startswith("symptoms not"):
                                        # Found symptoms, add to processed diseases
                                        new_diseases_processed[disease] = {
                                            'symptoms': existing,
                                            'matched_disease': None
                                        }
                                        continue
                        except Exception as e:
                            print(f"Error checking disease_symptoms.csv for {disease}: {e}")
                    
                    # If symptoms found, add to processed diseases
                    if existing and str(existing).strip() and not str(existing).lower().startswith("symptoms not"):
                        new_diseases_processed[disease] = {
                            'symptoms': existing,
                            'matched_disease': None
                        }
                        continue
                    
                    # Only try AI if symptoms not found
                    if not existing or str(existing).strip() == "" or str(existing).lower().startswith("symptoms not"):
                        if ai_available:
                            try:
                                # Use the unknown disease handler for comprehensive processing
                                handler = get_unknown_disease_handler()
                                result = handler.handle_unknown_disease_query(disease, save_to_db=True)
                                
                                if not result.get('error') and result.get('similar_disease'):
                                    # Successfully found similar disease
                                    similar_disease = result['similar_disease']
                                    matched_disease_name = similar_disease['disease_name']
                                    symptoms = result['symptoms']
                                    
                                    new_diseases_processed[disease] = {
                                        'symptoms': symptoms,
                                        'matched_disease': matched_disease_name,
                                        'similarity_score': similar_disease['similarity_score'],
                                        'myths_with_replacement': result.get('myths', []),
                                        'facts_with_replacement': result.get('facts', [])
                                    }
                                    
                                    # Disease already saved by handler
                                    fact_statement = f"{disease} symptoms include: {symptoms}"
                                    save_to_dataset(fact_statement, "credible", "ai_generated", "medical_fact", disease)
                                    
                                elif result.get('symptoms'):
                                    # Got symptoms but no similar disease found
                                    symptoms = result['symptoms']
                                    new_diseases_processed[disease] = {
                                        'symptoms': symptoms,
                                        'matched_disease': None
                                    }
                                    upsert_disease(disease, symptoms)
                                    fact_statement = f"{disease} symptoms include: {symptoms}"
                                    save_to_dataset(fact_statement, "credible", "ai_generated", "medical_fact", disease)
                                    
                            except Exception as e:
                                # Fallback to old method
                                import traceback
                                print(f"Error using unknown disease handler for {disease}: {e}")
                                traceback.print_exc()
                                
                                try:
                                    symptoms = gemini_list_symptoms(disease)
                                    if symptoms and symptoms.strip():
                                        matcher = get_matcher()
                                        matched_diseases = matcher.match_symptoms_to_diseases(symptoms.strip(), top_k=1, min_similarity=0.3)
                                        
                                        if matched_diseases:
                                            top_match = matched_diseases[0]
                                            matched_disease_name = top_match['disease_name']
                                            matched_symptoms = top_match['disease_symptoms']
                                            
                                            new_diseases_processed[disease] = {
                                                'symptoms': matched_symptoms,
                                                'matched_disease': matched_disease_name
                                            }
                                            
                                            upsert_disease(disease, matched_symptoms)
                                            fact_statement = f"{disease} symptoms include: {matched_symptoms}"
                                            save_to_dataset(fact_statement, "credible", "ai_generated", "medical_fact", disease)
                                        else:
                                            new_diseases_processed[disease] = {
                                                'symptoms': symptoms.strip(),
                                                'matched_disease': None
                                            }
                                            upsert_disease(disease, symptoms.strip())
                                            fact_statement = f"{disease} symptoms include: {symptoms.strip()}"
                                            save_to_dataset(fact_statement, "credible", "ai_generated", "medical_fact", disease)
                                except Exception:
                                    pass
                        else:
                            symptoms = ""
            except Exception as e:
                import traceback
                print(f"Error processing new disease: {e}")
                traceback.print_exc()
            
            all_diseases = []
            if context_result and context_result.get("diseases_found"):
                all_diseases = context_result["diseases_found"]
            
            for new_disease in new_diseases_processed.keys():
                if new_disease not in all_diseases:
                    all_diseases.append(new_disease)
            
            if all_diseases:
                st.write("**Diseases detected:**", ", ".join(all_diseases))

                for disease in all_diseases:
                    if disease in new_diseases_processed:
                        symptoms = new_diseases_processed[disease]['symptoms']
                        matched_disease = new_diseases_processed[disease].get('matched_disease')
                    else:
                        # Try to get from context_result first
                        symptoms = context_result.get("disease_symptoms", {}).get(disease) if context_result else None
                        # If not found, try direct lookup from CSV
                        if not symptoms:
                            try:
                                symptoms = get_symptoms(disease)
                            except Exception:
                                symptoms = None
                        matched_disease = None
                    
                    if symptoms and str(symptoms).strip() and not str(symptoms).lower().startswith("symptoms not"):
                        st.write(f"**{disease.title()}** symptoms: {symptoms}")
                    else:
                        st.write(f"**{disease.title()}**: Symptoms not available")

                for disease in all_diseases:
                    if disease in new_diseases_processed:
                        disease_info = new_diseases_processed[disease]
                        matched_disease = disease_info.get('matched_disease')
                        
                        # Check if we have pre-replaced myths and facts from unknown disease handler
                        if disease_info.get('myths_with_replacement') or disease_info.get('facts_with_replacement'):
                            # Display custom myths and facts with replaced names
                            st.markdown(f"### 📊 Information about {disease.title()}")
                            
                            if matched_disease:
                                similarity_score = disease_info.get('similarity_score', 0)
                                st.info(f"ℹ️ Based on {similarity_score:.0%} similarity with {matched_disease}. Disease names have been replaced in the information below.")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown(f"**🔴 Common Myths**")
                                myths = disease_info.get('myths_with_replacement', [])
                                if myths:
                                    for i, myth in enumerate(myths[:5], 1):
                                        st.markdown(f"{i}. {myth}", unsafe_allow_html=True)
                                else:
                                    st.info("No myths found in database.")
                            
                            with col2:
                                st.markdown(f"**✅ Medical Facts**")
                                facts = disease_info.get('facts_with_replacement', [])
                                if facts:
                                    for i, fact in enumerate(facts[:5], 1):
                                        st.markdown(f"{i}. {fact}", unsafe_allow_html=True)
                                else:
                                    st.info("No facts found in database.")
                        
                        elif matched_disease:
                            # Use regular display with display_name
                            display_disease_myths_and_facts(matched_disease, ai_available, ai_provider, display_name=disease)
                        else:
                            # Regular display
                            display_disease_myths_and_facts(disease, ai_available, ai_provider)
                    else:
                        display_disease_myths_and_facts(disease, ai_available, ai_provider)
                    
            else:
                st.info("No specific diseases detected in the statement.")
            
            best_overall_label = final_label
            best_overall_confidence = final_confidence
            
            try:
                detected_diseases = []
                try:
                    detected = classify_with_disease_context(statement, base_predictions) if base_predictions else None
                    if detected and detected.get("diseases_found"):
                        detected_diseases = detected["diseases_found"]
                except Exception:
                    detected_diseases = []
                extracted_for_save = extract_disease_from_text(statement)
                if isinstance(extracted_for_save, list) and extracted_for_save:
                    extracted_for_save = extracted_for_save[0]
                disease_tag = detected_diseases[0] if detected_diseases else (extracted_for_save if extracted_for_save else None)
                topic_tag = "user_classified"
                
                if auto_save and best_overall_label:
                    save_to_dataset(statement, best_overall_label, "user_input", topic_tag, disease_tag)
                
                if do_augment and best_overall_label:
                    augmenter = MedicalDataAugmenter()
                    variants = augmenter.create_semantic_variations(statement)
                    variants = variants[:int(augment_max)]
                    saved_count = 0
                    for v in variants:
                        if v and v.strip() and v.strip() != statement.strip():
                            save_to_dataset(v.strip(), best_overall_label, "user_augmented", topic_tag, disease_tag)
                            saved_count += 1
                    st.info(f"Augmented and saved {saved_count} variants")
            except Exception as e:
                st.warning(f"Could not save/augment dataset: {e}")
            
            if ai_available and best_overall_label and best_overall_confidence > 0.4:
                st.subheader("🤖 AI Explanation")
                with st.spinner("Generating AI explanation..."):
                    try:
                        explanation = gemini_explain_classification(statement, best_overall_label, best_overall_confidence)
                        
                        # Use custom styled containers for better appearance
                        if best_overall_label == "credible":
                            st.markdown("""
                            <div style='background-color: #1e3a5f; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50; margin: 10px 0;'>
                                <h4 style='color: #4CAF50; margin-top: 0;'>✅ Credible Statement Explanation</h4>
                                <p style='color: #e0e0e0; line-height: 1.6;'>{}</p>
                            </div>
                            """.format(explanation.replace('\n', '<br>')), unsafe_allow_html=True)
                        elif best_overall_label == "misleading":
                            st.markdown("""
                            <div style='background-color: #3d2f1f; padding: 20px; border-radius: 10px; border-left: 5px solid #FF9800; margin: 10px 0;'>
                                <h4 style='color: #FF9800; margin-top: 0;'>⚠️ Misleading Statement Explanation</h4>
                                <p style='color: #e0e0e0; line-height: 1.6;'>{}</p>
                            </div>
                            """.format(explanation.replace('\n', '<br>')), unsafe_allow_html=True)
                        else:  # false
                            st.markdown("""
                            <div style='background-color: #3d1f1f; padding: 20px; border-radius: 10px; border-left: 5px solid #f44336; margin: 10px 0;'>
                                <h4 style='color: #f44336; margin-top: 0;'>❌ False Statement Explanation</h4>
                                <p style='color: #e0e0e0; line-height: 1.6;'>{}</p>
                            </div>
                            """.format(explanation.replace('\n', '<br>')), unsafe_allow_html=True)
                        
                        with st.spinner("Getting AI classification for comparison..."):
                            try:
                                ai_label = gemini_classify_statement(statement)
                                if ai_label and ai_label != best_overall_label:
                                    st.markdown("""
                                    <div style='background-color: #3d2f1f; padding: 15px; border-radius: 8px; border-left: 5px solid #FF9800; margin: 10px 0;'>
                                        <p style='color: #FF9800; margin: 0;'><strong>⚠️ Discrepancy Detected:</strong> Model predicted '<strong>{}</strong>' but AI classified as '<strong>{}</strong>'. Storing AI's classification for future use.</p>
                                    </div>
                                    """.format(best_overall_label.title(), ai_label.title()), unsafe_allow_html=True)
                                    
                                    # Extract topic and disease for proper tagging
                                    # extract_disease_from_text and classify_with_disease_context already imported at top
                                    
                                    # Get topic classification (use same logic as in process_and_label_data)
                                    topic_keywords = {
                                        'covid_19': ['covid', 'coronavirus', 'pandemic', 'sars-cov-2'],
                                        'vaccines': ['vaccine', 'vaccination', 'immunization', 'shot'],
                                        'cancer': ['cancer', 'tumor', 'chemotherapy', 'oncology'],
                                        'mental_health': ['mental', 'depression', 'anxiety', 'psychology'],
                                        'diabetes': ['diabetes', 'diabetic', 'blood sugar', 'glucose'],
                                        'heart_disease': ['heart', 'cardiac', 'cardiovascular', 'hypertension'],
                                        'nutrition': ['diet', 'nutrition', 'food', 'vitamin'],
                                    }
                                    text_lower = statement.lower()
                                    topic = 'general_health'
                                    for topic_name, keywords in topic_keywords.items():
                                        if any(keyword in text_lower for keyword in keywords):
                                            topic = topic_name
                                            break
                                    
                                    # Extract disease
                                    diseases = extract_disease_from_text(statement)
                                    disease_tag = diseases[0] if diseases else None
                                    
                                    # Store with proper tagging
                                    from utils.feedback_storage import store_feedback_with_tags
                                    store_feedback_with_tags(statement, ai_label, "ai_feedback", topic, disease_tag)
                                    
                                    st.markdown("""
                                    <div style='background-color: #1e3a5f; padding: 15px; border-radius: 8px; border-left: 5px solid #4CAF50; margin: 10px 0;'>
                                        <p style='color: #4CAF50; margin: 0;'><strong>✅ Stored correct label:</strong> <strong>{}</strong> with topic: {}, disease: {} - This will be used for this statement in the future.</p>
                                    </div>
                                    """.format(ai_label.title(), topic, disease_tag or 'N/A'), unsafe_allow_html=True)
                                elif ai_label and ai_label == best_overall_label:
                                    st.info(f"✓ AI classification matches model prediction: **{ai_label.title()}**")
                            except Exception as e:
                                import traceback
                                print(f"Error in AI comparison: {e}")
                                traceback.print_exc()
                                pass
                            
                    except Exception as e:
                        st.error(f"Could not generate AI explanation: {e}")
            elif not ai_available:
                st.subheader("🤖 AI Explanation")
                st.info("💡 AI explanation not available. Please set up Groq API key to enable AI-powered explanations for all classification results.")



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
                                    st.warning(f"⚠️ Bulk content generation failed: {message}")
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
                st.warning("⚠️ AI not available: " + ai_message)
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
                                st.warning(f"⚠️ Bulk content generation failed: {message}")
                                st.info("💡 Disease symptoms were still saved successfully")
                    else:
                        st.info("💡 AI not available for bulk content generation, but disease was saved successfully")
                    
                    st.rerun()
                except Exception as e:
                    st.error(f"Error adding disease: {e}")
            else:
                st.warning("Please provide both disease name and symptoms.")


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
    st.header("🤖 RAG vs Non-RAG Evaluation")
    st.markdown("---")
    
    comparison_path = "data/processed/rag_vs_nonrag_comparison.csv"
    detailed_path = "data/processed/rag_vs_nonrag_detailed.csv"
    
    if not os.path.exists(comparison_path):
        st.error("RAG comparison results not found. Please run the RAG evaluation first.")
        st.info("Run `python evaluate_rag_local.py` to generate the comparison results.")
        return
    
    # Load comparison data
    comparison_df = pd.read_csv(comparison_path)
    detailed_df = pd.read_csv(detailed_path) if os.path.exists(detailed_path) else None
    
    st.subheader("📊 Summary Comparison")
    st.caption("Average metrics across 50 QA pairs evaluated")
    
    # Get data
    rag_row = comparison_df[comparison_df['Model'] == 'RAG'].iloc[0]
    baseline_row = comparison_df[comparison_df['Model'] == 'Non-RAG (Baseline)'].iloc[0]
    
    # Display metrics comparison in a better format
    col1, col2, col3, col4 = st.columns(4)
    
    metrics_data = [
        ("Factuality", rag_row['Factuality'], baseline_row['Factuality']),
        ("Completeness", rag_row['Completeness'], baseline_row['Completeness']),
        ("Faithfulness", rag_row['Faithfulness'], baseline_row['Faithfulness']),
        ("Safety", rag_row['Safety'], baseline_row['Safety'])
    ]
    
    for i, (metric_name, rag_val, baseline_val) in enumerate(metrics_data):
        with [col1, col2, col3, col4][i]:
            improvement = rag_val - baseline_val
            improvement_pct = (improvement / baseline_val * 100) if baseline_val > 0 else 0
            
            st.markdown(f"### {metric_name}")
            
            # RAG score
            st.markdown(f"""
            <div style='background-color: #1e3a5f; padding: 15px; border-radius: 8px; border-left: 4px solid #4CAF50; margin-bottom: 10px;'>
                <p style='color: #4CAF50; margin: 0; font-weight: bold; font-size: 18px;'>RAG: {rag_val:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Baseline score
            st.markdown(f"""
            <div style='background-color: #3d2f1f; padding: 15px; border-radius: 8px; border-left: 4px solid #FF9800; margin-bottom: 10px;'>
                <p style='color: #FF9800; margin: 0; font-weight: bold; font-size: 18px;'>Baseline: {baseline_val:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Improvement
            if improvement > 0:
                st.markdown(f"""
                <div style='background-color: #1e3a1e; padding: 10px; border-radius: 6px;'>
                    <p style='color: #4CAF50; margin: 0;'>📈 Improvement: +{improvement:.4f} ({improvement_pct:.1f}%)</p>
                </div>
                """, unsafe_allow_html=True)
            elif improvement < 0:
                st.markdown(f"""
                <div style='background-color: #3d1f1f; padding: 10px; border-radius: 6px;'>
                    <p style='color: #f44336; margin: 0;'>📉 Difference: {improvement:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background-color: #2d2d2d; padding: 10px; border-radius: 6px;'>
                    <p style='color: #9e9e9e; margin: 0;'>➡️ No difference</p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Summary table
    st.subheader("📋 Comparison Table")
    summary_table = pd.DataFrame({
        'Metric': ['Factuality', 'Completeness', 'Faithfulness', 'Safety'],
        'RAG': [
            f"{rag_row['Factuality']:.4f}",
            f"{rag_row['Completeness']:.4f}",
            f"{rag_row['Faithfulness']:.4f}",
            f"{rag_row['Safety']:.4f}"
        ],
        'Non-RAG Baseline': [
            f"{baseline_row['Factuality']:.4f}",
            f"{baseline_row['Completeness']:.4f}",
            f"{baseline_row['Faithfulness']:.4f}",
            f"{baseline_row['Safety']:.4f}"
        ],
        'Improvement': [
            f"+{rag_row['Factuality'] - baseline_row['Factuality']:.4f}",
            f"+{rag_row['Completeness'] - baseline_row['Completeness']:.4f}",
            f"+{rag_row['Faithfulness'] - baseline_row['Faithfulness']:.4f}",
            f"{rag_row['Safety'] - baseline_row['Safety']:.4f}"
        ]
    })
    st.dataframe(summary_table, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Visual comparison chart
    st.subheader("📈 Visual Comparison")
    
    metrics = ['Factuality', 'Completeness', 'Faithfulness', 'Safety']
    rag_values = [rag_row[m] for m in metrics]
    baseline_values = [baseline_row[m] for m in metrics]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='RAG',
        x=metrics,
        y=rag_values,
        marker_color='#1f77b4',
        text=[f'{v:.4f}' for v in rag_values],
        textposition='auto',
    ))
    
    fig.add_trace(go.Bar(
        name='Non-RAG Baseline',
        x=metrics,
        y=baseline_values,
        marker_color='#ff7f0e',
        text=[f'{v:.4f}' for v in baseline_values],
        textposition='auto',
    ))
    
    fig.update_layout(
        title='RAG vs Non-RAG Performance Comparison',
        xaxis_title='Metrics',
        yaxis_title='Score',
        barmode='group',
        height=450,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=12),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Key findings
    st.subheader("🔍 Key Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("**RAG Advantages:**")
        if rag_row['Factuality'] > baseline_row['Factuality']:
            st.write(f"✓ **Factuality**: {((rag_row['Factuality'] / baseline_row['Factuality'] - 1) * 100):.1f}% higher")
        if rag_row['Completeness'] > baseline_row['Completeness']:
            st.write(f"✓ **Completeness**: {((rag_row['Completeness'] / baseline_row['Completeness'] - 1) * 100):.1f}% higher")
        if rag_row['Faithfulness'] > baseline_row['Faithfulness']:
            st.write(f"✓ **Faithfulness**: Perfect (1.0) - All answers grounded in retrieved sources")
        if rag_row['Safety'] == baseline_row['Safety']:
            st.write(f"✓ **Safety**: Both models are safe (1.0)")
    
    with col2:
        st.info("**Non-RAG Baseline:**")
        st.write(f"• Generic safe response")
        st.write(f"• No retrieval mechanism")
        st.write(f"• Faithfulness: 0.0 (no source grounding)")
        st.write(f"• Lower factuality and completeness")
    
    st.markdown("---")
    
    # Detailed results
    if detailed_df is not None and len(detailed_df) > 0:
        st.subheader("📋 Detailed Results")
        st.caption(f"Showing results for {len(detailed_df)} QA pairs")
        
        # Allow user to filter/view specific QA pairs
        with st.expander("View All QA Pair Results", expanded=False):
            # Display in a more readable format
            for idx, row in detailed_df.iterrows():
                st.markdown(f"### QA Pair {idx + 1}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**RAG Model:**")
                    st.write(f"**Question:** {row['question']}")
                    st.write(f"**Answer:** {row['rag_answer'][:200]}..." if len(str(row['rag_answer'])) > 200 else f"**Answer:** {row['rag_answer']}")
                    st.metric("Factuality", f"{row['rag_factuality']:.4f}")
                    st.metric("Completeness", f"{row['rag_completeness']:.4f}")
                    st.metric("Faithfulness", f"{row['rag_faithfulness']:.4f}")
                    st.metric("Safety", f"{row['rag_safety']:.4f}")
                
                with col2:
                    st.markdown("**Non-RAG Baseline:**")
                    st.write(f"**Question:** {row['question']}")
                    st.write(f"**Answer:** {row['nonrag_answer']}")
                    st.metric("Factuality", f"{row['nonrag_factuality']:.4f}")
                    st.metric("Completeness", f"{row['nonrag_completeness']:.4f}")
                    st.metric("Faithfulness", f"{row['nonrag_faithfulness']:.4f}")
                    st.metric("Safety", f"{row['nonrag_safety']:.4f}")
                
                st.markdown("---")
        
        # Summary statistics
        st.subheader("📊 Summary Statistics")
        
        summary_cols = st.columns(4)
        
        with summary_cols[0]:
            st.metric("Average RAG Factuality", f"{detailed_df['rag_factuality'].mean():.4f}")
            st.metric("Average Baseline Factuality", f"{detailed_df['nonrag_factuality'].mean():.4f}")
        
        with summary_cols[1]:
            st.metric("Average RAG Completeness", f"{detailed_df['rag_completeness'].mean():.4f}")
            st.metric("Average Baseline Completeness", f"{detailed_df['nonrag_completeness'].mean():.4f}")
        
        with summary_cols[2]:
            st.metric("Average RAG Faithfulness", f"{detailed_df['rag_faithfulness'].mean():.4f}")
            st.metric("Average Baseline Faithfulness", f"{detailed_df['nonrag_faithfulness'].mean():.4f}")
        
        with summary_cols[3]:
            st.metric("RAG Safety Rate", f"{detailed_df['rag_safety'].mean():.2%}")
            st.metric("Baseline Safety Rate", f"{detailed_df['nonrag_safety'].mean():.2%}")
    
    st.markdown("---")
    st.info("💡 **Note:** RAG system uses TF-IDF retrieval from credible medical texts. Non-RAG baseline provides generic safe responses without retrieval.")


if __name__ == "__main__":
    main()
