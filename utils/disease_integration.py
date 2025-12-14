from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.disease_symptoms import get_symptoms, upsert_disease
from utils.labels import LABELS, normalize_label


def extract_disease_name_from_statement(text: str) -> Optional[str]:
    import re
    
    text_lower = text.lower()
    
    if len(text.split()) > 15:
        return None
    
    patterns = [
        r'(?:cures?|treats?|prevents?|causes?|helps?|fixes?|heals?)\s+([A-Z][a-zA-Z\s]{1,50}?)(?:\s+for\s+|\s+in\s+|\s|$|\.|,|!|\?)',
        r'([A-Z][a-zA-Z\s]{1,30}?)\s+(?:symptoms|disease|illness|condition)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            disease_name = matches[-1].strip()
            disease_name = re.sub(r'\s+(symptoms|disease|illness|condition|treatments?|remedies?)$', '', disease_name, flags=re.IGNORECASE)
            if disease_name and 2 < len(disease_name) < 50 and not any(word in disease_name.lower() for word in ['can', 'completely', 'replace', 'all', 'medical', 'treatments', 'for', 'serious', 'conditions']):
                return disease_name
    
    words = text.split()
    for i, word in enumerate(words):
        if word.lower() in ['cures', 'cure', 'treats', 'treat', 'prevents', 'prevent', 'causes', 'cause']:
            if i + 1 < len(words):
                potential_disease = []
                for j in range(i + 1, min(i + 5, len(words))):
                    next_word = words[j]
                    if next_word[0].isupper() or (potential_disease and next_word.lower() in ['disease', 'syndrome', 'sickness', 'virus', 'fever']):
                        if next_word.lower() not in ['can', 'completely', 'replace', 'all', 'medical', 'treatments', 'for', 'serious', 'conditions']:
                            potential_disease.append(next_word.rstrip('.,!?'))
                        else:
                            break
                    else:
                        break
                if potential_disease and len(' '.join(potential_disease)) < 50:
                    disease_name = ' '.join(potential_disease)
                    if len(disease_name) > 2:
                        return disease_name
    
    return None


def extract_disease_from_text(text: str) -> List[str]:
    # Disease mapping for better matching with database
    disease_mapping = {
        "covid-19": "COVID-19",
        "covid": "COVID-19", 
        "coronavirus": "COVID-19",
        "corona": "COVID-19",
        "sars-cov-2": "COVID-19",
        "influenza": "Influenza (Flu)",
        "flu": "Influenza (Flu)",
        "pneumonia": "Pneumonia",
        "tuberculosis": "Tuberculosis",
        "tb": "Tuberculosis",
        "diabetes": "Diabetes Type 2",
        "cancer": "Cancer",
        "heart disease": "Heart Disease",
        "hypertension": "Hypertension",
        "asthma": "Asthma",
        "copd": "COPD",
        "stroke": "Stroke",
        "alzheimer": "Alzheimer's Disease",
        "parkinson": "Parkinson's Disease",
        "hepatitis": "Hepatitis",
        "hiv": "HIV/AIDS",
        "aids": "HIV/AIDS",
        "malaria": "Malaria",
        "dengue": "Dengue Fever",
        "chickenpox": "Chickenpox",
        "measles": "Measles",
        "mumps": "Mumps",
        "rubella": "Rubella",
        "monkeypox": "Monkeypox",
        "ebola": "Ebola",
        "zika": "Zika Virus",
        "west nile": "West Nile Virus"
    }
    
    text_lower = text.lower()
    found_diseases = []
    
    for keyword, mapped_disease in disease_mapping.items():
        if keyword in text_lower:
            if mapped_disease not in found_diseases:
                found_diseases.append(mapped_disease)
    
    if not found_diseases:
        extracted_name = extract_disease_name_from_statement(text)
        if extracted_name and len(extracted_name.split()) <= 5:
            found_diseases.append(extracted_name)
    
    return found_diseases


def handle_unknown_disease(disease_name: str, statement: str) -> Tuple[str, str]:
    try:
        symptoms = "symptoms not available"
        explanation = "explanation not available"
        
        return symptoms, explanation
        
    except Exception as e:
        print(f"Error handling unknown disease {disease_name}: {e}")
        return "symptoms not available", "explanation not available"


def find_similar_diseases(symptoms_text: str, top_k: int = 3) -> List[Tuple[str, float]]:
    try:
        csv_path = "disease_symptoms.csv"
        if not os.path.exists(csv_path):
            return []
        df = pd.read_csv(csv_path)
        if df.empty or "disease_name" not in df.columns or "symptoms" not in df.columns:
            return []

        existing_symptoms = df["symptoms"].fillna("").astype(str).tolist()
        corpus = existing_symptoms + [symptoms_text or ""]

        # Vectorize
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
        X = vectorizer.fit_transform(corpus)

        # Compute cosine similarity of the last vector with all previous ones
        sims = cosine_similarity(X[-1], X[:-1]).flatten()

        # Rank top_k
        indices = sims.argsort()[::-1][:top_k]
        results: List[Tuple[str, float]] = []
        for idx in indices:
            results.append((str(df.iloc[idx]["disease_name"]), float(sims[idx])))
        return results
    except Exception as e:
        print(f"Error computing similar diseases: {e}")
        return []


def classify_with_disease_context(
    statement: str, 
    model_predictions: Dict[str, float],
    confidence_threshold: float = 0.7
) -> Dict[str, any]:
    # Extract diseases from statement
    diseases = extract_disease_from_text(statement)
    
    # Get best prediction from model
    best_label = max(model_predictions.keys(), key=lambda k: model_predictions[k])
    best_confidence = model_predictions[best_label]
    
    result = {
        "statement": statement,
        "predicted_label": best_label,
        "confidence": best_confidence,
        "diseases_found": diseases,
        "disease_symptoms": {},
        "explanation": None,
        "used_ai_fallback": False
    }
    
    # Check if model confidence is low
    if best_confidence < confidence_threshold:
        # No AI fallback available
        result["used_ai_fallback"] = False
    
    # Handle diseases
    for disease in diseases:
        symptoms = get_symptoms(disease)
        if symptoms is None:
            # Unknown disease - no AI fallback available
            symptoms, explanation = handle_unknown_disease(disease, statement)
            result["used_ai_fallback"] = True
        else:
            explanation = None
        
        result["disease_symptoms"][disease] = symptoms
        # Similar disease suggestions for context
        if symptoms and isinstance(symptoms, str):
            try:
                similar = find_similar_diseases(symptoms, top_k=3)
                if similar:
                    if "similar_diseases" not in result:
                        result["similar_diseases"] = {}
                    result["similar_diseases"][disease] = similar
            except Exception as e:
                print(f"Error finding similar diseases for {disease}: {e}")
        if explanation and not result["explanation"]:
            result["explanation"] = explanation
    
    # Generate explanation for misleading/false predictions
    if best_label in ["misleading", "false"] and not result["explanation"]:
        # No AI explanation available
        result["explanation"] = None
    
    return result


def get_top_myths_and_facts(dataset_path: str, top_k: int = 5) -> Dict[str, List[str]]:
    if not os.path.exists(dataset_path):
        return {"myths": [], "facts": []}
    
    df = pd.read_csv(dataset_path)
    if "text" not in df.columns or "label" not in df.columns:
        return {"myths": [], "facts": []}
    
    # Normalize labels
    df["label"] = df["label"].astype(str).map(lambda x: normalize_label(x))
    
    # Get misleading/false statements (myths)
    myths_df = df[df["label"].isin(["misleading", "false"])]
    myths = myths_df["text"].value_counts().head(top_k).index.tolist()
    
    # Get credible statements (facts)
    facts_df = df[df["label"] == "credible"]
    facts = facts_df["text"].value_counts().head(top_k).index.tolist()
    
    return {
        "myths": myths,
        "facts": facts
    }


def should_trigger_retraining(
    new_entries_count: int = 0,
    days_since_last_training: int = 0,
    retraining_threshold: int = 100,
    retraining_interval_days: int = 7
) -> bool:
    return (
        new_entries_count >= retraining_threshold or 
        days_since_last_training >= retraining_interval_days
    )


__all__ = [
    "extract_disease_name_from_statement",
    "extract_disease_from_text",
    "handle_unknown_disease", 
    "find_similar_diseases",
    "classify_with_disease_context",
    "get_top_myths_and_facts",
    "should_trigger_retraining"
]
