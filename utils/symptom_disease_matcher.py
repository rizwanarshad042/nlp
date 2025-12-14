from __future__ import annotations

import pandas as pd
import os
import re
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.disease_symptoms import get_symptoms, upsert_disease, CSV_PATH


class SymptomDiseaseMatcher:
    
    def __init__(self, disease_csv_path: str = CSV_PATH):
        self.disease_csv_path = disease_csv_path
        self.disease_df = None
        self.symptom_vectors = None
        self.vectorizer = None
        self._load_disease_data()
    
    def _load_disease_data(self) -> None:
        if os.path.exists(self.disease_csv_path):
            try:
                self.disease_df = pd.read_csv(self.disease_csv_path)
                if not self.disease_df.empty and 'symptoms' in self.disease_df.columns:
                    self.vectorizer = TfidfVectorizer(
                        ngram_range=(1, 3),
                        min_df=1,
                        stop_words='english',
                        lowercase=True
                    )
                    symptom_texts = self.disease_df['symptoms'].fillna('').astype(str).tolist()
                    self.symptom_vectors = self.vectorizer.fit_transform(symptom_texts)
            except Exception as e:
                print(f"Error loading disease data: {e}")
                self.disease_df = pd.DataFrame()
    
    def extract_symptoms_from_text(self, text: str) -> List[str]:
        symptom_keywords = [
            'fever', 'cough', 'pain', 'headache', 'nausea', 'vomiting', 'diarrhea',
            'fatigue', 'weakness', 'dizziness', 'shortness of breath', 'difficulty breathing',
            'chest pain', 'muscle pain', 'joint pain', 'sore throat', 'runny nose',
            'sneezing', 'congestion', 'rash', 'itching', 'swelling', 'bleeding',
            'weight loss', 'loss of appetite', 'difficulty swallowing', 'blurred vision',
            'numbness', 'tingling', 'seizures', 'confusion', 'memory loss',
            'depression', 'anxiety', 'insomnia', 'excessive thirst', 'frequent urination',
            'abdominal pain', 'constipation', 'blood in stool', 'blood in urine',
            'yellow skin', 'jaundice', 'palpitations', 'irregular heartbeat',
            'cold hands', 'hot flashes', 'night sweats', 'hair loss', 'skin changes',
            'joint stiffness', 'morning stiffness', 'wheezing', 'tightness',
            'back pain', 'neck pain', 'shoulder pain', 'knee pain', 'stiff neck'
        ]
        
        text_lower = text.lower()
        found_symptoms = []
        
        if ',' in text:
            parts = [part.strip() for part in text.split(',')]
            for part in parts:
                part_lower = part.lower()
                for symptom in symptom_keywords:
                    if symptom in part_lower and symptom not in found_symptoms:
                        found_symptoms.append(symptom)
                        break
                if part_lower in [s.lower() for s in symptom_keywords] and part_lower not in found_symptoms:
                    found_symptoms.append(part_lower)
        
        for symptom in symptom_keywords:
            if symptom in text_lower and symptom not in found_symptoms:
                found_symptoms.append(symptom)
        
        return found_symptoms
    
    def match_symptoms_to_diseases(
        self, 
        user_symptom_text: str, 
        top_k: int = 5,
        min_similarity: float = 0.3
    ) -> List[Dict]:
        if self.disease_df is None or self.disease_df.empty:
            return []
        
        try:
            extracted_symptoms = self.extract_symptoms_from_text(user_symptom_text)
            
            user_vector = self.vectorizer.transform([user_symptom_text])
            
            similarities = cosine_similarity(user_vector, self.symptom_vectors).flatten()
            
            top_indices = similarities.argsort()[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] >= min_similarity:
                    disease_name = str(self.disease_df.iloc[idx]['disease_name'])
                    disease_symptoms = str(self.disease_df.iloc[idx]['symptoms'])
                    
                    matching_symptoms = []
                    for symptom in extracted_symptoms:
                        if symptom in disease_symptoms.lower():
                            matching_symptoms.append(symptom)
                    
                    match_score = (len(matching_symptoms) / max(len(extracted_symptoms), 1) 
                                 if extracted_symptoms else 0)
                    
                    results.append({
                        'disease_name': disease_name,
                        'similarity_score': float(similarities[idx]),
                        'match_score': match_score,
                        'disease_symptoms': disease_symptoms,
                        'matching_symptoms': matching_symptoms,
                        'user_symptoms': extracted_symptoms
                    })
            
            results.sort(key=lambda x: (x['similarity_score'] + x['match_score']) / 2, reverse=True)
            
            return results
            
        except Exception as e:
            print(f"Error matching symptoms: {e}")
            return []
    
    def get_disease_evidence(
        self, 
        disease_name: str,
        dataset_path: str = "data/processed/medical_dataset.csv"
    ) -> Dict[str, List[str]]:
        evidence = {'myths': [], 'facts': []}
        
        if not os.path.exists(dataset_path):
            return evidence
        
        try:
            df = pd.read_csv(dataset_path)
            
            if df.empty or 'text' not in df.columns or 'label' not in df.columns:
                return evidence
            
            disease_lower = disease_name.lower()
            relevant_rows = df[df['text'].str.contains(disease_lower, case=False, na=False)]
            
            myths_df = relevant_rows[relevant_rows['label'].isin(['misleading', 'false'])]
            evidence['myths'] = myths_df['text'].head(10).tolist()
            
            facts_df = relevant_rows[relevant_rows['label'] == 'credible']
            evidence['facts'] = facts_df['text'].head(10).tolist()
            
        except Exception as e:
            print(f"Error getting disease evidence: {e}")
        
        return evidence
    
    def diagnose_from_text(
        self, 
        user_text: str,
        top_k: int = 3
    ) -> Dict:
        result = {
            'matched_diseases': [],
            'user_symptoms': [],
            'evidence': {},
            'recommendation': 'Please consult a healthcare professional for proper diagnosis.'
        }
        
        # Extract and match symptoms
        extracted_symptoms = self.extract_symptoms_from_text(user_text)
        result['user_symptoms'] = extracted_symptoms
        
        if not extracted_symptoms:
            result['recommendation'] = 'Could not identify clear symptoms. Please provide more detailed symptom descriptions.'
            return result
        
        matched_diseases = self.match_symptoms_to_diseases(user_text, top_k=top_k)
        result['matched_diseases'] = matched_diseases
        
        if matched_diseases:
            top_disease = matched_diseases[0]['disease_name']
            result['evidence'] = self.get_disease_evidence(top_disease)
            
            if matched_diseases[0]['similarity_score'] > 0.6:
                result['recommendation'] = (
                    f"Based on your symptoms, there is a strong possibility of {top_disease}. "
                    f"Matching symptoms: {', '.join(matched_diseases[0]['matching_symptoms'])}. "
                    f"Please seek professional medical advice immediately."
                )
            elif matched_diseases[0]['similarity_score'] > 0.4:
                result['recommendation'] = (
                    f"Your symptoms may be related to {top_disease} or similar conditions. "
                    f"Consult with a healthcare provider for proper evaluation."
                )
        
        return result


_matcher = None

def get_matcher() -> SymptomDiseaseMatcher:
    """Get or create global matcher instance"""
    global _matcher
    if _matcher is None:
        _matcher = SymptomDiseaseMatcher()
    return _matcher


def diagnose_symptoms(user_text: str, top_k: int = 3) -> Dict:
    """
    Convenience function for symptom-based diagnosis
    
    Args:
        user_text: User's symptom description
        top_k: Number of top diseases to return
        
    Returns:
        Dict with diagnosis results, evidence, and recommendations
    """
    matcher = get_matcher()
    return matcher.diagnose_from_text(user_text, top_k=top_k)
