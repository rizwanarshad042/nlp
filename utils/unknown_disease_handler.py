"""
Unknown Disease Handler
Handles queries about diseases not in the dataset by:
1. Getting symptoms from AI for the unknown disease
2. Finding the most similar disease from existing data
3. Showing myths/facts with the original disease name replaced by the unknown disease name
"""

import pandas as pd
import os
import re
from typing import Dict, List, Optional, Tuple
from utils.symptom_disease_matcher import get_matcher
from utils.gemini_integration import gemini_list_symptoms
from utils.disease_symptoms import get_symptoms, upsert_disease, CSV_PATH


class UnknownDiseaseHandler:
    """Handle queries about diseases not in the dataset"""
    
    def __init__(self, dataset_path: str = "data/processed/medical_dataset.csv"):
        self.dataset_path = dataset_path
        self.disease_csv_path = CSV_PATH
        self.matcher = get_matcher()
    
    def is_disease_known(self, disease_name: str) -> bool:
        """Check if disease exists in our database"""
        if not os.path.exists(self.disease_csv_path):
            return False
        
        try:
            df = pd.read_csv(self.disease_csv_path)
            disease_lower = disease_name.lower().strip()
            
            # Check for exact or partial match
            for existing_disease in df['disease_name'].tolist():
                if disease_lower in existing_disease.lower() or existing_disease.lower() in disease_lower:
                    return True
            
            return False
        except Exception as e:
            print(f"Error checking disease: {e}")
            return False
    
    def get_ai_symptoms(self, disease_name: str) -> Optional[str]:
        """Get symptoms for unknown disease from AI"""
        try:
            # Use Gemini to get symptoms
            symptoms = gemini_list_symptoms(disease_name)
            
            if symptoms and len(symptoms) > 10:
                return symptoms
            
            return None
        except Exception as e:
            print(f"Error getting AI symptoms: {e}")
            return None
    
    def find_most_similar_disease(
        self, 
        unknown_disease: str, 
        symptoms: str,
        top_k: int = 1
    ) -> Optional[Dict]:
        """
        Find the most similar disease based on symptoms
        
        Returns:
            Dict with similar disease info or None
        """
        try:
            # Use the symptom matcher to find similar diseases
            matches = self.matcher.match_symptoms_to_diseases(
                symptoms, 
                top_k=top_k,
                min_similarity=0.1  # Lower threshold for better matching
            )
            
            if matches:
                return matches[0]  # Return top match
            
            return None
        except Exception as e:
            print(f"Error finding similar disease: {e}")
            return None
    
    def replace_disease_name_in_text(
        self, 
        text: str, 
        original_disease: str, 
        new_disease: str
    ) -> str:
        """
        Replace disease name in text while preserving context
        Handles various forms of the disease name
        """
        if not text or not original_disease:
            return text
        
        # Create variations of the disease name to replace
        original_variations = [
            original_disease,
            original_disease.lower(),
            original_disease.upper(),
            original_disease.title()
        ]
        
        # Also handle abbreviated forms
        if '(' in original_disease:
            base_name = original_disease.split('(')[0].strip()
            original_variations.extend([
                base_name,
                base_name.lower(),
                base_name.upper(),
                base_name.title()
            ])
        
        # Replace all variations
        result_text = text
        for variation in original_variations:
            if variation in result_text:
                # Preserve the case of the first occurrence
                if variation[0].isupper():
                    replacement = new_disease.title() if ' ' in new_disease else new_disease.capitalize()
                elif variation.isupper():
                    replacement = new_disease.upper()
                else:
                    replacement = new_disease.lower()
                
                result_text = result_text.replace(variation, replacement)
        
        return result_text
    
    def get_myths_and_facts_for_similar_disease(
        self,
        unknown_disease: str,
        similar_disease_name: str,
        max_myths: int = 10,
        max_facts: int = 10
    ) -> Dict[str, List[str]]:
        """
        Get myths and facts from similar disease with name replacement
        
        Returns:
            Dict with 'myths' and 'facts' lists with replaced disease names
        """
        result = {
            'myths': [],
            'facts': [],
            'original_disease': similar_disease_name,
            'queried_disease': unknown_disease
        }
        
        if not os.path.exists(self.dataset_path):
            return result
        
        try:
            df = pd.read_csv(self.dataset_path)
            
            if df.empty or 'text' not in df.columns or 'label' not in df.columns:
                return result
            
            # Find records related to the similar disease
            disease_lower = similar_disease_name.lower()
            relevant_rows = df[
                (df['text'].str.contains(disease_lower, case=False, na=False)) |
                (df.get('disease', pd.Series()).str.contains(disease_lower, case=False, na=False))
            ]
            
            # Get myths (misleading and false)
            myths_df = relevant_rows[relevant_rows['label'].isin(['misleading', 'false'])]
            for text in myths_df['text'].head(max_myths):
                replaced_text = self.replace_disease_name_in_text(
                    text, 
                    similar_disease_name, 
                    unknown_disease
                )
                result['myths'].append(replaced_text)
            
            # Get facts (credible)
            facts_df = relevant_rows[relevant_rows['label'] == 'credible']
            for text in facts_df['text'].head(max_facts):
                replaced_text = self.replace_disease_name_in_text(
                    text, 
                    similar_disease_name, 
                    unknown_disease
                )
                result['facts'].append(replaced_text)
            
        except Exception as e:
            print(f"Error getting myths and facts: {e}")
        
        return result
    
    def handle_unknown_disease_query(
        self,
        disease_name: str,
        save_to_db: bool = True
    ) -> Dict:
        """
        Complete workflow for handling unknown disease query
        
        Args:
            disease_name: Name of the unknown disease
            save_to_db: Whether to save the new disease to database
        
        Returns:
            Dict with:
            - is_known: bool
            - symptoms: str (AI-generated)
            - similar_disease: Dict
            - myths: List[str] (with replaced names)
            - facts: List[str] (with replaced names)
            - recommendation: str
        """
        result = {
            'is_known': False,
            'disease_name': disease_name,
            'symptoms': None,
            'similar_disease': None,
            'myths': [],
            'facts': [],
            'recommendation': '',
            'error': None
        }
        
        # Check if disease is already known
        if self.is_disease_known(disease_name):
            result['is_known'] = True
            result['recommendation'] = f"{disease_name} is in our database. Use regular search."
            return result
        
        # Step 1: Get symptoms from AI
        print(f"Getting symptoms for {disease_name} from AI...")
        symptoms = self.get_ai_symptoms(disease_name)
        
        if not symptoms:
            result['error'] = "Could not retrieve symptoms from AI"
            result['recommendation'] = "Unable to process this disease. Please try again or consult a medical professional."
            return result
        
        result['symptoms'] = symptoms
        
        # Step 2: Find most similar disease
        print(f"Finding similar disease based on symptoms...")
        similar_disease = self.find_most_similar_disease(disease_name, symptoms)
        
        if not similar_disease:
            result['error'] = "Could not find similar disease"
            result['recommendation'] = f"Could not find a similar disease for {disease_name}. Please consult a medical professional."
            return result
        
        result['similar_disease'] = similar_disease
        similar_disease_name = similar_disease['disease_name']
        
        # Step 3: Get myths and facts with name replacement
        print(f"Getting myths and facts from {similar_disease_name}...")
        myths_facts = self.get_myths_and_facts_for_similar_disease(
            disease_name,
            similar_disease_name
        )
        
        result['myths'] = myths_facts['myths']
        result['facts'] = myths_facts['facts']
        
        # Step 4: Generate recommendation
        similarity_score = similar_disease['similarity_score']
        result['recommendation'] = (
            f"Based on symptom analysis, {disease_name} shows {similarity_score:.1%} similarity "
            f"to {similar_disease_name}. The information below is adapted from {similar_disease_name} "
            f"data with disease names replaced. Please consult a healthcare professional for accurate "
            f"information specific to {disease_name}."
        )
        
        # Step 5: Optionally save to database
        if save_to_db and symptoms:
            try:
                upsert_disease(disease_name, symptoms)
                print(f"Saved {disease_name} to database")
            except Exception as e:
                print(f"Could not save to database: {e}")
        
        return result


# Global instance
_handler = None

def get_unknown_disease_handler() -> UnknownDiseaseHandler:
    """Get or create global handler instance"""
    global _handler
    if _handler is None:
        _handler = UnknownDiseaseHandler()
    return _handler


def handle_unknown_disease(disease_name: str, save_to_db: bool = True) -> Dict:
    """
    Convenience function to handle unknown disease queries
    
    Args:
        disease_name: Name of the disease to query
        save_to_db: Whether to save the disease to database
    
    Returns:
        Dict with myths, facts, and recommendations
    """
    handler = get_unknown_disease_handler()
    return handler.handle_unknown_disease_query(disease_name, save_to_db)
