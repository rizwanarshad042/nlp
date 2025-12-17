import os
import pandas as pd
import numpy as np
import random
import re
from typing import List, Dict, Tuple
import json
from datetime import datetime

class MedicalDataAugmenter:
    def __init__(self):
        self.augmented_data = []
        
        # Medical terminology variations
        self.medical_synonyms = {
            'covid': ['covid-19', 'coronavirus', 'sars-cov-2', 'pandemic virus'],
            'vaccine': ['vaccination', 'immunization', 'shot', 'injection'],
            'doctor': ['physician', 'medical practitioner', 'healthcare provider', 'clinician'],
            'medicine': ['medication', 'drug', 'pharmaceutical', 'treatment'],
            'cancer': ['malignancy', 'tumor', 'neoplasm', 'oncology'],
            'depression': ['mental illness', 'psychological disorder', 'mood disorder'],
            'diabetes': ['diabetic condition', 'blood sugar disorder', 'glucose disorder'],
            'heart': ['cardiac', 'cardiovascular', 'circulatory system'],
            'brain': ['cerebral', 'neurological', 'cognitive'],
            'immune': ['immunity', 'immune system', 'defense system'],
            'infection': ['bacterial', 'viral', 'pathogenic', 'contagious'],
            'surgery': ['operation', 'surgical procedure', 'medical intervention'],
            'hospital': ['medical center', 'healthcare facility', 'clinic'],
            'patient': ['sick person', 'individual', 'person receiving treatment'],
            'treatment': ['therapy', 'intervention', 'medical care', 'healing process']
        }
        
        # Misinformation patterns to augment
        self.misinformation_patterns = [
            "This is a known fact that",
            "Studies have proven that",
            "Medical research shows",
            "Doctors are hiding the truth about",
            "The real cure for",
            "Big pharma doesn't want you to know",
            "Natural remedies are better than",
            "Alternative medicine can cure",
            "The government is lying about",
            "Evidence suggests that",
            "Many people don't know that",
            "The medical establishment won't tell you",
            "Scientists have discovered that",
            "Clinical trials prove that",
            "Patients have reported that"
        ]
        
        # Credible information patterns
        self.credible_patterns = [
            "According to medical research",
            "Scientific studies indicate",
            "Healthcare professionals recommend",
            "Evidence-based medicine shows",
            "Clinical guidelines suggest",
            "Medical consensus is that",
            "Peer-reviewed studies confirm",
            "Healthcare authorities state",
            "Medical professionals advise",
            "Research findings demonstrate",
            "Clinical evidence supports",
            "Medical literature indicates",
            "Healthcare guidelines recommend",
            "Scientific evidence shows",
            "Medical experts agree that"
        ]
        
    def load_comprehensive_data(self):
        comprehensive_path = "data/processed/medical_dataset.csv"
        if os.path.exists(comprehensive_path):
            df = pd.read_csv(comprehensive_path)
            print(f"Loaded comprehensive dataset: {len(df)} records")
            return df
        else:
            print("Comprehensive dataset not found. Please run data integration first.")
            return None
    
    def synonym_replacement(self, text: str, num_replacements: int = 2) -> str:
        words = text.split()
        augmented_text = words.copy()
        
        # Find medical terms to replace
        medical_terms_found = []
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?;:"')
            for medical_term, synonyms in self.medical_synonyms.items():
                if medical_term in word_lower:
                    medical_terms_found.append((i, synonyms))
                    break
        
        # Replace with synonyms
        if medical_terms_found and len(medical_terms_found) > 0:
            replacements = min(num_replacements, len(medical_terms_found))
            selected_terms = random.sample(medical_terms_found, replacements)
            
            for idx, synonyms in selected_terms:
                original_word = words[idx]
                synonym = random.choice(synonyms)
                if original_word.isupper():
                    synonym = synonym.upper()
                elif original_word.istitle():
                    synonym = synonym.title()
                augmented_text[idx] = synonym
        
        return ' '.join(augmented_text)
    
    def back_translation_style(self, text: str) -> str:
        variations = []
        
        patterns = [
            (r'\bI\b', 'We'),
            (r'\bmy\b', 'our'),
            (r'\bme\b', 'us'),
            (r'\bam\b', 'are'),
            (r'\bis\b', 'are'),
            (r'\bwas\b', 'were'),
            (r'\bthe\b', 'this'),
            (r'\ba\b', 'one'),
        ]
        
        augmented_text = text
        for pattern, replacement in random.sample(patterns, min(3, len(patterns))):
            if random.random() < 0.5:
                augmented_text = re.sub(pattern, replacement, augmented_text, flags=re.IGNORECASE)
        
        return augmented_text
    
    def add_context_variations(self, text: str, label: str) -> str:
        if label == 'false':
            pattern = random.choice(self.misinformation_patterns)
            return f"{pattern} {text.lower()}"
        elif label == 'credible':
            pattern = random.choice(self.credible_patterns)
            return f"{pattern} {text.lower()}"
        else:
            return text
    
    def create_semantic_variations(self, text: str) -> List[str]:
        variations = []
        
        variations.append(text)
        
        # Synonym replacement
        variations.append(self.synonym_replacement(text, num_replacements=2))
        variations.append(self.synonym_replacement(text, num_replacements=3))
        
        variations.append(self.back_translation_style(text))
        
        variations = list(set(variations))
        
        return variations
    
    def augment_dataset(self, df: pd.DataFrame, augmentation_factor: int = 5) -> pd.DataFrame:
        print(f"Starting data augmentation with factor {augmentation_factor}...")
        
        augmented_records = []
        
        for idx, row in df.iterrows():
            text = row['text']
            label = row['label']
            source = row.get('source', 'augmented')
            topic = row.get('topic', 'general_health')
            
            if label == 'unknown':
                continue
            
            augmented_records.append(row.to_dict())
            
            variations = self.create_semantic_variations(text)
            
            for i, variation in enumerate(variations):
                if i < augmentation_factor - 1:
                    context_variation = self.add_context_variations(variation, label)
                    
                    augmented_record = {
                        'text': context_variation,
                        'label': label,
                        'source': f"{source}_augmented_{i+1}",
                        'topic': topic,
                        'text_length': len(context_variation),
                        'word_count': len(context_variation.split()),
                        'sentence_count': len(context_variation.split('.')),
                        'avg_word_length': np.mean([len(word) for word in context_variation.split()]),
                        'exclamation_count': context_variation.count('!'),
                        'question_count': context_variation.count('?'),
                        'caps_ratio': sum(1 for c in context_variation if c.isupper()) / len(context_variation) if context_variation else 0
                    }
                    augmented_records.append(augmented_record)
        
        augmented_df = pd.DataFrame(augmented_records)
        print(f"Augmented dataset: {len(df)} -> {len(augmented_df)} records")
        
        return augmented_df
    
    def create_synthetic_medical_data(self) -> pd.DataFrame:
        synthetic_data = []
        
        false_claims = [
            "Drinking bleach cures COVID-19 infection completely",
            "Vaccines contain microchips for government tracking",
            "Cancer can be cured by avoiding all sugar intake",
            "Mental illness is just a sign of weak character",
            "All pharmaceutical drugs are designed to make you sick",
            "Natural immunity is always better than vaccine immunity",
            "Doctors are paid by drug companies to prescribe medications",
            "Essential oils can replace all medical treatments",
            "Chemotherapy always makes cancer worse",
            "Antibiotics work against all types of infections",
            "Homeopathy is scientifically proven to be effective",
            "Medical devices are designed to harm patients",
            "Clinical trials are just experiments on human guinea pigs",
            "The healthcare system is designed to keep people sick",
            "Emergency rooms are dangerous and should be avoided",
            "Surgery is never necessary and always causes more harm",
            "Pharmacists only care about making money from drugs",
            "Public health measures are government control tactics",
            "Telemedicine is never as good as in-person visits",
            "Medical ethics are just cover-ups for malpractice"
        ]
        
        # Synthetic credible medical information
        credible_info = [
            "Regular hand washing helps prevent the spread of infectious diseases",
            "Vaccines are tested extensively before approval for public use",
            "Early cancer detection improves treatment outcomes significantly",
            "Mental health conditions are real medical conditions that require treatment",
            "Prescription medications undergo rigorous safety testing before approval",
            "Natural immunity and vaccine immunity both provide protection against diseases",
            "Doctors follow evidence-based guidelines when prescribing medications",
            "Alternative medicine should be used as a complement to, not replacement for, conventional medicine",
            "Chemotherapy is an evidence-based treatment for many types of cancer",
            "Antibiotics are effective against bacterial infections, not viral infections",
            "Homeopathy lacks scientific evidence for effectiveness beyond placebo effects",
            "Medical devices are regulated and tested for safety before use",
            "Clinical trials follow strict ethical guidelines and safety protocols",
            "The healthcare system aims to improve patient health and outcomes",
            "Emergency rooms provide life-saving care for serious medical conditions",
            "Surgery is performed when medically necessary and beneficial",
            "Pharmacists provide important medication counseling and safety checks",
            "Public health measures are based on scientific evidence to protect communities",
            "Telemedicine can provide effective healthcare access in many situations",
            "Medical ethics ensure patient safety and professional standards"
        ]
        
        for i, claim in enumerate(false_claims):
            synthetic_data.append({
                'text': claim,
                'label': 'false',
                'source': 'synthetic_false',
                'topic': self._classify_topic(claim),
                'text_length': len(claim),
                'word_count': len(claim.split()),
                'sentence_count': len(claim.split('.')),
                'avg_word_length': np.mean([len(word) for word in claim.split()]),
                'exclamation_count': claim.count('!'),
                'question_count': claim.count('?'),
                'caps_ratio': sum(1 for c in claim if c.isupper()) / len(claim) if claim else 0
            })
        
        for i, info in enumerate(credible_info):
            synthetic_data.append({
                'text': info,
                'label': 'credible',
                'source': 'synthetic_credible',
                'topic': self._classify_topic(info),
                'text_length': len(info),
                'word_count': len(info.split()),
                'sentence_count': len(info.split('.')),
                'avg_word_length': np.mean([len(word) for word in info.split()]),
                'exclamation_count': info.count('!'),
                'question_count': info.count('?'),
                'caps_ratio': sum(1 for c in info if c.isupper()) / len(info) if info else 0
            })
        
        return pd.DataFrame(synthetic_data)
    
    def _classify_topic(self, text: str) -> str:
        text_lower = text.lower()
        
        topic_keywords = {
            'covid_19': ['covid', 'coronavirus', 'pandemic'],
            'vaccines': ['vaccine', 'vaccination', 'immunization'],
            'cancer': ['cancer', 'tumor', 'chemotherapy'],
            'mental_health': ['mental', 'depression', 'anxiety'],
            'pharmaceuticals': ['drug', 'medication', 'pharmaceutical'],
            'alternative_medicine': ['essential oils', 'homeopathy', 'natural'],
            'nutrition': ['diet', 'nutrition', 'food'],
            'emergency_medicine': ['emergency', 'trauma', 'urgent'],
            'surgery': ['surgery', 'operation', 'surgical'],
            'public_health': ['public health', 'community', 'epidemic']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return topic
        
        return 'general_health'
    
    def create_augmented_dataset(self) -> pd.DataFrame:
        print("Creating augmented dataset...")
        
        # Load original comprehensive data
        original_df = self.load_comprehensive_data()
        if original_df is None:
            return None
        
        # Augment original data with moderate factor
        print("Augmenting original dataset...")
        augmented_df = self.augment_dataset(original_df, augmentation_factor=4)
        
        # Create synthetic data
        print("Creating synthetic medical data...")
        synthetic_df = self.create_synthetic_medical_data()
        
        # Combine all datasets
        print("Combining all datasets...")
        combined_df = pd.concat([augmented_df, synthetic_df], ignore_index=True)
        
        # Remove duplicates
        combined_df = combined_df.drop_duplicates(subset=['text'], keep='first')
        
        # Shuffle the dataset
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Augmented dataset created: {len(combined_df)} total records")
        print(f"Label distribution:")
        print(combined_df['label'].value_counts())
        
        return combined_df


def main():
    """Main function to create augmented training dataset"""
    augmenter = MedicalDataAugmenter()
    augmented_df = augmenter.create_augmented_dataset()
    
    if augmented_df is not None:
        print("\n" + "="*60)
        print("AUGMENTED DATASET CREATION COMPLETE")
        print("="*60)
        print(f"Total records: {len(augmented_df)}")
        print(f"Original + Augmented + Synthetic data combined")
        print("Ready for advanced model training!")
    else:
        print("Failed to create augmented dataset")


if __name__ == "__main__":
    main()
