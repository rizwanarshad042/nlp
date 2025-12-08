"""
Medical Content Filter - Enhanced Version
Shared utility for filtering and validating medical-related content
Used by data_downloader.py and process_and_label_data.py

Features:
- 355+ disease names from disease_symptoms.csv
- Comprehensive medical terminology
- Advanced filtering for non-medical content
- News article detection
- Medical content scoring
"""

import re
import csv
import os
from typing import List, Set, Dict, Tuple

# Load disease names from disease_symptoms.csv
def load_disease_names() -> List[str]:
    """Load all disease names from disease_symptoms.csv"""
    disease_names = []
    csv_path = 'disease_symptoms.csv'
    
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    disease_name = row.get('disease_name', '').strip()
                    if disease_name and disease_name != 'disease_name':
                        # Add full name
                        disease_names.append(disease_name.lower())
                        
                        # Add variations (remove parentheses content, abbreviations)
                        if '(' in disease_name:
                            base_name = disease_name.split('(')[0].strip().lower()
                            disease_names.append(base_name)
                            
                            # Extract abbreviation
                            abbrev = disease_name[disease_name.find('(')+1:disease_name.find(')')].strip().lower()
                            if abbrev:
                                disease_names.append(abbrev)
        except Exception as e:
            print(f"Warning: Could not load disease names: {e}")
    
    return list(set(disease_names))  # Remove duplicates

# Load disease names
DISEASE_NAMES = load_disease_names()

# Comprehensive medical keywords
MEDICAL_KEYWORDS = [
    # Common diseases (in addition to loaded disease names)
    'covid', 'coronavirus', 'pandemic', 'sars-cov-2', 'influenza', 'flu', 'pneumonia',
    'tuberculosis', 'tb', 'diabetes', 'diabetic', 'cancer', 'tumor', 'tumour', 'oncology',
    'heart disease', 'cardiac', 'cardiovascular', 'hypertension', 'asthma', 'copd',
    'stroke', 'alzheimer', 'parkinson', 'dementia', 'epilepsy', 'seizure',
    'hepatitis', 'hiv', 'aids', 'malaria', 'dengue', 'chickenpox', 'measles',
    'mumps', 'rubella', 'monkeypox', 'ebola', 'zika', 'west nile',
    'disease', 'illness', 'condition', 'syndrome', 'disorder', 'infection',
    'virus', 'viral', 'bacteria', 'bacterial', 'pathogen', 'contagious', 'epidemic',
    'outbreak', 'transmission', 'communicable',
    
    # Medical terms and procedures
    'symptom', 'symptoms', 'diagnosis', 'diagnose', 'diagnostic', 'treatment', 'treat', 'cure',
    'medicine', 'medication', 'drug', 'pharmaceutical', 'prescription', 'dosage', 'dose',
    'doctor', 'physician', 'surgeon', 'nurse', 'patient', 'clinic', 'hospital',
    'medical', 'healthcare', 'health care', 'clinical', 'therapy', 'therapeutic',
    'surgery', 'surgical', 'operation', 'procedure', 'anesthesia', 'anesthetic',
    'immune', 'immunity', 'immunization', 'vaccine', 'vaccination', 'vaccinated',
    'antibiotic', 'antiviral', 'antifungal', 'antibacterial', 'antimicrobial',
    'prognosis', 'etiology', 'pathology', 'physiology', 'anatomy',
    
    # Body parts and systems
    'brain', 'cerebral', 'neurological', 'cognitive', 'liver', 'kidney', 'renal',
    'lung', 'respiratory', 'breathing', 'breathe', 'pulmonary', 'cardiac',
    'stomach', 'gastric', 'digestive', 'intestine', 'bowel', 'colon', 'rectum',
    'blood', 'circulatory', 'artery', 'vein', 'heart', 'pulse', 'blood pressure',
    'bone', 'skeletal', 'muscle', 'muscular', 'joint', 'arthritis', 'rheumatoid',
    'skin', 'dermatological', 'rash', 'wound', 'injury', 'fracture', 'trauma',
    'eye', 'vision', 'retinal', 'optic', 'ear', 'hearing', 'auditory',
    'thyroid', 'adrenal', 'pancreas', 'spleen', 'lymph', 'gland',
    'uterus', 'ovary', 'prostate', 'testicular', 'breast', 'cervical',
    
    # Health conditions and symptoms
    'fever', 'cough', 'pain', 'headache', 'nausea', 'vomiting', 'diarrhea',
    'fatigue', 'weakness', 'dizziness', 'shortness of breath', 'difficulty breathing',
    'chest pain', 'muscle pain', 'joint pain', 'sore throat', 'runny nose',
    'sneezing', 'congestion', 'itching', 'swelling', 'bleeding', 'inflammation',
    'weight loss', 'loss of appetite', 'difficulty swallowing', 'blurred vision',
    'numbness', 'tingling', 'confusion', 'memory loss', 'depression', 'anxiety',
    'insomnia', 'excessive thirst', 'frequent urination', 'abdominal pain',
    'constipation', 'blood in stool', 'blood in urine', 'yellow skin', 'jaundice',
    'palpitations', 'irregular heartbeat', 'cold hands', 'hot flashes',
    'night sweats', 'hair loss', 'skin changes', 'joint stiffness',
    'morning stiffness', 'wheezing', 'tightness', 'back pain', 'neck pain',
    'malaise', 'lethargy', 'discomfort', 'soreness', 'aching',
    
    # Medical procedures and tests
    'x-ray', 'mri', 'ct scan', 'ultrasound', 'biopsy', 'screening', 'test',
    'examination', 'exam', 'checkup', 'appointment', 'consultation',
    'laboratory', 'lab', 'blood test', 'urine test', 'stool test',
    'ecg', 'ekg', 'electrocardiogram', 'endoscopy', 'colonoscopy',
    'mammogram', 'pap smear', 'imaging', 'scan', 'radiology',
    
    # Health and wellness
    'health', 'wellness', 'wellbeing', 'fitness', 'exercise', 'diet', 'nutrition',
    'vitamin', 'mineral', 'supplement', 'herbal', 'alternative medicine',
    'prevention', 'prevent', 'heal', 'recovery', 'recover', 'rehabilitation',
    'chronic', 'acute', 'ailment', 'malady', 'affliction',
    'convalescence', 'remission', 'relapse', 'exacerbation',
    
    # Medical professionals and facilities
    'hospital', 'clinic', 'medical center', 'healthcare facility', 'emergency room',
    'er', 'icu', 'intensive care', 'ward', 'department', 'unit',
    'pharmacy', 'pharmacist', 'over the counter', 'otc',
    'specialist', 'cardiologist', 'neurologist', 'oncologist', 'pediatrician',
    'psychiatrist', 'dermatologist', 'radiologist', 'pathologist',
    
    # Medical research and evidence
    'clinical trial', 'study', 'research', 'evidence', 'peer reviewed',
    'medical journal', 'publication', 'findings', 'results', 'data',
    'randomized', 'placebo', 'control group', 'efficacy', 'safety',
    
    # Specialized medical terms
    'allergy', 'allergic', 'bronchitis', 'insulin', 'glucose', 'blood sugar',
    'malignancy', 'benign', 'malignant', 'metastasis', 'chemotherapy', 'chemo',
    'radiation', 'radiotherapy', 'transplant', 'organ', 'donor',
    'mental health', 'psychiatric', 'psychology', 'psychologist',
    'counseling', 'antidepressant', 'antipsychotic', 'sedative',
    'anemia', 'leukemia', 'lymphoma', 'sarcoma', 'carcinoma',
    'autoimmune', 'genetic', 'hereditary', 'congenital', 'acquired',
    'degenerative', 'progressive', 'terminal', 'palliative', 'hospice',
    
    # Public health terms
    'epidemic', 'pandemic', 'outbreak', 'quarantine', 'isolation',
    'contact tracing', 'social distancing', 'lockdown', 'public health',
    'cdc', 'who', 'world health organization', 'centers for disease control',
    'nih', 'national institutes of health', 'fda', 'food and drug administration',
    
    # Medical conditions categories
    'infectious', 'communicable', 'non-communicable', 'chronic disease',
    'rare disease', 'genetic disorder', 'metabolic disorder',
    'neurological disorder', 'cardiovascular disease', 'respiratory disease',
    'gastrointestinal', 'endocrine', 'hematological', 'immunological',
]

# Add disease names to medical keywords
MEDICAL_KEYWORDS.extend(DISEASE_NAMES)
MEDICAL_KEYWORDS = list(set(MEDICAL_KEYWORDS))  # Remove duplicates

# News article indicators
NEWS_INDICATORS = [
    # News agency names
    'reuters', 'associated press', 'ap news', 'bloomberg', 'cnn', 'bbc', 'fox news',
    'the new york times', 'washington post', 'wall street journal', 'usa today',
    'the guardian', 'independent', 'telegraph', 'daily mail', 'times of',
    'npr', 'abc news', 'cbs news', 'nbc news', 'msnbc', 'cnbc',
    
    # News article patterns
    'according to sources', 'officials said', 'spokesperson', 'press release',
    'breaking news', 'news report', 'news article', 'reported by', 'by reporter',
    'correspondent', 'journalist', 'news agency', 'news outlet',
    'exclusive interview', 'press conference', 'statement released',
    
    # News structure patterns
    'washington (reuters)', 'london (reuters)', 'new york (reuters)',
    '(reuters)', '(ap)', '(bloomberg)', '(cnn)', '(bbc)',
    'told reporters', 'told reuters', 'told the associated press',
    'in a statement', 'in a press conference', 'at a news conference',
    'sources familiar with', 'people with knowledge of',
]

# Non-medical keywords that indicate non-medical content
NON_MEDICAL_KEYWORDS = [
    # Legal/crime
    'police', 'officer', 'arrest', 'lawsuit', 'settlement', 'court', 'judge',
    'attorney', 'lawyer', 'legal', 'criminal', 'charged', 'trial', 'verdict',
    'prosecutor', 'defendant', 'plaintiff', 'litigation', 'indictment',
    'conviction', 'sentence', 'prison', 'jail', 'custody', 'bail',
    
    # Politics
    'politics', 'political', 'election', 'vote', 'voting', 'campaign',
    'senator', 'congressman', 'parliament', 'legislation', 'bill',
    'president', 'governor', 'mayor', 'democrat', 'republican',
    'congress', 'senate', 'house of representatives',
    
    # Sports
    'sports', 'game', 'player', 'team', 'score', 'match', 'tournament',
    'championship', 'league', 'season', 'coach', 'athlete', 'stadium',
    'playoffs', 'finals', 'quarterback', 'pitcher', 'goalkeeper',
    
    # Entertainment
    'entertainment', 'movie', 'film', 'actor', 'actress', 'celebrity', 'music', 'song',
    'concert', 'album', 'artist', 'performance', 'show', 'series', 'episode',
    'director', 'producer', 'oscar', 'emmy', 'grammy', 'box office',
    
    # Business/Finance
    'business', 'company', 'corporation', 'stock', 'market', 'finance',
    'investment', 'profit', 'revenue', 'earnings', 'shares', 'dividend',
    'merger', 'acquisition', 'ipo', 'nasdaq', 'dow jones', 'wall street',
    
    # Technology (non-medical)
    'software', 'computer', 'internet', 'website', 'app', 'smartphone',
    'technology', 'gadget', 'device', 'silicon valley', 'startup',
    'social media', 'facebook', 'twitter', 'instagram', 'youtube',
    
    # Food/Lifestyle (non-medical)
    'recipe', 'cooking', 'restaurant', 'food review', 'cuisine', 'chef',
    'travel', 'vacation', 'hotel', 'flight', 'tourism', 'destination',
    'fashion', 'style', 'clothing', 'designer', 'model',
    
    # Education (non-medical)
    'school', 'university', 'college', 'student', 'teacher', 'professor',
    'education', 'curriculum', 'graduation', 'degree',
    
    # Real estate
    'real estate', 'property', 'housing', 'mortgage', 'rent', 'landlord',
    
    # Weather
    'weather', 'forecast', 'temperature', 'rain', 'snow', 'storm', 'hurricane',
]

# Spam/promotional indicators
SPAM_INDICATORS = [
    'click here', 'buy now', 'limited time offer', 'act now', 'order today',
    'free shipping', 'discount', 'sale', 'promo code', 'coupon',
    'subscribe', 'follow us', 'like us', 'share this', 'retweet',
    'unsubscribe', 'opt out', 'terms and conditions', 'privacy policy',
    'cookie policy', 'copyright', '©', 'all rights reserved',
]


def is_medical_content(text: str, min_medical_keywords: int = 2, strict_mode: bool = False) -> bool:
    """
    Determine if text is medical-related by checking for medical keywords.
    Filters out news articles, spam, and non-medical content.
    
    Args:
        text: Text to check
        min_medical_keywords: Minimum number of medical keywords required
        strict_mode: If True, applies stricter filtering
    
    Returns:
        True if text is medical-related, False otherwise
    """
    if not text or len(text.strip()) < 20:
        return False
    
    text_lower = text.lower()
    
    # Check for spam/promotional content
    spam_count = sum(1 for indicator in SPAM_INDICATORS if indicator in text_lower)
    if spam_count >= 3:
        return False
    
    # Check for news article patterns
    news_indicator_count = sum(1 for indicator in NEWS_INDICATORS if indicator in text_lower)
    
    # If it looks like a news article, check if it's medical news
    if news_indicator_count >= 2:
        medical_count = sum(1 for keyword in MEDICAL_KEYWORDS if keyword in text_lower)
        
        # News articles need more medical content
        threshold = 5 if strict_mode else 4
        if medical_count < threshold:
            return False
        
        # Check if it's about medical topics
        medical_topic_indicators = [
            'medical', 'health', 'healthcare', 'treatment', 'disease', 'symptom',
            'diagnosis', 'patient', 'doctor', 'hospital', 'clinic', 'medicine',
            'research', 'study', 'clinical', 'therapy', 'vaccine', 'infection'
        ]
        medical_topic_count = sum(1 for indicator in medical_topic_indicators if indicator in text_lower)
        
        if medical_topic_count < 2:
            return False
    
    # Count medical keywords
    medical_count = sum(1 for keyword in MEDICAL_KEYWORDS if keyword in text_lower)
    
    # Check for non-medical keywords
    non_medical_count = sum(1 for keyword in NON_MEDICAL_KEYWORDS if keyword in text_lower)
    
    # If text has many non-medical keywords and few medical keywords, reject
    if non_medical_count >= 3 and medical_count < 2:
        return False
    
    # Check if text starts with news patterns
    first_200_chars = text_lower[:200]
    news_start_patterns = [
        'washington (reuters)', 'london (reuters)', 'new york (reuters)',
        '(reuters)', '(ap)', '(bloomberg)', '(cnn)', '(bbc)',
        'according to sources', 'officials said', 'breaking news'
    ]
    if any(pattern in first_200_chars for pattern in news_start_patterns):
        threshold = 5 if strict_mode else 4
        if medical_count < threshold:
            return False
    
    # Check if text is primarily about non-medical topics
    first_100_chars = text_lower[:100]
    if any(nm_keyword in first_100_chars for nm_keyword in ['police', 'lawsuit', 'settlement', 'attorney', 'court', 'arrest']):
        if medical_count < 3:
            return False
    
    # Check for news article structure
    if 'told reporters' in text_lower or 'told reuters' in text_lower or 'told the associated press' in text_lower:
        threshold = 5 if strict_mode else 4
        if medical_count < threshold:
            return False
    
    # Require minimum medical keywords
    return medical_count >= min_medical_keywords


def get_medical_keyword_count(text: str) -> int:
    """Count the number of medical keywords in text"""
    if not text:
        return 0
    
    text_lower = text.lower()
    return sum(1 for keyword in MEDICAL_KEYWORDS if keyword in text_lower)


def get_non_medical_keyword_count(text: str) -> int:
    """Count the number of non-medical keywords in text"""
    if not text:
        return 0
    
    text_lower = text.lower()
    return sum(1 for keyword in NON_MEDICAL_KEYWORDS if keyword in text_lower)


def is_news_article(text: str) -> bool:
    """Check if text appears to be a news article"""
    if not text:
        return False
    
    text_lower = text.lower()
    news_indicator_count = sum(1 for indicator in NEWS_INDICATORS if indicator in text_lower)
    
    return news_indicator_count >= 2


def is_spam_content(text: str) -> bool:
    """Check if text appears to be spam/promotional content"""
    if not text:
        return False
    
    text_lower = text.lower()
    spam_count = sum(1 for indicator in SPAM_INDICATORS if indicator in text_lower)
    
    return spam_count >= 3


def get_medical_content_score(text: str) -> float:
    """
    Calculate a medical content score (0.0 to 1.0)
    
    Returns:
        Score from 0.0 (non-medical) to 1.0 (highly medical)
    """
    if not text or len(text.strip()) < 20:
        return 0.0
    
    medical_count = get_medical_keyword_count(text)
    non_medical_count = get_non_medical_keyword_count(text)
    
    # Calculate score based on keyword ratio
    total_keywords = medical_count + non_medical_count
    if total_keywords == 0:
        return 0.0
    
    # Medical ratio
    medical_ratio = medical_count / total_keywords
    
    # Bonus for high medical keyword count
    if medical_count >= 5:
        medical_ratio = min(1.0, medical_ratio + 0.2)
    
    # Penalty for news articles
    if is_news_article(text):
        medical_ratio *= 0.7
    
    # Penalty for spam
    if is_spam_content(text):
        medical_ratio *= 0.5
    
    return medical_ratio


def filter_medical_texts(texts: List[str], min_keywords: int = 2, strict_mode: bool = False) -> List[str]:
    """Filter a list of texts to keep only medical content"""
    return [text for text in texts if is_medical_content(text, min_keywords, strict_mode)]


def get_medical_keywords() -> List[str]:
    """Get the list of medical keywords"""
    return MEDICAL_KEYWORDS.copy()


def get_disease_names() -> List[str]:
    """Get the list of disease names"""
    return DISEASE_NAMES.copy()


def get_non_medical_keywords() -> List[str]:
    """Get the list of non-medical keywords"""
    return NON_MEDICAL_KEYWORDS.copy()


def get_news_indicators() -> List[str]:
    """Get the list of news indicators"""
    return NEWS_INDICATORS.copy()


def analyze_text_medical_content(text: str) -> Dict:
    """
    Analyze text and return detailed medical content statistics
    
    Returns:
        Dictionary with analysis results
    """
    return {
        'is_medical': is_medical_content(text),
        'medical_keyword_count': get_medical_keyword_count(text),
        'non_medical_keyword_count': get_non_medical_keyword_count(text),
        'is_news_article': is_news_article(text),
        'is_spam': is_spam_content(text),
        'medical_score': get_medical_content_score(text),
        'text_length': len(text),
        'word_count': len(text.split())
    }


def get_statistics() -> Dict:
    """Get statistics about the filter"""
    return {
        'total_medical_keywords': len(MEDICAL_KEYWORDS),
        'total_disease_names': len(DISEASE_NAMES),
        'total_non_medical_keywords': len(NON_MEDICAL_KEYWORDS),
        'total_news_indicators': len(NEWS_INDICATORS),
        'total_spam_indicators': len(SPAM_INDICATORS)
    }


# Print statistics when module is loaded
if __name__ == '__main__':
    stats = get_statistics()
    print("="*60)
    print("MEDICAL CONTENT FILTER - STATISTICS")
    print("="*60)
    print(f"Medical Keywords: {stats['total_medical_keywords']}")
    print(f"Disease Names: {stats['total_disease_names']}")
    print(f"Non-Medical Keywords: {stats['total_non_medical_keywords']}")
    print(f"News Indicators: {stats['total_news_indicators']}")
    print(f"Spam Indicators: {stats['total_spam_indicators']}")
    print("="*60)
