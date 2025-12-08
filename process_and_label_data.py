#!/usr/bin/env python3
"""
Data Processing and Labeling Script
Processes downloaded datasets and assigns labels based on source credibility
"""

import os
import sys
import pandas as pd
import json
import glob
import numpy as np
from pathlib import Path
from datetime import datetime
import re
import random
from collections import Counter, defaultdict

# Add utils to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import shared medical content filter
from utils.medical_content_filter import (
    is_medical_content,
    get_medical_keyword_count,
    get_medical_content_score,
    analyze_text_medical_content
)

DATA_DIR = 'general_medical_misinformation_data'
RAW_DATA_INPUT = 'data/processed/raw_downloaded_data.csv'
OUTPUT_PATH = 'data/processed/medical_dataset.csv'
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
MYTHS_OUTPUT = 'data/processed/top_myths.csv'
FACTS_OUTPUT = 'data/processed/top_facts.csv'
QA_OUTPUT = 'data/processed/qa_pairs_100.csv'
MAX_DATASET_SIZE = 52000
LABEL_TARGETS = {
    'credible': 32000,
    'false': 12000,
    'misleading': 8000
}
SOURCE_SAMPLE_LIMITS = {
    'kaggle_unknown': 20000,
    'kaggle_fake_true_news': 15000,
    'medical_news': 8000,
    'fakenewsnet': 8000,
    'monant_misinfo': 8000,
    'med_mmhl': 6000,
    'covid_misinfo_claims': 6000
}
random.seed(42)
np.random.seed(42)

CLAIM_KEYWORDS = [
    'cure', 'treat', 'prevent', 'cause', 'spread', 'kill', 'boost immunity',
    'home remedy', 'herbal', 'miracle', 'proven', 'guaranteed', 'therapy',
    'vaccine', 'shot', 'mask', 'garlic', 'ginger', 'detox', '5g', 'drink', 'oil',
    'immune', 'dosage', 'variant', 'virus', 'bacteria', 'antibiotic'
]

# Source-based labeling rules - All sources are medical-related
SOURCE_LABEL_MAP = {
    # Credible sources - Official Health Organizations
    'who_mythbusters': 'credible',
    'who_general_facts': 'credible',
    'who_covid_myths': 'credible',
    'who_covid_qa': 'credible',
    'cdc_facts': 'credible',
    'cdc_facts_tagged': 'credible',
    'cdc_general_topics': 'credible',
    'cdc_vaccine_facts': 'credible',
    'cdc_vaccine_myths': 'credible',
    'nih_health_info': 'credible',
    'niaid_diseases': 'credible',
    'nhs_health_qa': 'credible',
    'factcheck_health': 'credible',
    'snopes_health': 'credible',
    
    # Credible sources - Research and Datasets
    'biosentvec': 'credible',
    'biowordvec': 'credible',
    'covid_chestxray': 'credible',
    'kaggle_amazon_reviews': 'credible',  # Medical product reviews
    'kaggle_sqlite': 'credible',
    'kaggle_real_news': 'credible',
    'kaggle_news_dataset': 'credible',
    'medical_news': 'credible',
    'scifact': 'credible',
    'pubhealth': 'credible',
    'manual_labeling': 'credible',
    'clinician_review': 'credible',
    
    # Misleading/False sources - Misinformation Datasets
    'kaggle_fake_true_news': 'false',
    'monant_misinfo': 'false',
    'healthfact_dataset': 'false',  # Contains false claims for fact-checking
    'facebook_health_misinfo': 'false',
    'healthlies': 'misleading',
    'med_mmhl': 'misleading',
    'synthetic_false': 'false',
    'synthetic_misleading': 'misleading',
    'synthetic_true': 'credible',
    'fakenewsnet': 'false',
    'medical_fact_checking': 'credible',
    'covid_misinfo_claims': 'misleading',
    
    # AI generated
    'ai_generated': 'credible',  # Default, can be overridden
    'user_input': None,  # Requires manual review
    'user_augmented': None,
    'ai_feedback': None,
    'template_generated': 'misleading',  # Template myths
    
    # Default fallback
    'unknown': 'credible',  # Default to credible if source unknown
}

# Medical content filtering is now handled by utils/medical_content_filter.py
# This ensures consistency across all scripts

# Topic classification keywords
TOPIC_KEYWORDS = {
    'covid_19': ['covid', 'coronavirus', 'pandemic', 'sars-cov-2'],
    'vaccines': ['vaccine', 'vaccination', 'immunization', 'shot'],
    'cancer': ['cancer', 'tumor', 'chemotherapy', 'oncology'],
    'mental_health': ['mental', 'depression', 'anxiety', 'psychology'],
    'diabetes': ['diabetes', 'diabetic', 'blood sugar', 'glucose'],
    'heart_disease': ['heart', 'cardiac', 'cardiovascular', 'hypertension'],
    'nutrition': ['diet', 'nutrition', 'food', 'vitamin'],
    'general_health': []  # Default
}

def classify_topic(text: str) -> str:
    """Classify text into health topics"""
    text_lower = text.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if topic == 'general_health':
            continue
        if any(keyword in text_lower for keyword in keywords):
            return topic
    return 'general_health'

def extract_disease_from_text(text: str) -> str:
    """Extract disease name from text"""
    disease_mapping = {
        "covid-19": "COVID-19", "covid": "COVID-19", "coronavirus": "COVID-19",
        "influenza": "Influenza (Flu)", "flu": "Influenza (Flu)",
        "diabetes": "Diabetes Type 2", "cancer": "Cancer",
        "heart disease": "Heart Disease", "hypertension": "Hypertension",
        "asthma": "Asthma", "copd": "COPD", "stroke": "Stroke",
        "alzheimer": "Alzheimer's Disease", "parkinson": "Parkinson's Disease"
    }
    
    text_lower = text.lower()
    for keyword, disease in disease_mapping.items():
        if keyword in text_lower:
            return disease
    return None

def split_into_sentences(text: str) -> list:
    """Lightweight sentence splitter for claim extraction."""
    if not isinstance(text, str):
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text.replace('\n', ' '))
    return [sent.strip() for sent in sentences if sent.strip()]

def is_claim_sentence(sentence: str) -> bool:
    """Heuristic check to see if a sentence makes a medical claim."""
    if len(sentence) < 30:
        return False
    lower = sentence.lower()
    return any(keyword in lower for keyword in CLAIM_KEYWORDS)

def expand_records_to_claims(records):
    """Explode long texts into individual claim sentences."""
    expanded = []
    for rec in records:
        sentences = split_into_sentences(rec.get('text', ''))
        claim_found = False
        for sentence in sentences:
            if is_claim_sentence(sentence):
                claim_found = True
                new_rec = rec.copy()
                new_rec['text'] = sentence[:10000]
                expanded.append(new_rec)
        if not claim_found and rec.get('text'):
            expanded.append(rec)
    return expanded

def normalize_label_value(label_text: str, default_label='credible'):
    """Normalize textual labels into credible/misleading/false."""
    if not label_text:
        return default_label
    label_text = str(label_text).lower()
    if label_text in ['true', 'support', 'supports', 'proven', 'accurate', 'positive']:
        return 'credible'
    if label_text in ['false', 'fake', 'refute', 'refutes', 'negative', 'hoax', 'inaccurate']:
        return 'false'
    if label_text in ['misleading', 'partly true', 'mixture', 'unproven', 'not enough info']:
        return 'misleading'
    return default_label

def generate_synthetic_claims(max_per_disease: int = 15):
    """Generate synthetic claims from disease_symptoms.csv for data balancing."""
    synthetic_records = []
    csv_path = Path('disease_symptoms.csv')
    if not csv_path.exists():
        return synthetic_records
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"   ⚠ Could not read {csv_path}: {exc}")
        return synthetic_records
    
    for _, row in df.iterrows():
        disease_name = str(row.get('disease_name', '')).strip()
        symptoms = str(row.get('symptoms', '')).strip()
        if not disease_name or not symptoms:
            continue
        
        symptom_list = [s.strip() for s in symptoms.split(',') if s.strip()]
        topic = classify_topic(disease_name)
        disease_label = disease_name
        
        credible_templates = [
            f"{disease_name} cannot be cured overnight; seek professional care.",
            f"Doctors recommend evidence-based treatment for {disease_name}, not home remedies.",
            f"Symptoms like {', '.join(symptom_list[:3])} require medical evaluation.",
            f"No herbal remedy has been proven to cure {disease_name}.",
            f"Vaccination and prescribed medication are the safest approach for {disease_name}."
        ]
        misleading_templates = [
            f"Healthy diet alone can reverse {disease_name} without medicine.",
            f"Breathing exercises are enough to treat {disease_name}.",
            f"Only people with weak immunity get {disease_name}.",
            f"{disease_name} is often misdiagnosed; most people do not really have it.",
            f"Natural supplements guarantee complete recovery from {disease_name}."
        ]
        false_templates = [
            f"Garlic water cures {disease_name} instantly.",
            f"5G towers spread {disease_name}.",
            f"Crystals protect you completely from {disease_name}.",
            f"{disease_name} is a hoax created by pharmaceutical companies.",
            f"Drinking bleach eliminates {disease_name} within hours."
        ]
        
        templates = [
            ('synthetic_true', 'credible', credible_templates),
            ('synthetic_misleading', 'misleading', misleading_templates),
            ('synthetic_false', 'false', false_templates),
        ]
        
        for source_name, label, sentence_list in templates:
            count = 0
            for sentence in sentence_list:
                synthetic_records.append({
                    'text': sentence[:10000],
                    'label': label,
                    'source': source_name,
                    'topic': topic,
                    'disease': disease_label,
                    'timestamp': datetime.now().isoformat()
                })
                count += 1
                if count >= max_per_disease // len(templates):
                    break
    print(f"   ✓ Generated {len(synthetic_records)} synthetic statements from disease_symptoms.csv")
    return synthetic_records

def apply_source_limits(df: pd.DataFrame) -> pd.DataFrame:
    """Limit the maximum number of samples per source to avoid dominance."""
    frames = []
    for source, group in df.groupby('source'):
        limit = SOURCE_SAMPLE_LIMITS.get(source)
        if limit and len(group) > limit:
            group = group.sample(n=limit, random_state=42)
        frames.append(group)
    limited_df = pd.concat(frames, ignore_index=True)
    print(f"   ✓ Applied per-source limits. Records after limiting: {len(limited_df)}")
    return limited_df

def rebalance_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Downsample/oversample labels to hit target counts."""
    balanced_frames = []
    for label, target in LABEL_TARGETS.items():
        label_df = df[df['label'] == label]
        if label_df.empty:
            print(f"   ⚠ No samples found for label '{label}'. Skipping target balancing for it.")
            continue
        if len(label_df) > target:
            label_df = label_df.sample(n=target, random_state=42)
        elif len(label_df) < target:
            needed = target - len(label_df)
            extra = label_df.sample(n=needed, replace=True, random_state=42)
            label_df = pd.concat([label_df, extra], ignore_index=True)
        balanced_frames.append(label_df)
    balanced_df = pd.concat(balanced_frames, ignore_index=True)
    if len(balanced_df) > MAX_DATASET_SIZE:
        balanced_df = balanced_df.sample(n=MAX_DATASET_SIZE, random_state=42).reset_index(drop=True)
    else:
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"   ✓ Rebalanced labels to targets (<= {MAX_DATASET_SIZE} samples).")
    return balanced_df

def print_label_summary(df: pd.DataFrame, heading: str):
    print("\n" + "="*60)
    print(heading)
    print("="*60)
    print(df['label'].value_counts())
    print("="*60)

def process_kaggle_datasets():
    """Process Kaggle datasets"""
    records = []
    kaggle_files = glob.glob(os.path.join(DATA_DIR, '**/*.csv'), recursive=True)
    
    for file_path in kaggle_files:
        try:
            # Determine source type from filename/path
            filename = os.path.basename(file_path).lower()
            if 'fake' in filename or 'false' in filename:
                source = 'kaggle_fake_true_news'
            elif 'real' in filename and 'news' in filename:
                source = 'kaggle_real_news'
            elif 'news' in filename or 'article' in filename:
                source = 'kaggle_news_dataset'
            elif 'review' in filename or 'amazon' in filename:
                source = 'kaggle_amazon_reviews'
            elif 'sqlite' in filename or 'database' in filename:
                source = 'kaggle_sqlite'
            else:
                source = 'kaggle_unknown'
            
            df = pd.read_csv(file_path, on_bad_lines='skip', low_memory=False)
            
            # Handle different column names
            text_col = None
            label_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if col_lower in ['text', 'statement', 'claim', 'content', 'message']:
                    text_col = col
                if col_lower in ['label', 'class', 'category', 'veracity']:
                    label_col = col
            
            if text_col:
                for _, row in df.iterrows():
                    text = str(row[text_col]).strip()
                    if len(text) < 10:
                        continue
                    
                    # Filter non-medical content
                    if not is_medical_content(text):
                        continue
                    
                    # Get label
                    if label_col and label_col in row:
                        label = str(row[label_col]).lower()
                        if label not in ['credible', 'misleading', 'false']:
                            label = SOURCE_LABEL_MAP.get(source, 'credible')
                    else:
                        label = SOURCE_LABEL_MAP.get(source, 'credible')
                    
                    # Normalize label
                    if 'false' in label or 'fake' in label:
                        label = 'false'
                    elif 'mislead' in label or 'myth' in label:
                        label = 'misleading'
                    else:
                        label = 'credible'
                    
                    records.append({
                        'text': text[:10000],  # Limit length
                        'label': label,
                        'source': source,
                        'topic': classify_topic(text),
                        'disease': extract_disease_from_text(text),
                        'timestamp': datetime.now().isoformat()
                    })
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    return expand_records_to_claims(records)

def process_who_cdc_scraped_data():
    """Process scraped WHO/CDC/NIH/NHS and fact-checking data"""
    records = []
    
    # Map of files to their source identifiers
    scraped_files = [
        ('who_general_facts.txt', 'who_general_facts'),
        ('who_covid_myths.txt', 'who_covid_myths'),
        ('who_covid_qa.txt', 'who_covid_qa'),
        ('cdc_general_topics.txt', 'cdc_general_topics'),
        ('cdc_vaccine_facts.txt', 'cdc_vaccine_facts'),
        ('cdc_vaccine_myths.txt', 'cdc_vaccine_myths'),
        ('nih_health_info.txt', 'nih_health_info'),
        ('niaid_diseases.txt', 'niaid_diseases'),
        ('nhs_health_qa.txt', 'nhs_health_qa'),
        ('factcheck_health.txt', 'factcheck_health'),
        ('snopes_health.txt', 'snopes_health'),
    ]
    
    for filename, source in scraped_files:
        file_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(file_path):
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Skip metadata lines (SOURCE: and URL:)
            lines = content.split('\n')
            content_lines = [line for line in lines if not line.startswith('SOURCE:') and not line.startswith('URL:')]
            content = '\n'.join(content_lines)
            
            # Split by claim/fact markers
            sections = re.split(r'---\s*CLAIM/FACT\s*---', content, flags=re.IGNORECASE)
            
            for section in sections:
                text = section.strip()
                if len(text) < 50:
                    continue
                
                # Filter non-medical content - all scraped data must be medical
                if not is_medical_content(text):
                    continue
                
                # Split into sentences/paragraphs
                sentences = re.split(r'[.!?]\s+', text)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) < 30:
                        continue
                    
                    # Double-check medical content for individual sentences
                    if not is_medical_content(sentence, min_medical_keywords=1):
                        continue
                    
                    records.append({
                        'text': sentence[:10000],
                        'label': SOURCE_LABEL_MAP.get(source, 'credible'),
                        'source': source,
                        'topic': classify_topic(sentence),
                        'disease': extract_disease_from_text(sentence),
                        'timestamp': datetime.now().isoformat()
                    })
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    return expand_records_to_claims(records)

def process_github_datasets():
    """Process datasets downloaded from GitHub"""
    records = []
    
    # Look for extracted datasets
    dataset_dirs = [
        os.path.join(DATA_DIR, 'medical-misinformation-dataset-main'),
        os.path.join(DATA_DIR, 'Med-MMHL-main'),
        os.path.join(DATA_DIR, 'monant_misinfo'),
        os.path.join(DATA_DIR, 'Medical-News-Dataset-main'),
        os.path.join(DATA_DIR, 'FakeNewsNet-master'),
        os.path.join(DATA_DIR, 'medical-fact-checking-master'),
        os.path.join(DATA_DIR, 'misinfo-claims-main'),
    ]
    
    for dataset_dir in dataset_dirs:
        if not os.path.exists(dataset_dir):
            continue
        
        # Determine source based on directory name
        dir_lower = dataset_dir.lower()
        if 'monant' in dir_lower or 'misinformation' in dir_lower:
            source = 'monant_misinfo'
            default_label = 'false'
        elif 'mmhl' in dir_lower:
            source = 'med_mmhl'
            default_label = 'misleading'
        elif 'healthfact' in dir_lower or 'health-fact' in dir_lower:
            source = 'healthfact_dataset'
            default_label = 'false'  # Contains false claims for fact-checking
        elif 'facebook' in dir_lower and 'health' in dir_lower:
            source = 'facebook_health_misinfo'
            default_label = 'false'
        elif 'medical-news' in dir_lower:
            source = 'medical_news'
            default_label = 'credible'
        elif 'fakenewsnet' in dir_lower:
            source = 'fakenewsnet'
            default_label = 'false'
        elif 'medical-fact-checking' in dir_lower:
            source = 'medical_fact_checking'
            default_label = 'credible'
        elif 'misinfo-claims' in dir_lower:
            source = 'covid_misinfo_claims'
            default_label = 'misleading'
        elif 'biosentvec' in dir_lower or 'biosent' in dir_lower:
            source = 'biosentvec'
            default_label = 'credible'
        elif 'biowordvec' in dir_lower or 'bioword' in dir_lower:
            source = 'biowordvec'
            default_label = 'credible'
        elif 'covid' in dir_lower and 'chest' in dir_lower:
            source = 'covid_chestxray'
            default_label = 'credible'
        else:
            source = 'github_unknown'
            default_label = 'credible'
        
        # Find CSV/JSON files
        csv_files = glob.glob(os.path.join(dataset_dir, '**/*.csv'), recursive=True)
        json_files = glob.glob(os.path.join(dataset_dir, '**/*.json'), recursive=True)
        
        jsonl_files = glob.glob(os.path.join(dataset_dir, '**/*.jsonl'), recursive=True)
        
        for file_path in csv_files + json_files + jsonl_files:
            try:
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path, on_bad_lines='skip', low_memory=False)
                elif file_path.endswith('.jsonl'):
                    rows = []
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                rows.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
                    if not rows:
                        continue
                    df = pd.DataFrame(rows)
                else:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        df = pd.DataFrame(data)
                    elif isinstance(data, dict):
                        df = pd.DataFrame([data])
                    else:
                        continue
                
                # Find text column
                text_col = None
                label_col = None
                
                for col in df.columns:
                    col_lower = col.lower()
                    if col_lower in ['text', 'statement', 'claim', 'content', 'article', 'title']:
                        text_col = col
                    if col_lower in ['label', 'class', 'category', 'veracity', 'truth']:
                        label_col = col
                
                if text_col:
                    for _, row in df.iterrows():
                        text = str(row[text_col]).strip()
                        if len(text) < 10:
                            continue
                        
                        # Filter non-medical content
                        if not is_medical_content(text):
                            continue
                        
                        # Get label
                        if label_col and label_col in row:
                            label = str(row[label_col]).lower()
                            if label not in ['credible', 'misleading', 'false']:
                                label = default_label
                        else:
                            label = default_label
                        
                        # Normalize
                        if 'false' in label or 'fake' in label or '0' in str(label):
                            label = 'false'
                        elif 'mislead' in label or 'myth' in label or '1' in str(label):
                            label = 'misleading'
                        else:
                            label = 'credible'
                        
                        records.append({
                            'text': text[:10000],
                            'label': label,
                            'source': source,
                            'topic': classify_topic(text),
                            'disease': extract_disease_from_text(text),
                            'timestamp': datetime.now().isoformat()
                        })
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
    
    return expand_records_to_claims(records)

def process_jsonl_datasets():
    """Process Hugging Face style JSONL datasets (SciFact, PubHealth, etc.)."""
    records = []
    jsonl_files = glob.glob(os.path.join(DATA_DIR, '*.jsonl'))
    if not jsonl_files:
    return records
    
    file_source_map = {
        'scifact_train.jsonl': ('scifact', 'credible'),
        'scifact_dev.jsonl': ('scifact', 'credible'),
        'scifact_test.jsonl': ('scifact', 'credible'),
        'pubhealth_train.jsonl': ('pubhealth', 'credible'),
        'pubhealth_dev.jsonl': ('pubhealth', 'credible'),
        'pubhealth_test.jsonl': ('pubhealth', 'credible'),
    }
    
    for file_path in jsonl_files:
        filename = os.path.basename(file_path)
        source, default_label = file_source_map.get(filename, ('jsonl_unknown', 'credible'))
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    text = data.get('claim') or data.get('text') or data.get('question') or data.get('title')
                    if not text or len(text) < 10:
                        continue
                    if not is_medical_content(text):
                        continue
                    
                    label_raw = data.get('label') or data.get('verdict') or data.get('answer')
                    label = normalize_label_value(label_raw, default_label)
                    
                    records.append({
                        'text': text[:10000],
                        'label': label,
                        'source': source,
                        'topic': classify_topic(text),
                        'disease': extract_disease_from_text(text),
                        'timestamp': datetime.now().isoformat()
                    })
        except Exception as exc:
            print(f"Error processing JSONL {filename}: {exc}")
            continue
    
    return expand_records_to_claims(records)

def process_raw_csv_data():
    """Process raw downloaded CSV data and assign labels"""
    records = []
    
    if not os.path.exists(RAW_DATA_INPUT):
        print(f"   ⚠ Raw data file not found: {RAW_DATA_INPUT}")
        print(f"   Run data_downloader.py first to download and save raw data")
        return records
    
    try:
        raw_df = pd.read_csv(RAW_DATA_INPUT, on_bad_lines='skip', low_memory=False)
        print(f"   Found {len(raw_df)} raw records to label")
        
        for _, row in raw_df.iterrows():
            text = str(row['text']).strip()
            source_file = str(row.get('source_file', 'unknown')).lower()
            source_type = str(row.get('source_type', 'unknown'))
            
            # Determine source and label based on source_file name
            source = 'unknown'
            label = 'credible'  # Default
            
            # Map source file to source identifier and label
            if 'who' in source_file:
                if 'myth' in source_file:
                    source = 'who_covid_myths'
                    label = 'credible'  # WHO debunking myths
                elif 'fact' in source_file:
                    source = 'who_general_facts'
                    label = 'credible'
                elif 'qa' in source_file:
                    source = 'who_covid_qa'
                    label = 'credible'
                else:
                    source = 'who_general_facts'
                    label = 'credible'
            
            elif 'cdc' in source_file:
                if 'myth' in source_file:
                    source = 'cdc_vaccine_myths'
                    label = 'credible'  # CDC debunking myths
                elif 'vaccine' in source_file:
                    source = 'cdc_vaccine_facts'
                    label = 'credible'
                elif 'topic' in source_file:
                    source = 'cdc_general_topics'
                    label = 'credible'
                else:
                    source = 'cdc_facts'
                    label = 'credible'
            
            elif 'nih' in source_file or 'niaid' in source_file:
                source = 'nih_health_info'
                label = 'credible'
            
            elif 'nhs' in source_file:
                source = 'nhs_health_qa'
                label = 'credible'
            
            elif 'factcheck' in source_file:
                source = 'factcheck_health'
                label = 'credible'
            
            elif 'snopes' in source_file:
                source = 'snopes_health'
                label = 'credible'
            
            elif 'monant' in source_file or 'misinformation' in source_file:
                source = 'monant_misinfo'
                label = 'false'
            
            elif 'mmhl' in source_file or 'med-mmhl' in source_file:
                source = 'med_mmhl'
                label = 'misleading'
            
            elif 'healthfact' in source_file or 'health-fact' in source_file:
                source = 'healthfact_dataset'
                label = 'false'
            
            elif 'facebook' in source_file and 'health' in source_file:
                source = 'facebook_health_misinfo'
                label = 'false'
            
            elif 'fake' in source_file or 'false' in source_file:
                source = 'kaggle_fake_true_news'
                label = 'false'
            
            elif 'biosent' in source_file or 'bioword' in source_file:
                source = 'biosentvec' if 'sent' in source_file else 'biowordvec'
                label = 'credible'
            
            elif 'covid' in source_file and 'chest' in source_file:
                source = 'covid_chestxray'
                label = 'credible'
            
            elif 'review' in source_file or 'amazon' in source_file:
                source = 'kaggle_amazon_reviews'
                label = 'credible'
            
            elif 'medical-news' in source_file:
                source = 'medical_news'
                label = 'credible'
            
            elif 'fakenewsnet' in source_file:
                source = 'fakenewsnet'
                label = 'false'
            
            elif 'scifact' in source_file:
                source = 'scifact'
                label = 'credible'
            
            elif 'pubhealth' in source_file:
                source = 'pubhealth'
                label = 'credible'
            
            elif 'medical-fact-checking' in source_file:
                source = 'medical_fact_checking'
                label = 'credible'
            
            elif 'misinfo-claims' in source_file:
                source = 'covid_misinfo_claims'
                label = 'misleading'
            
            else:
                # Try to infer from text content
                text_lower = text.lower()
                if any(word in text_lower for word in ['myth', 'false claim', 'misinformation', 'debunk']):
                    label = 'misleading'
                else:
                    label = 'credible'
                source = 'kaggle_unknown'
            
            # Use SOURCE_LABEL_MAP if available
            if source in SOURCE_LABEL_MAP:
                label = SOURCE_LABEL_MAP[source]
            
            records.append({
                'text': text[:10000],
                'label': label,
                'source': source,
                'topic': classify_topic(text),
                'disease': extract_disease_from_text(text),
                'timestamp': datetime.now().isoformat()
            })
        
        print(f"   ✓ Labeled {len(records)} records from raw CSV")
        
    except Exception as e:
        print(f"   ✗ Error processing raw CSV: {e}")
    
    return records

def integrate_all_data():
    """Main function to integrate all data sources"""
    print("="*60)
    print("DATA PROCESSING AND LABELING")
    print("="*60)
    
    all_records = []
    
    # First, try to process raw CSV data (preferred method)
    print("\n1. Processing raw downloaded CSV data...")
    raw_csv_records = process_raw_csv_data()
    if raw_csv_records:
        print(f"   Found {len(raw_csv_records)} records from raw CSV")
        all_records.extend(raw_csv_records)
    else:
        # Fallback to processing individual sources
        print("\n   No raw CSV found. Processing individual data sources...")
        
        # Process different data sources
        print("\n1a. Processing Kaggle datasets...")
        kaggle_records = process_kaggle_datasets()
        print(f"   Found {len(kaggle_records)} records from Kaggle")
        all_records.extend(kaggle_records)
        
        print("\n1b. Processing WHO/CDC scraped data...")
        who_cdc_records = process_who_cdc_scraped_data()
        print(f"   Found {len(who_cdc_records)} records from WHO/CDC")
        all_records.extend(who_cdc_records)
        
        print("\n1c. Processing GitHub datasets...")
        github_records = process_github_datasets()
        print(f"   Found {len(github_records)} records from GitHub")
        all_records.extend(github_records)
        
        print("\n1d. Processing JSONL datasets...")
        jsonl_records = process_jsonl_datasets()
        print(f"   Found {len(jsonl_records)} records from JSONL files")
        all_records.extend(jsonl_records)
    
    print("\n2. Generating synthetic statements...")
    synthetic_records = generate_synthetic_claims(max_per_disease=30)
    all_records.extend(synthetic_records)
    
    # Load existing dataset if it exists
    existing_df = pd.DataFrame()
    if os.path.exists(OUTPUT_PATH):
        try:
            existing_df = pd.read_csv(OUTPUT_PATH, on_bad_lines='skip', low_memory=False)
            print(f"\n2. Found existing dataset with {len(existing_df)} records")
        except Exception as e:
            print(f"\n2. Error loading existing dataset: {e}")
    
    # Combine all records
    new_df = pd.DataFrame(all_records)
    
    if not new_df.empty:
        print(f"\nTotal collected statements before limiting: {len(new_df)}")
        # Remove duplicates based on text
        new_df = new_df.drop_duplicates(subset=['text'], keep='first')
        print(f"\n3. After deduplication: {len(new_df)} unique records")
        
        # Apply per-source limits
        new_df = apply_source_limits(new_df)
        
        # Rebalance labels
        new_df = rebalance_labels(new_df)
        print_label_summary(new_df, "POST-BALANCING LABEL COUNTS")
        
        # Combine with existing
        if not existing_df.empty:
            # Only add new records that don't exist
            existing_texts = set(existing_df['text'].astype(str).str.lower())
            new_df = new_df[~new_df['text'].str.lower().isin(existing_texts)]
            print(f"4. After removing duplicates with existing: {len(new_df)} new records")
            
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df
        
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        combined_df = apply_source_limits(combined_df)
        combined_df = rebalance_labels(combined_df)
        print_label_summary(combined_df, "FINAL LABEL COUNTS")
        
        # Save to medical_dataset.csv (labeled data)
        combined_df.to_csv(OUTPUT_PATH, index=False, quoting=1, escapechar='\\')
        print(f"\n✓ Saved {len(combined_df)} total labeled records to {OUTPUT_PATH}")
        
        # Print statistics
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        print(f"\nTotal records: {len(combined_df)}")
        print(f"\nLabel distribution:")
        print(combined_df['label'].value_counts())
        print(f"\nSource distribution:")
        print(combined_df['source'].value_counts().head(10))
        print(f"\nTopic distribution:")
        print(combined_df['topic'].value_counts().head(10))
        
        print("\n" + "="*60)
        print("LABELING COMPLETE")
        print("="*60)
        print(f"\nRaw data was read from: {RAW_DATA_INPUT}")
        print(f"Labeled data saved to: {OUTPUT_PATH}")
        print("\n✓ All data has been labeled and is ready for model training")
        print("="*60)
    else:
        print("\n⚠ No new records found. Dataset may already be processed or data directory is empty.")
        if not existing_df.empty:
            print(f"Using existing dataset with {len(existing_df)} records")

if __name__ == '__main__':
    integrate_all_data()

