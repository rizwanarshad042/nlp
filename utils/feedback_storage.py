"""
Utility for storing and retrieving statement-label feedback pairs
Stores feedback directly in the training dataset (medical_dataset.csv)
Supports similarity-based matching for similar statements
"""

import pandas as pd
import os
from typing import Optional, Tuple
import csv
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATASET_PATH = "data/processed/medical_dataset.csv"
FEEDBACK_SOURCE = "ai_feedback"
SIMILARITY_THRESHOLD = 0.75

def _normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    return text.strip().lower()

def get_stored_label(statement: str, use_similarity: bool = True) -> Optional[Tuple[str, float]]:
    """
    Retrieve stored label for a statement from the training dataset
    Supports exact matching and similarity-based matching
    
    Args:
        statement: The medical statement to check
        use_similarity: If True, also check for similar statements (default: True)
    
    Returns:
        Tuple of (label, similarity_score) or None if not found
        For exact matches, similarity_score is 1.0
    """
    dataset_path = DATASET_PATH
    if not os.path.exists(dataset_path):
        return None
    
    try:
        normalized_statement = _normalize_text(statement)
        
        try:
            df = pd.read_csv(dataset_path, on_bad_lines='warn', sep=',')
        except pd.errors.ParserError as e:
            try:
                with open(dataset_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                
                parsed_data = []
                if lines:
                    header = lines[0].strip().split(',')
                    for line in lines[1:]:
                        try:
                            reader = csv.reader(io.StringIO(line))
                            row = next(reader)
                            if len(row) == len(header):
                                parsed_data.append(row)
                        except Exception:
                            pass
                
                df = pd.DataFrame(parsed_data, columns=header) if parsed_data else pd.DataFrame()
            except Exception:
                return None
        
        if df.empty or 'text' not in df.columns or 'label' not in df.columns:
            return None
        
        if 'source' in df.columns:
            feedback_rows = df[df['source'] == FEEDBACK_SOURCE].copy()
        else:
            feedback_rows = df.copy()
        
        if feedback_rows.empty:
            return None
        
        for _, row in feedback_rows.iterrows():
            stored_text = str(row['text'])
            stored_normalized = _normalize_text(stored_text)
            
            if stored_normalized == normalized_statement:
                return (str(row['label']).lower(), 1.0)
        
        if use_similarity and len(feedback_rows) > 0:
            try:
                feedback_texts = feedback_rows['text'].fillna('').astype(str).tolist()
                corpus = feedback_texts + [statement]
                
                vectorizer = TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    stop_words='english',
                    lowercase=True
                )
                X = vectorizer.fit_transform(corpus)
                
                similarities = cosine_similarity(X[-1:], X[:-1]).flatten()
                
                best_idx = similarities.argmax()
                best_similarity = float(similarities[best_idx])
                
                if best_similarity >= SIMILARITY_THRESHOLD:
                    best_label = str(feedback_rows.iloc[best_idx]['label']).lower()
                    return (best_label, best_similarity)
            except Exception as e:
                print(f"Error computing similarity: {e}")
        
        return None
    except Exception as e:
        print(f"Error reading feedback from dataset: {e}")
        return None

def store_feedback(statement: str, label: str, source: str = FEEDBACK_SOURCE, topic: str = None, disease: str = None) -> bool:
    """
    Store a statement-label pair in the training dataset
    
    Args:
        statement: The medical statement
        label: The correct label (credible, misleading, false)
        source: Source of the feedback (default: "ai_feedback")
        topic: Health topic classification (optional)
        disease: Disease name if applicable (optional)
    
    Returns:
        True if stored successfully, False otherwise
    """
    return store_feedback_with_tags(statement, label, source, topic, disease)

def store_feedback_with_tags(statement: str, label: str, source: str = FEEDBACK_SOURCE, topic: str = None, disease: str = None) -> bool:
    """
    Store a statement-label pair in the training dataset with proper tagging
    
    Args:
        statement: The medical statement
        label: The correct label (credible, misleading, false)
        source: Source of the feedback (default: "ai_feedback")
        topic: Health topic classification (optional, will be auto-detected if None)
        disease: Disease name if applicable (optional, will be auto-detected if None)
    
    Returns:
        True if stored successfully, False otherwise
    """
    try:
        dataset_path = DATASET_PATH
        normalized_statement = _normalize_text(statement)
        
        if os.path.exists(dataset_path):
            try:
                df = pd.read_csv(dataset_path, on_bad_lines='warn', sep=',')
            except pd.errors.ParserError as e:
                try:
                    with open(dataset_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                    
                    parsed_data = []
                    if lines:
                        header = lines[0].strip().split(',')
                        for line in lines[1:]:
                            try:
                                reader = csv.reader(io.StringIO(line))
                                row = next(reader)
                                if len(row) == len(header):
                                    parsed_data.append(row)
                            except Exception:
                                pass
                    
                    df = pd.DataFrame(parsed_data, columns=header) if parsed_data else pd.DataFrame(columns=['text', 'label', 'source', 'topic', 'disease', 'timestamp'])
                except Exception:
                    df = pd.DataFrame(columns=['text', 'label', 'source', 'topic', 'disease', 'timestamp'])
        else:
            df = pd.DataFrame(columns=['text', 'label', 'source', 'topic', 'disease', 'timestamp'])
        
        if 'text' not in df.columns:
            df = pd.DataFrame(columns=['text', 'label', 'source', 'topic', 'disease', 'timestamp'])
        
        from datetime import datetime
        
        # Auto-detect topic and disease if not provided
        if topic is None:
            try:
                # Use same topic classification logic
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
            except Exception:
                topic = 'feedback'
        
        if disease is None:
            try:
                from utils.disease_integration import extract_disease_from_text
                diseases = extract_disease_from_text(statement)
                disease = diseases[0] if diseases else None
            except Exception:
                disease = None
        
        if 'source' in df.columns:
            feedback_mask = df['source'] == source
            existing_feedback = df[feedback_mask]
            
            for idx, row in existing_feedback.iterrows():
                if _normalize_text(str(row['text'])) == normalized_statement:
                    df.at[idx, 'label'] = str(label)[:50]
                    df.at[idx, 'topic'] = str(topic)[:50] if topic else 'feedback'
                    df.at[idx, 'disease'] = str(disease)[:100] if disease else None
                    df.at[idx, 'timestamp'] = datetime.now().isoformat()
                    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
                    df.to_csv(dataset_path, index=False, quoting=csv.QUOTE_ALL)
                    return True
        
        new_row = {
            'text': str(statement).replace('\n', ' ').replace('\r', ' ')[:10000],
            'label': str(label)[:50],
            'source': str(source)[:50],
            'topic': str(topic)[:50] if topic else 'feedback',
            'disease': str(disease)[:100] if disease else None,
            'timestamp': datetime.now().isoformat()
        }
        
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        df.to_csv(dataset_path, index=False, quoting=csv.QUOTE_ALL)
        
        return True
    except Exception as e:
        print(f"Error storing feedback in dataset: {e}")
        return False

def get_similar_feedback(statement: str, top_k: int = 3, min_similarity: float = None) -> list:
    """
    Get similar feedback entries for a statement
    
    Args:
        statement: The medical statement to find similar feedback for
        top_k: Number of similar entries to return
        min_similarity: Minimum similarity threshold (defaults to SIMILARITY_THRESHOLD)
    
    Returns:
        List of tuples: (statement_text, label, similarity_score)
    """
    if min_similarity is None:
        min_similarity = SIMILARITY_THRESHOLD
    
    dataset_path = DATASET_PATH
    if not os.path.exists(dataset_path):
        return []
    
    try:
        df = pd.read_csv(dataset_path, on_bad_lines='warn', sep=',')
        if df.empty or 'text' not in df.columns or 'label' not in df.columns:
            return []
        
        if 'source' in df.columns:
            feedback_rows = df[df['source'] == FEEDBACK_SOURCE].copy()
        else:
            feedback_rows = df.copy()
        
        if feedback_rows.empty:
            return []
        
        feedback_texts = feedback_rows['text'].fillna('').astype(str).tolist()
        corpus = feedback_texts + [statement]
        
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            stop_words='english',
            lowercase=True
        )
        X = vectorizer.fit_transform(corpus)
        
        similarities = cosine_similarity(X[-1:], X[:-1]).flatten()
        
        results = []
        for idx, similarity in enumerate(similarities):
            if similarity >= min_similarity:
                results.append((
                    str(feedback_rows.iloc[idx]['text']),
                    str(feedback_rows.iloc[idx]['label']).lower(),
                    float(similarity)
                ))
        
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]
    except Exception as e:
        print(f"Error finding similar feedback: {e}")
        return []

def get_all_feedback() -> pd.DataFrame:
    """Get all stored feedback entries from the training dataset"""
    dataset_path = DATASET_PATH
    if not os.path.exists(dataset_path):
        return pd.DataFrame(columns=['text', 'label', 'source', 'topic', 'disease', 'timestamp'])
    
    try:
        df = pd.read_csv(dataset_path, on_bad_lines='warn', sep=',')
        if 'source' in df.columns:
            return df[df['source'] == FEEDBACK_SOURCE]
        return pd.DataFrame(columns=['text', 'label', 'source', 'topic', 'disease', 'timestamp'])
    except Exception as e:
        print(f"Error reading feedback from dataset: {e}")
        return pd.DataFrame(columns=['text', 'label', 'source', 'topic', 'disease', 'timestamp'])

