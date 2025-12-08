#!/usr/bin/env python3
"""
RAG vs Non-RAG Evaluation Script (Local)
Evaluates RAG and non-RAG baseline models on QA pairs.
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 60)
print("RAG vs Non-RAG Evaluation (Local)")
print("=" * 60)

# Find files
qa_path = None
kb_path = None

# Check common locations
possible_qa_paths = [
    'data/processed/qa_pairs_100.csv',
    'qa_pairs_100.csv',
    os.path.join(os.path.dirname(__file__), 'data/processed/qa_pairs_100.csv'),
    os.path.join(os.path.dirname(__file__), 'qa_pairs_100.csv')
]

possible_kb_paths = [
    'data/processed/medical_dataset.csv',
    'medical_dataset.csv',
    os.path.join(os.path.dirname(__file__), 'data/processed/medical_dataset.csv'),
    os.path.join(os.path.dirname(__file__), 'medical_dataset.csv')
]

for path in possible_qa_paths:
    if os.path.exists(path):
        qa_path = path
        print(f"✓ Found QA pairs: {qa_path}")
        break

for path in possible_kb_paths:
    if os.path.exists(path):
        kb_path = path
        print(f"✓ Found knowledge base: {kb_path}")
        break

if not qa_path:
    print("\n✗ ERROR: qa_pairs_100.csv not found!")
    print("Please place qa_pairs_100.csv in one of these locations:")
    for path in possible_qa_paths:
        print(f"  - {path}")
    print("\nNote: If you downloaded from Google Drive, make sure the file is in your project directory.")
    print("You can also generate it using create_rag_dataset.py if available.")
    exit(1)

if not kb_path:
    print("\n✗ ERROR: medical_dataset.csv not found!")
    print("Please place medical_dataset.csv in one of these locations:")
    for path in possible_kb_paths:
        print(f"  - {path}")
    exit(1)

print("\n" + "=" * 60)
print("Loading Data")
print("=" * 60)

qa_df = pd.read_csv(qa_path)
print(f"Loaded {len(qa_df)} QA pairs")

kb_df = pd.read_csv(kb_path)
print(f"Loaded {len(kb_df)} knowledge base records")

# Filter credible texts for RAG retrieval
credible_texts = kb_df[kb_df['label'] == 'credible']['text'].dropna().tolist()
print(f"Using {len(credible_texts)} credible texts for RAG retrieval")

if len(credible_texts) > 20000:
    credible_texts = credible_texts[:20000]
    print(f"Limited to top 20,000 texts for efficiency")

print("\n" + "=" * 60)
print("Building RAG System")
print("=" * 60)

vectorizer = TfidfVectorizer(stop_words='english', max_features=50000)
kb_matrix = vectorizer.fit_transform(credible_texts)
print("✓ TF-IDF vectorizer trained on knowledge base")

harmful_terms = ['drink bleach', 'poison', 'dangerous advice', 'no treatment needed']


def rag_answer(question, top_k=3):
    """RAG system: retrieves relevant passages and returns concatenated answer."""
    q_vec = vectorizer.transform([question])
    sims = cosine_similarity(q_vec, kb_matrix)[0]
    top_idx = sims.argsort()[-top_k:][::-1]
    retrieved = [credible_texts[i] for i in top_idx]
    answer = " ".join(retrieved)
    return answer, retrieved


def baseline_answer(question):
    """Non-RAG baseline: returns generic safe response."""
    answer = ("Further medical evaluation is required. Consult verified health sources "
              "such as WHO/CDC for precise guidance.")
    return answer, []


def evaluate_answer(pred, reference, retrieved):
    """Evaluate answer on Factuality, Completeness, Faithfulness, Safety."""
    tokens_pred = set(pred.lower().split())
    tokens_ref = set(str(reference).lower().split())
    overlap = tokens_pred & tokens_ref
    precision = len(overlap) / (len(tokens_pred) + 1e-6)
    recall = len(overlap) / (len(tokens_ref) + 1e-6)
    f1 = (2 * precision * recall) / (precision + recall + 1e-6)
    factuality = f1
    completeness = recall
    if retrieved:
        ref_vec = vectorizer.transform([' '.join(retrieved)])
        pred_vec = vectorizer.transform([pred])
        faithfulness = float(cosine_similarity(ref_vec, pred_vec)[0][0])
    else:
        faithfulness = 0.0
    safety = 0 if any(term in pred.lower() for term in harmful_terms) else 1
    return {
        'factuality': factuality,
        'completeness': completeness,
        'faithfulness': min(1.0, faithfulness),
        'safety': safety
    }

print("\n" + "=" * 60)
print("Evaluating Models")
print("=" * 60)

rag_scores = []
baseline_scores = []
subset = qa_df.head(50)  # Evaluate on first 50 QA pairs

print(f"Evaluating on {len(subset)} QA pairs...")

for idx, row in subset.iterrows():
    question = row['question']
    reference = row.get('answer', '')
    
    rag_pred, retrieved = rag_answer(question)
    baseline_pred, _ = baseline_answer(question)
    
    rag_scores.append(evaluate_answer(rag_pred, reference, retrieved))
    baseline_scores.append(evaluate_answer(baseline_pred, reference, []))
    
    if (idx + 1) % 10 == 0:
        print(f"  Processed {idx + 1}/{len(subset)} pairs...")

rag_avg = {k: np.mean([score[k] for score in rag_scores]) for k in rag_scores[0]}
baseline_avg = {k: np.mean([score[k] for score in baseline_scores]) for k in baseline_scores[0]}

print("\n" + "=" * 60)
print("Results Summary")
print("=" * 60)

print("\nRAG Metrics (averaged over sample):")
for metric, value in rag_avg.items():
    print(f"  {metric.title()}: {value:.4f}")

print("\nNon-RAG Baseline Metrics (averaged over sample):")
for metric, value in baseline_avg.items():
    print(f"  {metric.title()}: {value:.4f}")

print("\n" + "=" * 60)
print("Saving Results")
print("=" * 60)

# Create output directory
output_dir = 'data/processed'
os.makedirs(output_dir, exist_ok=True)

# Save summary comparison
comparison_data = {
    'Model': ['RAG', 'Non-RAG (Baseline)'],
    'Factuality': [rag_avg['factuality'], baseline_avg['factuality']],
    'Completeness': [rag_avg['completeness'], baseline_avg['completeness']],
    'Faithfulness': [rag_avg['faithfulness'], baseline_avg['faithfulness']],
    'Safety': [rag_avg['safety'], baseline_avg['safety']]
}

comparison_df = pd.DataFrame(comparison_data)
comparison_path = os.path.join(output_dir, 'rag_vs_nonrag_comparison.csv')
comparison_df.to_csv(comparison_path, index=False)
print(f"✓ Saved summary comparison to {comparison_path}")

# Save detailed results for each QA pair
detailed_results = []
for idx, row in subset.iterrows():
    question = row['question']
    reference = row.get('answer', '')
    
    rag_pred, retrieved = rag_answer(question)
    baseline_pred, _ = baseline_answer(question)
    
    rag_scores_detailed = evaluate_answer(rag_pred, reference, retrieved)
    baseline_scores_detailed = evaluate_answer(baseline_pred, reference, [])
    
    detailed_results.append({
        'question': question,
        'reference_answer': reference,
        'rag_answer': rag_pred,
        'rag_factuality': rag_scores_detailed['factuality'],
        'rag_completeness': rag_scores_detailed['completeness'],
        'rag_faithfulness': rag_scores_detailed['faithfulness'],
        'rag_safety': rag_scores_detailed['safety'],
        'nonrag_answer': baseline_pred,
        'nonrag_factuality': baseline_scores_detailed['factuality'],
        'nonrag_completeness': baseline_scores_detailed['completeness'],
        'nonrag_faithfulness': baseline_scores_detailed['faithfulness'],
        'nonrag_safety': baseline_scores_detailed['safety']
    })

detailed_df = pd.DataFrame(detailed_results)
detailed_path = os.path.join(output_dir, 'rag_vs_nonrag_detailed.csv')
detailed_df.to_csv(detailed_path, index=False)
print(f"✓ Saved detailed results for {len(detailed_df)} QA pairs to {detailed_path}")

print("\n" + "=" * 60)
print("Evaluation Complete!")
print("=" * 60)
print("\nFiles created:")
print(f"  - {comparison_path}")
print(f"  - {detailed_path}")
print("\nYou can now use these files for your project report.")

