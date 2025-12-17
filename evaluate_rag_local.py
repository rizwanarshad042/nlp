# RAG Evaluation: Compare retrieval-based answers vs hardcoded responses
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 60)
print("RAG vs Non-RAG Evaluation (Local)")
print("=" * 60)

# Locate QA pairs and knowledge base files
qa_path = None
kb_path = None
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
        print(f"Found QA pairs: {qa_path}")
        break

for path in possible_kb_paths:
    if os.path.exists(path):
        kb_path = path
        print(f"Found knowledge base: {kb_path}")
        break

if not qa_path:
    print("\nERROR: qa_pairs_100.csv not found!")
    print("Please place qa_pairs_100.csv in one of these locations:")
    for path in possible_qa_paths:
        print(f"  - {path}")
    print("\nNote: If you downloaded from Google Drive, make sure the file is in your project directory.")
    print("You can also generate it using create_rag_dataset.py if available.")
    exit(1)

if not kb_path:
    print("\nERROR: medical_dataset.csv not found!")
    print("Please place medical_dataset.csv in one of these locations:")
    for path in possible_kb_paths:
        print(f"  - {path}")
    exit(1)

print("\n" + "=" * 60)
print("Loading Data")
print("=" * 60)

# Load questions and answers for testing
qa_df = pd.read_csv(qa_path)
print(f"Loaded {len(qa_df)} QA pairs")

# Load medical dataset (knowledge base for RAG)
kb_df = pd.read_csv(kb_path)
print(f"Loaded {len(kb_df)} knowledge base records")

# Keep only credible medical information for retrieval
credible_df = kb_df[kb_df['label'] == 'credible'].copy()
print(f"Found {len(credible_df)} credible records")

# Clean the knowledge base: remove synthetic content
if 'source' in credible_df.columns:
    credible_df = credible_df[~credible_df['source'].str.contains('synthetic|ai_generated|augmented', case=False, na=False)]
    print(f"After removing synthetic sources: {len(credible_df)} records")

# Remove template texts
credible_df = credible_df[~credible_df['text'].str.contains('Medical fact about|This information is supported by scientific evidence', case=False, na=False)]
print(f"After removing template texts: {len(credible_df)} records")

# Keep only substantial content
credible_df = credible_df[credible_df['text'].str.len() > 100]
print(f"After length filter (>100 chars): {len(credible_df)} records")



# Extract clean credible texts for knowledge base
credible_texts = credible_df['text'].dropna().tolist()

# Limit size for performance (TF-IDF is memory-intensive)
if len(credible_texts) > 15000:
    credible_texts = credible_texts[:15000]
    print(f"Limited to top 15,000 texts for efficiency")
    
print(f"Using {len(credible_texts)} high-quality credible texts for RAG retrieval")

print("\n" + "=" * 60)
print("Building RAG System")
print("=" * 60)

# Build TF-IDF index for fast similarity search
vectorizer = TfidfVectorizer(stop_words='english', max_features=50000)

# Convert all texts to vectors for similarity comparison
kb_matrix = vectorizer.fit_transform(credible_texts)
print("TF-IDF vectorizer trained on knowledge base")
print(f"Knowledge base matrix shape: {kb_matrix.shape} (documents x features)")

# Safety check: terms that indicate harmful medical advice
harmful_terms = ['drink bleach', 'poison', 'dangerous advice', 'no treatment needed']


# RAG function: retrieves relevant texts and generates answer
def rag_answer(question, top_k=5):
    # Convert question to same vector space as knowledge base
    q_vec = vectorizer.transform([question])
    # Find similarity scores with all knowledge base texts
    sims = cosine_similarity(q_vec, kb_matrix)[0]
    # Get indices of top 5 most similar texts
    top_idx = sims.argsort()[-top_k:][::-1]
    retrieved = [credible_texts[i] for i in top_idx]
    
    # Keep only texts with good similarity (>0.1)
    best_matches = []
    for i, text in enumerate(retrieved):
        if sims[top_idx[i]] > 0.1:
            best_matches.append(text[:500])
    
    # Combine top matches into final answer
    if best_matches:
        answer = " ".join(best_matches[:3])
    else:
        answer = "Based on medical evidence: " + retrieved[0][:300]
    
    return answer, retrieved


# Baseline function: keyword matching with pre-written responses
def baseline_answer(question):
    question_lower = question.lower()
    # Hardcoded responses for common medical topics
    topic_responses = {
        'covid': 'COVID-19 is a respiratory illness caused by SARS-CoV-2 virus. Symptoms include fever, cough, difficulty breathing. Vaccines are available and effective. Consult WHO/CDC for latest guidance.',
        'vaccine': 'Vaccines are safe and effective at preventing serious illness. Common side effects are mild and temporary. Consult healthcare providers for vaccination schedule.',
        'mask': 'Masks help prevent spread of respiratory illnesses by blocking droplets. Medical and N95 masks are most effective. Follow CDC guidelines for proper mask usage.',
        'cancer': 'Cancer treatment should be supervised by oncologists. Options include surgery, chemotherapy, radiation, and immunotherapy. Early detection improves outcomes.',
        'diabetes': 'Diabetes requires medical management including blood sugar monitoring, medication, diet, and exercise. Consult endocrinologist for treatment plan.',
        'heart': 'Heart disease requires medical evaluation and treatment. Risk factors include high blood pressure, cholesterol, smoking. Follow cardiologist recommendations.',
        'flu': 'Influenza is treated with rest, fluids, and antiviral medications if prescribed. Annual flu vaccine is recommended for prevention.',
        'symptom': 'Medical symptoms require proper evaluation by healthcare professionals. Seek medical attention for persistent or severe symptoms.'
    }
    
    # Return response if keyword found
    for keyword, response in topic_responses.items():
        if keyword in question_lower:
            return response, []  # Return hardcoded response, no retrieval
    
    # Generic fallback response if no keyword matches
    answer = ("Medical information varies by condition. Consult healthcare professionals and verified sources "
              "such as WHO, CDC, or medical journals for evidence-based guidance on specific health topics.")
    return answer, []


# Evaluation: measures answer quality on 4 metrics
def evaluate_answer(pred, reference, retrieved):
    # Tokenize answers
    tokens_pred = set(pred.lower().split())
    tokens_ref = set(str(reference).lower().split())
    overlap = tokens_pred & tokens_ref
    
    # Calculate precision and recall
    precision = len(overlap) / (len(tokens_pred) + 1e-6)
    recall = len(overlap) / (len(tokens_ref) + 1e-6)
    f1 = (2 * precision * recall) / (precision + recall + 1e-6)
    
    # Metric 1: Factuality (F1 score)
    factuality = f1
    # Metric 2: Completeness (recall)
    completeness = recall
    
    # Metric 3: Faithfulness (similarity to retrieved context)
    if retrieved:
        ref_vec = vectorizer.transform([' '.join(retrieved)])
        pred_vec = vectorizer.transform([pred])
        faithfulness = float(cosine_similarity(ref_vec, pred_vec)[0][0])
    else:
        faithfulness = 0.0
    
    # Metric 4: Safety (check for harmful advice)
    safety = 0 if any(term in pred.lower() for term in harmful_terms) else 1
    
    return {
        'factuality': factuality,
        'completeness': completeness,
        'faithfulness': min(1.0, faithfulness),  # Cap at 1.0
        'safety': safety
    }


# Run evaluation on 50 QA pairs
print("\n" + "=" * 60)
print("Evaluating Models")
print("=" * 60)

rag_scores = []
baseline_scores = []
subset = qa_df.head(50)  # Test on first 50 pairs

print(f"Evaluating on {len(subset)} QA pairs...")

# Test each QA pair with both approaches
for idx, row in subset.iterrows():
    question = row['question']
    reference = row.get('answer', '')
    
    # Get answers from both systems
    rag_pred, retrieved = rag_answer(question)
    baseline_pred, _ = baseline_answer(question)
    
    # Evaluate both answers
    rag_scores.append(evaluate_answer(rag_pred, reference, retrieved))
    baseline_scores.append(evaluate_answer(baseline_pred, reference, []))
    
    if (idx + 1) % 10 == 0:
        print(f"  Processed {idx + 1}/{len(subset)} pairs...")

# Calculate average scores
rag_avg = {k: np.mean([score[k] for score in rag_scores]) for k in rag_scores[0]}
baseline_avg = {k: np.mean([score[k] for score in baseline_scores]) for k in baseline_scores[0]}

# Display results
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

output_dir = 'data/processed'
os.makedirs(output_dir, exist_ok=True)

# Save summary: average metrics for both approaches
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
print(f"Saved summary comparison to {comparison_path}")

# Save detailed results: individual scores for each QA pair
detailed_results = []
for idx, row in subset.iterrows():
    question = row['question']
    reference = row.get('answer', '')
    
    # Generate answers
    rag_pred, retrieved = rag_answer(question)
    baseline_pred, _ = baseline_answer(question)
    
    # Evaluate
    rag_scores_detailed = evaluate_answer(rag_pred, reference, retrieved)
    baseline_scores_detailed = evaluate_answer(baseline_pred, reference, [])
    
    # Store all information
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
print(f"Saved detailed results for {len(detailed_df)} QA pairs to {detailed_path}")

print("\n" + "=" * 60)
print("Evaluation Complete!")
print("=" * 60)
print("\nFiles created:")
print(f"  - {comparison_path}")
print(f"  - {detailed_path}")
print("\nYou can now use these files for your project report.")

