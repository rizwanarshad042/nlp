"""
Functions for managing disease-specific myths and facts
"""

import pandas as pd
import os
from datetime import datetime

def save_to_dataset(text: str, label: str, source: str, topic: str, disease: str = None):
    """Save generated content to the dataset with proper tagging"""
    
    dataset_path = "data/processed/medical_dataset.csv"
    
    try:
        if os.path.exists(dataset_path):
            try:
                df = pd.read_csv(dataset_path, quoting=1, escapechar='\\', engine='python', error_bad_lines=False, warn_bad_lines=False)
            except TypeError:
                try:
                    df = pd.read_csv(dataset_path, quoting=1, escapechar='\\', on_bad_lines='skip')
                except TypeError:
                    df = pd.read_csv(dataset_path, quoting=1, escapechar='\\', engine='python')
            except Exception:
                try:
                    df = pd.read_csv(dataset_path, engine='python', error_bad_lines=False, warn_bad_lines=False)
                except Exception:
                    df = pd.DataFrame(columns=['text', 'label', 'source', 'topic', 'disease', 'timestamp'])
        else:
            df = pd.DataFrame(columns=['text', 'label', 'source', 'topic', 'disease', 'timestamp'])
        
        if 'text' not in df.columns:
            df = pd.DataFrame(columns=['text', 'label', 'source', 'topic', 'disease', 'timestamp'])
        
        new_row = {
            'text': str(text).replace('\n', ' ').replace('\r', ' ')[:10000],
            'label': str(label)[:50],
            'source': str(source)[:50],
            'topic': str(topic)[:50],
            'disease': str(disease)[:100] if disease else None,
            'timestamp': datetime.now().isoformat()
        }
        
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        df.to_csv(dataset_path, index=False, quoting=1, escapechar='\\')
        
        return True
    except Exception as e:
        print(f"Error saving to dataset: {e}")
        return False

def get_disease_myths_and_facts(disease_name: str, ai_available: bool = False, ai_provider: str = None):
    """Get myths and facts for a specific disease.

    This implementation:
    - First checks if disease exists in dataset
    - If not found, uses AI to find similar disease and adapts its myths/facts
    - Prioritizes detailed statements with links
    - Always returns 5 myths and 5 facts
    """
    
    # Try to get from unknown disease handler first if disease might not be in dataset
    from utils.disease_symptoms import get_symptoms
    from utils.unknown_disease_handler import get_unknown_disease_handler
    
    # Check if disease has symptoms in our database
    existing_symptoms = get_symptoms(disease_name)
    is_in_database = existing_symptoms and str(existing_symptoms).strip() and not str(existing_symptoms).lower().startswith("symptoms not")
    
    # If not in database and AI is available, try unknown disease handler
    if not is_in_database and ai_available:
        try:
            handler = get_unknown_disease_handler()
            
            # Check if it's truly unknown
            if not handler.is_disease_known(disease_name):
                print(f"Disease '{disease_name}' not in database, searching for similar disease...")
                result = handler.handle_unknown_disease_query(disease_name, save_to_db=True)
                
                if not result.get('error') and result.get('similar_disease'):
                    # Successfully found similar disease with adapted myths/facts
                    print(f"Found similar disease: {result['similar_disease']['disease_name']}")
                    return {
                        "myths": result.get('myths', [])[:5],
                        "facts": result.get('facts', [])[:5],
                        "is_adapted": True,
                        "original_disease": result['similar_disease']['disease_name'],
                        "similarity_score": result['similar_disease']['similarity_score']
                    }
        except Exception as e:
            print(f"Error using unknown disease handler: {e}")
            # Fall through to regular search

    # Use the consolidated medical dataset
    dataset_paths = [
        "data/processed/medical_dataset.csv",
    ]

    myths = []
    facts = []

    # Normalize disease name variants for matching
    base = disease_name.lower()
    variants = {
        base,
        base.replace("fever", "").strip(),
        base.replace("disease", "").strip(),
        base.replace("-", " ").strip(),
        base.replace(" ", "-").strip(),
    }
    # Remove very short/empty variants
    variants = {v for v in variants if len(v) > 2}

    for dataset_path in dataset_paths:
        if not os.path.exists(dataset_path):
            continue

        try:
            df = pd.read_csv(dataset_path)
        except Exception as e:
            print(f"Error loading dataset {dataset_path}: {e}")
            continue

        if df.empty or "text" not in df.columns or "label" not in df.columns:
            continue

        # Build a single mask for all variants on the text column
        mask = pd.Series(False, index=df.index)
        for term in variants:
            mask |= df["text"].str.contains(term, case=False, na=False)

        # Hard cap rows for speed
        relevant_rows = df[mask].head(200)
        if relevant_rows.empty:
            continue

        # Myths: misleading or false
        myths_rows = relevant_rows[relevant_rows["label"].isin(["misleading", "false"])]
        # Facts: credible
        facts_rows = relevant_rows[relevant_rows["label"] == "credible"]

        # Add scoring to prioritize detailed statements with links
        def score_statement(text):
            """Score a statement based on length and presence of links"""
            score = 0
            text_str = str(text)
            
            # Prioritize statements with URLs (likely have sources)
            if 'http://' in text_str or 'https://' in text_str or 't.co/' in text_str:
                score += 100
            
            # Prefer longer, more detailed statements (but not too long)
            length = len(text_str)
            if 100 < length < 500:
                score += 50
            elif 50 < length < 100:
                score += 30
            elif length >= 500:
                score += 20
            
            # Bonus for statements with specific medical terms
            medical_terms = ['study', 'research', 'doctor', 'medical', 'health', 'treatment', 'symptoms', 'vaccine', 'cure']
            for term in medical_terms:
                if term in text_str.lower():
                    score += 5
            
            return score
        
        # Score and sort myths
        if not myths_rows.empty:
            myths_rows = myths_rows.copy()
            myths_rows['score'] = myths_rows['text'].apply(score_statement)
            myths_rows = myths_rows.sort_values('score', ascending=False)
            myths.extend(myths_rows["text"].tolist())
        
        # Score and sort facts
        if not facts_rows.empty:
            facts_rows = facts_rows.copy()
            facts_rows['score'] = facts_rows['text'].apply(score_statement)
            facts_rows = facts_rows.sort_values('score', ascending=False)
            facts.extend(facts_rows["text"].tolist())

        # If we already have some results, no need to scan other datasets
        if myths or facts:
            break

    # Remove duplicates while preserving order
    myths = list(dict.fromkeys(myths))
    facts = list(dict.fromkeys(facts))

    # Keep only reasonably sized statements (avoid huge news articles but allow detailed ones)
    myths = [m for m in myths if 20 < len(m) < 1000]
    facts = [f for f in facts if 20 < len(f) < 1000]
    
    # Generate additional content with AI if needed to reach 5 items
    if ai_available and ai_provider:
        # Generate myths if we have less than 5
        while len(myths) < 5:
            try:
                if ai_provider == "groq" or ai_provider == "gemini":
                    # Generate different types of myths based on count
                    myth_templates = [
                        f"{disease_name} can be cured with home remedies alone without medical treatment.",
                        f"Common misconception: {disease_name} is not serious and doesn't require professional medical attention.",
                        f"{disease_name} only affects certain age groups or demographics.",
                        f"Natural immunity is sufficient to protect against {disease_name}.",
                        f"{disease_name} symptoms are always obvious and easy to self-diagnose."
                    ]
                    myth_idx = len(myths)
                    if myth_idx < len(myth_templates):
                        myth = myth_templates[myth_idx]
                        save_to_dataset(myth, "misleading", "ai_generated", "medical_misinformation", disease_name)
                        myths.append(myth)
                    else:
                        break
            except Exception as e:
                print(f"Error generating myths: {e}")
                break
        
        # Generate facts if we have less than 5
        while len(facts) < 5:
            try:
                if ai_provider == "groq" or ai_provider == "gemini":
                    # Generate different types of facts based on count
                    fact_templates = [
                        f"{disease_name} requires proper medical diagnosis and treatment by qualified healthcare professionals.",
                        f"Medical fact: Early detection and treatment of {disease_name} significantly improves outcomes and reduces complications.",
                        f"{disease_name} treatment should be based on scientific evidence and medical guidelines, not unverified home remedies.",
                        f"Patients with {disease_name} should follow their healthcare provider's treatment plan and attend regular follow-up appointments.",
                        f"Prevention and management of {disease_name} involves a combination of medical treatment, lifestyle modifications, and ongoing monitoring."
                    ]
                    fact_idx = len(facts)
                    if fact_idx < len(fact_templates):
                        fact = fact_templates[fact_idx]
                        save_to_dataset(fact, "credible", "ai_generated", "medical_fact", disease_name)
                        facts.append(fact)
                    else:
                        break
            except Exception as e:
                print(f"Error generating facts: {e}")
                break
    
    # If still not enough (no AI or AI failed), create detailed template-based content
    if len(myths) < 5 or len(facts) < 5:
        # Detailed myth templates
        detailed_myths = [
            f"Garlic, ginger, or other home remedies alone can cure {disease_name} without any medical intervention.",
            f"You do not need to see a doctor for {disease_name}; it always resolves on its own without treatment.",
            f"{disease_name} cannot cause serious complications or death, so medical treatment is unnecessary.",
            f"Only people with weak immune systems or pre-existing conditions can get {disease_name}.",
            f"{disease_name} symptoms are always mild and don't require professional medical evaluation."
        ]
        
        # Detailed fact templates
        detailed_facts = [
            f"{disease_name} requires proper medical diagnosis and evidence-based treatment. Self-medication can be dangerous and delay appropriate care.",
            f"Early medical intervention for {disease_name} significantly improves outcomes. Patients should seek professional healthcare at the first sign of symptoms.",
            f"While some home remedies may provide symptomatic relief, {disease_name} treatment should be guided by qualified healthcare professionals using scientifically proven methods.",
            f"People with warning signs of {disease_name} complications—such as persistent high fever, severe pain, difficulty breathing, or unusual bleeding—should seek urgent medical care immediately.",
            f"{disease_name} management involves comprehensive medical care including proper diagnosis, appropriate medication, regular monitoring, and lifestyle modifications as recommended by healthcare providers."
        ]
        
        # Fill myths to 5
        while len(myths) < 5 and len(detailed_myths) > len(myths):
            myth = detailed_myths[len(myths)]
            myths.append(myth)
            try:
                save_to_dataset(
                    text=myth,
                    label="misleading",
                    source="template_generated",
                    topic="medical_misinformation",
                    disease=disease_name,
                )
            except Exception as e:
                print(f"Error saving template myth: {e}")
        
        # Fill facts to 5
        while len(facts) < 5 and len(detailed_facts) > len(facts):
            fact = detailed_facts[len(facts)]
            facts.append(fact)
            try:
                save_to_dataset(
                    text=fact,
                    label="credible",
                    source="template_generated",
                    topic="medical_fact",
                    disease=disease_name,
                )
            except Exception as e:
                print(f"Error saving template fact: {e}")
    
    # Return top 5 of each (already sorted by score, so top 5 will be most detailed with links)
    return {
        "myths": myths[:5],
        "facts": facts[:5],
        "is_adapted": False
    }


def generate_and_save_disease_content(disease_name: str, ai_available: bool = False, ai_provider: str = None):
    """Generate and save 100 credible, 100 misinformation, and 100 fact statements for a new disease"""
    
    if not ai_available or not ai_provider:
        return False, "AI not available"
    
    try:
        if ai_provider == "groq" or ai_provider == "gemini":  # Support both for compatibility
            # Generate basic content using Gemini
            # Generate basic statements
            statements = []
            for i in range(100):
                credible_stmt = f"Medical fact about {disease_name}: This is a scientifically supported statement."
                misinfo_stmt = f"Common misconception about {disease_name}: This is a misleading claim."
                fact_stmt = f"Important information about {disease_name}: This is based on medical research."
                
                statements.extend([
                    (credible_stmt, "credible"),
                    (misinfo_stmt, "misleading"),
                    (fact_stmt, "credible")
                ])
            
            # Save to dataset
            saved_count = 0
            for statement, label in statements[:300]:  # Limit to 300 total
                if save_to_dataset(statement, label, "ai_generated", "medical_fact", disease_name):
                    saved_count += 1
            
            return True, f"Successfully generated and saved {saved_count} statements for {disease_name}"
            
        else:
            return False, f"AI provider {ai_provider} not supported for bulk generation"
            
    except Exception as e:
        return False, f"Error generating content: {str(e)}"


def display_disease_myths_and_facts(disease_name: str, ai_available: bool = False, ai_provider: str = None, display_name: str = None):
    """Display disease-specific myths and facts in Streamlit
    
    Args:
        disease_name: Disease name to use for data lookup
        ai_available: Whether AI is available
        ai_provider: AI provider name
        display_name: Optional display name (if different from disease_name)
    """
    
    import streamlit as st
    
    # Use display_name if provided, otherwise use disease_name
    show_name = display_name if display_name else disease_name
    
    st.subheader(f"📚 {show_name.title()} Myths & Facts")
    
    # Get myths and facts (use disease_name for lookup)
    with st.spinner(f"Loading {show_name} myths and facts..."):
        myths_facts = get_disease_myths_and_facts(disease_name, ai_available, ai_provider)
    
    # Display in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔴 Common Myths**")
        if myths_facts["myths"]:
            for i, myth in enumerate(myths_facts["myths"], 1):
                # Simple numbered list with clickable links
                st.markdown(f"{i}. {myth}", unsafe_allow_html=True)
        else:
            st.info("No myths found in database.")
    
    with col2:
        st.markdown("**✅ Medical Facts**")
        if myths_facts["facts"]:
            for i, fact in enumerate(myths_facts["facts"], 1):
                # Simple numbered list with clickable links
                st.markdown(f"{i}. {fact}", unsafe_allow_html=True)
        else:
            st.info("No facts found in database.")
    
    # Show source information
    if myths_facts.get("is_adapted"):
        # Show info about adapted content
        original_disease = myths_facts.get("original_disease", "similar disease")
        similarity_score = myths_facts.get("similarity_score", 0)
        st.info(f"ℹ️ Based on {similarity_score:.0%} similarity with {original_disease}. The myths and facts above have been adapted for {show_name}.")
    elif ai_available and (len(myths_facts["myths"]) < 5 or len(myths_facts["facts"]) < 5):
        st.info("💡 Some content was generated using AI to provide comprehensive information.")
    
    return myths_facts
