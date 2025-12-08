"""
Groq API Integration for Medical Misinformation Detection System
Provides free AI explanations using Groq API (replaces Gemini)
"""

import os
import requests
import json
from typing import Optional

def get_gemini_client():
    """Get Groq API client (kept name for compatibility)"""
    try:
        from api_keys import GROQ_API_KEY
        api_key = GROQ_API_KEY
    except ImportError:
        raise RuntimeError(
            "api_keys.py not found. Please create this file with your API keys.\n"
            "Get your API key from: https://console.groq.com/keys"
        )
    
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found in api_keys.py. Please set it in your api_keys.py file.\n"
            "Get your API key from: https://console.groq.com/keys"
        )
    
    return api_key

def gemini_classify_statement(statement: str) -> str:
    """
    Get AI classification of a medical statement
    
    Args:
        statement: The medical statement to classify
    
    Returns:
        The AI's classification label: "credible", "misleading", or "false"
    """
    try:
        api_key = get_gemini_client()
        
        prompt = (
            f"Classify the following medical statement as exactly one of: 'credible', 'misleading', or 'false'. "
            f"Respond with ONLY the single word label (credible, misleading, or false), nothing else.\n\n"
            f"Statement: '{statement}'"
        )
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a medical fact-checker. Classify medical statements as 'credible', 'misleading', or 'false'. Respond with only the label word."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model": "llama-3.1-8b-instant",
            "temperature": 0.2,
            "max_tokens": 10
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content'].strip().lower()
                if content in ["credible", "misleading", "false"]:
                    return content
                elif "credible" in content:
                    return "credible"
                elif "misleading" in content:
                    return "misleading"
                elif "false" in content:
                    return "false"
        return None
    except Exception as e:
        print(f"Error getting AI classification: {e}")
        return None

def gemini_explain_classification(statement: str, predicted_label: str, confidence: float) -> str:
    """
    Generate explanation using Groq API (kept name for compatibility)
    
    Args:
        statement: The medical statement being classified
        predicted_label: The predicted label (credible, misleading, or false)
        confidence: The confidence score of the prediction
    
    Returns:
        A detailed explanation of why the statement was classified as such
    """
    try:
        api_key = get_gemini_client()
        
        if predicted_label == "credible":
            prompt = (
                f"Explain why the medical statement '{statement}' is considered credible and accurate. "
                f"Provide evidence from reliable medical sources (WHO, CDC, medical journals) and explain "
                f"the scientific basis. Keep it informative but concise (2-3 sentences)."
            )
            system_content = "You are a helpful medical assistant providing accurate, evidence-based explanations for credible medical statements."
        elif predicted_label == "misleading":
            prompt = (
                f"Explain why the medical statement '{statement}' is misleading. Point out what aspects "
                f"are incorrect or oversimplified, and provide the correct medical information from reliable "
                f"sources (WHO, CDC, medical journals). Keep it educational and concise (2-3 sentences)."
            )
            system_content = "You are a helpful medical assistant providing accurate, evidence-based explanations for misleading medical statements."
        else:  # false
            prompt = (
                f"Explain why the medical statement '{statement}' is medically false or incorrect. "
                f"Provide the correct medical information from reliable sources (WHO, CDC, medical journals) "
                f"and explain why this claim is dangerous or harmful. Keep it educational and concise (2-3 sentences)."
            )
            system_content = "You are a helpful medical assistant providing accurate, evidence-based explanations for false medical statements."
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model": "llama-3.1-8b-instant",
            "temperature": 0.7,
            "max_tokens": 300
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                return content.strip()
            else:
                return "Explanation generated but no content returned."
        else:
            return f"Groq API error: {response.status_code} - {response.text}"
            
    except requests.exceptions.Timeout:
        return "Groq API request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return f"Groq API request failed: {str(e)}"
    except Exception as e:
        return f"Error generating explanation: {str(e)}"

def gemini_list_symptoms(disease_name: str) -> str:
    """
    Generate symptoms list using Groq API (kept name for compatibility)
    
    Args:
        disease_name: The name of the disease
    
    Returns:
        A comma-separated list of symptoms
    """
    try:
        api_key = get_gemini_client()
        
        prompt = f"List the major symptoms of {disease_name} in comma-separated format. Only list symptoms, no explanations."
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful medical assistant. Provide concise, accurate medical information."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model": "llama-3.1-8b-instant",
            "temperature": 0.2,
            "max_tokens": 150
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                return content.strip()
            else:
                return "symptoms not available"
        else:
            return "symptoms not available"
            
    except Exception as e:
        return "symptoms not available"

def check_gemini_api_key():
    """Check if Groq API key is available and valid (kept name for compatibility)"""
    try:
        from api_keys import GROQ_API_KEY
        api_key = GROQ_API_KEY
    except ImportError:
        return False, "api_keys.py not found. Please create this file with your API keys."
    
    if not api_key or api_key.strip() == "":
        return False, "GROQ_API_KEY not found in api_keys.py. Please set it in your api_keys.py file."
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": "test"
                }
            ],
            "model": "llama-3.1-8b-instant",
            "max_tokens": 5
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            return True, "API key is valid"
        else:
            error_msg = response.text
            try:
                error_json = response.json()
                error_detail = error_json.get('error', {}).get('message', error_msg)
                error_code = error_json.get('error', {}).get('code', response.status_code)
            except:
                error_detail = error_msg
                error_code = response.status_code
            
            if response.status_code == 429 or "quota" in error_detail.lower() or "rate limit" in error_detail.lower():
                return False, f"API key valid but rate limit exceeded. Check usage: https://console.groq.com/usage"
            elif "invalid" in error_detail.lower() or "unauthorized" in error_detail.lower() or "401" in str(error_code) or "403" in str(error_code):
                return False, f"Invalid API key or access denied. Please check your Groq API key. Error: {error_detail[:200]}"
            else:
                return False, f"API key error (Status {error_code}): {error_detail[:300]}"
                
    except requests.exceptions.Timeout:
        return False, "API key test timed out. Please check your internet connection."
    except Exception as e:
        return False, f"API key test failed: {str(e)}"

__all__ = ["gemini_explain_classification", "check_gemini_api_key", "gemini_list_symptoms", "gemini_classify_statement"]