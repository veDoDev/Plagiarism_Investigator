import json
import time
from django.shortcuts import render
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import hashlib
import openai
from django.conf import settings
from django.core.files.storage import default_storage
from google import genai
from hashlib import sha256
import re
import fitz  # PyMuPDF
from docx import Document
import nltk
from nltk.corpus import stopwords
import trafilatura

# Download stopwords once
nltk.download('stopwords', quiet=True)

def preprocess_text(text):
    """
    Cleans and preprocesses text for plagiarism detection.
    Steps: lowercase, remove special chars, extra spaces, stopwords (optional)
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S*@\S*\s?', '', text)
    
    # Remove numbers (optional - you can keep if needed)
    # text = re.sub(r'\d+', '', text)
    
    # Remove special characters (keep only letters, numbers, spaces)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Optional: Remove stopwords
    # Uncomment if you want stopword removal
    # stop_words = set(stopwords.words('english'))
    # words = text.split()
    # text = ' '.join([word for word in words if word not in stop_words])
    
    return text


def plagiarism_score(text1, text2):
    """
    Calculate plagiarism similarity with preprocessing
    """
    # Preprocess both texts
    text1_clean = preprocess_text(text1)
    text2_clean = preprocess_text(text2)
    
    # Check for empty text after preprocessing
    if not text1_clean or not text2_clean:
        return 0.0, 0.0  # Return 0% similarity and confidence for empty text
    
    # Vectorize and calculate similarity
    vectorizer = TfidfVectorizer().fit_transform([text1_clean, text2_clean])
    similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0]
    similarity_percent = round(similarity * 100, 2)
    
    # Confidence calculation
    tfidf_array = vectorizer.toarray()
    mean = np.mean(tfidf_array)
    std = np.std(tfidf_array)
    
    if similarity_percent == 100.0:
        confidence = 100.0
    else:
        confidence = max(0, min(100, round((mean / (std + 1e-6)) * similarity_percent, 2)))
    
    return similarity_percent, confidence




openai.api_key = os.getenv("OPENAI_API_KEY")

def gemini_generate(prompt):
    """
    Calls the Gemini API to generate content based on the given prompt.
    """
    # Initialize the Gemini client
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # Use the correct model for text generation
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    # Return the generated text
    return response.text

def _hash_text(s: str) -> str:
    return sha256(s.encode("utf-8")).hexdigest()

def cached_solution_path(assignment_text: str) -> str:
    h = _hash_text(assignment_text)[:16]
    dirpath = os.path.join(settings.MEDIA_ROOT, "generated_solutions")
    os.makedirs(dirpath, exist_ok=True)
    return os.path.join(dirpath, f"solution_{h}.txt")

def generate_solution_with_ai(assignment_text: str, max_tokens: int = 1000) -> str:
    """
    Calls the Gemini API to generate a solution string.
    - Uses a system prompt.
    - Caches solutions on disk (by hash) to avoid repeated billing.
    """
    # Check cache
    path = cached_solution_path(assignment_text)
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read()

    # Build prompt
    system_prompt = (
        "You are a helpful assistant that writes clear, correct, and concise solutions to assignment questions. "
        "Give step-by-step explanations, include code blocks if applicable, and keep answers academic and well-structured."
    )
    prompt = f"{system_prompt}\n\nAssignment:\n{assignment_text}"

    # Generate solution using Gemini
    response = gemini_generate(prompt)

    # Save to cache
    with open(path, "w") as f:
        f.write(response)

    return response


def cached_urls_path(document_text: str) -> str:
    """
    Returns the cache file path for fetched URLs based on document hash.
    """
    h = _hash_text(document_text[:3000])[:16]
    dirpath = os.path.join(settings.MEDIA_ROOT, "cached_urls")
    os.makedirs(dirpath, exist_ok=True)
    return os.path.join(dirpath, f"urls_{h}.json")

def fetch_similar_urls(document_text: str, max_chars: int = 3000) -> list:
    """
    Use Gemini API to fetch 5-8 URLs containing similar content.
    """
    truncated_text = document_text[:max_chars]
    
    # Check cache first
    cache_path = cached_urls_path(truncated_text)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                return cached_data.get('urls', [])
        except Exception as e:
            print(f"Cache read error: {e}")
    
    # Build the prompt
    prompt = f"""You are an academic investigator. Given the following text, return 5-8 URLs that contain similar or related content.

Requirements:
- Return ONLY URLs, one per line
- No explanations or additional text
- URLs should be from academic sources, educational sites, or reputable content platforms

Text to analyze:
{truncated_text}

URLs:"""
    
    try:
        response = gemini_generate(prompt)
        
        # Extract URLs
        urls = []
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('http://') or line.startswith('https://'):
                urls.append(line)
            elif line and ('http://' in line or 'https://' in line):
                url_match = re.search(r'https?://[^\s]+', line)
                if url_match:
                    urls.append(url_match.group(0))
        
        urls = urls[:8]
        
        # Cache results
        cache_data = {'urls': urls, 'document_hash': _hash_text(truncated_text)[:16]}
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2)
        
        return urls
        
    except Exception as e:
        print(f"Gemini API error: {e}")
        return []
    
try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False
    print("Warning: trafilatura not installed. Web scraping will be disabled.")

    
def extract_text_from_url(url: str, timeout: int = 10) -> str:
    """
    Extract clean text from URL using trafilatura.
    """
    if not TRAFILATURA_AVAILABLE:
        return ""
    
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded)
            return text.strip() if text else ""
        return ""
    except Exception as e:
        print(f"Failed to extract from {url}: {e}")
        return ""
    
def scrape_multiple_urls(urls: list) -> dict:
    """
    Scrape text from multiple URLs.
    Returns: {url: extracted_text}
    """
    results = {}
    for url in urls:
        print(f"Scraping: {url}")
        text = extract_text_from_url(url)
        if text:
            results[url] = text
        time.sleep(1)  # Be polite to servers
    return results

def analyze_research_paper(document_text: str, source_urls_dict: dict) -> dict:
    """
    Compare document against multiple web sources.
    Returns detailed analysis with similarity breakdown.
    """
    results = []
    all_similarities = []
    
    for url, source_text in source_urls_dict.items():
        if source_text:
            similarity, _ = plagiarism_score(document_text, source_text)
            results.append({
                'url': url,
                'similarity': similarity,
                'source_length': len(source_text),
            })
            all_similarities.append(similarity)
    
    # Calculate confidence scores
    if all_similarities:
        mean_sim = np.mean(all_similarities)
        std_sim = np.std(all_similarities)
        for result in results:
            if result['similarity'] > mean_sim:
                confidence = min(100, round((result['similarity'] / (std_sim + 1e-6)) * 10, 2))
            else:
                confidence = max(0, round(result['similarity'] * 0.8, 2))
            result['confidence'] = confidence
            
            # Add risk level classification
            if result['similarity'] >= 70:
                result['risk_level'] = 'HIGH'
                result['risk_class'] = 'danger'
            elif result['similarity'] >= 40:
                result['risk_level'] = 'MEDIUM'
                result['risk_class'] = 'warning'
            else:
                result['risk_level'] = 'LOW'
                result['risk_class'] = 'success'
    
    # Sort by similarity (highest first)
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Calculate overall metrics
    max_similarity = max(all_similarities) if all_similarities else 0
    avg_similarity = mean_sim if all_similarities else 0
    
    # Determine overall verdict
    if max_similarity >= 70:
        verdict = "HIGH RISK - Significant similarity detected"
        verdict_class = "danger"
    elif max_similarity >= 40:
        verdict = "MODERATE RISK - Some similarity found"
        verdict_class = "warning"
    else:
        verdict = "LOW RISK - Minimal similarity"
        verdict_class = "success"
    
    return {
        'sources': results,
        'max_similarity': round(max_similarity, 2),
        'avg_similarity': round(avg_similarity, 2),
        'total_sources_checked': len(results),
        'high_risk_sources': len([r for r in results if r['similarity'] > 70]),
        'medium_risk_sources': len([r for r in results if 40 <= r['similarity'] <= 70]),
        'low_risk_sources': len([r for r in results if r['similarity'] < 40]),
        'verdict': verdict,
        'verdict_class': verdict_class,
    }
