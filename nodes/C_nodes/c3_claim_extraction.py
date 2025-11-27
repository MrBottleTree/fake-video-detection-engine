import logging
import sys
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o"

def extract_claims_openai(transcript: str, ocr_results: list) -> list:
    """
    Uses OpenAI API to extract factual claims from transcript and OCR text.
    """
    if not OPENAI_API_KEY:
        logger.warning("C3: OPENAI_API_KEY not found. Skipping extraction.")
        return []

    logger.info("C3: Attempting OpenAI Claim Extraction...")
    
    ocr_text = ""
    for item in ocr_results:
        if isinstance(item, str):
            ocr_text += item + "\n"
        elif isinstance(item, dict):
            ocr_text += item.get("text", "") + "\n"
            
    prompt = f"""
    You are a fact-checking assistant. Extract verifiable factual claims from the following text sources.
    
    TRANSCRIPT:
    {transcript[:4000]}  # Limit context window
    
    ON-SCREEN TEXT (OCR):
    {ocr_text[:2000]}
    
    Instructions:
    1. Identify specific, factual claims that can be verified (e.g., statistics, events, quotes, scientific facts).
    2. Ignore opinions, questions, or vague statements.
    3. Return a JSON object with a key "claims" containing a list of strings.
    4. If no claims are found, return {{"claims": []}}.
    """
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        result = json.loads(content)
        claims = result.get("claims", [])
        
        formatted_claims = []
        for txt in claims:
            formatted_claims.append({
                "claim_text": txt,
                "source": "openai",
                "confidence": 0.95
            })
        return formatted_claims
        
    except Exception as e:
        logger.error(f"C3: OpenAI API request failed: {e}")
        return []

def run(state: dict) -> dict:
    """
    C3 Node: Claim Extraction (OpenAI Only)
    
    Extracts claims from:
    1. Transcript (from A2)
    2. OCR Results (from V2)
    """
    print("--- C3: Claim Extraction (OpenAI Only) ---")
    
    transcript = state.get("transcript", "")
    ocr_results = state.get("ocr_results", [])
    
    print("C3: Attempting OpenAI Claim Extraction...")
    final_claims = extract_claims_openai(transcript, ocr_results)
    
    if final_claims:
        print(f"C3: OpenAI successfully extracted {len(final_claims)} claims.")
    else:
        print("C3: OpenAI extraction failed or returned no claims. Returning empty list.")

    print(f"Extracted {len(final_claims)} claims.")
    for c in final_claims:
        print(f" - {c['claim_text'][:50]}... (Conf: {c['confidence']})")

    state["claims"] = final_claims
    return state

