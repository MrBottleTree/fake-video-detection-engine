import logging
import spacy
import torch
import subprocess
import sys
import os
import json
import google.generativeai as genai
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global model cache
nlp_model = None

def get_spacy_model():
    global nlp_model
    model_name = "en_core_web_trf"
    
    if nlp_model is None:
        try:
            if torch.cuda.is_available():
                spacy.prefer_gpu()
            nlp_model = spacy.load(model_name)
        except OSError:
            logger.warning(f"C3: '{model_name}' not found. Falling back to 'en_core_web_sm'.")
            try:
                nlp_model = spacy.load("en_core_web_sm")
            except OSError:
                subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], capture_output=True)
                nlp_model = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.error(f"C3: Failed to load Spacy: {e}")
            nlp_model = spacy.blank("en")
            nlp_model.add_pipe("sentencizer")
            
    return nlp_model

class LLMClaimExtractor:
    """Extracts claims using Google Gemini LLM for context awareness."""
    
    def __init__(self):
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None
            logger.warning("C3: GOOGLE_API_KEY not found. LLM extraction disabled.")

    def extract(self, transcript: str, ocr_text: str) -> List[Dict[str, Any]]:
        if not self.model:
            return []

        prompt = f"""
        You are an expert fact-checker. Your task is to extract factual claims from the following video transcript and on-screen text (OCR).
        
        CRITICAL INSTRUCTION:
        - If the transcript says "this graph" or "as seen here", use the OCR text to resolve what is being shown.
        - Combine the spoken claim with the visual context to create a standalone, verifiable statement.
        - Extract only factual claims that can be verified (stats, events, scientific facts).
        - Ignore opinions or subjective statements.
        
        TRANSCRIPT:
        {transcript[:15000]}
        
        OCR / ON-SCREEN TEXT:
        {ocr_text[:5000]}
        
        Return a JSON list of objects with keys: "claim_text", "confidence" (0.0-1.0), "source" ("transcript+ocr", "transcript", or "ocr").
        Example: [{{"claim_text": "The US inflation rate in 2022 was 8.5%", "confidence": 0.95, "source": "transcript+ocr"}}]
        """
        
        try:
            response = self.model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            return json.loads(response.text)
        except Exception as e:
            logger.error(f"C3: LLM Extraction failed: {e}")
            return []

def run(state: dict) -> dict:
    """
    C3 Node: Claim Extraction (LLM + Robust Fallback)
    """
    print("--- C3: Claim Extraction (LLM Enhanced) ---")
    
    transcript = state.get("transcript", "")
    ocr_results = state.get("ocr_results", [])
    
    # Flatten OCR results to single string for context
    ocr_text_list = []
    if ocr_results:
        for item in ocr_results:
            if isinstance(item, str): ocr_text_list.append(item)
            elif isinstance(item, dict): ocr_text_list.append(item.get("text", ""))
    ocr_context = "\n".join(ocr_text_list)
    
    claims = []
    
    # 1. Try LLM Extraction (Primary)
    llm_extractor = LLMClaimExtractor()
    if llm_extractor.model:
        print("   Attempting LLM extraction...")
        claims = llm_extractor.extract(transcript, ocr_context)
        if claims:
            print(f"   ✅ LLM extracted {len(claims)} claims.")
    
    # 2. Fallback to Spacy (Secondary)
    if not claims:
        print("   ⚠️ LLM failed or returned nothing. Falling back to Spacy.")
        nlp = get_spacy_model()
        doc = nlp(transcript)
        for sent in doc.sents:
            # Simple heuristic: > 5 words, has verb/noun
            if len(sent) > 5 and any(t.pos_ == "VERB" for t in sent):
                claims.append({
                    "claim_text": sent.text.strip(),
                    "source": "transcript_fallback",
                    "confidence": 0.6
                })
        
        # Add OCR lines as claims if they look like sentences
        for line in ocr_text_list:
            if len(line.split()) > 4:
                claims.append({
                    "claim_text": line,
                    "source": "ocr_fallback",
                    "confidence": 0.5
                })

    # Deduplicate
    unique_claims = []
    seen = set()
    for c in claims:
        txt = c.get("claim_text", "").strip()
        if txt and txt not in seen:
            unique_claims.append(c)
            seen.add(txt)
            
    # Limit to top 5
    final_claims = unique_claims[:5]
    
    print(f"Extracted {len(final_claims)} claims.")
    for c in final_claims:
        print(f" - {c['claim_text'][:60]}... (Conf: {c.get('confidence', 0)})")

    state["claims"] = final_claims
    return state
