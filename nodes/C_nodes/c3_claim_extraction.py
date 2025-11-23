import logging
import spacy
import torch
import subprocess
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global model cache to avoid reloading
nlp_model = None

def get_spacy_model():
    global nlp_model
    if nlp_model is None:
        try:
            # Use GPU if available
            if torch.cuda.is_available():
                spacy.prefer_gpu()
                logger.info("C3: Using GPU for Spacy.")
            
            logger.info("C3: Loading Spacy model 'en_core_web_sm'...")
            nlp_model = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("C3: 'en_core_web_sm' not found. Attempting download...")
            try:
                # Use subprocess to avoid sys.exit() from spacy.cli.download
                result = subprocess.run(
                    [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                if result.returncode == 0:
                    logger.info("C3: Successfully downloaded 'en_core_web_sm'.")
                    nlp_model = spacy.load("en_core_web_sm")
                else:
                    logger.error(f"C3: Failed to download 'en_core_web_sm': {result.stderr}. Falling back to blank model.")
                    nlp_model = spacy.blank("en")
                    nlp_model.add_pipe("sentencizer")
            except Exception as download_error:
                logger.error(f"C3: Failed to download 'en_core_web_sm': {download_error}. Falling back to blank model.")
                nlp_model = spacy.blank("en")
                nlp_model.add_pipe("sentencizer")
        except Exception as e:
            logger.error(f"C3: Failed to load 'en_core_web_sm': {e}. Falling back to blank model.")
            nlp_model = spacy.blank("en")
            nlp_model.add_pipe("sentencizer")
            
    return nlp_model

def is_claim_like(sent) -> bool:
    """
    Heuristic check if a spacy Span/Doc looks like a claim.
    Criteria:
    - Length > 4 tokens (ignoring punct)
    - Has a verb (if POS tags available)
    - Not a question
    """
    # Filter out punctuation/space
    tokens = [t for t in sent if not t.is_punct and not t.is_space]
    
    if len(tokens) < 4:
        return False
    
    # Check if it ends with a question mark (Spacy handles this in sent.text usually, but let's check the last token)
    if sent[-1].text == '?':
        return False
        
    # POS Tag check (if model supports it)
    if sent[0].has_vector or sent[0].pos_: # Check if model has POS capabilities
        has_verb = any(t.pos_ == "VERB" or t.pos_ == "AUX" for t in sent)
        if not has_verb:
            return False
            
    return True

def run(state: dict) -> dict:
    """
    C3 Node: Claim Extraction (Robust Spacy Version)
    
    Extracts claims from:
    1. Transcript (from A2)
    2. OCR Results (from V2)
    """
    print("--- C3: Claim Extraction (Robust) ---")
    
    transcript = state.get("transcript", "")
    ocr_results = state.get("ocr_results", [])
    
    nlp = get_spacy_model()
    claims = []
    
    # 1. Extract from Transcript
    if transcript:
        doc = nlp(transcript)
        for sent in doc.sents:
            if is_claim_like(sent):
                claims.append({
                    "claim_text": sent.text.strip(),
                    "source": "transcript",
                    "confidence": 0.85
                })
    
    # 2. Extract from OCR
    if ocr_results:
        for item in ocr_results:
            text = ""
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict):
                text = item.get("text", "")
            
            if text:
                # OCR text might be fragmented. Process it as a doc.
                doc = nlp(text)
                for sent in doc.sents:
                    if is_claim_like(sent):
                         claims.append({
                            "claim_text": sent.text.strip(),
                            "source": "ocr",
                            "confidence": 0.75
                        })

    # Deduplicate claims
    unique_claims = []
    seen_texts = set()
    for c in claims:
        txt = c["claim_text"]
        if txt not in seen_texts:
            unique_claims.append(c)
            seen_texts.add(txt)
            
    # Limit to top N
    final_claims = unique_claims[:5]
    
    if not final_claims and transcript:
        # Fallback
        doc = nlp(transcript)
        sents = list(doc.sents)
        if sents:
             final_claims.append({
                 "claim_text": sents[0].text.strip(),
                 "source": "transcript_fallback",
                 "confidence": 0.5
             })
             print(f"Fallback: Using first sentence: {sents[0].text.strip()}")

    print(f"Extracted {len(final_claims)} claims.")
    for c in final_claims:
        print(f" - {c['claim_text'][:50]}...")

    state["claims"] = final_claims
    return state
