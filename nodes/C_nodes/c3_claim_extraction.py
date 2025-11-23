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
    model_name = "en_core_web_trf"
    
    if nlp_model is None:
        try:
            # Use GPU if available
            if torch.cuda.is_available():
                spacy.prefer_gpu()
                logger.info("C3: Using GPU for Spacy.")
            else:
                logger.info("C3: GPU not available, using CPU.")
            
            logger.info(f"C3: Loading Spacy model '{model_name}'...")
            nlp_model = spacy.load(model_name)
        except OSError:
            logger.warning(f"C3: '{model_name}' not found. Attempting download...")
            try:
                # Use subprocess to avoid sys.exit() from spacy.cli.download
                result = subprocess.run(
                    [sys.executable, "-m", "spacy", "download", model_name],
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout for larger model
                )
                if result.returncode == 0:
                    logger.info(f"C3: Successfully downloaded '{model_name}'.")
                    nlp_model = spacy.load(model_name)
                else:
                    logger.error(f"C3: Failed to download '{model_name}': {result.stderr}. Falling back to 'en_core_web_sm'.")
                    try:
                        nlp_model = spacy.load("en_core_web_sm")
                    except OSError:
                        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], capture_output=True)
                        nlp_model = spacy.load("en_core_web_sm")
            except Exception as download_error:
                logger.error(f"C3: Failed to download '{model_name}': {download_error}. Falling back to blank model.")
                nlp_model = spacy.blank("en")
                nlp_model.add_pipe("sentencizer")
        except Exception as e:
            logger.error(f"C3: Failed to load '{model_name}': {e}. Falling back to blank model.")
            nlp_model = spacy.blank("en")
            nlp_model.add_pipe("sentencizer")
            
    return nlp_model

def is_claim_like(sent) -> bool:
    """
    Robust check if a spacy Span/Doc looks like a claim.
    Criteria:
    - Length > 5 tokens (ignoring punct)
    - Must have a Subject and a Verb (ROOT/AUX)
    - Not a question
    - Not a fragment (must be a complete sentence)
    - Bonus: Contains Named Entities (PERSON, ORG, GPE, DATE, EVENT)
    """
    # Filter out punctuation/space
    tokens = [t for t in sent if not t.is_punct and not t.is_space]
    
    if len(tokens) < 5:
        return False
    
    # Check if it ends with a question mark
    if sent.text.strip().endswith('?'):
        return False
        
    # POS Tag and Dependency check
    if sent[0].has_vector or sent[0].pos_: 
        has_verb = any(t.pos_ in ["VERB", "AUX"] for t in sent)
        has_subj = any(t.dep_ in ["nsubj", "nsubjpass", "csubj", "csubjpass"] for t in sent)
        
        if not (has_verb and has_subj):
            return False
            
    # Named Entity Check - Claims often involve entities
    has_entity = len(sent.ents) > 0
    
    # If no entities, be stricter about length
    if not has_entity and len(tokens) < 8:
        return False
            
    return True

def run(state: dict) -> dict:
    """
    C3 Node: Claim Extraction (Robust Transformer Version)
    
    Extracts claims from:
    1. Transcript (from A2)
    2. OCR Results (from V2)
    """
    print("--- C3: Claim Extraction (Robust Transformer) ---")
    
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
                    "confidence": 0.9 if len(sent.ents) > 0 else 0.8
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
                            "confidence": 0.8 if len(sent.ents) > 0 else 0.7
                        })

    # Deduplicate claims
    unique_claims = []
    seen_texts = set()
    # Sort by confidence first
    claims.sort(key=lambda x: x["confidence"], reverse=True)
    
    for c in claims:
        txt = c["claim_text"]
        if txt not in seen_texts:
            unique_claims.append(c)
            seen_texts.add(txt)
            
    # Limit to top N
    final_claims = unique_claims[:5]
    
    if not final_claims and transcript:
        # Fallback: Just take the first sentence if it's long enough
        doc = nlp(transcript)
        sents = list(doc.sents)
        if sents and len(sents[0]) > 3:
             final_claims.append({
                 "claim_text": sents[0].text.strip(),
                 "source": "transcript_fallback",
                 "confidence": 0.5
             })
             print(f"Fallback: Using first sentence: {sents[0].text.strip()}")

    print(f"Extracted {len(final_claims)} claims.")
    for c in final_claims:
        print(f" - {c['claim_text'][:50]}... (Conf: {c['confidence']})")

    state["claims"] = final_claims
    return state
