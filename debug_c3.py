import sys
import os

# Add project root to sys.path
sys.path.append(os.getcwd())

from nodes.C_nodes import c3_claim_extraction

state = {
    "transcript": "This is a test transcript. Google pays 70 Lakhs.",
    "ocr_results": [{"text": "70 Lakhs CTC"}],
    "debug": True
}

print("Running C3 in isolation...")
try:
    new_state = c3_claim_extraction.run(state)
    print("C3 finished successfully.")
    print("Claims:", new_state.get("claims"))
except Exception as e:
    print(f"C3 failed: {e}")
