
import os
import sys
import json
from nodes.E_nodes import e2_source_reliability

# Mock state
state = {
    "evidence": [
        {
            "url": "https://www.google.com",
            "claim_text": "Test claim",
            "snippet": "Test snippet"
        },
        {
            "url": "https://www.bbc.com/news",
            "claim_text": "Test claim 2",
            "snippet": "BBC snippet"
        }
    ],
    "data_dir": "."
}

# Mock dump_node_debug to avoid errors
def mock_dump(state, node, data):
    print(f"DEBUG DUMP {node}: {data}")

import nodes.E_nodes.e2_source_reliability as e2
e2.dump_node_debug = mock_dump

print("Running E2 reproduction...")
try:
    new_state = e2.run(state)
    print("E2 finished.")
    for ev in new_state["evidence"]:
        print(f"URL: {ev.get('url')} - Score: {ev.get('reliability_score')}")
except Exception as e:
    print(f"E2 failed: {e}")
