import sys
import os
import json
from unittest.mock import MagicMock, patch

# Helper to create mocks that satisfy importlib checks
def create_mock_module(version="2.0.0"):
    m = MagicMock()
    m.__spec__ = MagicMock()
    m.__version__ = version
    return m

# Mock dependencies that might be missing in the test environment
sys.modules["moviepy"] = create_mock_module()
sys.modules["moviepy.editor"] = create_mock_module()
sys.modules["librosa"] = create_mock_module()
sys.modules["yt_dlp"] = create_mock_module()
sys.modules["imageio_ffmpeg"] = create_mock_module()
sys.modules["cv2"] = create_mock_module()
sys.modules["numpy"] = create_mock_module()
sys.modules["face_alignment"] = create_mock_module()
sys.modules["torch"] = create_mock_module()
sys.modules["easyocr"] = create_mock_module()
sys.modules["whisper"] = create_mock_module()
sys.modules["scipy"] = create_mock_module()
sys.modules["scipy.spatial"] = create_mock_module()
sys.modules["sklearn"] = create_mock_module()
sys.modules["sklearn.linear_model"] = create_mock_module()
# requests, duckduckgo_search, and bs4 are installed, so we don't mock them in sys.modules
# to avoid breaking their internal imports (like requests.exceptions)

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nodes.E_nodes import e1_web_evidence, e2_source_reliability, e3_claim_evidence_scorer

def test_e_nodes():
    print("Testing E Nodes (Production-Ready Suite)...")
    
    # --- Setup Mocks ---
    # Mock DDGS to return realistic results
    mock_ddgs_instance = MagicMock()
    # We need to mock the generator returned by ddgs.text()
    def mock_text_search(query, max_results=5):
        # Return different results based on query to test logic
        if "Person A" in query:
            yield {
                "href": "https://www.bbc.com/news/person-a",
                "title": "Person A did something confirmed",
                "body": "It is confirmed that Person A did something yesterday in London. This is a reliable report."
            }
            yield {
                "href": "https://fake-news-network.xyz/person-a",
                "title": "Lies about Person A",
                "body": "Person A is an alien. This is fake news."
            }
        else:
            yield {
                "href": "https://www.reuters.com/article/person-b",
                "title": "Person B statement",
                "body": "Person B said something today. This is another reliable report."
            }

    mock_ddgs_instance.text.side_effect = mock_text_search
    
    mock_ddgs_instance.text.side_effect = mock_text_search
    
    # Use patch for DDGS and requests within the E-node modules
    # We patch where they are USED, not where they are defined
    with patch("nodes.E_nodes.e1_web_evidence.DDGS", return_value=mock_ddgs_instance), \
         patch("nodes.E_nodes.e1_web_evidence.requests.Session") as mock_session_cls, \
         patch("nodes.E_nodes.e2_source_reliability.requests.head") as mock_head:
        
        # Setup requests mocks
        mock_session = mock_session_cls.return_value
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {} # Default json
        mock_session.post.return_value = mock_response
        mock_session.get.return_value = mock_response
        
        # Mock head for E2 about page check
        mock_head.return_value = mock_response

    
        # --- Test Data ---
        state = {
            "claims": [
                {"who": "Person A", "what": "did something", "when": "yesterday", "where": "London"},
                {"who": "Person B", "what": "said something", "when": "today"}
            ],
            "transcript": "Person A did something yesterday in London and Person B said something",
            "ocr_results": [
                {"detections": [{"text": "Person A"}]},
                {"detections": [{"text": "London"}]}
            ],
            "debug": True
        }
        
        # --- Run E1 (Web Search) ---
        print("\n--- Running E1 ---")
        state = e1_web_evidence.run(state)
        assert "evidence" in state
        assert len(state["evidence"]) == 2
        # Verify results structure
        assert state["evidence"][0]["results"][0]["source"] == "duckduckgo"
        print("E1 Passed (Mocked DDG returned results)")
        
        # --- Run E2 (Source Reliability) ---
        print("\n--- Running E2 ---")
        state = e2_source_reliability.run(state)
        
        # Verify Trusted Source Loading (BBC should be 1.0, Fake News should be 0.2)
        results_claim_1 = state["evidence"][0]["results"]
        bbc_result = next(r for r in results_claim_1 if "bbc.com" in r["url"])
        fake_result = next(r for r in results_claim_1 if "fake-news-network.xyz" in r["url"])
        
        print(f"BBC Score: {bbc_result['reliability_score']}")
        print(f"Fake News Score: {fake_result['reliability_score']}")
        
        # BBC is Tier 1 -> 1.0. With about page boost (mocked) + HTTPS + freshness, it might be capped at 1.0
        # Reliability score is clamped to [0, 1], so it should be exactly 1.0 or very close
        assert bbc_result["reliability_score"] >= 0.95
        
        # Fake News is Tier 5 -> 0.2. 
        # Base 0.2 + SSL (0.05) = 0.25. Should be <= 0.3
        assert fake_result["reliability_score"] <= 0.3
        print("E2 Passed (Reliability Scores Correct)")
        
        # --- Run E3 (Scoring) ---
        print("\n--- Running E3 ---")
        state = e3_claim_evidence_scorer.run(state)
        features = state["features"]
        
        print("Features:", json.dumps(features, indent=2))
        
        # Assertions for robustness
        # f8: Support Ratio. Both claims have at least one reliable source (BBC, Reuters). Should be 1.0.
        assert features["claim_support_ratio"] == 1.0
        
        # f9: Median Reliability. Should be high since we have BBC(1.0) and Reuters(1.0).
        assert features["median_source_reliability"] >= 0.9
        
        # f10: ASR-OCR Consistency. 
        # If sentence-transformers is installed, this will use semantic similarity.
        # "Person A did something..." vs "Person A London" -> Should be > 0.3
        assert features["asr_ocr_consistency"] > 0.2
        
        # New Features
        assert "entity_match_score" in features
        assert "temporal_consistency" in features
        assert "confidence_score" in features
        
        print("E3 Passed (Features Calculated Correctly)")
        
        print("\nAll Production-Ready E-Node tests passed!")

if __name__ == "__main__":
    test_e_nodes()
