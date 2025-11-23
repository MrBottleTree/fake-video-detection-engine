import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes.E_nodes.e1_web_evidence import run, WebSearcher, Claim

@pytest.fixture
def mock_state():
    return {
        "claims": [
            {"claim_text": "Python is a programming language.", "who": "Guido", "what": "Python", "when": "1991", "where": "Netherlands"},
            "Simple string claim"
        ],
        "debug": True,
        "use_cache": False
    }

def test_e1_run_structure(mock_state):
    """Test that E1 run function returns state with evidence key and IDs."""
    with patch('nodes.E_nodes.e1_web_evidence.WebSearcher') as MockSearcher:
        # Mock searcher instance
        instance = MockSearcher.return_value
        instance.construct_queries.return_value = ["query1"]
        instance.search_robust.return_value = [
            {"url": "http://test.com", "title": "Test", "snippet": "Snippet", "source": "test", "date": None, "relevance_score": 0.9}
        ]
        instance.deduplicate.return_value = instance.search_robust.return_value
        instance.rank_results.return_value = instance.search_robust.return_value

        result_state = run(mock_state)
        
        assert "evidence" in result_state
        assert len(result_state["evidence"]) == 2
        
        # Check ID generation
        first_claim = result_state["evidence"][0]["claim"]
        assert "id" in first_claim
        assert first_claim["claim_text"] == "Python is a programming language."

def test_smart_queries():
    """Test that smart queries (debunking/supporting) are generated."""
    searcher = WebSearcher(debug=True, use_cache=False)
    claim = {
        "id": "123",
        "claim_text": "Earth is flat",
        "who": None, "what": None, "when": None, "where": None
    }
    
    queries = searcher.construct_queries(claim)
    
    # Check for variety
    assert any("debunked" in q for q in queries)
    assert any("proof" in q for q in queries)
    assert any("fact check" in q for q in queries)

def test_web_searcher_fallback():
    """Test that WebSearcher falls back to different APIs."""
    searcher = WebSearcher(debug=True, use_cache=False)
    
    # Mock all search methods
    searcher._search_serper = MagicMock(side_effect=Exception("Serper failed"))
    searcher._search_google = MagicMock(side_effect=Exception("Google failed"))
    searcher._search_ddg = MagicMock(return_value=[{"url": "ddg.com", "title": "DDG", "snippet": "DDG", "source": "ddg", "date": None, "relevance_score": 0.0}])
    
    # Set keys to trigger attempts
    searcher.serper_key = "fake_key"
    searcher.google_key = "fake_key"
    searcher.google_cx = "fake_cx"
    
    results = searcher.search_robust("test query")
    
    # Verify fallback chain
    searcher._search_serper.assert_called_once()
    searcher._search_google.assert_called_once()
    searcher._search_ddg.assert_called_once()
    
    assert len(results) == 1
    assert results[0]["source"] == "ddg"
