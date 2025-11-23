import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nodes.E_nodes import e3_claim_evidence_scorer

class TestE3Scorer(unittest.TestCase):

    def test_verdict_thresholds(self):
        self.assertEqual(e3_claim_evidence_scorer.get_verdict(0.8), "Highly Likely")
        self.assertEqual(e3_claim_evidence_scorer.get_verdict(0.5), "Likely")
        self.assertEqual(e3_claim_evidence_scorer.get_verdict(0.3), "Possible")
        self.assertEqual(e3_claim_evidence_scorer.get_verdict(0.1), "Unverified")

    def test_aggregation_logic(self):
        # Claim with high reliability evidence
        claims = ["Claim 1"]
        evidence = [
            {"claim_text": "Claim 1", "reliability_score": 0.9},
            {"claim_text": "Claim 1", "reliability_score": 0.7}
        ]
        state = {"claims": claims, "evidence": evidence}
        
        result = e3_claim_evidence_scorer.run(state)
        
        scored_claim = result["claims"][0]
        self.assertEqual(scored_claim["text"], "Claim 1")
        self.assertEqual(scored_claim["evidence_count"], 2)
        # Average of 0.9 and 0.7 is 0.8
        self.assertAlmostEqual(scored_claim["evidence_score"], 0.8)
        self.assertEqual(scored_claim["verdict"], "Highly Likely")

    def test_no_evidence(self):
        claims = ["Claim 1"]
        evidence = []
        state = {"claims": claims, "evidence": evidence}
        
        result = e3_claim_evidence_scorer.run(state)
        
        scored_claim = result["claims"][0]
        self.assertEqual(scored_claim["evidence_count"], 0)
        self.assertEqual(scored_claim["evidence_score"], 0.0)
        self.assertEqual(scored_claim["verdict"], "Unverified")

    def test_mixed_claims(self):
        claims = ["Claim 1", "Claim 2"]
        evidence = [
            {"claim_text": "Claim 1", "reliability_score": 0.8},
            {"claim_text": "Claim 2", "reliability_score": 0.1}
        ]
        state = {"claims": claims, "evidence": evidence}
        
        result = e3_claim_evidence_scorer.run(state)
        
        c1 = next(c for c in result["claims"] if c["text"] == "Claim 1")
        c2 = next(c for c in result["claims"] if c["text"] == "Claim 2")
        
        self.assertEqual(c1["verdict"], "Highly Likely")
        self.assertEqual(c2["verdict"], "Unverified")

if __name__ == '__main__':
    unittest.main()
