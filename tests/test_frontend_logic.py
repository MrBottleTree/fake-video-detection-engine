
import unittest

class TestFrontendLogic(unittest.TestCase):
    def test_logic(self):
        edges = [("IN", "V1"), ("IN", "A1"), ("A1", "A2"), ("V1", "C2"), ("A2", "C2")]
        nodes_order = ["IN", "V1", "A1", "A2", "C2"]
        
        node_parents = {}
        for src, dst in edges:
            node_parents.setdefault(dst, set()).add(src)
            
        done_nodes = set()
        
        output_0 = {"IN": {}}
        valid_0 = self.get_valid(output_0, node_parents, done_nodes)
        self.assertEqual(valid_0, {"IN"})
        done_nodes.update(valid_0)
        
        statuses_0 = self.get_statuses(nodes_order, done_nodes, node_parents)
        self.assertEqual(statuses_0["V1"], "running")
        self.assertEqual(statuses_0["A1"], "running")
        self.assertEqual(statuses_0["A2"], "queued")
        
        output_1 = {"V1": {}}
        valid_1 = self.get_valid(output_1, node_parents, done_nodes)
        self.assertEqual(valid_1, {"V1"})
        done_nodes.update(valid_1)
        
        statuses_1 = self.get_statuses(nodes_order, done_nodes, node_parents)
        self.assertEqual(statuses_1["V1"], "done")
        self.assertEqual(statuses_1["A1"], "running")
        self.assertEqual(statuses_1["C2"], "queued")
        
        output_2 = {"A1": {}}
        valid_2 = self.get_valid(output_2, node_parents, done_nodes)
        done_nodes.update(valid_2)
        
        statuses_2 = self.get_statuses(nodes_order, done_nodes, node_parents)
        self.assertEqual(statuses_2["A2"], "running")
        
        print("Test passed!")

    def get_valid(self, output, node_parents, done_nodes):
        current_batch = set(output.keys())
        valid_batch = set()
        for node in current_batch:
            parents = node_parents.get(node, set())
            if parents.issubset(done_nodes):
                valid_batch.add(node)
        return valid_batch

    def get_statuses(self, nodes_order, done_nodes, node_parents):
        statuses = {}
        for n in nodes_order:
            if n in done_nodes:
                statuses[n] = "done"
                continue
            parents = node_parents.get(n, set())
            if not parents:
                statuses[n] = "running"
            elif parents.issubset(done_nodes):
                statuses[n] = "running"
            else:
                statuses[n] = "queued"
        return statuses

if __name__ == "__main__":
    unittest.main()
