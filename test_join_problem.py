"""Cross-check QUBO minima against independent recursive tree enumeration."""

import unittest
import numpy as np
from join_problem import audit, build_join_qubo, subsets, tree_selections


class JoinProblemTests(unittest.TestCase):
    def test_tree_counts(self):
        self.assertEqual(len(set(tree_selections(3))), 3)
        self.assertEqual(len(set(tree_selections(4))), 15)
        self.assertTrue(all(len(t) == 2 for t in tree_selections(4)))

    def test_random_costs_and_ties(self):
        rng = np.random.default_rng(2026)
        for n in (3, 4):
            with self.assertRaises(ValueError):
                build_join_qubo(n, np.zeros(len(subsets(n))), 1)
            for costs in [np.ones(len(subsets(n)))] + [rng.integers(1, 30, len(subsets(n))) for _ in range(10)]:
                for penalty in (costs.max() + 1, 2 * costs.max() + 1):
                    reference = audit(n, costs, build_join_qubo(n, costs, penalty))
                    self.assertEqual(int(reference["valid"].sum()), len(tree_selections(n)))

    def test_known_bushy_and_deep(self):
        for weights, optimum in (([2,8,9,7,8,3,12,13,14,15], "1000010000"),
                                 ([2,8,9,7,8,10,3,12,13,14], "1000001000")):
            costs = np.array(weights, dtype=float)
            reference = audit(4, costs, build_join_qubo(4, costs, costs.max()+1))
            self.assertEqual(reference["exact_cost"], 5)
            self.assertEqual([label for label, yes in zip(reference["labels"], reference["optimal"]) if yes], [optimum])


if __name__ == "__main__":
    unittest.main()
