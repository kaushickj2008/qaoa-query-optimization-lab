"""Independent cut/distribution checks, including a zero-optimum graph."""

import unittest
import numpy as np

from evaluate import evaluate
from problem import from_adjacency


class EvaluationTests(unittest.TestCase):
    def test_uniform_single_edge(self) -> None:
        problem = from_adjacency(np.array([[0., 2.], [2., 0.]]))
        result = evaluate(problem, np.zeros((2, 1)), shots=1000, seed=7)
        np.testing.assert_allclose(result.probabilities, [0.25] * 4)
        np.testing.assert_array_equal(result.cut_values, [0., 2., 2., 0.])
        self.assertAlmostEqual(result.expected_ratio, 0.5)
        self.assertAlmostEqual(result.optimal_probability, 0.5)
        self.assertEqual(result.exact_bitstrings, ["01", "10"])
        self.assertEqual(result.counts.sum(), 1000)
        self.assertEqual(result.best_sampled_ratio, 1.0)
        self.assertAlmostEqual(result.sampled_mean_cut, 2 * result.optimal_sample_fraction)
        again = evaluate(problem, np.zeros((2, 1)), shots=1000, seed=7)
        np.testing.assert_array_equal(result.counts, again.counts)

    def test_empty_graph_and_invalid_shots(self) -> None:
        problem = from_adjacency(np.zeros((2, 2)))
        result = evaluate(problem, np.zeros((2, 1)), shots=1)
        self.assertIsNone(result.expected_ratio)
        self.assertIsNone(result.best_sampled_ratio)
        self.assertAlmostEqual(result.optimal_probability, 1.)
        self.assertEqual(result.best_sampled_cut, 0.)
        for shots in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                evaluate(problem, np.zeros((2, 1)), shots=shots)


if __name__ == "__main__":
    unittest.main()
