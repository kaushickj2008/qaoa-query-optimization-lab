"""Integration checks for the Step 3 hybrid loop."""

import unittest

import networkx as nx
import numpy as np

from ansatz import make_energy_qnode
from problem import from_adjacency, make_graph
from train import train


class TrainingTests(unittest.TestCase):
    def test_optimizers_improve_actual_qaoa(self) -> None:
        problem = from_adjacency(nx.to_numpy_array(make_graph(), nodelist=range(5)))
        energy = make_energy_qnode(problem)
        for method in ("L-BFGS-B", "COBYLA", "ADAM"):
            with self.subTest(method=method):
                result = train(energy, method=method, maxiter=150)
                self.assertLess(result.energy, result.energy_history[0] - 0.1)
                self.assertGreaterEqual(result.energy, -5.0 - 1e-9)
                self.assertEqual(result.parameters.shape, (2, 2))
                self.assertAlmostEqual(float(energy(result.parameters)), result.energy)
                self.assertAlmostEqual(result.energy, min(result.energy_history))
                self.assertEqual(result.gradient_evaluations == 0, method == "COBYLA")

    def test_reproducibility_and_budget_status(self) -> None:
        energy = make_energy_qnode(from_adjacency(np.array([[0., 1.], [1., 0.]])))
        first = train(energy, p=1, method="ADAM", maxiter=1)
        second = train(energy, p=1, method="ADAM", maxiter=1)
        np.testing.assert_allclose(first.parameters, second.parameters)
        np.testing.assert_allclose(first.energy_history, second.energy_history)
        self.assertFalse(first.success)
        self.assertEqual(len(first.energy_history), 2)
        with self.assertRaises(ValueError):
            train(energy, p=2, initial_parameters=np.zeros((2, 1)))


if __name__ == "__main__":
    unittest.main()
