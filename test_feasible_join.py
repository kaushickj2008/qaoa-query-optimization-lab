"""Verify plan costs, validity preservation, gradients and exact evolution."""

import unittest
import numpy as np
import pennylane as qml
from scipy.linalg import expm
from feasible_join import make_plan_table, make_feasible_qnode


class FeasibleJoinTests(unittest.TestCase):
    def test_plan_costs_and_padding(self):
        for n, weights, size in ((3, [2,4,7], 3), (4, [2,8,9,7,8,3,12,13,14,15], 15)):
            table = make_plan_table(n, np.array(weights))
            self.assertEqual(len(table.plans), size)
            self.assertEqual(len(set(table.plans)), size)
            self.assertEqual(table.costs.min(), 2 if n == 3 else 5)
            distribution = make_feasible_qnode(table, True)
            for p in (1,2,3):
                angles = np.random.default_rng(p).uniform(-4,4,(2,p))
                probs = distribution(angles)
                self.assertAlmostEqual(float(probs[:size].sum()), 1.0)
                np.testing.assert_allclose(probs[size:], 0, atol=1e-12)

    def test_dense_evolution_and_gradients(self):
        table = make_plan_table(3, np.array([2,4,7]))
        s = np.array([1,1,1,0], dtype=complex) / np.sqrt(3)
        cost = np.diag([2/7,4/7,1,0])
        mixer = np.outer(s, s.conj())
        angles = qml.numpy.array([[0.4,-0.3],[0.2,0.7]], requires_grad=True)
        state = s.copy()
        for gamma, beta in angles.T:
            state = expm(-1j*beta*mixer) @ expm(-1j*gamma*cost) @ state
        np.testing.assert_allclose(make_feasible_qnode(table,True)(angles), abs(state)**2, atol=1e-10)
        energy = make_feasible_qnode(table)
        gradient = qml.grad(energy)(angles)
        for index in np.ndindex(2,2):
            delta=np.zeros((2,2)); delta[index]=1e-6
            self.assertAlmostEqual(float(gradient[index]),float((energy(angles+delta)-energy(angles-delta))/2e-6),places=6)


if __name__ == "__main__":
    unittest.main()
