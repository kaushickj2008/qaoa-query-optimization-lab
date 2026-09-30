"""Validate generic QUBO gates independently using dense matrix evolution."""

import unittest
import numpy as np
import pennylane as qml
from scipy.linalg import expm
from qubo import from_qubo, make_qubo_qnode
from reproduce_join import paper_example


class QuboTests(unittest.TestCase):
    def test_conversion_and_evolution(self):
        for problem in (paper_example(), from_qubo(np.array([[1., -0.3], [-0.3, 2.]]))):
            n = len(problem.matrix)
            bits = np.array([[int(c) for c in format(k, f"0{n}b")] for k in range(2**n)])
            hc = qml.matrix(problem.hamiltonian, wire_order=range(n))
            np.testing.assert_allclose(hc, np.diag(np.einsum("bi,ij,bj->b", bits, problem.matrix, bits)), atol=1e-12)
            hm = qml.matrix(qml.Hamiltonian([1.] * n, [qml.X(i) for i in range(n)]), wire_order=range(n))
            angles = qml.numpy.array([[0.23, -0.31], [0.17, 0.42]], requires_grad=True)
            state = np.ones(2**n, dtype=complex) / np.sqrt(2**n)
            for gamma, beta in angles.T:
                state = expm(-1j * beta * hm) @ expm(-1j * gamma * hc) @ state
            np.testing.assert_allclose(make_qubo_qnode(problem, True)(angles), abs(state)**2, atol=1e-10)
            energy = make_qubo_qnode(problem)
            np.testing.assert_allclose(energy(angles), np.vdot(state, hc @ state).real, atol=1e-10)
            grad = qml.grad(energy)(angles)
            for index in np.ndindex(2, 2):
                delta = np.zeros((2, 2)); delta[index] = 1e-6
                self.assertAlmostEqual(float(grad[index]), float((energy(angles+delta)-energy(angles-delta))/2e-6), places=6)


if __name__ == "__main__":
    unittest.main()
