"""Independent dense-matrix checks for the small Step 2 circuit."""

import unittest

import networkx as nx
import numpy as np
import pennylane as qml
from scipy.linalg import expm

from ansatz import make_energy_qnode, qaoa_ansatz
from problem import from_adjacency, make_graph


class AnsatzTests(unittest.TestCase):
    def test_matrix_evolution(self) -> None:
        """Detect wrong signs, factors of two, wire order and layer order."""
        for a in (nx.to_numpy_array(make_graph(), nodelist=range(5)),
                  np.array([[0., 0.7, 1.3], [0.7, 0., 0.], [1.3, 0., 0.]]),
                  np.zeros((2, 2))):
            problem = from_adjacency(a)
            n = len(a)
            hc = qml.matrix(problem.hamiltonian, wire_order=range(n))
            hm = qml.matrix(qml.Hamiltonian([1.] * n, [qml.X(i) for i in range(n)]),
                            wire_order=range(n))
            device = qml.device("default.qubit", wires=n)

            @qml.qnode(device)
            def state(parameters):
                qaoa_ansatz(parameters, problem)
                return qml.state()

            for p in (1, 2, 3):
                angles = np.random.default_rng(42 + p).uniform(-0.8, 0.8, (2, p))
                reference = np.ones(2**n, dtype=complex) / np.sqrt(2**n)
                for gamma, beta in angles.T:
                    reference = expm(-1j * beta * hm) @ (expm(-1j * gamma * hc) @ reference)
                actual = state(angles)
                # Projectors remove the intentionally omitted global phase.
                np.testing.assert_allclose(np.outer(actual, actual.conj()),
                                           np.outer(reference, reference.conj()), atol=1e-10)
                np.testing.assert_allclose(make_energy_qnode(problem)(angles),
                                           np.vdot(reference, hc @ reference).real, atol=1e-10)

    def test_zero_angles_and_gradients(self) -> None:
        problem = from_adjacency(nx.to_numpy_array(make_graph(), nodelist=range(5)))
        energy = make_energy_qnode(problem)
        self.assertAlmostEqual(float(energy(np.zeros((2, 2)))), -3.0)
        angles = qml.numpy.array([[0.2, -0.4], [0.3, 0.1]], requires_grad=True)
        gradient = qml.grad(energy)(angles)
        numeric = np.zeros((2, 2))
        for index in np.ndindex(2, 2):
            delta = np.zeros((2, 2))
            delta[index] = 1e-6
            numeric[index] = (energy(angles + delta) - energy(angles - delta)) / 2e-6
        np.testing.assert_allclose(gradient, numeric, atol=1e-7)
        for shape in ((2, 0), (3, 2), (4,)):
            with self.assertRaises(ValueError):
                energy(np.zeros(shape))


if __name__ == "__main__":
    unittest.main()
