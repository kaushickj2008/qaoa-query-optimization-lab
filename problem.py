"""Step 1: deterministic Max-Cut instance and equivalent minimization models."""

from dataclasses import dataclass
from itertools import product

import networkx as nx
import numpy as np
import pennylane as qml
from numpy.typing import NDArray


@dataclass(frozen=True)
class MaxCutProblem:
    """Wire i corresponds to adjacency row i; Q uses the x.T @ Q @ x convention."""

    adjacency: NDArray[np.float64]
    qubo: NDArray[np.float64]
    hamiltonian: qml.operation.Operator


def make_graph() -> nx.Graph:
    """Return an unweighted five-cycle with one chord, using wires 0,...,4."""
    graph = nx.cycle_graph(5)
    graph.add_edge(0, 2)
    return graph


def from_adjacency(adjacency: NDArray[np.float64]) -> MaxCutProblem:
    """Convert a nonnegative, symmetric, loop-free adjacency matrix to Max-Cut.

    For binary x, C(x) = sum_{i<j} A[i,j]*(x[i]+x[j]-2*x[i]*x[j]).
    We minimize -C: Q = A - diag(A @ 1), with energy x.T @ Q @ x.
    Substitution x_i=(1-Z_i)/2 gives H_C=sum_{i<j} A[i,j]*(Z_i Z_j-I)/2.
    Thus each computational-basis eigenvalue is the negative cut weight.
    """
    a = np.array(adjacency, dtype=float, copy=True)
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[0] != a.shape[1]:
        raise ValueError("Adjacency must be a nonempty square matrix.")
    if not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError("Weights must be finite and nonnegative.")
    if not np.array_equal(a, a.T) or np.any(np.diag(a) != 0):
        raise ValueError("Adjacency must be symmetric with zero diagonal.")

    qubo = a - np.diag(a.sum(axis=1))
    coefficients = [-float(np.triu(a, k=1).sum()) / 2.0]
    operators = [qml.Identity(0)]
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if a[i, j] != 0:
                coefficients.append(float(a[i, j]) / 2.0)
                operators.append(qml.PauliZ(i) @ qml.PauliZ(j))
    return MaxCutProblem(a, qubo, qml.Hamiltonian(coefficients, operators))


def verify_small_instance(problem: MaxCutProblem) -> tuple[float, list[str]]:
    """Check every basis state against direct cuts; exponential, limited to 10 nodes.

    Explicit wire_order makes the leftmost bit wire 0. This is a formulation
    check and exact reference, not a scalable classical optimization algorithm.
    """
    n = len(problem.adjacency)
    if n > 10:
        raise ValueError("Dense exhaustive verification is limited to 10 nodes.")
    matrix = qml.matrix(problem.hamiltonian, wire_order=list(range(n)))
    np.testing.assert_allclose(matrix, np.diag(np.diag(matrix)))
    best_cut, solutions = -1.0, []
    for index, bits in enumerate(product((0, 1), repeat=n)):
        x = np.array(bits, dtype=float)
        cut = sum(problem.adjacency[i, j] for i in range(n)
                  for j in range(i + 1, n) if bits[i] != bits[j])
        np.testing.assert_allclose(x @ problem.qubo @ x, -cut, atol=1e-10)
        np.testing.assert_allclose(matrix[index, index], -cut, atol=1e-10)
        bitstring = "".join(map(str, bits))
        if cut > best_cut:
            best_cut, solutions = float(cut), [bitstring]
        elif cut == best_cut:
            solutions.append(bitstring)
    return best_cut, solutions


if __name__ == "__main__":
    graph = make_graph()
    problem = from_adjacency(nx.to_numpy_array(graph, nodelist=range(5)))
    optimum, bitstrings = verify_small_instance(problem)
    print("Adjacency:\n", problem.adjacency)
    print("QUBO (minimize x.T @ Q @ x):\n", problem.qubo)
    print("Cost Hamiltonian:", problem.hamiltonian)
    print("Verified all 32 basis states.")
    print("Maximum cut:", optimum)
    print("Optimal bitstrings (wire 0 first):", bitstrings)
