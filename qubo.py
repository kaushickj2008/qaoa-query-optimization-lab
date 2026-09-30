"""General symmetric QUBO support, independent of the Max-Cut cost layer."""

from dataclasses import dataclass
import numpy as np
import pennylane as qml
from numpy.typing import NDArray


@dataclass(frozen=True)
class Qubo:
    """E(x)=x.T Q x = offset + sum h_i Z_i + sum_{i<j} J_ij Z_i Z_j."""

    matrix: NDArray[np.float64]
    offset: float
    fields: NDArray[np.float64]
    couplings: NDArray[np.float64]
    hamiltonian: qml.operation.Operator


def from_qubo(matrix: NDArray[np.float64]) -> Qubo:
    """Use x_i=(1-Z_i)/2; symmetric off-diagonals count twice in x.T Q x."""
    q = np.array(matrix, dtype=float, copy=True)
    if q.ndim != 2 or q.shape[0] == 0 or q.shape[0] != q.shape[1]:
        raise ValueError("Q must be a nonempty square matrix.")
    if not np.isfinite(q).all() or not np.array_equal(q, q.T):
        raise ValueError("Q must be finite and symmetric.")
    offset = float(np.trace(q) / 2 + np.triu(q, 1).sum() / 2)
    fields = -q.sum(axis=1) / 2
    couplings = np.triu(q, 1) / 2
    coefficients = [offset] + fields.tolist()
    operators = [qml.Identity(0)] + [qml.Z(i) for i in range(len(q))]
    for i, j in zip(*np.nonzero(couplings)):
        coefficients.append(float(couplings[i, j]))
        operators.append(qml.Z(int(i)) @ qml.Z(int(j)))
    return Qubo(q, offset, fields, couplings, qml.Hamiltonian(coefficients, operators))


def make_qubo_qnode(problem: Qubo, probabilities: bool = False) -> qml.QNode:
    """Analytic QAOA with exact commuting Z/ZZ evolution and positive X mixer.

    Identity evolution is omitted as a global phase, but measured energy
    includes it. Parameter shape and order match train.py: (gamma, beta).
    """
    n = len(problem.matrix)
    device = qml.device("default.qubit", wires=n, shots=None)

    @qml.qnode(device, interface="autograd", diff_method="backprop")
    def circuit(parameters):
        if len(parameters.shape) != 2 or parameters.shape[0] != 2 or parameters.shape[1] < 1:
            raise ValueError("parameters must have shape (2, p), p >= 1.")
        for i in range(n):
            qml.Hadamard(i)
        for layer in range(parameters.shape[1]):
            gamma, beta = parameters[0, layer], parameters[1, layer]
            for i in range(n):
                qml.RZ(2 * gamma * problem.fields[i], wires=i)
            for i, j in zip(*np.nonzero(problem.couplings)):
                qml.IsingZZ(2 * gamma * problem.couplings[i, j], wires=(int(i), int(j)))
            for i in range(n):
                qml.RX(2 * beta, wires=i)
        return qml.probs(wires=range(n)) if probabilities else qml.expval(problem.hamiltonian)

    return circuit
