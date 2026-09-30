"""Step 2: QAOA state preparation and an analytic energy QNode."""

from collections.abc import Callable, Sequence

import networkx as nx
import numpy as np
import pennylane as qml
from numpy.typing import NDArray

from problem import MaxCutProblem, from_adjacency, make_graph

MixerLayer = Callable[[float, Sequence[int]], None]


def transverse_field_mixer(beta: float, wires: Sequence[int]) -> None:
    """Apply exp(-i beta H_M), H_M = sum_i X_i, using RX(2 beta)."""
    for wire in wires:
        qml.RX(2 * beta, wires=wire)


def maxcut_cost_layer(gamma: float, problem: MaxCutProblem) -> None:
    """Apply exp(-i gamma H_C), omitting only its global identity phase.

    IsingZZ(theta) = exp(-i theta Z_i Z_j / 2). Since H_C has
    coefficient A[i,j]/2, each edge uses theta = gamma * A[i,j].
    The full H_C, including its constant, is still used for measurement.
    """
    a = problem.adjacency
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if a[i, j] != 0:
                qml.IsingZZ(gamma * a[i, j], wires=(i, j))


def qaoa_ansatz(
    parameters: NDArray[np.float64],
    problem: MaxCutProblem,
    mixer_layer: MixerLayer = transverse_field_mixer,
) -> None:
    """Prepare p cost-then-mixer layers acting on |+>^n.

    Parameters have shape (2, p): row 0 is gamma, row 1 is beta.
    Exactly 2p angles are shared across edges/wires within each layer.
    A custom mixer callback must queue exp(-i beta H_M) on the given wires.
    This function queues gates and must run inside a PennyLane QNode.
    """
    if len(parameters.shape) != 2 or parameters.shape[0] != 2 or parameters.shape[1] < 1:
        raise ValueError("parameters must have shape (2, p), with p >= 1.")
    wires = tuple(range(len(problem.adjacency)))
    for wire in wires:
        qml.Hadamard(wires=wire)
    for layer in range(parameters.shape[1]):
        maxcut_cost_layer(parameters[0, layer], problem)
        mixer_layer(parameters[1, layer], wires)


def make_energy_qnode(
    problem: MaxCutProblem,
    mixer_layer: MixerLayer = transverse_field_mixer,
) -> qml.QNode:
    """Return differentiable <H_C> on an exact statevector simulator.

    Uses Autograd/backprop with shots=None. Lower energy means larger
    expected cut weight. No optimizer or parameter updates occur here.
    """
    device = qml.device("default.qubit", wires=len(problem.adjacency), shots=None)

    @qml.qnode(device, interface="autograd", diff_method="backprop")
    def energy(parameters: NDArray[np.float64]) -> float:
        qaoa_ansatz(parameters, problem, mixer_layer)
        return qml.expval(problem.hamiltonian)

    return energy


if __name__ == "__main__":
    problem = from_adjacency(nx.to_numpy_array(make_graph(), nodelist=range(5)))
    energy = make_energy_qnode(problem)
    # Illustrative angles only; Step 3 will train them.
    parameters = qml.numpy.array([[0.2, 0.4], [0.3, 0.1]], requires_grad=True)
    print(qml.draw(energy)(parameters))
    print(f"Energy: {float(energy(parameters)):.8f}")
    print(f"Expected cut: {-float(energy(parameters)):.8f}")
