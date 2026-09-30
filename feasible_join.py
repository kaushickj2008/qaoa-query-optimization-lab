"""Enumerated-plan Grover-mixer baseline for tiny join problems.

Inspired by https://arxiv.org/abs/2006.00354 (not a new mixer proposal).
Classical enumeration constructs the entire plan/cost table first, so it has
already done enough work to solve the problem exactly. This is a controlled
feasibility experiment, not a scalable algorithm or speedup demonstration.
"""

from dataclasses import dataclass
import math
from time import perf_counter

import numpy as np
import pennylane as qml

from join_problem import subsets, tree_selections


@dataclass(frozen=True)
class PlanTable:
    plans: tuple[str, ...]
    costs: np.ndarray
    qubits: int
    preparation_seconds: float


def make_plan_table(n: int, costs: np.ndarray) -> PlanTable:
    """Enumerate every unordered binary tree and its additive retained cost."""
    start = perf_counter()
    variables = subsets(n)
    costs = np.asarray(costs, dtype=float)
    if costs.shape != (len(variables),) or not np.isfinite(costs).all() or np.any(costs <= 0):
        raise ValueError("One strictly positive finite cost per subset is required.")
    trees = sorted(tree_selections(n), key=lambda tree: tuple(sorted(tree)))

    def decode(mask: int, tree: frozenset[int]) -> str:
        if mask.bit_count() == 1:
            return chr(65 + mask.bit_length() - 1)
        children = {1 << i for i in range(n) if mask & (1 << i)} | set(tree)
        children = {x for x in children if x != mask and x & mask == x}
        children = sorted(x for x in children if not any(x != y and x & y == x for y in children))
        if len(children) != 2:
            raise ValueError("Invalid binary join tree.")
        return f"({decode(children[0], tree)} ⋈ {decode(children[1], tree)})"

    labels = tuple(decode((1 << n)-1, tree) for tree in trees)
    values = np.array([sum(costs[variables.index(v)] for v in tree) for tree in trees])
    return PlanTable(labels, values, math.ceil(math.log2(len(trees))), perf_counter()-start)


def make_feasible_qnode(table: PlanTable, probabilities: bool = False) -> qml.QNode:
    """Prepare uniform valid-plan indices and alternate cost and projector mixer.

    H_M=|s><s|, U_M=I+(exp(-i beta)-1)|s><s|. Since |s> has zero
    amplitude on padding states, this mixer and the diagonal cost preserve
    validity exactly in ideal arithmetic. Cost is normalized by max cost.
    Uses dense gates on 2 or 4 qubits; no hardware gate-efficiency claim.
    """
    dimension = 2**table.qubits
    state = np.zeros(dimension)
    state[:len(table.costs)] = 1 / np.sqrt(len(table.costs))
    diagonal = np.zeros(dimension)
    diagonal[:len(table.costs)] = table.costs / table.costs.max()
    projector = np.outer(state, state)
    device = qml.device("default.qubit", wires=table.qubits)

    @qml.qnode(device, interface="autograd", diff_method="backprop")
    def circuit(parameters):
        if len(parameters.shape) != 2 or parameters.shape[0] != 2 or parameters.shape[1] < 1:
            raise ValueError("parameters must have shape (2, p), p >= 1.")
        qml.StatePrep(state, wires=range(table.qubits))
        for gamma, beta in parameters.T:
            qml.DiagonalQubitUnitary(qml.numpy.exp(-1j * gamma * diagonal), wires=range(table.qubits))
            mixer = np.eye(dimension) + (qml.numpy.exp(-1j * beta)-1) * projector
            qml.QubitUnitary(mixer, wires=range(table.qubits))
        return qml.probs(wires=range(table.qubits)) if probabilities else qml.expval(qml.Hermitian(np.diag(diagonal), wires=range(table.qubits)))

    return circuit
