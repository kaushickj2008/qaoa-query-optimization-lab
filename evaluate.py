"""Step 4: exact probabilities, simulated measurement, and Max-Cut metrics."""

from dataclasses import dataclass

import numpy as np
import pennylane as qml
from numpy.typing import NDArray

from ansatz import MixerLayer, qaoa_ansatz, transverse_field_mixer
from problem import MaxCutProblem


@dataclass(frozen=True)
class Evaluation:
    """Ratios use positive cut weights; None denotes a zero exact optimum."""

    probabilities: NDArray[np.float64]
    counts: NDArray[np.int64]
    cut_values: NDArray[np.float64]
    exact_cut: float
    expected_cut: float
    sampled_mean_cut: float
    best_sampled_cut: float
    expected_ratio: float | None
    best_sampled_ratio: float | None
    optimal_probability: float
    optimal_sample_fraction: float
    exact_bitstrings: list[str]
    best_sampled_bitstrings: list[str]


def evaluate(
    problem: MaxCutProblem,
    parameters: NDArray[np.float64],
    shots: int = 1000,
    seed: int = 123,
    mixer_layer: MixerLayer = transverse_field_mixer,
) -> Evaluation:
    """Evaluate a small trained circuit, enumerating all 2**n cuts (n <= 10).

    Basis index k uses wire 0 as its most significant bit. NumPy draws IID
    simulated computational-basis measurements from PennyLane probabilities;
    this is ideal sampling, not a hardware/noise simulation. Expected ratio
    is sum_x P(x) C(x) / C*, distinct from best-of-shots cut / C*.
    """
    n = len(problem.adjacency)
    if n > 10:
        raise ValueError("Exact evaluation is limited to 10 nodes.")
    if not isinstance(shots, int) or isinstance(shots, bool) or shots < 1:
        raise ValueError("shots must be a positive integer.")
    parameters = np.asarray(parameters, dtype=float)
    if not np.isfinite(parameters).all():
        raise ValueError("parameters must be finite.")
    device = qml.device("default.qubit", wires=n, shots=None)

    @qml.qnode(device)
    def distribution(angles):
        qaoa_ansatz(angles, problem, mixer_layer)
        return qml.probs(wires=range(n))

    probabilities = np.asarray(distribution(parameters), dtype=float)
    np.testing.assert_allclose(probabilities.sum(), 1.0, atol=1e-10)
    probabilities = probabilities / probabilities.sum()
    labels = [format(k, f"0{n}b") for k in range(2**n)]
    bits = np.array([[int(bit) for bit in label] for label in labels])
    cuts = np.zeros(2**n)
    for i in range(n):
        for j in range(i + 1, n):
            cuts += problem.adjacency[i, j] * (bits[:, i] != bits[:, j])
    draws = np.random.default_rng(seed).choice(2**n, size=shots, p=probabilities)
    counts = np.bincount(draws, minlength=2**n)
    exact = float(cuts.max())
    # Equal cut values are computed in the same edge order for every state.
    optimal = cuts == exact
    best = float(cuts[counts > 0].max())
    expected = float(probabilities @ cuts)
    return Evaluation(
        probabilities, counts, cuts, exact, expected, float(counts @ cuts / shots), best,
        expected / exact if exact > 0 else None,
        best / exact if exact > 0 else None,
        float(probabilities[optimal].sum()), float(counts[optimal].sum() / shots),
        [labels[k] for k in np.flatnonzero(optimal)],
        [labels[k] for k in np.flatnonzero((counts > 0) & (cuts == best))],
    )
