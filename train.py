"""Step 3: reproducible classical optimization of an analytic QAOA energy."""

import argparse
from collections.abc import Callable
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pennylane as qml
from numpy.typing import NDArray
from scipy.optimize import minimize

from ansatz import make_energy_qnode
from problem import from_adjacency, make_graph


@dataclass(frozen=True)
class TrainingResult:
    """Best evaluated angles and raw objective-evaluation history.

    History includes initialization and trial points (not optimizer iterations).
    Gradient evaluations are counted separately. Success denotes the stopping
    criterion, never proof of global optimality.
    """

    parameters: NDArray[np.float64]
    energy: float
    energy_history: NDArray[np.float64]
    gradient_evaluations: int
    success: bool
    message: str


def train(
    energy: Callable,
    p: int = 2,
    method: str = "L-BFGS-B",
    seed: int = 42,
    maxiter: int = 200,
    tolerance: float = 1e-6,
    learning_rate: float = 0.05,
    initial_parameters: NDArray[np.float64] | None = None,
) -> TrainingResult:
    """Minimize <H_C> using L-BFGS-B, COBYLA, or Adam.

    The callable must accept Autograd arrays of shape (2, p). Gradient-based
    methods differentiate it with qml.grad; COBYLA uses only scalar energies.
    Seeded nonzero initialization avoids the stationary all-zero angles.
    Adam stops when the gradient infinity norm is <= tolerance. SciPy owns
    its stopping criteria; COBYLA's maxiter is a function-evaluation budget.
    Best observed parameters are retained even if the budget is exhausted.
    """
    method = method.upper()
    if method not in {"L-BFGS-B", "COBYLA", "ADAM"}:
        raise ValueError("method must be L-BFGS-B, COBYLA, or ADAM.")
    if not isinstance(p, int) or isinstance(p, bool) or p < 1:
        raise ValueError("p must be a positive integer.")
    if not isinstance(maxiter, int) or isinstance(maxiter, bool) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer.")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive.")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    angles = (np.random.default_rng(seed).uniform(-0.5, 0.5, (2, p))
              if initial_parameters is None else np.array(initial_parameters, dtype=float, copy=True))
    if angles.shape != (2, p) or not np.isfinite(angles).all():
        raise ValueError("initial_parameters must be finite with shape (2, p).")

    history: list[float] = []
    best_energy = float("inf")
    best_angles = angles.copy()
    gradient_evaluations = 0

    def objective(flat: NDArray[np.float64]) -> float:
        nonlocal best_energy, best_angles
        value = float(energy(qml.numpy.array(flat.reshape(2, p), requires_grad=False)))
        if not np.isfinite(value):
            raise FloatingPointError("Nonfinite energy encountered.")
        history.append(value)
        if value < best_energy:
            best_energy, best_angles = value, flat.reshape(2, p).copy()
        return value

    derivative = qml.grad(energy)

    def gradient(flat: NDArray[np.float64]) -> NDArray[np.float64]:
        nonlocal gradient_evaluations
        gradient_evaluations += 1
        value = np.asarray(derivative(qml.numpy.array(flat.reshape(2, p), requires_grad=True)),
                           dtype=float).reshape(-1)
        if not np.isfinite(value).all():
            raise FloatingPointError("Nonfinite gradient encountered.")
        return value

    flat = angles.reshape(-1).copy()
    objective(flat)
    if method == "ADAM":
        # Bias-corrected first and second moments, with standard decay values.
        first, second = np.zeros_like(flat), np.zeros_like(flat)
        success, message = False, "Adam update budget exhausted."
        for step in range(1, maxiter + 1):
            grad = gradient(flat)
            if np.linalg.norm(grad, ord=np.inf) <= tolerance:
                success, message = True, "Adam gradient tolerance reached."
                break
            first = 0.9 * first + 0.1 * grad
            second = 0.999 * second + 0.001 * grad**2
            flat -= learning_rate * (first / (1 - 0.9**step)) / (
                np.sqrt(second / (1 - 0.999**step)) + 1e-8)
            objective(flat)
    else:
        result = minimize(objective, flat, method=method,
                          jac=gradient if method == "L-BFGS-B" else None,
                          tol=tolerance, options={"maxiter": maxiter})
        success, message = bool(result.success), str(result.message)

    return TrainingResult(best_angles, best_energy, np.array(history),
                          gradient_evaluations, success, message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["L-BFGS-B", "COBYLA", "ADAM"], default="L-BFGS-B")
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--maxiter", type=int, default=200)
    args = parser.parse_args()
    problem = from_adjacency(nx.to_numpy_array(make_graph(), nodelist=range(5)))
    result = train(make_energy_qnode(problem), p=args.p, method=args.method,
                   seed=args.seed, maxiter=args.maxiter)
    print(f"Initial energy: {result.energy_history[0]:.8f}")
    print(f"Best energy: {result.energy:.8f}; expected cut: {-result.energy:.8f}")
    print(f"Objective evaluations: {len(result.energy_history)}; gradient evaluations: {result.gradient_evaluations}")
    print(f"Optimizer success: {result.success}; {result.message}")
    print("Best parameters (gamma row, beta row):\n", result.parameters)
