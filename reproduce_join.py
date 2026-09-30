"""Partial replication: Nayak et al. (2024), section 3.3, equation (7).

Source: https://doi.org/10.1007/s13222-024-00468-3
Only the worked QUBO is replicated. QAOA settings below are our experiment,
not an attempt to reproduce the paper's five-relation performance figure.
"""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np
import pennylane as qml

from qubo import from_qubo, make_qubo_qnode
from train import train


def paper_example():
    """Variables select joins R1R2, R1R3, R2R3; costs 2,4,7; w_max=8.

    Each unordered conflict pair receives 8, following explicit equation (7).
    Hence each symmetric Q off-diagonal is 4, not 8.
    """
    return from_qubo(np.array([[-6., 4., 4.], [4., -4., 4.], [4., 4., -1.]]))


def run() -> dict:
    """Audit eight bitstrings, then run 5 seeds at p=1,2 with two optimizers."""
    problem = paper_example()
    labels = [format(k, "03b") for k in range(8)]
    bits = np.array([[int(c) for c in label] for label in labels])
    energies = np.einsum("bi,ij,bj->b", bits, problem.matrix, bits)
    # Independently hand-evaluated equation (7), in ascending basis order.
    np.testing.assert_allclose(energies, [0, -1, -4, 3, -6, 1, -2, 13])
    np.testing.assert_allclose(np.diag(qml.matrix(problem.hamiltonian, wire_order=range(3))), energies)
    valid = bits.sum(axis=1) == 1
    costs = bits @ np.array([2., 4., 7.])
    rows = []
    for method in ("L-BFGS-B", "COBYLA"):
        for p in (1, 2):
            for seed in range(5):
                result = train(make_qubo_qnode(problem), p=p, method=method, seed=seed, maxiter=200)
                probs = np.asarray(make_qubo_qnode(problem, probabilities=True)(result.parameters))
                np.testing.assert_allclose(probs @ energies, result.energy, atol=1e-9)
                feasible = float(probs[valid].sum())
                rows.append({"method": method, "p": p, "seed": seed,
                             "energy": result.energy, "energy_gap": result.energy + 6,
                             "optimal_probability": float(probs[4]), "feasible_probability": feasible,
                             "conditional_valid_join_cost": float(probs[valid] @ costs[valid] / feasible) if feasible > 0 else None,
                             "parameters": result.parameters.tolist(), "probabilities": probs.tolist(),
                             "energy_history": result.energy_history.tolist(),
                             "objective_evaluations": len(result.energy_history),
                             "gradient_evaluations": result.gradient_evaluations,
                             "success": result.success, "message": result.message})
    summary = []
    for method in ("L-BFGS-B", "COBYLA"):
        for p in (1, 2):
            group = [r for r in rows if r["method"] == method and r["p"] == p]
            values = [r["optimal_probability"] for r in group]
            summary.append({"method": method, "p": p, "seeds": 5,
                            "mean_optimal_probability": float(np.mean(values)),
                            "sample_std_optimal_probability": float(np.std(values, ddof=1)),
                            "mean_feasible_probability": float(np.mean([r["feasible_probability"] for r in group])),
                            "optimizer_successes": sum(r["success"] for r in group)})
    return {"source": "https://doi.org/10.1007/s13222-024-00468-3",
            "scope": "Section 3.3 equation (7) formulation replication; independent QAOA experiment",
            "configuration": {"maxiter": 200, "tolerance": 1e-6, "shots": None,
                              "initialization": "numpy default_rng(seed), uniform(-0.5,0.5), shape (2,p)"},
            "python": platform.python_version(),
            "versions": {name: version(name) for name in ("numpy", "scipy", "pennylane")},
            "qubo": problem.matrix.tolist(), "bitstrings": labels, "energies": energies.tolist(),
            "exact_solution": "100", "retained_join_cost": 2,
            "uniform_baseline": {"optimal_probability": 1/8, "feasible_probability": 3/8,
                                 "conditional_valid_join_cost": 13/3},
            "summary": summary, "runs": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/join_replication/report.json"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path; existing report will not be overwritten.")
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(report["summary"], indent=2))
    print(f"Exact solution: 100, energy -6, retained join cost 2. Report: {args.output}")
