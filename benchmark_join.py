"""Four-table synthetic benchmark: an extension, not a paper-results replication."""

import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np

from join_problem import audit, build_join_qubo, subsets
from qubo import make_qubo_qnode
from train import train


def run() -> dict:
    """Compare two penalty choices over 24 fixed-seed QAOA runs.

    Divide each Hamiltonian by its penalty to control overall energy scale.
    Report unscaled energy and feasible plan cost separately. This is a small
    diagnostic, not a statistically powered or time-to-solution comparison.
    """
    # Variable order: AB AC AD BC BD CD ABC ABD ACD BCD.
    instances = {"bushy_favored": [2, 8, 9, 7, 8, 3, 12, 13, 14, 15],
                 "deep_favored": [2, 8, 9, 7, 8, 10, 3, 12, 13, 14]}
    reports = []
    for name, weights in instances.items():
        costs = np.array(weights, dtype=float)
        for rule, penalty in (("max_plus_one", costs.max() + 1), ("twice_max", 2 * costs.max())):
            problem = build_join_qubo(4, costs, float(penalty))
            reference = audit(4, costs, problem)
            energy_node = make_qubo_qnode(problem)
            probability_node = make_qubo_qnode(problem, probabilities=True)

            def normalized_energy(angles):
                return energy_node(angles) / penalty

            # Scale cost evolution too, not only the measured objective.
            def scaled_angles(angles):
                import pennylane as qml
                return qml.numpy.stack((angles[0] / penalty, angles[1]))

            def objective(angles):
                return normalized_energy(scaled_angles(angles))

            runs = []
            for p in (1, 2):
                for seed in range(3):
                    result = train(objective, p=p, seed=seed, maxiter=100)
                    probs = np.asarray(probability_node(scaled_angles(result.parameters)))
                    np.testing.assert_allclose(probs.sum(), 1, atol=1e-9)
                    np.testing.assert_allclose(probs @ reference["energies"], result.energy * penalty, atol=1e-8)
                    valid = reference["valid"]
                    feasible = float(probs[valid].sum())
                    runs.append({"p": p, "seed": seed, "training": asdict(result),
                                 "probabilities": probs,
                                 "optimal_probability": float(probs[reference["optimal"]].sum()),
                                 "feasible_probability": feasible,
                                 "conditional_valid_cost": float(probs[valid] @ reference["costs"][valid] / feasible) if feasible > 0 else None})
            summary = []
            for p in (1, 2):
                group = [r for r in runs if r["p"] == p]
                optimal = [r["optimal_probability"] for r in group]
                summary.append({"p": p, "mean_optimal_probability": float(np.mean(optimal)),
                                "sample_std_optimal_probability": float(np.std(optimal, ddof=1)),
                                "mean_feasible_probability": float(np.mean([r["feasible_probability"] for r in group])),
                                "successes": sum(r["training"]["success"] for r in group)})
            reports.append({"instance": name, "subset_costs": weights, "penalty_rule": rule,
                            "penalty": float(penalty), "qubo": problem.matrix,
                            "exact_cost": reference["exact_cost"], "tree_count": reference["tree_count"],
                            "optimal_bitstrings": [label for label, yes in zip(reference["labels"], reference["optimal"]) if yes],
                            "uniform_optimal_probability": float(reference["optimal"].mean()),
                            "uniform_feasible_probability": float(reference["valid"].mean()),
                            "summary": summary, "runs": runs})
            print(name, rule, summary, flush=True)
    return {"scope": "Synthetic four-table unsplit subset encoding; not paper performance replication",
            "variables": ["".join(chr(65+i) for i in range(4) if mask & (1 << i)) for mask in subsets(4)],
            "configuration": {"optimizer": "L-BFGS-B", "maxiter": 100, "tolerance": 1e-6,
                              "seeds": [0, 1, 2], "depths": [1, 2], "shots": None,
                              "cost_evolution_and_objective_scale": "1/penalty",
                              "initialization": "uniform(-0.5,0.5), numpy default_rng(seed)"},
            "versions": {name: version(name) for name in ("numpy", "scipy", "pennylane")},
            "python": platform.python_version(), "experiments": reports}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/four_table/report.json"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path.")
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False, default=lambda x: x.tolist())
    print(args.output.resolve())
