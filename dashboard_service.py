"""Validated experiment adapter for the local dashboard; reuses research modules."""

from dataclasses import asdict
from importlib.metadata import version
import time

import networkx as nx
import numpy as np
import pennylane as qml

from ansatz import make_energy_qnode
from evaluate import evaluate
from feasible_join import make_plan_table, make_feasible_qnode
from join_problem import audit, build_join_qubo, subsets
from problem import from_adjacency, make_graph
from qubo import make_qubo_qnode
from train import train

PROBLEMS = {
    "maxcut": {"name": "Five-node Max-Cut", "qubits": 5, "description": "Split a five-node graph into two groups to maximize crossing edges."},
    "join3": {"name": "Three-table join · paper example", "qubits": 3, "description": "Choose the first join among AB, AC and BC. Retained costs: 2, 4 and 7."},
    "bushy4": {"name": "Four-table join · bushy", "qubits": 10, "description": "Synthetic costs favor (A ⋈ B) ⋈ (C ⋈ D). Enumerated reference: 15 valid plans."},
    "deep4": {"name": "Four-table join · deep", "qubits": 10, "description": "Synthetic costs favor ((A ⋈ B) ⋈ C) ⋈ D. Enumerated reference: 15 valid plans."},
    "bushy4_valid": {"name": "Four-table bushy · valid-plan mixer", "qubits": 4, "description": "Experimental: enumerate 15 valid plans, then mix only their indices. Enumeration already enables an exact classical solution."},
    "deep4_valid": {"name": "Four-table deep · valid-plan mixer", "qubits": 4, "description": "Experimental: enumerate 15 valid plans, then mix only their indices. Enumeration already enables an exact classical solution."},
}


def validate(data: dict) -> dict:
    """Allow only bounded inputs suitable for a local, single-worker simulator."""
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a JSON object.")
    defaults = dict(problem="maxcut", method="L-BFGS-B", p=2, seed=42,
                    sample_seed=123, shots=1000, maxiter=100, penalty="max_plus_one")
    if set(data) - set(defaults):
        raise ValueError("Unknown configuration fields.")
    config = defaults | data
    for key, options in (("problem", PROBLEMS), ("method", ("L-BFGS-B", "COBYLA", "ADAM")),
                         ("penalty", ("max_plus_one", "twice_max"))):
        if not isinstance(config[key], str) or config[key] not in options:
            raise ValueError(f"Invalid {key}.")
    for key, low, high in (("p", 1, 3), ("seed", 0, 2**32-1), ("sample_seed", 0, 2**32-1),
                           ("shots", 100, 10000), ("maxiter", 1, 200)):
        value = config[key]
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{key} must be an integer between {low} and {high}.")
    return config


def run_experiment(config: dict) -> dict:
    """Return provenance, full distribution, sampled counts and training history."""
    config = validate(config)
    start = time.perf_counter()
    options = {key: config[key] for key in ("method", "p", "seed", "maxiter")}
    plan_labels = None
    if config["problem"] == "maxcut":
        problem = from_adjacency(nx.to_numpy_array(make_graph(), nodelist=range(5)))
        result = train(make_energy_qnode(problem), **options)
        metrics = evaluate(problem, result.parameters, config["shots"], config["sample_seed"])
        labels = [format(k, "05b") for k in range(32)]
        probabilities, counts = metrics.probabilities, metrics.counts
        valid = np.ones(32, dtype=bool)
        optimal = metrics.cut_values == metrics.exact_cut
        values = metrics.cut_values
        exact = metrics.exact_cut
        expected = metrics.expected_cut
        expected_label = "Expected cut"
        extra = {"expected_approximation_ratio": metrics.expected_ratio,
                 "sampled_mean_cut": metrics.sampled_mean_cut,
                 "best_sampled_cut": metrics.best_sampled_cut}
        model = {"adjacency": problem.adjacency, "qubo": problem.qubo, "scale": 1.0,
                 "variables": [f"Node {i}" for i in range(5)]}
        uniform_valid_optimal = float(optimal.mean())
    elif config["problem"].endswith("_valid"):
        weights = [2,8,9,7,8,3,12,13,14,15] if config["problem"] == "bushy4_valid" else [2,8,9,7,8,10,3,12,13,14]
        table = make_plan_table(4, np.array(weights, dtype=float))
        lookup_start = time.perf_counter()
        best_index = int(np.argmin(table.costs))
        exact_seconds = table.preparation_seconds + time.perf_counter() - lookup_start
        result = train(make_feasible_qnode(table), **options)
        probabilities = np.asarray(make_feasible_qnode(table, True)(result.parameters))
        probabilities = probabilities / probabilities.sum()
        counts = np.random.default_rng(config["sample_seed"]).multinomial(config["shots"], probabilities)
        labels = [format(k, f"0{table.qubits}b") for k in range(len(probabilities))]
        valid = np.arange(len(probabilities)) < len(table.costs)
        values = np.zeros(len(probabilities)); values[valid] = table.costs
        exact = float(table.costs[best_index])
        optimal = valid & (values == exact)
        expected = float(probabilities[valid] @ values[valid] / probabilities[valid].sum())
        expected_label = "Expected retained plan cost"
        uniform_valid_optimal = float(np.mean(table.costs == exact))
        plan_labels = list(table.plans) + ["Unused code"] * (len(probabilities)-len(table.plans))
        extra = {"exact_enumeration_seconds": exact_seconds, "exact_plan": table.plans[best_index],
                 "preprocessing_seconds": table.preparation_seconds,
                 "uniform_valid_mean_cost": float(table.costs.mean())}
        model = {"encoding": "Enumerated valid-plan index", "scale": float(table.costs.max()),
                 "variables": [f"Index bit {i}" for i in range(table.qubits)],
                 "plan_table": list(table.plans), "plan_costs": table.costs,
                 "limitation": "Classical preprocessing enumerates and costs every plan. No speedup claim."}
    else:
        weights = {"join3": [2, 4, 7], "bushy4": [2,8,9,7,8,3,12,13,14,15],
                   "deep4": [2,8,9,7,8,10,3,12,13,14]}[config["problem"]]
        n = 3 if config["problem"] == "join3" else 4
        costs = np.array(weights, dtype=float)
        penalty = float(costs.max()+1 if config["penalty"] == "max_plus_one" else costs.max()*2)
        problem = build_join_qubo(n, costs, penalty)
        reference = audit(n, costs, problem)
        energy = make_qubo_qnode(problem)

        def angles(parameters):
            return qml.numpy.stack((parameters[0] / penalty, parameters[1]))

        def objective(parameters):
            return energy(angles(parameters)) / penalty

        result = train(objective, **options)
        probabilities = np.asarray(make_qubo_qnode(problem, True)(angles(result.parameters)))
        probabilities = probabilities / probabilities.sum()
        counts = np.random.default_rng(config["sample_seed"]).multinomial(config["shots"], probabilities)
        labels = reference["labels"]
        valid, optimal, values = reference["valid"], reference["optimal"], reference["costs"]
        exact = reference["exact_cost"]
        feasible = float(probabilities[valid].sum())
        expected = float(probabilities[valid] @ values[valid] / feasible) if feasible else None
        expected_label = "Cost given a valid plan"
        extra = {"raw_expected_qubo_energy": result.energy * penalty,
                 "retained_cost_excludes_root": True}
        model = {"qubo": problem.matrix, "scale": penalty, "subset_costs": weights,
                 "variables": ["".join(chr(65+i) for i in range(n) if mask & (1 << i)) for mask in subsets(n)]}
        uniform_valid_optimal = float(optimal.sum() / valid.sum())
    distribution = [{"bits": label, "probability": float(probabilities[k]), "count": int(counts[k]),
                     "valid": bool(valid[k]), "optimal": bool(optimal[k]),
                     "value": float(values[k]) if valid[k] else None,
                     "plan": plan_labels[k] if plan_labels else None} for k, label in enumerate(labels)]
    return {"config": config, "problem_name": PROBLEMS[config["problem"]]["name"],
            "simulation": "Ideal statevector; simulated samples, not quantum hardware",
            "elapsed_seconds": time.perf_counter()-start,
            "versions": {name: version(name) for name in ("numpy", "scipy", "pennylane")},
            "training": asdict(result), "model": model, "distribution": distribution,
            "metrics": {"exact_value": exact, "expected_value": expected, "expected_label": expected_label,
                        "optimal_probability": float(probabilities[optimal].sum()),
                        "valid_probability": float(probabilities[valid].sum()),
                        "optimal_samples": int(counts[optimal].sum()),
                        "uniform_valid_optimal_probability": uniform_valid_optimal, **extra}}
