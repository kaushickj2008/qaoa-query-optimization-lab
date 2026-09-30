"""Run Steps 1–4 and save a reproducible report and plots to a new directory."""

import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from ansatz import make_energy_qnode
from evaluate import Evaluation, evaluate
from problem import from_adjacency, make_graph
from train import TrainingResult, train


def plot_results(result: TrainingResult, metrics: Evaluation, directory: Path) -> None:
    """Save convergence and basis probabilities; history counts objective calls."""
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    calls = np.arange(1, len(result.energy_history) + 1)
    ax.plot(calls, -result.energy_history, "o-", alpha=0.65, label="Evaluated expected cut")
    ax.plot(calls, -np.minimum.accumulate(result.energy_history), label="Best so far")
    ax.axhline(metrics.exact_cut, color="black", linestyle="--", label="Exact maximum cut")
    ax.set(xlabel="Objective evaluation (not iteration)", ylabel="Expected cut weight",
           title="QAOA training — ideal statevector")
    ax.legend()
    fig.savefig(directory / "convergence.png", dpi=180)
    plt.close(fig)

    # Show at most 32 states for readable plots; JSON retains the full distribution.
    indices = np.argsort(-metrics.probabilities, kind="stable")[:32]
    n = int(np.log2(len(metrics.probabilities)))
    positions = np.arange(len(indices))
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    ax.bar(positions - 0.2, metrics.probabilities[indices], 0.4, label="Exact probability")
    ax.bar(positions + 0.2, metrics.counts[indices] / metrics.counts.sum(), 0.4,
           label="Sample frequency")
    ax.set_xticks(positions, [format(int(k), f"0{n}b") for k in indices], rotation=90)
    ax.set(xlabel="Bitstring (wire 0 first)", ylabel="Probability / frequency",
           title="Final distribution — up to 32 most probable states")
    ax.legend()
    fig.savefig(directory / "probabilities.png", dpi=180)
    plt.close(fig)


def main() -> None:
    """Train and evaluate the five-node example; refuse to overwrite a run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["L-BFGS-B", "COBYLA", "ADAM"], default="L-BFGS-B")
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-seed", type=int, default=123)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("results/demo"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists; choose a new --output path.")
    problem = from_adjacency(nx.to_numpy_array(make_graph(), nodelist=range(5)))
    result = train(make_energy_qnode(problem), p=args.p, method=args.method,
                   seed=args.seed, maxiter=args.maxiter)
    metrics = evaluate(problem, result.parameters, shots=args.shots, seed=args.sample_seed)
    np.testing.assert_allclose(metrics.expected_cut, -result.energy, atol=1e-9)
    args.output.mkdir(parents=True)
    report = {
        "configuration": {**vars(args), "output": str(args.output),
                          "tolerance": 1e-6, "learning_rate": 0.05},
        "versions": {name: version(name) for name in ("numpy", "scipy", "pennylane", "networkx", "matplotlib")},
        "python": platform.python_version(), "adjacency": problem.adjacency,
        "training": asdict(result), "evaluation": asdict(metrics),
        "conventions": "Array index is basis integer; wire 0 is most significant bit. Ideal simulator.",
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False,
                   default=lambda value: value.tolist()) + "\n", encoding="utf-8")
    plot_results(result, metrics, args.output)
    print(f"Expected cut: {metrics.expected_cut:.6f} / exact {metrics.exact_cut:g}")
    print(f"Expected approximation ratio: {metrics.expected_ratio}")
    print(f"Best sampled cut: {metrics.best_sampled_cut:g}; ratio: {metrics.best_sampled_ratio}")
    print(f"Optimal-state probability: {metrics.optimal_probability:.6f}")
    print(f"Sampled mean cut: {metrics.sampled_mean_cut:.6f}")
    print(f"Best sampled bitstrings: {metrics.best_sampled_bitstrings}")
    print(f"Optimizer success: {result.success}; {result.message}")
    print(f"Artifacts: {args.output.resolve()}")


if __name__ == "__main__":
    main()
