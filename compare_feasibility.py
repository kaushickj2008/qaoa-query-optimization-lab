"""Compare tiny enumerated-plan QAOA with penalty QAOA and classical references."""

import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pennylane as qml

from feasible_join import make_plan_table, make_feasible_qnode
from join_problem import audit, build_join_qubo
from qubo import make_qubo_qnode
from train import train


def run() -> dict:
    instances = {"bushy4":[2,8,9,7,8,3,12,13,14,15], "deep4":[2,8,9,7,8,10,3,12,13,14]}
    experiments=[]
    for name, weights in instances.items():
        costs=np.array(weights,dtype=float)
        # Build the complete feasible table first; never hide this preprocessing.
        table=make_plan_table(4,costs)
        begin=perf_counter()
        best_index=int(np.argmin(table.costs))
        exact_seconds=table.preparation_seconds + perf_counter()-begin
        exact=float(table.costs[best_index])
        penalty=float(costs.max()+1)
        begin=perf_counter()
        problem=build_join_qubo(4,costs,penalty)
        reference=audit(4,costs,problem)
        penalty_setup_seconds=perf_counter()-begin
        raw_energy=make_qubo_qnode(problem)
        raw_probs=make_qubo_qnode(problem,True)

        def angles(x):
            return qml.numpy.stack((x[0]/penalty,x[1]))

        def energy(x):
            return raw_energy(angles(x))/penalty

        rows=[]
        for mode in ("penalty_qubo","enumerated_plan_grover"):
            for seed in range(3):
                start=perf_counter()
                if mode == "penalty_qubo":
                    result=train(energy,p=2,seed=seed,maxiter=100)
                    probs=np.asarray(raw_probs(angles(result.parameters)))
                    valid=reference["valid"]
                    optimum=reference["optimal"]
                    values=reference["costs"]
                    setup=penalty_setup_seconds
                else:
                    result=train(make_feasible_qnode(table),p=2,seed=seed,maxiter=100)
                    probs=np.asarray(make_feasible_qnode(table,True)(result.parameters))
                    valid=np.arange(len(probs))<len(table.costs)
                    values=np.zeros(len(probs)); values[valid]=table.costs
                    optimum=valid & (values==exact)
                    setup=table.preparation_seconds
                wall=perf_counter()-start
                valid_prob=float(probs[valid].sum())
                rows.append({"mode":mode,"seed":seed,"training":asdict(result),
                             "optimal_probability":float(probs[optimum].sum()),
                             "valid_probability":valid_prob,
                             "conditional_valid_cost":float(probs[valid]@values[valid]/valid_prob),
                             "setup_seconds":setup,"training_and_distribution_seconds":wall,
                             "total_seconds":setup+wall,"probabilities":probs})
        summary=[]
        for mode in ("penalty_qubo","enumerated_plan_grover"):
            group=[r for r in rows if r["mode"]==mode]
            summary.append({"mode":mode,"mean_optimal_probability":float(np.mean([r["optimal_probability"] for r in group])),
                            "mean_valid_probability":float(np.mean([r["valid_probability"] for r in group])),
                            "mean_total_seconds":float(np.mean([r["total_seconds"] for r in group])),
                            "successes":sum(r["training"]["success"] for r in group)})
        random_runs=[]
        for seed in range(3):
            start=perf_counter()
            indices=np.random.default_rng(seed).integers(len(table.costs),size=1000)
            random_runs.append({"seed":seed,"best_cost":float(table.costs[indices].min()),
                                "mean_cost":float(table.costs[indices].mean()),
                                "total_seconds":table.preparation_seconds+perf_counter()-start})
        experiments.append({"instance":name,"costs":weights,"plans":table.plans,
                            "plan_costs":table.costs,"exact_cost":exact,"exact_plan":table.plans[best_index],
                            "exact_enumeration_seconds":exact_seconds,
                            "uniform_valid_optimal_probability":float(np.mean(table.costs==exact)),
                            "uniform_valid_mean_cost":float(table.costs.mean()),
                            "uniform_valid_1000_draws":random_runs,"summary":summary,"runs":rows})
        print(name,summary,flush=True)
    return {"scope":"Synthetic simulator diagnostic, not evidence of quantum advantage",
            "configuration":{"p":2,"optimizer":"L-BFGS-B","maxiter":100,"tolerance":1e-6,"seeds":[0,1,2],
                             "penalty_cost_scale":"max(costs)+1","feasible_cost_scale":"max(plan_costs)"},
            "limitations":["Feasible codebook preprocessing enumerates and costs every plan, already enabling exact solution.",
                           "Encoding, initial state, objective and mixer change together; this is not a mixer-only ablation.",
                           "Wall times are warm-process local simulator measurements; no hardware or equal gate-budget comparison.",
                           "Identical iteration ceilings do not imply identical computational work."],
            "versions":{x:version(x) for x in ("numpy","scipy","pennylane")},"experiments":experiments}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("results/feasibility_comparison/report.json"))
    args=parser.parse_args()
    if args.output.exists(): parser.error("Choose a new output file.")
    report=run()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as file:
        json.dump(report,file,indent=2,allow_nan=False,default=lambda x:x.tolist())
