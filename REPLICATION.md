# Join-order replication audit

Status: partial formulation replication completed; full-paper reproduction not completed.
Sources inspected on 2026-09-27.

Selected source: Nayak et al., [Quantum Join Ordering by Splitting the Search Space
of QUBO Problems (2024)](https://doi.org/10.1007/s13222-024-00468-3), section 3.3,
equation (7). We independently implemented the explicit three-relation example.
This is not a reproduction of the larger experimental results.

## Comparison table

| Aspect | Selected paper | Previous local baseline | This replication |
|---|---|---|---|
| Problem | Join ordering | Five-node Max-Cut | Three-relation join example |
| Encoding | Subset variables with conflict penalties | Cut adjacency | Explicit equation (7) |
| Search-space splitting | Included | Absent | Not implemented |
| Database integration | PostgreSQL | None | None |
| Execution | Annealing and circuit methods | Ideal PennyLane | Ideal PennyLane |
| Validation | Paper experiments | Exact 32-state check | Exact eight-state audit |

The paper's equations (5)–(6), read literally, count equal-size conflicting
sets in both directions. Equation (7) gives each unordered pair one penalty.
We follow equation (7). A subsequent author-code audit confirmed that its
conflict loops use j >= i and exclude i == j, counting each pair once.
This observation alone is not a novelty claim.

The [author repository](https://github.com/Nnayak3110/Quantum-Query-Optimization-by-splitting-QUBO)
was located. Its README identifies it as the reference implementation.
The initial web fetch failed, but the subsequent read-only source audit succeeded
at commit `ede754f0139ce7de2fc310ea2bc7ef301b7cc919`. The D-Wave model builders
use twice the maximum cost, whereas the worked example uses maximum plus one.
No author code was executed. Our three-relation results follow the worked
example, not the builder's alternative penalty setting.

## Exact mathematical check

Variable order is x0=R1R2, x1=R1R3, x2=R2R3; wire 0 is the leftmost bit.
The retained costs are 2, 4, 7. The explicit polynomial is

E(x) = -6x0 -4x1 -x2 +8x0x1 +8x0x2 +8x1x2.

The symmetric matrix has diagonal (-6,-4,-1) and off-diagonals 4.
Its Ising form under x=(1-Z)/2 is

H = 0.5 I - Z0 - 2 Z1 - 3.5 Z2 + 2(Z0Z1 + Z0Z2 + Z1Z2).

| Bitstring | Energy | Valid complete plan? |
|---|---:|---|
| 000 | 0 | No |
| 001 | -1 | Yes |
| 010 | -4 | Yes |
| 011 | 3 | No |
| 100 | -6 | Yes, optimum |
| 101 | 1 | No |
| 110 | -2 | No |
| 111 | 13 | No |

All eight values match direct polynomial evaluation and Hamiltonian diagonals.
The retained cost 2 omits common costs and is not database wall-clock runtime.

## Independent QAOA experiment

These are new local measurements on the worked example, not numbers replicated
from a performance figure. We used p=1,2, seeds 0–4, uniform initial angles in
[-0.5,0.5], tolerance 1e-6, maxiter=200, and exact analytic probabilities.
There are no finite-shot or hardware measurements in this experiment.

| Optimizer | Depth | Mean P(optimum), sample SD | Mean P(valid) | Successful termination |
|---|---:|---:|---:|---:|
| L-BFGS-B | 1 | 18.51%, 11.10 percentage points | 64.26% | 5/5 |
| L-BFGS-B | 2 | 23.16%, 20.99 percentage points | 71.60% | 5/5 |
| COBYLA | 1 | 30.67%, 3.31 percentage points | 53.53% | 5/5 |
| COBYLA | 2 | 34.30%, 15.98 percentage points | 59.79% | 0/5 |

Uniform sampling over eight bitstrings gives 12.5% optimal and 37.5% valid
probability. A classical enumeration of the three valid plans finds the exact
answer; this example establishes no quantum advantage. Equal maxiter values
are not equal resource budgets across these optimizers. Standard deviation
describes seed variability, not a confidence interval or measurement uncertainty.
All depth-2 COBYLA runs stopped at their budget; their results are best-so-far.

We report feasibility separately from energy because an invalid selection is
not a query plan. The JSON also records expected retained cost conditional on
validity. That conditional cost must be read alongside the validity probability.
An energy ratio would be misleading because shifts/penalties change it.

## Reproduce locally

```sh
.venv312/bin/python -m unittest -v test_qubo.py
.venv312/bin/python reproduce_join.py --output results/join_replication_repeat/report.json
```

`results/join_replication/report.json` contains all 20 runs, parameters,
probabilities, histories, statuses, seeds, versions and the exact energy table.
`qubo.py` adds general linear Z terms that the Max-Cut-only circuit lacks.
Independent dense exponentials verify both its circuit and gradients.

## Research decision

No defensible new contribution has been established. A candidate question is
how penalty scale and initialization affect feasibility versus retained plan
cost under matched evaluation budgets. This must be checked against the
[2026 structured-initialization preprint](https://arxiv.org/abs/2608.20683)
and broader literature before calling it a gap.

Before a full replication claim: recover benchmark instances and exact solver settings,
and reproduce a larger reported experiment. Then extend to multiple instances
and compare with exact/classical methods using comparable resource budgets.
The current comparison supports continuing investigation, not paper submission.
