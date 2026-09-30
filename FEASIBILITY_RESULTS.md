# Valid-plan experiment

The new enumerated-plan baseline prepares a uniform superposition of valid
plan indices and applies a Grover-style projector mixer. This is an application
of an existing approach, not a new mixer: [Bärtschi and Eidenbenz (2020)](https://arxiv.org/abs/2006.00354).

We first enumerate all 15 four-table trees and compute every cost. Four qubits
encode their indices; the sixteenth index is unused. The diagonal cost and
projector mixer cannot move amplitude into that unused state. Dense simulation
gates are used; no hardware compilation or gate-efficiency claim is made.

## Results

Two synthetic instances, p=2, L-BFGS-B, maxiter=100, seeds 0–2:

| Instance | Method | Mean valid probability | Mean optimal probability | Mean local total seconds |
|---|---|---:|---:|---:|
| Bushy | Original penalty QUBO | 27.94% | 4.28% | 1.227 |
| Bushy | Enumerated-plan mixer | 100% | 82.26% | 0.040 |
| Deep | Original penalty QUBO | 28.51% | 3.92% | 1.258 |
| Deep | Enumerated-plan mixer | 100% | 54.81% | 0.051 |

Classical enumeration found the exact optimum with certainty in approximately
0.20 ms and 0.17 ms respectively, including tree/cost-table construction.
Uniform valid-plan sampling has 100% validity and 6.67% optimal probability
per draw. The report includes three 1,000-draw classical runs for each instance.

These timings are single-process measurements on this machine, not robust
hardware benchmarks. Quantum runs include local preprocessing, training and
final distribution evaluation. Python import/startup time is excluded. All
12 optimizer runs met their termination conditions; this does not prove global
optimality. Objective and gradient call counts are in the report.

## Interpretation

Validity improved as intended. This does not establish quantum advantage:
the plan table already contains enough information to select the exact answer
classically, and enumeration was much faster here. Both approaches use the
same iteration ceilings, not equal work or gate budgets. Encoding, objective,
initial state and mixer all change, so this is not a mixer-only ablation.
Two instances and three seeds do not support general performance claims.

The new baseline is useful for separating feasibility loss from optimization
quality. A scalable next investigation needs state preparation/cost evaluation
that does not enumerate every candidate, plus stronger multi-instance studies.
No research novelty is claimed.

## Use

In the dashboard choose either four-table “valid-plan mixer” option. These
show readable plans beside index bitstrings and the classical exact reference.
Original subset bitstrings and new plan-index bitstrings have different meanings.
New angles refer to cost normalized by maximum plan cost; the original baseline
uses cost normalized by penalty. The existing penalty setting is unused for
valid-plan modes and hidden in the interface.

```sh
.venv312/bin/python -m unittest discover -v
.venv312/bin/python compare_feasibility.py --output results/feasibility_repeat/report.json
```

Full results: `results/feasibility_comparison/report.json`.
Tests verify validity through depth 3, plan counts/costs, direct matrix
evolution and finite-difference gradients. Results remain simulator-only.
