# Four-table benchmark: findings and limits

This is a synthetic extension of the subset encoding, not a reproduction of
the paper's benchmark results. The saved run is `results/four_table/report.json`.

## What was verified

- Four tables require 10 proper, non-singleton subset variables (10 qubits).
- An independent recursive enumerator finds 15 unordered binary join trees.
- All 1,024 assignments were checked against those trees for each instance
  and penalty; all ground states match the optimal valid plans.
- Additional checks cover 44 positive-cost instance/penalty combinations,
  including equal-cost ties. All 10 project tests passed.
- The bushy-favored optimum is (A join B) join (C join D), retained cost 5.
- The deep-favored optimum is ((A join B) join C) join D, retained cost 5.

Costs are synthetic additive costs assigned to relation subsets. They are not
CPU measurements, cardinality estimates, or database execution times. We omit
the root as an assumed constant and allow all joins, including Cartesian ones.

## QAOA results

24 ideal statevector runs: 2 instances, 2 penalties, 2 depths, 3 initialization
seeds. All used L-BFGS-B with maxiter=100 and tolerance=1e-6; all reported
successful termination, which does not establish global optimality.

| Instance | Penalty rule | Depth | Mean P(optimal plan) | Mean P(valid plan) |
|---|---|---:|---:|---:|
| Bushy | max + 1 | 1 | 2.383% | 22.871% |
| Bushy | max + 1 | 2 | 4.276% | 27.943% |
| Bushy | 2 max | 1 | 1.952% | 23.453% |
| Bushy | 2 max | 2 | 3.573% | 35.134% |
| Deep | max + 1 | 1 | 2.233% | 23.035% |
| Deep | max + 1 | 2 | 3.916% | 28.513% |
| Deep | 2 max | 1 | 1.880% | 23.529% |
| Deep | 2 max | 2 | 3.348% | 35.291% |

We divide both cost evolution and objective by the penalty. This removes an
overall energy-scale difference but retains different relative cost/penalty
weights. Raw energies can be recovered by multiplying recorded training
energies by the penalty. The JSON contains parameters, distributions, histories,
per-seed variability, conditional valid-plan cost, versions and settings.

## Classical reference points

Uniform sampling over 1,024 bitstrings has P(optimal)=1/1024=0.09766% and
P(valid)=15/1024=1.46484%. QAOA improves both over that unconstrained reference.
But uniform sampling over the 15 valid trees has P(optimal)=1/15=6.66667% and
P(valid)=100%. Every listed unconditional QAOA optimum probability is below
that simple feasibility-aware reference. Exact enumeration of just 15 plans
finds the optimum outright. These are probability comparisons, not claims of
equal computational effort or measured time-to-solution.

In these runs, increasing depth improves optimal probability; increasing the
penalty improves validity but lowers optimal probability. Three seeds and two
hand-selected instances are insufficient to generalize either observation.
There is no demonstrated quantum advantage or established research novelty.

## Edge case found during verification

With three zero-cost subsets and penalty 1, the encoding is
E=-x0-x1-x2+x0x1+x0x2+x1x2. Both one-bit (valid) and two-bit (invalid)
selections have energy -1. The builder therefore rejects zero costs instead
of claiming all ground states are valid. This is a boundary condition of our
implementation, not a claim that the paper's positive-cost experiments fail.

## Next research decision

Investigate feasibility-preserving state preparation/mixers or another compact
encoding, starting with a literature check. Include uniform valid-plan sampling,
exact enumeration, and larger classical heuristics as appropriate baselines.
Keep conditional-on-validity metrics separate from unconditional success and
account for discarded measurements and classical preprocessing costs.
Full paper replication still requires the original instances and settings.
