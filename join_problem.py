"""Small subset-based join-order QUBOs with an independent tree enumerator.

Generalizes the three-table example; synthetic subset costs are not SQL runtimes.
All binary trees are allowed, including Cartesian joins. Root cost is omitted
as an assumed plan-independent constant. Supports only three or four tables.
"""

from itertools import combinations
from functools import lru_cache
import numpy as np

from qubo import Qubo, from_qubo


def subsets(n: int) -> tuple[int, ...]:
    """Return bitmask subsets ordered by cardinality, then table combination."""
    if n not in (3, 4):
        raise ValueError("This exhaustive prototype supports 3 or 4 tables.")
    return tuple(sum(1 << i for i in group)
                 for size in range(2, n) for group in combinations(range(n), size))


def tree_selections(n: int) -> tuple[frozenset[int], ...]:
    """Enumerate unordered binary join trees recursively, independent of QUBO.

    Each tree is represented by its proper internal subsets; swapping children
    is not a distinct plan. This yields 3 trees for n=3 and 15 for n=4.
    """
    subsets(n)  # validate the supported size

    @lru_cache(None)
    def descend(mask: int) -> tuple[frozenset[int], ...]:
        if mask.bit_count() == 1:
            return (frozenset(),)
        trees = []
        left = (mask - 1) & mask
        while left:
            right = mask ^ left
            if right and left & (mask & -mask):
                for a in descend(left):
                    for b in descend(right):
                        trees.append(a | b | {mask})
            left = (left - 1) & mask
        return tuple(trees)

    root = (1 << n) - 1
    return tuple(tree - {root} for tree in descend(root))


def build_join_qubo(n: int, costs: np.ndarray, penalty: float) -> Qubo:
    """Reward each subset by cost-penalty; penalize each crossing pair once.

    Crossing sets overlap but neither contains the other. Symmetric Q stores
    half the polynomial pair coefficient in each off-diagonal entry.
    """
    variables = subsets(n)
    costs = np.asarray(costs, dtype=float)
    if costs.shape != (len(variables),) or not np.isfinite(costs).all() or np.any(costs <= 0):
        raise ValueError("Supply one finite, strictly positive cost per subset; zero costs can tie invalid states.")
    if not np.isfinite(penalty) or penalty <= costs.max():
        raise ValueError("Penalty must exceed every subset cost.")
    q = np.diag(costs - penalty)
    for i, a in enumerate(variables):
        for j in range(i + 1, len(variables)):
            b = variables[j]
            overlap = a & b
            if overlap and overlap != a and overlap != b:
                q[i, j] = q[j, i] = penalty / 2
    return from_qubo(q)


def audit(n: int, costs: np.ndarray, problem: Qubo) -> dict:
    """Compare all QUBO ground states with independently enumerated plans."""
    variables = subsets(n)
    labels = [format(k, f"0{len(variables)}b") for k in range(2**len(variables))]
    bits = np.array([[int(c) for c in label] for label in labels])
    trees = set(tree_selections(n))
    selected = [frozenset(v for v, bit in zip(variables, row) if bit) for row in bits]
    valid = np.array([s in trees for s in selected])
    raw_costs = bits @ costs
    energies = np.einsum("bi,ij,bj->b", bits, problem.matrix, bits)
    best = float(min(sum(costs[variables.index(v)] for v in tree) for tree in trees))
    optimal = valid & np.isclose(raw_costs, best, rtol=0, atol=1e-9)
    ground = np.isclose(energies, energies.min(), rtol=0, atol=1e-9)
    np.testing.assert_array_equal(ground, optimal)
    return {"labels": labels, "valid": valid, "optimal": optimal, "costs": raw_costs,
            "energies": energies, "exact_cost": best, "tree_count": len(trees)}
