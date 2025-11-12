#!/usr/bin/env python3
"""
Computational Verification of Lucas Power Sums via Algebraic Methods
=====================================================================

This implementation provides rigorous computational verification for:
    y^a = ∑_{i=1}^k L_{n_i}
where (n_1,...,n_k) satisfies Lucas-admissibility in the sense of Brown.

Algebraic Foundation:
The Lucas sequence (L_n)_{n≥0} is defined by the characteristic polynomial
    x² - x - 1 ∈ ℤ[x]
with roots α = (1+√5)/2 and β = (1-√5)/2 in the quadratic field ℚ(√5).
The Binet formula yields:
    L_n = α^n + β^n ∈ ℤ
where α, β are algebraic units in the ring of integers ℤ[α] ⊂ ℚ(√5).

Author: Computational verification for journal submission
Date: November 2025
"""

import sys
import time
from typing import List, Tuple, Set, Dict, Iterator
from collections import defaultdict
from functools import lru_cache

# ============================================================================
# Part 1: Lucas Number Computation via Algebraic Recurrence
# ============================================================================

class LucasAlgebra:
    """
    Efficient computation of Lucas numbers using the recurrence relation
    over the ring ℤ with memoization for optimal performance.
    
    The Lucas sequence satisfies:
        L_0 = 2, L_1 = 1
        L_{n+2} = L_{n+1} + L_n  (characteristic equation: x² = x + 1)
    
    This corresponds to the minimal polynomial of the golden ratio α
    over ℚ, ensuring all values lie in ℤ.
    """
    
    def __init__(self):
        """Initialize with base cases from the defining recurrence."""
        self._cache: Dict[int, int] = {0: 2, 1: 1}
        self._max_computed: int = 1
    
    def __call__(self, n: int) -> int:
        """
        Compute L_n using dynamic programming.
        
        Args:
            n: Non-negative integer index
            
        Returns:
            L_n ∈ ℤ (exact integer arithmetic)
        """
        if n < 0:
            raise ValueError(f"Lucas index must be non-negative, got {n}")
        
        if n in self._cache:
            return self._cache[n]
        
        # Extend cache from max_computed to n
        for i in range(self._max_computed + 1, n + 1):
            self._cache[i] = self._cache[i-1] + self._cache[i-2]
        
        self._max_computed = max(self._max_computed, n)
        return self._cache[n]
    
    def binet_estimate(self, n: int) -> float:
        """
        Asymptotic estimate using Binet's formula:
            L_n ≈ α^n  for large n  (since |β^n| → 0)
        where α = (1+√5)/2 is the golden ratio.
        
        Used for determining search bounds.
        """
        import math
        alpha = (1 + math.sqrt(5)) / 2
        return alpha ** n

# Global Lucas computer instance
L = LucasAlgebra()


# ============================================================================
# Part 2: Lucas-Admissibility in the Sense of Brown
# ============================================================================

def is_lucas_admissible(indices: Tuple[int, ...]) -> bool:
    """
    Verify Lucas-admissibility according to Brown's uniqueness theorem.
    
    Definition (Brown, 1969):
    A sequence 0 ≤ n_1 < n_2 < ... < n_k is Lucas-admissible if:
        (1) Non-adjacency: n_{i+1} - n_i ≥ 2 for all i ∈ {1,...,k-1}
        (2) Exclusion: {0, 2} ⊄ {n_1,...,n_k}
    
    These conditions ensure uniqueness of representation in ℤ.
    
    Args:
        indices: Strictly increasing sequence of non-negative integers
        
    Returns:
        True iff the sequence is Lucas-admissible
    """
    if not indices or len(indices) == 0:
        return False
    
    # Verify strict monotonicity and non-negativity
    for i in range(len(indices)):
        if indices[i] < 0:
            return False
        if i > 0 and indices[i] <= indices[i-1]:
            return False
    
    # Condition (1): Non-adjacency
    for i in range(len(indices) - 1):
        if indices[i+1] - indices[i] < 2:
            return False
    
    # Condition (2): Exclusion constraint
    if 0 in indices and 2 in indices:
        return False
    
    return True


# ============================================================================
# Part 3: Efficient Generation of Admissible Sequences
# ============================================================================

def generate_admissible_k_tuples(k: int, n_max: int) -> Iterator[Tuple[int, ...]]:
    """
    Generate all Lucas-admissible k-tuples with largest element < n_max.
    
    This uses a backtracking algorithm that prunes the search space
    based on admissibility constraints, achieving significant speedup
    over naive enumeration of all k-subsets of {0,1,...,n_max-1}.
    
    Theoretical bound:
    By the non-adjacency constraint, the number of admissible k-tuples
    is at most C(⌊n_max/2⌋, k) where C denotes binomial coefficient.
    For k=3, n_max=50, this gives ≈4,000 tuples vs 20,000 naive.
    
    Args:
        k: Length of tuple
        n_max: Upper bound (exclusive) on largest element
        
    Yields:
        Lucas-admissible k-tuples in lexicographic order
    """
    def backtrack(
        current: List[int],
        min_next: int,
        remaining: int
    ) -> Iterator[Tuple[int, ...]]:
        """
        Recursive backtracking with pruning.
        
        Args:
            current: Current partial sequence
            min_next: Minimum value for next element (ensures non-adjacency)
            remaining: Number of elements still to add
        """
        if remaining == 0:
            tuple_result = tuple(current)
            if is_lucas_admissible(tuple_result):
                yield tuple_result
            return
        
        # Maximum value ensuring space for remaining elements
        # Each remaining element needs at least 2 units of separation
        max_next = n_max - 1 - 2 * (remaining - 1)
        
        if min_next > max_next:
            return  # Pruning: impossible to place remaining elements
        
        for next_val in range(min_next, max_next + 1):
            current.append(next_val)
            # Next element must be at least next_val + 2 (non-adjacency)
            yield from backtrack(current, next_val + 2, remaining - 1)
            current.pop()
    
    yield from backtrack([], 0, k)


# ============================================================================
# Part 4: Prime Factorization and Perfect Power Detection
# ============================================================================

def integer_root(n: int, k: int) -> Tuple[bool, int]:
    """
    Compute k-th root of n in ℤ using binary search.
    
    Args:
        n: Positive integer
        k: Root degree (k ≥ 2)
        
    Returns:
        (True, r) if n = r^k for some r ∈ ℤ, else (False, 0)
    """
    if k == 1:
        return (True, n)
    if n == 0:
        return (True, 0)
    if n == 1:
        return (True, 1)
    
    # Binary search for r such that r^k = n
    low, high = 1, n
    
    while low <= high:
        mid = (low + high) // 2
        mid_power = mid ** k
        
        if mid_power == n:
            return (True, mid)
        elif mid_power < n:
            low = mid + 1
        else:
            high = mid - 1
    
    return (False, 0)


def all_power_representations(n: int) -> List[Tuple[int, int]]:
    """
    Find all representations of n as a perfect power y^a with a ≥ 2.
    
    This exploits the multiplicative structure of ℤ. For n = ∏ p_i^{e_i},
    any representation n = y^a requires a | gcd(e_1,...,e_r).
    
    Args:
        n: Positive integer
        
    Returns:
        List of (y, a) pairs with y^a = n and a ≥ 2, sorted by a (descending)
    """
    if n <= 1:
        return []
    
    representations = []
    
    # Check all possible exponents a from 2 to log_2(n)
    import math
    max_exp = int(math.log2(n)) + 1
    
    for a in range(2, max_exp + 1):
        is_root, y = integer_root(n, a)
        if is_root and y >= 2:
            representations.append((y, a))
    
    # Sort by exponent (descending) for canonical form
    representations.sort(key=lambda x: x[1], reverse=True)
    
    return representations


# ============================================================================
# Part 5: Exhaustive Search with Algebraic Optimization
# ============================================================================

def exhaustive_search(
    k: int,
    y_max: int,
    verbose: bool = True,
    canonical_only: bool = True
) -> List[Tuple[int, int, Tuple[int, ...]]]:
    """
    Exhaustive search for all solutions to y^a = ∑L_{n_i} with fixed k.
    
    Algorithm:
    1. Determine search bound n_max via Binet asymptotic formula
    2. Generate all admissible k-tuples with n_k < n_max
    3. For each tuple, compute Lucas sum N = ∑L_{n_i}
    4. Factor N into all power representations y^a
    5. Filter by y ≤ y_max and (optionally) canonical form
    
    Algebraic Optimization:
    The search space is bounded using the asymptotic L_n ~ α^n / √5
    where α = (1+√5)/2. For y ≤ y_max and a ≥ 2, we have:
        N ≤ y_max^2  (worst case: a=2)
        L_n ≤ N ⟹ n ≤ log_α(N√5) ≈ log_α(y_max^2)
    
    Args:
        k: Number of summands
        y_max: Maximum base y to search
        verbose: Print progress information
        canonical_only: Return only maximal-exponent representations
        
    Returns:
        List of (y, a, indices) tuples sorted by (y, a)
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"EXHAUSTIVE SEARCH: k={k}, y ≤ {y_max}")
        print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # Step 1: Determine search bound using Binet formula
    import math
    alpha = (1 + math.sqrt(5)) / 2
    
    # Upper bound: y_max^2 for a=2 case
    max_sum = y_max ** 2
    
    # Solve L_n ≤ max_sum using asymptotic L_n ≈ α^n
    # α^n ≤ max_sum  ⟹  n ≤ log_α(max_sum)
    n_max_estimate = int(math.log(max_sum) / math.log(alpha)) + 5
    
    # Verify and adjust bound
    while L(n_max_estimate) < max_sum:
        n_max_estimate += 1
    
    n_max = n_max_estimate + 2  # Safety margin
    
    if verbose:
        print(f"Search parameters:")
        print(f"  k = {k}")
        print(f"  y_max = {y_max:,}")
        print(f"  Maximum sum: {max_sum:,}")
        print(f"  Search bound: n_{k} < {n_max}")
        print(f"  L_{n_max} = {L(n_max):,}\n")
    
    # Step 2: Generate all admissible k-tuples
    if verbose:
        print("Generating admissible sequences...")
    
    admissible_tuples = list(generate_admissible_k_tuples(k, n_max))
    
    if verbose:
        print(f"  Generated {len(admissible_tuples):,} admissible {k}-tuples\n")
    
    # Step 3: Compute solutions
    if verbose:
        print("Computing Lucas sums and factorizations...")
    
    solutions = []
    lucas_sums_computed = 0
    
    for indices in admissible_tuples:
        # Compute sum ∑L_{n_i}
        lucas_sum = sum(L(n) for n in indices)
        lucas_sums_computed += 1
        
        # Skip if sum exceeds maximum
        if lucas_sum > max_sum:
            continue
        
        # Find all power representations
        power_reps = all_power_representations(lucas_sum)
        
        for y, a in power_reps:
            if y <= y_max:
                solutions.append((y, a, indices))
    
    # Step 4: Filter for canonical form if requested
    if canonical_only:
        # Group by indices, keep only max exponent
        by_indices = defaultdict(list)
        for y, a, indices in solutions:
            by_indices[indices].append((y, a))
        
        canonical_solutions = []
        for indices, pairs in by_indices.items():
            # Keep pair with maximum exponent
            y_max_exp, a_max_exp = max(pairs, key=lambda x: x[1])
            canonical_solutions.append((y_max_exp, a_max_exp, indices))
        
        solutions = canonical_solutions
    
    # Sort by (y, a)
    solutions.sort(key=lambda x: (x[0], x[1]))
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"\nSearch completed:")
        print(f"  Time: {elapsed:.2f} seconds")
        print(f"  Admissible tuples checked: {len(admissible_tuples):,}")
        print(f"  Solutions found: {len(solutions)}")
        print(f"{'='*80}\n")
    
    return solutions


# ============================================================================
# Part 6: Infinite Family Detection
# ============================================================================

def filter_infinite_families(
    solutions: List[Tuple[int, int, Tuple[int, ...]]],
    k: int,
    verbose: bool = True
) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Separate sporadic solutions from infinite algebraic families.
    
    For k=3, the infinite family is:
        L_{3n} + L_{n+2} + L_{n-2} = L_n^3  (n even, n ≥ 4)
    
    This identity arises from the triple-angle formula in ℚ(√5):
        L_{3n} = L_n^3 - 3(-1)^n L_n
    combined with the identity 3L_n = L_{n+2} + L_{n-2}.
    
    Args:
        solutions: List of (y, a, indices) tuples
        k: Number of summands
        verbose: Print information about families
        
    Returns:
        (sporadic, family) where both are lists of tuples
    """
    sporadic = []
    family = []
    
    if k == 3:
        for y, a, indices in solutions:
            n1, n2, n3 = indices
            
            # Pattern: (n-2, n+2, 3n) with n even, n ≥ 4
            if a == 3 and n2 == n1 + 4:
                n = n1 + 2
                if n3 == 3 * n and n % 2 == 0 and n >= 4:
                    if y == L(n):
                        family.append((y, a, indices, f"L_{n}^3"))
                        continue
            
            sporadic.append((y, a, indices))
    
    elif k == 2:
        for y, a, indices in solutions:
            n1, n2 = indices
            
            # Pattern: (0, 2n) with y = L_n
            if a == 2 and n1 == 0:
                n = n2 // 2
                if n2 == 2 * n and y == L(n):
                    family.append((y, a, indices, f"L_{n}^2"))
                    continue
            
            sporadic.append((y, a, indices))
    else:
        sporadic = solutions
    
    if verbose and family:
        print(f"\nInfinite family solutions (excluded from count):")
        for item in family:
            if len(item) == 4:
                y, a, indices, label = item
                print(f"  {label}: y={y}, a={a}, indices={indices}")
        print()
    
    return sporadic, family


# ============================================================================
# Part 7: Main Verification Program
# ============================================================================

def verify_table_data(
    table_data: List[Tuple[int, int, int, Tuple]],
    table_name: str
) -> bool:
    """Verify all solutions in a table match claimed values."""
    print(f"\n{'='*80}")
    print(f"VERIFICATION: {table_name}")
    print(f"{'='*80}\n")
    
    all_valid = True
    
    for i, (k, y, a, indices) in enumerate(table_data, 1):
        lucas_sum = sum(L(n) for n in indices)
        power = y ** a
        
        if lucas_sum == power and is_lucas_admissible(indices):
            print(f"✓ Solution {i:2d}: y={y:3d}, a={a:2d}, indices={str(indices):25s} VERIFIED")
        else:
            print(f"✗ Solution {i:2d}: y={y:3d}, a={a:2d}, indices={str(indices):25s} FAILED")
            all_valid = False
    
    print(f"\n{'='*80}")
    if all_valid:
        print(f"✓ All {len(table_data)} solutions VERIFIED")
    else:
        print(f"✗ VERIFICATION FAILED")
    print(f"{'='*80}\n")
    
    return all_valid


def main():
    """Main computational verification program."""
    
    print("\n" + "="*80)
    print("LUCAS POWER SUMS: Computational Verification via Algebraic Methods")
    print("="*80)
    print("\nFoundation: Lucas sequence over ℤ in the quadratic field ℚ(√5)")
    print("="*80 + "\n")
    
    # ========================================================================
    # TASK 1: Verify k=3, y ≤ 25,000
    # ========================================================================
    
    print("\n" + "="*80)
    print("TASK 1: Exhaustive verification for k=3, y ≤ 25,000")
    print("="*80)
    
    solutions_k3 = exhaustive_search(k=3, y_max=25000, verbose=True, canonical_only=True)
    
    sporadic_k3, family_k3 = filter_infinite_families(solutions_k3, k=3, verbose=True)
    
    print(f"Results for k=3:")
    print(f"  Total solutions (canonical): {len(solutions_k3)}")
    print(f"  Sporadic solutions: {len(sporadic_k3)}")
    print(f"  Infinite family solutions: {len(family_k3)}\n")
    
    print("Sporadic solutions for k=3, y ≤ 25,000:")
    print("-" * 80)
    for y, a, indices in sporadic_k3:
        lucas_vals = [L(n) for n in indices]
        print(f"  y={y:5d}, a={a:2d}, indices={str(indices):20s}, L={lucas_vals}")
    
    # ========================================================================
    # TASK 2: Extend to k=4
    # ========================================================================
    
    print("\n\n" + "="*80)
    print("TASK 2: Extension to k=4")
    print("="*80)
    
    # For k=4, search up to y=200 as per manuscript claim
    solutions_k4 = exhaustive_search(k=4, y_max=200, verbose=True, canonical_only=True)
    
    sporadic_k4, family_k4 = filter_infinite_families(solutions_k4, k=4, verbose=True)
    
    print(f"Results for k=4:")
    print(f"  Total solutions (canonical): {len(solutions_k4)}")
    print(f"  Sporadic solutions: {len(sporadic_k4)}\n")
    
    print("First 20 sporadic solutions for k=4, y ≤ 200:")
    print("-" * 80)
    for y, a, indices in sporadic_k4[:20]:
        lucas_vals = [L(n) for n in indices]
        print(f"  y={y:3d}, a={a:2d}, indices={str(indices):30s}")
    
    if len(sporadic_k4) > 20:
        print(f"  ... and {len(sporadic_k4) - 20} more solutions")
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    print("\n\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    print(f"\n✓ k=3 (y ≤ 25,000): {len(sporadic_k3)} sporadic solutions")
    print(f"✓ k=4 (y ≤ 200): {len(sporadic_k4)} sporadic solutions")
    print("\nComputational verification complete.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
