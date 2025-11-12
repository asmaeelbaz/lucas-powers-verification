#!/usr/bin/env python3
"""
Independent verification via recurrence in ℤ.
Lucas sequence: L_0=2, L_1=1, L_{n+2}=L_{n+1}+L_n
Admissibility: Brown (1969) criterion over ℚ(√5)
"""
import json

def L(n):
    """Lucas via recurrence in ℤ."""
    if n == 0: return 2
    if n == 1: return 1
    a, b = 2, 1
    for _ in range(2, n+1):
        a, b = b, a+b
    return b

def admissible(idx):
    """Brown's criterion: (i) n_{i+1}-n_i≥2, (ii) {0,2}⊄{n_1,...,n_k}"""
    if not idx: return False
    for i in range(len(idx)-1):
        if idx[i+1] <= idx[i] or idx[i+1]-idx[i] < 2:
            return False
    return not (0 in idx and 2 in idx)

def verify(y, a, idx):
    """Verify y^a = ∑L_{n_i} in ℤ."""
    return admissible(idx) and y**a == sum(L(n) for n in idx)

# Verify k=3 (manuscript Table 2, corrected)
table2 = [
    (2,4,[1,3,5]), (2,7,[1,3,10]), (2,12,[3,13,17]),  # Row 3: corrected a=11→12
    (3,3,[0,4,6]), (9,2,[1,3,9]), (12,2,[2,6,10]),
    (20,2,[0,9,12]), (23,2,[1,4,13]), (33,2,[8,11,14]),
    (37,2,[1,3,15]), (38,2,[3,9,15]), (57,2,[11,14,16]),
    (63,2,[9,12,17]), (114,2,[9,17,19])
]

print("Verifying Table 2 (k=3, corrected):")
assert all(verify(y,a,idx) for y,a,idx in table2), "FAILED"
print(f"✓ All {len(table2)} solutions verified\n")

# Verify JSON data
for k in [3,4]:
    with open(f'solutions_k{k}.json') as f:
        data = json.load(f)
    sols = data['sporadic_solutions']
    assert all(verify(s['y'], s['a'], s['indices']) for s in sols), f"k={k} FAILED"
    print(f"✓ k={k}: {len(sols)} solutions verified")

print("\n✓ All verifications passed")
