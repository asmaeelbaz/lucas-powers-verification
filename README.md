# Lucas Power Sums: Computational Verification

Computational verification for the Diophantine equation y^a = ∑L_{n_i} over ℤ in ℚ(√5).

## Results

**k=3, y ≤ 25000**: 14 sporadic solutions (1.5s, 16170 admissible triples)  
**k=4, y ≤ 200**: 33 sporadic solutions (0.14s, 14674 admissible 4-tuples)

Infinite family (k=3): L_{3n} + L_{n+2} + L_{n-2} = L_n^3 for n even, n ≥ 4.

## Algorithm

1. Bound n_k via Binet asymptotic L_n ~ φ^n/√5
2. Generate admissible k-tuples (Brown's criterion)
3. Factorize sums into y^a (canonical form: maximal exponent)

Admissibility: (i) n_{i+1} - n_i ≥ 2, (ii) {0,2} ⊄ {n_1,...,n_k}

## Usage

```bash
python3 lucas_search.py  # Full search k ∈ {3,4}
python3 verify.py        # Independent verification
```

Extension to k=5:
```python
from lucas_search import exhaustive_search
sols = exhaustive_search(k=5, y_max=500)
```

## Data

- `solutions_k3.json`: Complete k=3 results
- `solutions_k4.json`: Complete k=4 results

## Methods

Combines Matveev's theorem (linear forms in logarithms over ℚ(√5)) with Baker-Davenport reduction via continued fraction convergents of κ = log y / log φ.

## Reference

C. Bouyacoub, A. El-Baz, O. Kihel (2025). Perfect Powers Expressible as Sums of Lucas Numbers. *Quaestiones Mathematicae* (submitted).

## License

MIT. Citation requested for academic use.
