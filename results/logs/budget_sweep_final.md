# Budget Sweep — Revenue Comparison (Fixed k-budget semantics)
*Generated: 2026-06-24 12:19*

**Note:** µ/σ/Greedy-discount are k-independent by design (Babaei 2013 — pure pricing, no seeding phase).
S2V-DQN/ToupleGDD: first k seeds FREE (method's GNN selects them), then greedy-discount pricing on remaining n-k nodes.

| Method | SBM/k=5 | SBM/k=10 | SBM/k=20 | SBM/k=30 | rice_fb/k=5 | rice_fb/k=10 | rice_fb/k=20 | rice_fb/k=30 |
|--------|--------:|---------:|---------:|---------:|------------:|-------------:|-------------:|-------------:|
| **IE-Strategy** | 15.60 | 28.17 | 41.64 | 47.92 | 26.54 | 48.74 | 86.61 | 114.90 |
| **µ-Discount** | 60.97 | 60.97 | 60.97 | 60.97 | 241.28 | 241.28 | 241.28 | 241.28 |
| σ-Discount | 48.13 | 48.13 | 48.13 | 48.13 | 190.28 | 190.28 | 190.28 | 190.28 |
| Greedy-Discount | 42.06 | 42.06 | 42.06 | 42.06 | 158.19 | 158.19 | 158.19 | 158.19 |
| S2V-DQN (dec.) | 41.19 | 41.00 | 40.18 | 38.70 | 166.04 | 165.89 | 165.43 | 164.76 |
| ToupleGDD (dec.) | 41.17 | 40.93 | 40.10 | 38.13 | 166.04 | 165.88 | 165.34 | 164.69 |