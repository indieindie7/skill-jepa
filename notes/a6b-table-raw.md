# A6b nudger replay 2026-10-31 14:36 (a6b_nudger.py, seeds=3, w=0.3, temp=0.1, cold=10, softmax T=0.05, route rows=2298, days with >= 20 rows: 15)

| arm | hit1 cold | hit1 warm | hit1 all | top-1 entropy bits/day | distinct/day | KL med bits | KL p90 bits | KL med warm | KL med cold |
|---|---|---|---|---|---|---|---|---|---|
| N0 | 23.0+-1.4 | 33.0+-1.5 | 31.1+-1.5 | 2.61+-0.02 | 13.9 | 0.000 | 0.000 | 0.000 | nan |
| N1 | 58.1+-0.5 | 60.2+-0.3 | 61.0+-0.1 | 1.94+-0.01 | 6.5 | 1.625 | 11.576 | 1.625 | nan |
| N2 | 57.8+-0.0 | 62.8+-0.7 | 62.9+-0.6 | 2.48+-0.02 | 13.0 | 1.098 | 9.301 | 0.556 | 5.786 |

Production router top-1 (route.log col 3) same days: entropy 2.13 bits, distinct 14.0.

Rule: N2 hit1(all) >= N0 + .05 -> gain +31.8 pts (ok); N2 entropy >= N0 -> 2.48 vs 2.61 (FAIL); N2 median KL <= 1 -> 1.098 (FAIL). VERDICT FAIL
