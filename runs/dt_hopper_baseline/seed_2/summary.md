# Decision Transformer — Hopper Minari Training Summary

| metric | value |
|--------|-------|
| parameter count | 727,695 |
| norm_position | pre |
| train trajectories | 1194 |
| validation trajectories | 133 |
| total transitions | 999,404 |
| dataset mean return | 2817.8 |
| dataset 95th percentile return | 3678.3 |
| random policy mean return | 22.4 |
| best eval mean return, target 1800 | 3514.5 |
| best eval mean return, target 3600 | 3549.5 |
| final eval mean return, target 1800 | 3504.3 |
| final eval mean return, target 3600 | 3546.1 |
| final train action MSE | 0.054273 |
| final validation action MSE | 0.049607 |

## Conclusion: **PASS**

Training converged and online policy clearly outperforms random.
