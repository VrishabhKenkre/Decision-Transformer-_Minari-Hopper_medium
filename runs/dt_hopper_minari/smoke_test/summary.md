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
| random policy mean return | 25.0 |
| best eval mean return, target 1800 | 2197.5 |
| best eval mean return, target 3600 | 2395.5 |
| final eval mean return, target 1800 | 2197.5 |
| final eval mean return, target 3600 | 2395.5 |
| final train action MSE | 0.110702 |
| final validation action MSE | 0.098808 |

## Conclusion: **PASS**

Training converged and online policy clearly outperforms random.
