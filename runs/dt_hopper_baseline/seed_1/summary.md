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
| random policy mean return | 17.7 |
| best eval mean return, target 1800 | 3530.6 |
| best eval mean return, target 3600 | 3412.0 |
| final eval mean return, target 1800 | 2516.3 |
| final eval mean return, target 3600 | 2926.7 |
| final train action MSE | 0.052325 |
| final validation action MSE | 0.055625 |

## Conclusion: **PASS**

Training converged and online policy clearly outperforms random.
