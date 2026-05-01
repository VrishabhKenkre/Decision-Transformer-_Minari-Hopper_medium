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
| random policy mean return | 14.8 |
| best eval mean return, target 1800 | 3557.9 |
| best eval mean return, target 3600 | 3570.4 |
| final eval mean return, target 1800 | 3555.1 |
| final eval mean return, target 3600 | 3535.5 |
| final train action MSE | 0.046888 |
| final validation action MSE | 0.052306 |

## Conclusion: **PASS**

Training converged and online policy clearly outperforms random.
