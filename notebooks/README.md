# notebooks/ — analysis notebooks (load-only)

One notebook per config group. Each loads its group's saved results and draws the figures and
tables; **no optimiser ever runs here**.

| config group                           | notebook                         | results                                              |
|----------------------------------------|----------------------------------|------------------------------------------------------|
| `configs/Circle.toml`                  | `Circle_benchmark.ipynb`         | `results/Circle.npz`, `results/Circle__walnuts.npz`  |
| `configs/Rosenbrock.toml`              | `Rosenbrock_benchmark.ipynb`     | `results/Rosenbrock.npz`, `results/Rosenbrock__walnuts.npz` |
| `configs/breast_cancer.toml`           | `breast_cancer_benchmark.ipynb`  | `results/breast_cancer.npz`                          |
| `configs/MoG40/base.toml` + `MoG40/*`  | `MoG40_benchmark.ipynb`          | `results/MoG40.npz`, `results/MoG40/*.npz`           |
| `configs/nice_digits/base.toml` + var. | `nice_digits_benchmark.ipynb`    | `results/nice_digits.npz`, `results/nice_digits/*.npz` |
| `configs/darcy_symmetry/base.toml`     | `darcy_symmetry_benchmark.ipynb` | `results/darcy_symmetry/base.npz`                    |

The workflow is strictly one-directional:

```
python -m gmmvi.benchmark <config> --seeds 0,...     # CLI run -> results/<config>.npz
        |
        v
notebook loads the .npz (benchmark.require_results)  # missing file -> the command to run
        |
        v
plots + tables; figures embedded in the notebook (PNG + vector PDF), none written to disk
```

Post-hoc metric evaluation (TV / EMC / ESS / MMD / entropic W2, the shared reverse-KL
re-estimator) may rebuild the *target* — that is allowed; rerunning an *optimiser* is not.
