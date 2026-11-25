# WFR GMMVI

Gaussian mixture variational inference driven by Wasserstein-Fisher-Rao (WFR) gradient flows,
written in JAX. The repository also carries three variational baselines (IBW, GMBBVI, SEMTRUX)
and one MCMC baseline (WALNUTS). All five run on the same benchmark targets through one runner,
one result format and one evaluator.

## Requirements

Python 3.10 or newer and [JAX](https://github.com/jax-ml/jax) (`jax` and `jaxlib`; install the
CUDA build to run on a GPU). The remaining runtime packages are listed in
[requirements.txt](requirements.txt). One of them, `umap-learn`, is pinned to an exact version:
only the NICE digits notebook uses it, and UMAP's layout changes between releases, so an
unpinned install would redraw that figure.

The entropic W2 and Sinkhorn divergence metrics use `ott-jax`, which is pinned to the version
the results were produced with, 0.6.0 on JAX 0.10.1.

## Install

```bash
git clone git@github.com:Jonathanjtt/GMMVI.git
cd GMMVI

# 1. JAX first, following the official instructions for your platform:
#    https://docs.jax.dev/en/latest/installation.html
pip install "jax[cuda12]"          # GPU (CUDA 12); or `pip install jax` for CPU only

# 2. The remaining dependencies
pip install -r requirements.txt

# 3. The gmmvi package itself, editable. It declares no dependencies of its own,
#    so this step never touches the JAX install.
pip install -e .
```

## Targets

One config file per benchmark lives in [configs/](configs/); the target definitions are in
[gmmvi/targets/](gmmvi/targets/).

| Config ID | Target |
|---|---|
| [Circle](configs/Circle.toml) | 2-D ring: a continuum of modes on a circle |
| [Rosenbrock](configs/Rosenbrock.toml) | 2-D Rosenbrock banana: one curved, ill-conditioned ridge |
| [MoG40](configs/MoG40/base.toml) | MoG40, a 2-D mixture of 40 well-separated Gaussians (fab-jax `MoG40`) with means drawn uniformly in (-40, 40)²; the mode-coverage stress test |
| [breast_cancer](configs/breast_cancer.toml) | 31-D Bayesian logistic regression on the WDBC breast-cancer data; a unimodal, log-concave posterior |
| [darcy_symmetry](configs/darcy_symmetry/base.toml) | 64-D PDE-constrained Bayesian inverse problem: Darcy-flow permeability with symmetrised observations, which make the posterior bimodal |
| [nice_digits](configs/nice_digits/base.toml) | 196-D NICE normalizing flow trained on 14×14 MNIST digits; high dimensional, with an exact sampler for evaluation |

The config ID is what every command below takes. Two groups have variants with a different
initial condition: `MoG40/init_far`, `MoG40/init_far_small_spread` and `MoG40/init_standard`
for MoG40, and `nice_digits/init_standard` for the NICE flow. A group with variants keeps its base config in the subfolder as `base.toml`, and you run
it by the bare group name: `MoG40`, never `MoG40/base`. The second form is the same config
under a different name and would save its results to `results/MoG40/base.npz`.

## Running experiments

### Configuration

Each benchmark is one TOML file. It fixes the target, the mixture size `K`, the iteration
horizon, the initial mixture, one `[methods.<method ID>]` table of hyperparameters per method,
and the metric budgets. A variant declares `target = "<base>"`, inherits the base config and
overrides only what it changes. [configs/MoG40/base.toml](configs/MoG40/base.toml) shows the
full schema.

### A single run

```bash
python -m gmmvi.benchmark <config ID> [--methods a,b] [--seeds 0,1,2] [--out path] [--eval]
```

With no flags this runs every method in the config on seed 0 and saves the trajectories to
`results/<config ID>.npz`. The method IDs are `WFR`, `ibw`, `gmbbvi`, `semtrux` and
`walnuts`.

The Rosenbrock banana, for example, runs 40 components for 600 iterations with all five
methods, the last of them a 5,000-transition WALNUTS chain:

```bash
python -m gmmvi.benchmark Rosenbrock
```

```
[Rosenbrock] gmbbvi seed=0 ...
[Rosenbrock] semtrux seed=0 ...
[Rosenbrock] ibw seed=0 ...
[Rosenbrock] WFR seed=0 ...
[Rosenbrock] walnuts seed=0 ...
saved 5 runs -> results/Rosenbrock.npz
```

One shared reverse-KL evaluator scores every run with the same random numbers, so the
`-ELBO` traces in one file are comparable across methods. Each run is stamped with the
evaluator version, and the notebooks recompute the traces post hoc for files that predate the
stamp. Open [notebooks/Rosenbrock_benchmark.ipynb](notebooks/Rosenbrock_benchmark.ipynb) to
turn the file into figures and tables.

### Several seeds, a subset of methods, variants

```bash
python -m gmmvi.benchmark MoG40 --seeds 0,1,2,3,4                 # 5 seeds, all methods
python -m gmmvi.benchmark MoG40 --seeds 0,1,2 --methods WFR,ibw   # 3 seeds, 2 methods
python -m gmmvi.benchmark MoG40/init_standard --seeds 0,1,2,3,4   # -> results/MoG40/init_standard.npz
```

A run seed is the Monte Carlo stream the optimiser consumes. The initial mixture is pinned
separately by the config's `init_seed`, so runs with different seeds start from one common
mixture and differ only in that stream. Reported numbers (ESS, entropic W2, the negative ELBO)
are means over the run seeds, and error bars are one sample standard deviation across them.
The metric seeds (`metrics.ess_seeds`, `metrics.emc_seeds`, `metrics.ot.seed`) are a different
thing: they resample a single finished mixture. `evaluate` averages over them internally and
`evaluate_seeds` reports their spread as `<metric>_mcstd`, so the two kinds of seed never stand
in for one another.

The WALNUTS chains in `results/` live in their own files, `results/<config ID>__walnuts.npz`,
which the notebooks load next to the VI file. `--out` writes there:

```bash
python -m gmmvi.benchmark Circle --methods walnuts --seeds 0,1,2,3,4 --out results/Circle__walnuts.npz
```

`--eval` scores the finished runs (TV, EMC, ESS, MMD and entropic W2, where the target
supports each) and writes `results/<config ID>__metrics.json` next to the `.npz`. It scores only
the runs in the file it just wrote, so a group split across a VI file and a WALNUTS file needs
both scored.

### Precision

`breast_cancer`, `nice_digits` and `darcy_symmetry` need float64: the breast-cancer covariance
is too ill-conditioned for float32 to factor late in the run, and the Darcy PDE solve needs
double precision. Their registry builders turn on `jax_enable_x64` when the target is built,
so the CLI needs no environment variable. The 2-D targets run in float32.

## What results/ holds

| File | Methods | Seeds in file | In git |
|---|---|---|---|
| `breast_cancer.npz`, `MoG40.npz`, `MoG40/init_*.npz` | the four VI methods | 0 to 4 | yes |
| `Circle.npz`, `Rosenbrock.npz` | the four VI methods | 0 to 14; the notebooks report 0 to 4 and the rest were tuning seeds | yes |
| `Circle__walnuts.npz`, `Rosenbrock__walnuts.npz` | `walnuts` | 0 to 4 | yes |
| `MoG40/ablation_w_fr.npz` | `WFR_W_only`, `WFR_FR_only` | 0 to 4 | yes |
| `nice_digits.npz` | the four VI methods | 0 to 4 | no |
| `nice_digits/init_standard.npz` | the four VI methods | 0 to 29 | no |
| `darcy_symmetry/base.npz` | the four VI methods | 0 to 4 | no |

The ablation file holds WFR with one branch only: the base MoG40 config with `n_w_steps`
set to 300 (W-only) or 0 (FR-only). Section 7b of the MoG40 notebook draws it next to the
full method.

The NICE digits files are gitignored because the 196×196 covariance trajectories make them
large (0.3 GB and 1.1 GB). Regenerate them with the CLI:

```bash
python -m gmmvi.benchmark nice_digits --seeds 0,1,2,3,4
```

The Darcy file is gitignored because its five-seed file is 169 MB, past GitHub's 100 MB
limit. The Darcy notebook reads it from `results/darcy_symmetry/base.npz`, one level below
where the runner would put it by default, so both routes below name that path explicitly.
This group's results are distributed as GitHub Release assets:

```bash
gh release download results-2026-08-13 -R Jonathanjtt/GMMVI -p darcy_symmetry.npz -D results/
mv results/darcy_symmetry.npz results/darcy_symmetry/base.npz
```

That asset was produced with an earlier base config (`J = 256`, unit initial covariance). The
current base config uses `J = 512` and a point-like initial covariance and dates from
2026-09-20, so the released file does not match the current config. Regenerating the current
base takes a while, since every `log_prob` is a PDE solve:

```bash
python -m gmmvi.benchmark darcy_symmetry --seeds 0,1,2,3,4 --out results/darcy_symmetry/base.npz
```

## The WALNUTS sampler baseline

`walnuts` is the one MCMC method among four variational ones. It answers a question the VI methods cannot
ask of themselves: how much of the remaining gap to the target comes from the mixture family
and how much from the optimiser. The sampler is
[WALNUTS](https://github.com/bob-carpenter/walnuts), the Within-Orbit Adaptive Step-Length
No-U-Turn Sampler of Bou-Rabee, Carpenter, Kleppe and Liu, with its randomised `R2P`
adaptive leapfrog integrator.

Upstream's `WALNUTSpy` is vendored under
[gmmvi/algorithms/walnuts/_vendor/](gmmvi/algorithms/walnuts/_vendor/) (MIT). The copy
differs from upstream only in relative imports, a dropped plotting import, `R2P` as the
default integrator, and the trimming of the other integrators. Everything in
[sampler.py](gmmvi/algorithms/walnuts/sampler.py) is plumbing: it adapts a target, counts
log-density calls and seeds the NumPy stream the sampler draws from.

The sampler needs only the unnormalised log density and its gradient, which the VI methods
already need, so it runs on every target here. The chain starts from one draw of the same
shared initial mixture the VI methods receive, so no method starts from a better point.

It returns draws and no mixture, which limits what can be reported. `MethodResult` carries the
chain in `.draws` with empty `(0, d)` mixture slots and `is_sample_only` set. `evaluate` scores
it on the metrics a cloud of points supports and reports `nan` for the rest:

| Metric | `walnuts` | Why |
| --- | --- | --- |
| EMC, `modes` | yes | a histogram over target modes; needs only samples |
| entropic W2, MMD | yes | two-sample metrics against exact target draws |
| `n_target_evals`, runtime | yes | an exact call counter; the VI methods use an analytic formula |
| TV | `nan` | needs `q(x)` on the grid; a chain histogram is a different estimator |
| reverse KL, -ELBO | `nan` | `E_q[log q - log pi]` needs `log q` |
| ESS | `nan` | the reported ESS is self-normalised importance sampling with `q` as the proposal; MCMC's ESS is an autocorrelation quantity, a different number under the same name |
| weight entropy, coverage | `nan` | no mixture weights and no component means |

For `walnuts`, `n_iterations` counts MCMC transitions, so it is not comparable to an
optimiser horizon and the config sets the chain length directly. Compare methods on the cost
counters instead. For the sampler those are an exact count of log-density calls, warmup
included, because warmup is target work the sampler needs; the warmup/sampling split is kept
in `.extra`. Every call returns a value and a gradient, so `n_grad_evals` equals
`n_target_evals`, and `n_grad_evals` is the axis to use against `semtrux` and `WFR`. `gmbbvi`
is gradient free, so comparing it on `n_target_evals` prices two different things under one
name.

The result keeps upstream's per-transition diagnostics table (step size, tree depth, energy
error). `store_diagnostics = false` in the method's config table drops it and keeps only the
scalar summaries; the tracked configs set it that way.

## Plotting the results

The notebooks in [notebooks/](notebooks/) only load. One notebook per config group opens
`results/<config ID>.npz` and produces the figures and tables; no optimiser runs inside a
notebook. If the file is missing, the notebook prints the CLI command that produces it.
Figures are embedded in the notebook as PNG and vector PDF and are not written to disk.

Each notebook pins one physical GPU index at the top, before importing JAX. The indices change
between sessions, so check `nvidia-smi` and pick an idle card before running all cells.

Aggregates are over the reported seeds: convergence traces are a mean line with a shaded
±1 s.d. band (`plotting.trace_panel` takes a `(n_seeds, n_checkpoints)` block from
`benchmark.stack_traces`), and scalar metrics appear as `mean ± s.d.` in the tables and in the
`plotting.metric_bars` panels. Means are taken over the seeds whose run stayed finite. Views of
a single mixture (density snapshots, sample grids, field reconstructions) show one seed, named
in the notebook, since a snapshot has no meaningful seed average.

## Package layout

```
gmmvi/benchmark.py      runner: shared init, one adapter per method, result IO, evaluation
gmmvi/metrics.py        post-hoc metrics shared by every benchmark
gmmvi/plotting.py       figure code the notebooks call
gmmvi/jaxvi/            shared JAX helpers: target resolution, mixture log-prob, KL traces
gmmvi/targets/          target definitions, their data files, and the registry
gmmvi/algorithms/
    WFR/                the WFR driver and its W -> FR schedule
    ibw/                IBW (inverse Bures-Wasserstein)
    gmbbvi/             GMBBVI
    semtrux/            SEMTRUX
    walnuts/            WALNUTS sampler: vendored upstream code plus the adapter
```

Importing `gmmvi` pulls in no JAX on purpose, so `gmmvi.metrics` works in a plain NumPy test
run.
