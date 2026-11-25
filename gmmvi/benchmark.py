"""Experiment runner: one shared init per (target, K, seed), one adapter per method, one result structure.

Per-target settings live in ``configs/<target>.toml`` (any ``configs/*/base.toml`` shows the full schema); a variant
declares ``target = "<base>"`` and deep-merges its own keys on top.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np

from gmmvi import metrics
from gmmvi.targets import registry

__all__ = ["InitState", "MethodResult", "METHODS", "shared_init",
           "run_one", "run_benchmark", "evaluate", "save_results", "load_results",
           "require_results", "load_config", "config_path", "RESULTS_DIR", "CONFIG_DIR",
           "ot_epsilon", "group_seeds", "reduce_seeds", "evaluate_seeds", "stack_traces",
           "kl_evaluator", "kl_traces", "KL_EVALUATOR", "Cost", "grad_evals", "timed",
           "vi_fit_budget"]

_REPO = Path(__file__).resolve().parent.parent
CONFIG_DIR = _REPO / "configs"
RESULTS_DIR = _REPO / "results"


_FALLBACK = dict(K=40, n_iterations=300, init_seed=0,
                 init=dict(offset=0.0, spread=2.0, cov_scale=1e-5),
                 methods={}, metrics=dict(n_emc=50_000, emc_seeds=5, n_ess=50_000,
                                          n_emc_trace=20_000, emc_trace_seeds=3,
                                          coverage_radius=3.0))


def config_path(target_name):
    """``configs/<name>.{toml,json}`` or ``configs/<name>/base.{toml,json}``, else ``None``."""
    for stem in (f"{target_name}", f"{target_name}/base"):
        for suffix in (".toml", ".json"):
            p = CONFIG_DIR / f"{stem}{suffix}"
            if p.exists():
                return p
    return None


def load_config(target_name, overrides=None, _seen=()):
    """Settings for ``target_name``, else :data:`_FALLBACK`; a variant inherits ``cfg["target"]`` and deep-merges.

    ``overrides`` is deep-merged last.  ``cfg["target"]`` is always the real registry name.
    """
    cfg = None
    p = config_path(target_name)
    if p is not None:
        cfg = (_load_toml if p.suffix == ".toml" else _load_json)(p)
    if cfg is None:
        cfg = dict(_FALLBACK)
    cfg = _deep_copy(cfg)
    base = cfg.get("target")
    if base and base != target_name:
        if target_name in _seen:
            raise ValueError("config inheritance cycle: "
                             + " -> ".join((*_seen, target_name, base)))
        parent = load_config(base, _seen=(*_seen, target_name))
        real_target = parent.get("target", base)
        cfg = _deep_merge(parent, {k: v for k, v in cfg.items() if k != "target"})
        cfg["target"] = real_target
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg


def _load_toml(path):
    import tomllib

    with open(path, "rb") as fh:
        return tomllib.load(fh)


def _load_json(path):
    return json.loads(path.read_text())


def _deep_copy(d):
    return {k: _deep_copy(v) if isinstance(v, dict) else v for k, v in d.items()}


def _deep_merge(base, extra):
    out = _deep_copy(base)
    for k, v in extra.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


@dataclass
class InitState:
    """One initial mixture, shared by every method in a run."""
    means: np.ndarray
    covariances: np.ndarray
    weights: np.ndarray

    def copy(self):
        return InitState(self.means.copy(), self.covariances.copy(), self.weights.copy())


def shared_init(K, dim, offset=0.0, spread=2.0, cov_scale=1e-5, seed=0, negate_pairs=False):
    """``means = offset + spread * RandomState(seed).randn(K, dim)``, ``covs = cov_scale * I``, uniform weights.

    ``negate_pairs=True`` sets ``means[K-1-i] = -means[i]`` for ``i < K // 2`` after the draw.
    """
    rng = np.random.RandomState(seed)
    means = np.asarray(offset, dtype=float) + spread * rng.randn(K, dim)
    if negate_pairs:
        for i in range(K // 2):
            means[K - 1 - i] = -means[i]
    covs = cov_scale * np.broadcast_to(np.eye(dim), (K, dim, dim)).copy()
    weights = np.full(K, 1.0 / K)
    return InitState(means, covs, weights)


def _n_logged(n_iterations, compute_kl):
    return len(range(0, n_iterations + 1, max(1, int(compute_kl or 1))))


class Cost(NamedTuple):
    """What one run spent on the target: ``target_evals`` (every value or score, KL diagnostic included),
    ``grad_evals`` (value-gradient pairs, diagnostic excluded; 0 for the gradient-free GMBBVI) and
    ``hess_evals`` (no current method evaluates a Hessian).
    """
    target_evals: int
    grad_evals: int
    hess_evals: int = 0


def _score_batch(method, hp, dim):
    """Points per component per iteration costing one value-gradient pair (``4 * dim`` when ``J`` is unset)."""
    if method == "gmbbvi":
        return 0
    if method == "ibw":
        return int(hp.get("BG", 10))
    J = hp.get("J")
    return int(J) if J is not None else 4 * int(dim)


KL_EVALUATOR = "shared-crn/v1"

_KL_KEY_SEED, _KL_Z_SEED = 20250806, 20250807


def kl_evaluator(dim, BKL, cfg=None):
    """The ``(key, z)`` every reverse-KL trace shares (common random numbers); ``[metrics] kl_seed`` / ``kl_z_seed`` override."""
    import jax

    mcfg = (cfg or {}).get("metrics", {})
    key = jax.random.PRNGKey(int(mcfg.get("kl_seed", _KL_KEY_SEED)))
    z = jax.random.normal(jax.random.PRNGKey(int(mcfg.get("kl_z_seed", _KL_Z_SEED))),
                          (int(BKL), int(dim)))
    return key, z


def _shared_kl_trace(raw, spec, cfg, BKL):
    """Reverse-KL trace on the shared evaluator; None when the driver logged no trajectory."""
    if raw.get("means_log") is None or raw.get("covs_log") is None:
        return None
    key, z = kl_evaluator(spec.dim, BKL, cfg)
    return metrics.reverse_kl_trace(
        {"means_log": raw["means_log"], "covs_log": raw["covs_log"],
         "weights_log": raw["weights_log"], "elbo_iters": raw.get("elbo_iters")},
        spec.target, key, z)


def _fit_gmbbvi(spec, init, K, n_iterations, seed, hp):
    from gmmvi.algorithms.gmbbvi.optim import fit

    hp = dict(hp)
    J, BKL, ckl = hp.get("J"), hp.get("BKL", 1000), hp.get("compute_kl")
    res = fit(spec.target, n_components=K, n_iterations=n_iterations,
              init_means=init.means.copy(), init_covs=init.covariances.copy(),
              seed=seed, compute_kl_trace=False, **hp)
    J = J if J is not None else 4 * spec.dim
    n = n_iterations * K * J + _n_logged(n_iterations, ckl) * BKL
    return res, Cost(n, 0)


def _fit_semtrux(spec, init, K, n_iterations, seed, hp):
    from gmmvi.algorithms.semtrux.optim import fit

    hp = dict(hp)
    J, BKL, ckl = hp.get("J"), hp.get("BKL", 1000), hp.get("compute_kl")
    res = fit(spec.target, n_components=K, n_iterations=n_iterations,
              init_means=init.means.copy(), init_covs=init.covariances.copy(),
              seed=seed, compute_kl_trace=False, **hp)
    J = J if J is not None else 4 * spec.dim
    n = n_iterations * K * J + _n_logged(n_iterations, ckl) * BKL
    return res, Cost(n, n_iterations * K * J)


def _fit_ibw(spec, init, K, n_iterations, seed, hp):
    from gmmvi.algorithms.ibw.optim import fit

    hp = dict(hp)
    BG, BKL, ckl = hp.pop("BG", 10), hp.get("BKL", 1000), hp.get("compute_kl")
    res = fit(spec.target, n_components=K, method="ibw", n_iterations=n_iterations,
              BG=BG, init_means=init.means.copy(), init_covs=init.covariances.copy(),
              seed=seed, compute_kl_trace=False, **hp)
    n = n_iterations * K * BG + _n_logged(n_iterations, ckl) * BKL
    return res, Cost(n, n_iterations * K * BG)


_WFR_FIXED = ("b_fr", "dt_weight")


def _wfr_fixed_defaults():
    import inspect

    from gmmvi.algorithms.WFR.WFR import fit

    sig = inspect.signature(fit).parameters
    return {k: sig[k].default for k in _WFR_FIXED}


def _fit_wfr(spec, init, K, n_iterations, seed, hp):
    from gmmvi.algorithms.WFR.WFR import fit

    hp = dict(hp)
    bad = [k for k in _WFR_FIXED if k in hp]
    if bad:
        fixed = _wfr_fixed_defaults()
        raise ValueError(
            "WFR hyperparameters " + ", ".join(repr(k) for k in bad) + " are fixed at the "
            "driver defaults (" + ", ".join(f"{k}={fixed[k]!r}" for k in bad) + ") and "
            "cannot be set from a config. Remove them, or call "
            "gmmvi.algorithms.WFR.WFR.fit directly to sweep them.")
    J, BKL, ckl = hp.get("J"), hp.get("BKL", 1000), hp.get("compute_kl")
    res = fit(spec.target, n_iterations=n_iterations,
              init_means=init.means.copy(), init_covs=init.covariances.copy(),
              seed=seed, compute_kl_trace=False, **hp)
    J = J if J is not None else 4 * spec.dim
    n = n_iterations * K * J + _n_logged(n_iterations, ckl) * BKL
    return res, Cost(n, n_iterations * K * J, 0)


def _fit_walnuts(spec, init, K, n_iterations, seed, hp):
    """WALNUTS-R2P sampler: ``n_iterations`` are MCMC transitions, the cost is its exact ``lpFun`` counter."""
    from gmmvi.algorithms.walnuts import sampler

    hp = dict(hp)
    hp.pop("BKL", None)
    hp.pop("compute_kl", None)
    q0 = hp.pop("q0", None)
    if q0 is None:
        q0 = metrics.sample_gmm(init.means, init.covariances, init.weights, 1, seed)[0]
    res = sampler.fit(spec.target, seed=seed, q0=q0, dim=spec.dim, **hp)
    n = int(res["n_target_evals"])
    return res, Cost(n, n)


METHODS = {
    "gmbbvi": _fit_gmbbvi,
    "semtrux": _fit_semtrux,
    "ibw": _fit_ibw,
    "WFR": _fit_wfr,
    "walnuts": _fit_walnuts,
}

SAMPLE_ONLY_METHODS = frozenset({"walnuts"})

_CORE_KEYS = {"means", "covariances", "weights", "epsilons", "kls", "elbo", "elbo_iters",
              "means_log", "covs_log", "weights_log", "draws", "n_iterations"}
_UNSAVEABLE = {"vi"}


_COMPILE_EVENT_PREFIX = "/jax/core/compile/"
_timer_lock = threading.Lock()
_timer_spans = []
_compile_listener_installed = False


def _record_compile_span(event, start_time, end_time, **_):
    """``jax.monitoring`` callback: file one compile phase against every open timer."""
    if not event.startswith(_COMPILE_EVENT_PREFIX):
        return
    with _timer_lock:
        for spans in _timer_spans:
            spans.append((start_time, end_time))


def _install_compile_listener():
    """Subscribe to JAX's compile events once per process; False when JAX is absent."""
    global _compile_listener_installed
    if _compile_listener_installed:
        return True
    try:
        from jax import monitoring
    except ImportError:
        return False
    register_span = getattr(monitoring, "register_event_time_span_listener", None)
    if register_span is not None:
        register_span(_record_compile_span)
    else:
        monitoring.register_event_duration_secs_listener(
            lambda event, duration, **kw: _record_compile_span(
                event, time.time() - duration, time.time()))
    _compile_listener_installed = True
    return True


def _union_seconds(spans, t0, t1):
    """Seconds covered by ``spans``, clipped to ``[t0, t1]``, counting overlap ONCE."""
    total, covered_to = 0.0, t0
    for start, end in sorted((max(s, t0), min(e, t1)) for s, e in spans):
        if end <= start:
            continue
        start = max(start, covered_to)
        if end > start:
            total += end - start
            covered_to = end
    return total


@contextmanager
def timed():
    """Time a block; yields a dict with ``wall_s``, ``compile_s`` (JAX compile phases) and ``runtime_s = wall_s - compile_s``."""
    out, spans = {}, []
    have_jax = _install_compile_listener()
    with _timer_lock:
        _timer_spans.append(spans)
    t0 = time.time()
    try:
        yield out
    finally:
        t1 = time.time()
        with _timer_lock:
            for i, open_spans in enumerate(_timer_spans):
                if open_spans is spans:
                    del _timer_spans[i]
                    break
            mine = list(spans)
        wall = max(t1 - t0, 0.0)
        compiled = min(_union_seconds(mine, t0, t1), wall) if have_jax else 0.0
        out.update(wall_s=wall, compile_s=compiled, runtime_s=wall - compiled)


@dataclass
class MethodResult:
    """One method x one target x one seed, normalised across the four drivers."""
    method: str
    target: str
    seed: int
    K: int
    n_iterations: int
    means: np.ndarray
    covariances: np.ndarray
    weights: np.ndarray
    epsilons: Optional[np.ndarray] = None
    draws: Optional[np.ndarray] = None
    iters: Optional[np.ndarray] = None
    means_log: Optional[np.ndarray] = None
    covs_log: Optional[np.ndarray] = None
    weights_log: Optional[np.ndarray] = None
    kls: Optional[np.ndarray] = None
    elbo: Optional[np.ndarray] = None
    runtime_s: float = float("nan")
    compile_s: float = float("nan")
    n_target_evals: int = 0
    n_grad_evals: Optional[int] = None
    n_hess_evals: Optional[int] = None
    hparams: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw, *, method, target, seed, K, n_iterations, runtime_s,
                 cost, hparams, compile_s=float("nan")):
        extra = {k: v for k, v in raw.items() if k not in _CORE_KEYS and k not in _UNSAVEABLE}
        return cls(
            method=method, target=target, seed=seed, K=K, n_iterations=n_iterations,
            means=np.asarray(raw["means"]), covariances=np.asarray(raw["covariances"]),
            weights=np.asarray(raw["weights"]),
            epsilons=None if raw.get("epsilons") is None else np.asarray(raw["epsilons"]),
            draws=_maybe(raw.get("draws")),
            iters=_maybe(raw.get("elbo_iters")), means_log=_maybe(raw.get("means_log")),
            covs_log=_maybe(raw.get("covs_log")), weights_log=_maybe(raw.get("weights_log")),
            kls=_maybe(raw.get("kls")), elbo=_maybe(raw.get("elbo")),
            runtime_s=runtime_s, compile_s=compile_s,
            n_target_evals=int(cost.target_evals),
            n_grad_evals=int(cost.grad_evals), n_hess_evals=int(cost.hess_evals),
            hparams=dict(hparams), extra=extra,
        )

    def as_trace_dict(self):
        """The ``res``-shaped dict :mod:`gmmvi.metrics` trace helpers expect."""
        return {"means_log": self.means_log, "covs_log": self.covs_log,
                "weights_log": self.weights_log, "elbo_iters": self.iters}

    @property
    def is_sample_only(self):
        """True when this result is a cloud of draws with no fitted mixture."""
        return self.draws is not None

    @property
    def diverged(self):
        if self.is_sample_only:
            return not np.isfinite(np.asarray(self.draws, float)).all()
        return not np.isfinite(np.asarray(self.means, float)).all()

    def q_sample(self, n, seed):
        """``n`` draws from this run's q -> (n, d): ancestral for a mixture, a subsample for a chain."""
        if self.is_sample_only:
            return metrics.subsample(self.draws, n, seed)
        return metrics.sample_gmm(self.means, self.covariances, self.weights, n, seed)


def _maybe(x):
    return None if x is None else np.asarray(x)


def vi_fit_budget(cfg):
    """``n_iterations * K * J``: the value-gradient pairs the variational methods spend fitting (GMBBVI and samplers excluded)."""
    Js = {m: hp["J"] for m, hp in cfg.get("methods", {}).items()
          if m in METHODS and m not in SAMPLE_ONLY_METHODS and m != "gmbbvi" and "J" in hp}
    if not Js:
        raise ValueError("no variational method in this config declares J, so there is no "
                         "fit budget to hold a sampler to")
    if len(set(Js.values())) != 1:
        raise ValueError(f"the variational methods disagree on J ({Js}); no single fit "
                         "budget exists, so a sampler cannot be cost-matched to them")
    return int(cfg["n_iterations"]) * int(cfg["K"]) * int(next(iter(Js.values())))


def grad_evals(result, dim=None):
    """Value-gradient pair evaluations of one run; reconstructed from ``(method, K, n_iterations, hparams)`` for old files."""
    if result.n_grad_evals is not None:
        return int(result.n_grad_evals)
    if result.method in SAMPLE_ONLY_METHODS:
        return int(result.n_target_evals)
    if dim is None:
        arr = result.means if result.means is not None and result.means.size else result.draws
        dim = np.shape(arr)[-1]
    return (int(result.n_iterations) * int(result.K)
            * _score_batch(result.method, result.hparams or {}, dim))


def run_one(target_name, method, seed=0, config=None, spec=None, init=None):
    """One method x one target x one seed -> :class:`MethodResult`; ``target_name`` may be a variant config name."""
    cfg = config if config is not None else load_config(target_name)
    spec = spec if spec is not None else registry.build(cfg.get("target", target_name),
                                                        **cfg.get("target_kwargs", {}))
    K, n_iterations = int(cfg["K"]), int(cfg["n_iterations"])
    if init is None:
        init = make_init(cfg, spec, seed)
    hp = dict(cfg.get("methods", {}).get(method, {}))
    if not cfg.get("forward_init_weights", False):
        hp.pop("init_weights", None)

    with timed() as clock:
        raw, cost = METHODS[method](spec, init.copy(), K, n_iterations, seed, hp)
        kls = _shared_kl_trace(raw, spec, cfg, int(hp.get("BKL", 1000)))
        if kls is not None:
            raw["kls"] = kls
            raw["elbo"] = -np.asarray(kls)
            raw["kl_evaluator"] = KL_EVALUATOR

    if cfg.get("log_covs", True) is False:
        raw["covs_log"] = None

    return MethodResult.from_raw(raw, method=method, target=target_name, seed=seed, K=K,
                                 n_iterations=int(raw.get("n_iterations", n_iterations)),
                                 runtime_s=clock["runtime_s"],
                                 compile_s=clock["compile_s"], cost=cost, hparams=hp)


def make_init(cfg, spec, seed):
    """Shared init for one seed, honouring the config's ``init_seed`` policy."""
    ic = cfg.get("init", {})
    init_seed = cfg.get("init_seed", 0)
    return shared_init(K=int(cfg["K"]), dim=spec.dim,
                       offset=ic.get("offset", 0.0), spread=ic.get("spread", 2.0),
                       cov_scale=ic.get("cov_scale", 1e-5),
                       seed=seed if init_seed is None else int(init_seed),
                       negate_pairs=bool(ic.get("negate_pairs", False)))


def run_benchmark(target_name, methods=None, seeds=(0,), config=None, overrides=None,
                  spec=None, verbose=True):
    """Loop over methods x seeds -> ``{(method, seed): MethodResult}``, one shared init per seed."""
    cfg = config if config is not None else load_config(target_name, overrides)
    spec = spec if spec is not None else registry.build(cfg.get("target", target_name),
                                                        **cfg.get("target_kwargs", {}))
    methods = list(methods if methods is not None else cfg.get("methods", {}).keys())
    out = {}
    for seed in seeds:
        init = make_init(cfg, spec, seed)
        for m in methods:
            if verbose:
                print(f"[{target_name}] {m} seed={seed} ...", flush=True)
            out[(m, seed)] = run_one(target_name, m, seed=seed, config=cfg, spec=spec,
                                     init=init)
    return out


_OT_EPSILON_CACHE = {}


def ot_epsilon(spec, config=None, force=False):
    """Entropic-OT epsilon for ``spec`` (from target samples only), computed once and cached."""
    cfg = config if config is not None else load_config(spec.name)
    o = cfg.get("metrics", {}).get("ot", {})
    key = (spec.name, o.get("epsilon"), o.get("relative_epsilon", 0.01),
           o.get("max_scale_samples", 2000), o.get("scale_seed", 0),
           o.get("normalize_by_dim", False))
    if force or key not in _OT_EPSILON_CACHE:
        n_scale = int(o.get("max_scale_samples", 2000))
        Y = spec.sample_fn(n_scale, int(o.get("scale_seed", 0)))
        _OT_EPSILON_CACHE[key] = metrics.resolve_ot_epsilon(
            Y, epsilon=o.get("epsilon"),
            relative_epsilon=float(o.get("relative_epsilon", 0.01)),
            max_scale_samples=n_scale, scale_seed=int(o.get("scale_seed", 0)),
            normalize_by_dim=bool(o.get("normalize_by_dim", False)))
    return _OT_EPSILON_CACHE[key]


_MMD_BANDWIDTH_CACHE = {}


def mmd_bandwidth(spec, config=None, force=False):
    """MMD bandwidth for ``spec`` (median heuristic on target samples), computed once and cached."""
    cfg = config if config is not None else load_config(spec.name)
    o = cfg.get("metrics", {}).get("mmd", {})
    key = (spec.name, o.get("bandwidth"), o.get("max_points", 2048),
           o.get("bandwidth_seed", 99))
    if force or key not in _MMD_BANDWIDTH_CACHE:
        max_points = int(o.get("max_points", 2048))
        Y = spec.sample_fn(max_points, int(o.get("target_seed", 0))) if spec.can_sample else None
        _MMD_BANDWIDTH_CACHE[key] = metrics.resolve_mmd_bandwidth(
            Y, bandwidth=o.get("bandwidth"), max_points=max_points,
            seed=int(o.get("bandwidth_seed", 99)))
    return _MMD_BANDWIDTH_CACHE[key]


def evaluate(result, spec, config=None, traces=False, ot_info=None, mmd_info=None):
    """Post-hoc metrics for one :class:`MethodResult`; keys the target cannot support are omitted.

    ``traces=True`` adds the per-iteration TV / EMC / ESS traces (needs ``covs_log``).
    """
    if traces and result.covs_log is None and not result.is_sample_only:
        raise ValueError(
            f"{result.method} seed {result.seed}: per-iteration traces were requested, but this "
            "run carries no covariance trajectory -- it was fitted with `log_covs = false`, "
            "which drops the (n_logged, K, d, d) snapshots.  Scalar metrics and the shared "
            "reverse-KL trace are unaffected; for TV / EMC / ESS traces re-run the benchmark "
            "with `log_covs = true`.")
    cfg = config if config is not None else load_config(result.target)
    mcfg = cfg.get("metrics", {})
    return _evaluate(result, spec, cfg, mcfg, traces, ot_info, mmd_info)


def _score_at_budget(result, spec, cfg, mcfg, ocfg, kcfg, info, kinfo):
    """Score a sample-only run on the chain prefix it had produced at the VI methods' pair budget (``*_at_budget`` keys)."""
    from gmmvi.algorithms.walnuts import sampler

    cum = result.extra.get("evals_cumulative")
    if cum is None or not np.size(cum):
        return {"budget_pairs": np.nan, "budget_note": "no evals_cumulative in this run"}

    try:
        budget = int(mcfg.get("budget_pairs", 0)) or vi_fit_budget(cfg)
    except (ValueError, KeyError) as exc:
        return {"budget_pairs": np.nan, "budget_note": f"no VI fit budget in this config: {exc}"}
    res = {"evals_cumulative": np.asarray(cum), "draws": result.draws,
           "n_iterations": int(result.n_iterations), "n_draws": int(len(result.draws)),
           "thin": int(result.hparams.get("thin", 1)),
           "n_target_evals": int(result.n_target_evals),
           "n_target_evals_warmup": int(result.extra.get("n_target_evals_warmup", 0))}
    draws, meta = sampler.draws_at_budget(res, budget)

    out = {"budget_pairs": budget,
           "pairs_at_budget": meta["pairs_spent"],
           "transitions_at_budget": meta["transitions"],
           "n_draws_at_budget": meta["n_draws"],
           "budget_exhausted_by_warmup": meta["exhausted_by_warmup"],
           "budget_exceeds_chain": meta["budget_exceeds_chain"],
           "entropic_w2_at_budget": np.nan, "mmd_at_budget": np.nan}
    if not len(draws):
        return out

    if info is not None:
        n_ot, ot_seed = int(ocfg.get("n", 2000)), int(ocfg.get("seed", 0))
        val, conv = metrics.entropic_w2(
            None, None, None, spec.sample_fn, info["epsilon"], n=n_ot, seed=ot_seed,
            target_seed=int(ocfg.get("target_seed", 0)),
            q_samples=metrics.subsample(draws, n_ot, ot_seed),
            max_iterations=int(ocfg.get("max_iterations", 20_000)))
        out["entropic_w2_at_budget"] = val
        out["ot_converged_at_budget"] = conv
    if kinfo is not None:
        n_mmd, mmd_seed = int(kcfg.get("n", 2000)), int(kcfg.get("seed", 0))
        out["mmd_at_budget"] = metrics.mmd(
            None, None, None, spec.sample_fn, bandwidth=kinfo["bandwidth"], n=n_mmd,
            seed=mmd_seed, target_seed=int(kcfg.get("target_seed", 0)),
            biased=bool(kcfg.get("biased", True)),
            q_samples=metrics.subsample(draws, n_mmd, mmd_seed))
    return out


def _evaluate(result, spec, cfg, mcfg, traces, ot_info, mmd_info):
    m, c, w = result.means, result.covariances, result.weights
    sample_only = result.is_sample_only
    out = {"weight_entropy": np.nan if sample_only else metrics.weight_entropy(w),
           "runtime_s": result.runtime_s,
           "n_target_evals": result.n_target_evals,
           "n_grad_evals": grad_evals(result)}
    if sample_only:
        out.update(tv=np.nan, reverse_kl=np.nan, ess=np.nan, coverage=np.nan)
    if result.kls is not None and len(result.kls):
        out["reverse_kl"] = float(np.asarray(result.kls, float)[-1])

    if spec.has_grid and not sample_only:
        out["tv"] = metrics.total_variation(spec.grid_pts, spec.ref_density, m, c, w)
        if traces:
            out["tv_trace"] = metrics.trace(
                result.as_trace_dict(),
                lambda mm, cc, ww: metrics.total_variation(spec.grid_pts, spec.ref_density,
                                                           mm, cc, ww),
                skip_nonfinite=False)

    ocfg = mcfg.get("ot", {})
    info = None
    if ocfg.get("enabled", True) and spec.can_sample and not result.diverged:
        info = ot_epsilon(spec, cfg) if ot_info is None else ot_info
        n_ot, ot_seed = int(ocfg.get("n", 2000)), int(ocfg.get("seed", 0))
        val, converged = metrics.entropic_w2(
            m, c, w, spec.sample_fn, info["epsilon"],
            n=n_ot, seed=ot_seed,
            target_seed=int(ocfg.get("target_seed", 0)),
            q_samples=result.q_sample(n_ot, ot_seed) if sample_only else None,
            max_iterations=int(ocfg.get("max_iterations", 20_000)))
        out["entropic_w2"] = val
        out["ot_converged"] = converged
        out["ot_epsilon"] = info["epsilon"]
        out["ot_epsilon_mode"] = info["epsilon_mode"]
        out["ot_relative_epsilon"] = info["relative_epsilon"]
        out["ot_cost_scale"] = info["cost_scale"]

    kcfg = mcfg.get("mmd", {})
    kinfo = None
    if kcfg.get("enabled", True) and spec.can_sample and not result.diverged:
        kinfo = mmd_bandwidth(spec, cfg) if mmd_info is None else mmd_info
        n_mmd, mmd_seed = int(kcfg.get("n", 2000)), int(kcfg.get("seed", 0))
        out["mmd"] = metrics.mmd(
            m, c, w, spec.sample_fn, bandwidth=kinfo["bandwidth"],
            n=n_mmd, seed=mmd_seed,
            target_seed=int(kcfg.get("target_seed", 0)),
            biased=bool(kcfg.get("biased", True)),
            q_samples=result.q_sample(n_mmd, mmd_seed) if sample_only else None)
        out["mmd_bandwidth"] = kinfo["bandwidth"]
        out["mmd_bandwidth_mode"] = kinfo["bandwidth_mode"]

    if sample_only and not result.diverged and (info is not None or kinfo is not None):
        out.update(_score_at_budget(result, spec, cfg, mcfg, ocfg, kcfg, info, kinfo))

    if spec.mode_means is not None and not result.diverged:
        n_emc = int(mcfg.get("n_emc", 50_000))
        seeds = range(int(mcfg.get("emc_seeds", 5)))
        emcs, modes, esss = [], [], []
        for s in seeds:
            X = result.q_sample(n_emc, s)
            e, nm = metrics.emc_from_samples(X, spec.mode_means, spec.mode_precisions)
            emcs.append(e)
            modes.append(nm)
            if not sample_only:
                esss.append(metrics.ess_from_samples(X, m, c, w, spec.log_prob_np))
        out.update(emc=float(np.mean(emcs)), emc_std=float(np.std(emcs)),
                   modes=float(np.mean(modes)))
        if not sample_only:
            out.update(ess=float(np.nanmean(esss)), ess_std=float(np.nanstd(esss)))
            out["coverage"] = metrics.mode_coverage(m, spec.mode_means, spec.mode_precisions,
                                                    r=float(mcfg.get("coverage_radius", 3.0)))
        if traces and not sample_only:
            n_tr = int(mcfg.get("n_emc_trace", 20_000))
            tr_seeds = range(int(mcfg.get("emc_trace_seeds", 3)))
            out["emc_trace"] = metrics.trace(
                result.as_trace_dict(),
                lambda mm, cc, ww, s: metrics.emc(mm, cc, ww, spec.mode_means, n_tr, s,
                                                  spec.mode_precisions)[0],
                seeds=tr_seeds)
            out["ess_trace"] = metrics.trace(
                result.as_trace_dict(),
                lambda mm, cc, ww, s: metrics.ess_fraction(mm, cc, ww, spec.log_prob_np,
                                                           n_tr, s),
                seeds=tr_seeds)
    elif spec.mode_means is not None:
        out.update(emc=np.nan, modes=np.nan, ess=np.nan)
        if not sample_only:
            out["coverage"] = 0
    elif sample_only:
        pass
    elif not result.diverged:
        n_ess = int(mcfg.get("n_ess", 50_000))
        seeds = range(int(mcfg.get("ess_seeds", mcfg.get("emc_seeds", 5))))
        esss = [metrics.ess_fraction(m, c, w, spec.log_prob_np, n_ess, s) for s in seeds]
        out.update(ess=float(np.nanmean(esss)), ess_std=float(np.nanstd(esss)))
    else:
        out.update(ess=np.nan)
    return out


_MC_SPREAD_KEYS = ("ess_std", "emc_std")


def group_seeds(results, methods=None):
    """``{(method, seed): MethodResult}`` -> ``{method: {seed: MethodResult}}``, seeds sorted, ``methods`` fixes the order."""
    order = methods if methods is not None else list(dict.fromkeys(m for m, _ in results))
    out = {}
    for m in order:
        runs = {s: r for (mm, s), r in results.items() if mm == m}
        if runs:
            out[m] = {s: runs[s] for s in sorted(runs)}
    return out


def reduce_seeds(values, spread="std", ddof=1):
    """``(mean, spread, n_finite)`` over the finite entries; ``spread`` is ``"std"`` (ddof), ``"sem"`` or ``"minmax"``."""
    v = np.asarray(values, float).ravel()
    fin = v[np.isfinite(v)]
    if fin.size == 0:
        return np.nan, np.nan, 0
    mean = float(fin.mean())
    if fin.size == 1:
        return mean, 0.0, 1
    if spread == "minmax":
        sd = float(fin.max() - fin.min()) / 2.0
    else:
        sd = float(fin.std(ddof=ddof))
        if spread == "sem":
            sd /= np.sqrt(fin.size)
    return mean, sd, int(fin.size)


def evaluate_seeds(runs, spec, config=None, *, ot_info=None, mmd_info=None,
                   spread="std", ddof=1, per_seed=False, scores=None):
    """:func:`evaluate` every seed of one method -> row with ``<metric>``, ``<metric>_std`` (across run seeds),
    ``<metric>_n`` and ``<metric>_mcstd`` (within-run metric-seed spread); traces are skipped.
    """
    scored = scores if scores is not None else {
        s: evaluate(r, spec, config, ot_info=ot_info, mmd_info=mmd_info)
        for s, r in runs.items()}

    keys = list(dict.fromkeys(k for d in scored.values() for k in d))
    row = {"n_seeds": len(runs), "seeds": sorted(runs),
           "diverged_seeds": sorted(s for s, r in runs.items() if r.diverged)}
    for k in keys:
        vals = [d[k] for d in scored.values() if k in d]
        if any(np.ndim(v) > 0 for v in vals):
            continue
        if all(isinstance(v, str) for v in vals):
            uniq = set(vals)
            row[k] = vals[0] if len(uniq) == 1 else "<mixed>"
            continue
        if k in _MC_SPREAD_KEYS:
            m, _, _ = reduce_seeds(vals, spread, ddof)
            row[k[:-4] + "_mcstd"] = m
            continue
        mean, sd, n = reduce_seeds(vals, spread, ddof)
        row[k] = mean
        row[k + "_std"] = sd
        row[k + "_n"] = n
    out = {k: v for k, v in row.items() if not k.endswith("_mcstd") or np.isfinite(v)}
    return (out, scored) if per_seed else out


def kl_traces(grouped, spec=None, config=None, verbose=True):
    """``(iters, {method: (n_seeds, n_points)})`` reverse-KL traces; recomputed post hoc (needs ``spec``) for pre-stamp files."""
    grouped = {m: runs for m, runs in grouped.items()
               if not all(r.is_sample_only for r in runs.values())}
    if not grouped:
        return None, {}
    methods = list(grouped)
    stamps = {str(np.asarray(r.extra.get("kl_evaluator", "")).item() or "")
              for runs in grouped.values() for r in runs.values()}

    if stamps == {KL_EVALUATOR}:
        iters = None
        out = {}
        for m in methods:
            iters, out[m] = stack_traces(grouped[m], "kls")
        return iters, out

    BKL = int(next(iter(grouped[methods[0]].values())).hparams.get("BKL", 1000))
    key, z = kl_evaluator(spec.dim, BKL, config)
    out = {m: np.stack([np.asarray(metrics.reverse_kl_trace(r.as_trace_dict(), spec.target,
                                                            key, z), float)
                        for _, r in sorted(grouped[m].items())])
           for m in methods}
    iters = np.asarray(next(iter(grouped[methods[0]].values())).iters)
    return iters, out


def stack_traces(runs, key="kls"):
    """``{seed: MethodResult}`` -> ``(iters, Y)`` with ``Y`` of shape ``(n_seeds, n_points)``."""
    seeds = sorted(runs)
    rows = [np.asarray(getattr(runs[s], key), float).ravel() for s in seeds]
    it = runs[seeds[0]].iters
    it = np.arange(rows[0].size) if it is None else np.asarray(it)
    return it, np.stack(rows)


def _key(method, seed):
    return f"{method}__seed{seed}"


def save_results(results, target_name, path=None):
    """Write ``{(method, seed): MethodResult}`` to a single ``.npz`` under ``results/``."""
    path = Path(path) if path is not None else RESULTS_DIR / f"{target_name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays, meta = {}, {}
    for (method, seed), r in results.items():
        k = _key(method, seed)
        d = asdict(r)
        meta[k] = {f: d.pop(f) for f in ("method", "target", "seed", "K", "n_iterations",
                                         "runtime_s", "compile_s",
                                         "n_target_evals", "n_grad_evals",
                                         "n_hess_evals", "hparams")}
        d.pop("extra")
        for name, v in d.items():
            if v is not None:
                arrays[f"{k}/{name}"] = np.asarray(v)
        for name, v in r.extra.items():
            v = np.asarray(v)
            if v.dtype.kind in "biufU":
                arrays[f"{k}/extra/{name}"] = v
    arrays["__meta__"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    np.savez_compressed(path, **arrays)
    return path


def require_results(target_name, path=None):
    """:func:`load_results`, but a missing file names the CLI command that produces it."""
    p = Path(path) if path is not None else RESULTS_DIR / f"{target_name}.npz"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. This notebook only loads saved results; produce them with:\n"
            f"    python -m gmmvi.benchmark {target_name} --seeds 0\n"
            f"(add more seeds / --methods as needed), then re-run this cell.")
    return load_results(target_name, path=p)


def load_results(target_name, path=None):
    """Inverse of :func:`save_results` -> ``{(method, seed): MethodResult}``."""
    path = Path(path) if path is not None else RESULTS_DIR / f"{target_name}.npz"
    z = np.load(path, allow_pickle=False)
    meta = json.loads(bytes(z["__meta__"]).decode())
    out = {}
    for k, mk in meta.items():
        fields, extra = {}, {}
        prefix = f"{k}/"
        for name in z.files:
            if not name.startswith(prefix):
                continue
            rest = name[len(prefix):]
            (extra if rest.startswith("extra/") else fields)[rest.replace("extra/", "")] = z[name]
        out[(mk["method"], mk["seed"])] = MethodResult(**mk, **fields, extra=extra)
    return out


def main(argv=None):
    """``python -m gmmvi.benchmark <config> [--methods a,b] [--seeds 0,1,2] [--eval]`` -> ``results/<config>.npz``."""
    import argparse

    ap = argparse.ArgumentParser(prog="python -m gmmvi.benchmark",
                                 description="Run one benchmark config and save results/")
    ap.add_argument("config", help="config name: configs/<name>.toml|.json; may contain "
                                   "'/' to group results into folders")
    ap.add_argument("--methods", default=None,
                    help="comma-separated subset (default: every method in the config)")
    ap.add_argument("--seeds", default="0", help="comma-separated run seeds (default: 0)")
    ap.add_argument("--out", default=None, help="results path (default results/<config>.npz)")
    ap.add_argument("--eval", action="store_true",
                    help="also run evaluate() per run and write <config>__metrics.json")
    args = ap.parse_args(argv)

    seeds = tuple(int(s) for s in args.seeds.split(",") if s != "")
    methods = args.methods.split(",") if args.methods else None
    cfg = load_config(args.config)
    spec = registry.build(cfg.get("target", args.config), **cfg.get("target_kwargs", {}))
    results = run_benchmark(args.config, methods=methods, seeds=seeds, config=cfg, spec=spec)
    path = save_results(results, args.config, path=args.out)
    print(f"saved {len(results)} runs -> {path}", flush=True)

    if args.eval:
        scores = {_key(m, s): {k: v for k, v in evaluate(r, spec, cfg).items()
                               if np.isscalar(v) or isinstance(v, (bool, str))}
                  for (m, s), r in results.items()}
        mpath = Path(path).with_name(Path(path).stem + "__metrics.json")
        mpath.write_text(json.dumps(scores, indent=2, default=float))
        print(f"metrics -> {mpath}", flush=True)
    return path


if __name__ == "__main__":
    main()
