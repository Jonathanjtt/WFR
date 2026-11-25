"""Post-hoc metrics on a fitted mixture ``(means (K, d), covs (K, d, d), weights (K,))``.

A collapsed / non-finite / non-PSD component is dropped rather than raising; grid densities are renormalised on the grid.
"""

from __future__ import annotations

import warnings

import numpy as np

__all__ = [
    "gmm_pdf",
    "mixture_log_prob",
    "sample_gmm",
    "subsample",
    "grid_density",
    "total_variation",
    "emc",
    "emc_from_samples",
    "mode_assignment",
    "mode_coverage",
    "ess_fraction",
    "ess_from_samples",
    "weight_entropy",
    "mixture_marginal_moments",
    "estimate_ot_cost_scale",
    "resolve_ot_epsilon",
    "entropic_ot_cost",
    "transport_cost",
    "sinkhorn_divergence",
    "entropic_w2",
    "entropic_w2_floor",
    "pilot_relative_epsilon",
    "median_bandwidth",
    "resolve_mmd_bandwidth",
    "mmd_from_samples",
    "mmd",
    "mmd_floor",
    "reverse_kl_trace",
    "trace",
]


def gmm_pdf(pts, means, covs, weights):
    """Mixture density q(x) on ``pts`` (N, d) -> (N,); degenerate components contribute 0."""
    pts = np.asarray(pts, float)
    means = np.asarray(means, float)
    covs = np.asarray(covs, float)
    weights = np.asarray(weights, float)
    d = pts.shape[1]
    norm = (2.0 * np.pi) ** (0.5 * d)
    out = np.zeros(len(pts))
    for k in range(len(weights)):
        det = np.linalg.det(covs[k])
        if not np.isfinite(det) or det <= 0:
            continue
        try:
            Ci = np.linalg.inv(covs[k])
        except np.linalg.LinAlgError:
            continue
        diff = pts - means[k]
        out += weights[k] * np.exp(-0.5 * np.einsum("bi,ij,bj->b", diff, Ci, diff)) / (
            norm * np.sqrt(det)
        )
    return out


def mixture_log_prob(X, means, covs, weights):
    """log q(x) on ``X`` (N, d) -> (N,); degenerate components contribute -inf."""
    X = np.asarray(X, float)
    means = np.asarray(means, float)
    covs = np.asarray(covs, float)
    w = np.clip(np.asarray(weights, float), 0.0, None)
    w = w / w.sum()
    d = X.shape[1]
    comp = np.full((X.shape[0], len(w)), -np.inf)
    for k in range(len(w)):
        if w[k] <= 0 or not np.isfinite(covs[k]).all() or not np.isfinite(means[k]).all():
            continue
        C = 0.5 * (covs[k] + covs[k].T)
        sign, logdet = np.linalg.slogdet(C)
        if sign <= 0 or not np.isfinite(logdet):
            continue
        try:
            Ci = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            continue
        diff = X - means[k]
        maha = np.einsum("bi,ij,bj->b", diff, Ci, diff)
        comp[:, k] = np.log(w[k]) - 0.5 * (d * np.log(2.0 * np.pi) + logdet + maha)
    mx = comp.max(axis=1)
    out = np.full(X.shape[0], -np.inf)
    ok = np.isfinite(mx)
    out[ok] = mx[ok] + np.log(np.exp(comp[ok] - mx[ok, None]).sum(axis=1))
    return out


def sample_gmm(means, covs, weights, n, seed):
    """Ancestral sampling from the mixture -> (n, d); the RNG consumption order is part of the metric."""
    means = np.asarray(means, float)
    covs = np.asarray(covs, float)
    rng = np.random.RandomState(seed)
    w = np.clip(np.asarray(weights, float), 0, None)
    w = w / w.sum()
    comp = rng.choice(len(w), size=n, p=w)
    out = np.empty((n, means.shape[1]))
    for k in range(len(w)):
        idx = np.where(comp == k)[0]
        if idx.size == 0:
            continue
        C = np.asarray(covs[k], float)
        C = 0.5 * (C + C.T) + 1e-12 * np.eye(C.shape[0])
        out[idx] = rng.multivariate_normal(np.asarray(means[k], float), C, size=idx.size)
    return out


def subsample(draws, n, seed):
    """``n`` rows of an MCMC chain -> (n, d): without replacement when it is long enough, else with replacement and a warning."""
    X = np.asarray(draws, float)
    n, N = int(n), len(X)
    if N == 0:
        return np.zeros((0, X.shape[1] if X.ndim == 2 else 0))
    rng = np.random.RandomState(seed)
    if n > N:
        warnings.warn(
            f"chain has {N} draws but the metric asked for {n}; sampling WITH replacement, "
            f"so the cloud carries only {N} distinct points. Lengthen the chain "
            "(n_iterations) or lower the metric's sample budget to remove the resampling.",
            RuntimeWarning, stacklevel=2)
        return X[rng.randint(0, N, size=n)]
    return X[rng.choice(N, size=n, replace=False)]


def grid_density(pts, means, covs, weights):
    """Mixture density on ``pts`` renormalised to sum to 1 over the grid; nan if degenerate."""
    f = gmm_pdf(pts, means, covs, weights)
    s = f.sum()
    if not np.isfinite(f).all() or not np.isfinite(s) or s <= 0:
        return np.full_like(f, np.nan)
    return f / s


def total_variation(pts, ref_density, means, covs, weights):
    """TV(q, pi) = 1/2 sum |q - pi| on ``pts``; ``ref_density`` is normalised on the same grid.  nan if degenerate."""
    f = grid_density(pts, means, covs, weights)
    if not np.isfinite(f).all():
        return np.nan
    return 0.5 * np.abs(f - np.asarray(ref_density, float)).sum()


def mode_assignment(X, mode_means, mode_precisions=None):
    """Nearest target mode per sample -> (N,) ints (Mahalanobis when ``mode_precisions`` is given)."""
    X = np.asarray(X, float)
    M = np.asarray(mode_means, float)
    diff = X[:, None, :] - M[None]
    if mode_precisions is None:
        return (diff ** 2).sum(-1).argmin(1)
    P = np.asarray(mode_precisions, float)
    return np.einsum("nmi,mij,nmj->nm", diff, P, diff).argmin(1)


def emc_from_samples(X, mode_means, mode_precisions=None):
    """(EMC = base-M entropy of the nearest-mode histogram, #modes covered) of drawn samples."""
    n_modes = len(np.asarray(mode_means))
    assign = mode_assignment(X, mode_means, mode_precisions)
    counts = np.bincount(assign, minlength=n_modes).astype(float)
    p = counts / counts.sum()
    nz = p > 0
    value = -(p[nz] * (np.log(p[nz]) / np.log(n_modes))).sum()
    return value, int((counts > 0).sum())


def emc(means, covs, weights, mode_means, n, seed, mode_precisions=None):
    """(EMC, #modes covered) of one fitted mixture for one sampling seed."""
    X = sample_gmm(means, covs, weights, n, seed)
    return emc_from_samples(X, mode_means, mode_precisions)


def mode_coverage(means, mode_means, mode_precisions=None, r=3.0):
    """#target modes with a component mean within (Mahalanobis) radius ``r``."""
    diff = np.asarray(means, float)[:, None, :] - np.asarray(mode_means, float)[None]
    if mode_precisions is None:
        dm = np.sqrt((diff ** 2).sum(-1))
    else:
        P = np.asarray(mode_precisions, float)
        dm = np.sqrt(np.einsum("kmi,mij,kmj->km", diff, P, diff))
    return int((dm.min(axis=0) <= r).sum())


def ess_from_samples(X, means, covs, weights, log_prob):
    """Normalised self-normalised-IS ESS in (0, 1] of drawn samples; nan if no weight is finite."""
    X = np.asarray(X, float)
    logw = np.asarray(log_prob(X), float).reshape(-1) - mixture_log_prob(X, means, covs, weights)
    ok = np.isfinite(logw)
    if not ok.any():
        return np.nan
    w = np.zeros(len(X))
    w[ok] = np.exp(logw[ok] - logw[ok].max())
    denom = len(X) * (w ** 2).sum()
    return float(w.sum() ** 2 / denom) if denom > 0 else np.nan


def ess_fraction(means, covs, weights, log_prob, n, seed):
    """Normalised ESS of one fitted mixture for one sampling seed."""
    X = sample_gmm(means, covs, weights, n, seed)
    return ess_from_samples(X, means, covs, weights, log_prob)


def weight_entropy(weights):
    """Base-K entropy of the mixture weights in [0, 1]: 1 = uniform, 0 = one live component."""
    w = np.asarray(weights, float)
    return float(-(w * np.log(w + 1e-12)).sum() / np.log(len(w)))


def mixture_marginal_moments(means, covs, weights, i):
    """(mean, std) of the i-th coordinate under the mixture, from the exact mixture law."""
    w = np.asarray(weights, float)
    mu = np.asarray(means, float)[:, i]
    v = np.asarray(covs, float)[:, i, i]
    mean = (w * mu).sum()
    var = (w * (v + mu ** 2)).sum() - mean ** 2
    return mean, np.sqrt(var)


_SCALE_DEFINITION = "median_target_pairwise_sqeuclidean"


def estimate_ot_cost_scale(target_samples, max_samples=2000, seed=0, normalize_by_dim=False):
    """Median pairwise squared Euclidean distance among (at most ``max_samples``) target samples."""
    from scipy.spatial.distance import pdist

    Y = np.asarray(target_samples, dtype=np.float64)
    if Y.ndim != 2:
        raise ValueError(f"target_samples must be (n_samples, dim), got shape {Y.shape}")
    n, dim = Y.shape
    if n < 2:
        raise ValueError(f"need at least 2 target samples to estimate a cost scale, got {n}")
    if n > int(max_samples):
        idx = np.random.RandomState(int(seed)).choice(n, size=int(max_samples), replace=False)
        Y = Y[np.sort(idx)]
    d2 = pdist(Y, metric="sqeuclidean")
    if normalize_by_dim:
        d2 = d2 / float(dim)
    d2 = d2[np.isfinite(d2) & (d2 > 0)]
    if d2.size == 0:
        raise ValueError(
            "no positive finite pairwise squared distance among the target samples; "
            "the cost scale is undefined (are all samples identical?)")
    return float(np.median(d2))


def resolve_ot_epsilon(target_samples, epsilon=None, relative_epsilon=0.01,
                       max_scale_samples=2000, scale_seed=0, normalize_by_dim=False):
    """Entropic-OT epsilon with its provenance: ``epsilon`` as given, else ``relative_epsilon * cost_scale``."""
    if epsilon is not None:
        eps = float(epsilon)
        if not (np.isfinite(eps) and eps > 0):
            raise ValueError(f"absolute epsilon must be finite and > 0, got {epsilon!r}")
        return {"epsilon": eps, "epsilon_mode": "absolute", "relative_epsilon": None,
                "cost_scale": None, "scale_definition": None, "scale_sample_size": None,
                "scale_seed": None, "normalize_by_dim": bool(normalize_by_dim),
                "ground_cost": "sqeuclidean"}

    rel = float(relative_epsilon)
    if not (np.isfinite(rel) and rel > 0):
        raise ValueError(f"relative_epsilon must be finite and > 0, got {relative_epsilon!r}")
    Y = np.asarray(target_samples, dtype=np.float64)
    scale = estimate_ot_cost_scale(Y, max_samples=max_scale_samples, seed=scale_seed,
                                   normalize_by_dim=normalize_by_dim)
    return {"epsilon": rel * scale, "epsilon_mode": "relative", "relative_epsilon": rel,
            "cost_scale": scale, "scale_definition": _SCALE_DEFINITION,
            "scale_sample_size": int(min(len(Y), int(max_scale_samples))),
            "scale_seed": int(scale_seed), "normalize_by_dim": bool(normalize_by_dim),
            "ground_cost": "sqeuclidean"}


def _solve_entropic(X, Y, epsilon, a=None, b=None, **solve_kwargs):
    """Run ott's Sinkhorn on PointCloud(X, Y) and return its output object."""
    import jax.numpy as jnp
    from ott.geometry import pointcloud
    from ott.problems.linear import linear_problem
    from ott.solvers.linear import sinkhorn

    eps = float(epsilon)
    geom = pointcloud.PointCloud(jnp.asarray(X, float), jnp.asarray(Y, float), epsilon=eps)
    prob = linear_problem.LinearProblem(geom, a=None if a is None else jnp.asarray(a, float),
                                        b=None if b is None else jnp.asarray(b, float))
    return sinkhorn.Sinkhorn(**solve_kwargs)(prob)


def entropic_ot_cost(X, Y, epsilon, a=None, b=None, return_output=False, **solve_kwargs):
    """``reg_ot_cost`` = <P*, C> + eps * KL(P* || a (x) b) between point clouds X and Y."""
    out = _solve_entropic(X, Y, epsilon, a=a, b=b, **solve_kwargs)
    cost = float(out.reg_ot_cost)
    return (cost, out) if return_output else cost


def transport_cost(X, Y, epsilon, a=None, b=None, return_output=False, **solve_kwargs):
    """Bare <P*, C> under the regularised coupling (a biased W2^2 estimate)."""
    out = _solve_entropic(X, Y, epsilon, a=a, b=b, **solve_kwargs)
    cost = float(out.primal_cost)
    return (cost, out) if return_output else cost


def sinkhorn_divergence(X, Y, epsilon, a=None, b=None, return_output=False, **solve_kwargs):
    """Debiased Sinkhorn divergence S_eps -- the only variant here that is 0 when X ~ Y."""
    import jax.numpy as jnp
    from ott.geometry import pointcloud
    from ott.tools import sinkhorn_divergence as _sd

    eps = float(epsilon)
    out = _sd.sinkhorn_divergence(
        pointcloud.PointCloud, jnp.asarray(X, float), jnp.asarray(Y, float), epsilon=eps,
        a=None if a is None else jnp.asarray(a, float),
        b=None if b is None else jnp.asarray(b, float),
        solve_kwargs=solve_kwargs)
    div = float(out[0] if isinstance(out, tuple) else out.divergence)
    return (div, out) if return_output else div


def entropic_w2(means, covs, weights, target_sample_fn, epsilon, n=2000, seed=0,
                target_seed=0, q_samples=None, **solve_kwargs):
    """``(sqrt(reg_ot_cost), converged)`` between n samples of q (or ``q_samples``) and n exact samples of pi."""
    if q_samples is None:
        means = np.asarray(means, float)
        covs = np.asarray(covs, float)
        if not (np.isfinite(means).all() and np.isfinite(covs).all()
                and np.isfinite(np.asarray(weights, float)).all()):
            return np.nan, False
        Xq = sample_gmm(means, covs, weights, n, seed)
    else:
        Xq = np.asarray(q_samples, float)
        if not np.isfinite(Xq).all():
            return np.nan, False
    Xp = np.asarray(target_sample_fn(n, target_seed), float)
    cost, out = entropic_ot_cost(Xp, Xq, epsilon, return_output=True, **solve_kwargs)
    converged = bool(out.converged)
    if not converged:
        warnings.warn(
            f"Sinkhorn did not converge at epsilon={float(epsilon):g} after "
            f"{int(out.n_iters)} iterations; the value still drifts with the iteration "
            "budget. Reported with converged=False -- epsilon is NOT adjusted per run. Use "
            "metrics.pilot_relative_epsilon to pick a relative_epsilon that converges for "
            "every method and seed, then rerun all of them at that one value.",
            RuntimeWarning, stacklevel=2)
    return float(np.sqrt(max(cost, 0.0))), converged


def entropic_w2_floor(target_sample_fn, epsilon, n=2000, seeds=(0, 1), **solve_kwargs):
    """:func:`entropic_w2` between two independent draws from pi: the floor a perfect fit scores."""
    Xa = np.asarray(target_sample_fn(n, seeds[0]), float)
    Xb = np.asarray(target_sample_fn(n, seeds[1]), float)
    cost = entropic_ot_cost(Xa, Xb, epsilon, **solve_kwargs)
    return float(np.sqrt(max(cost, 0.0)))


PILOT_RELATIVE_EPSILONS = (0.001, 0.003, 0.01, 0.03, 0.1)


def pilot_relative_epsilon(target_sample_fn, fits, candidates=PILOT_RELATIVE_EPSILONS,
                           n=2000, seed=0, target_seed=0, max_scale_samples=2000,
                           scale_seed=0, normalize_by_dim=False, **solve_kwargs):
    """Smallest ``relative_epsilon`` at which every ``(label, means, covs, weights)`` fit converges -> ``(chosen, report)``."""
    Xp = np.asarray(target_sample_fn(max(n, max_scale_samples), target_seed), float)
    report, chosen = {}, None
    for rel in candidates:
        info = resolve_ot_epsilon(Xp, epsilon=None, relative_epsilon=rel,
                                  max_scale_samples=max_scale_samples, scale_seed=scale_seed,
                                  normalize_by_dim=normalize_by_dim)
        failed = []
        for label, m, c, w in fits:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                _, ok = entropic_w2(m, c, w, target_sample_fn, info["epsilon"], n=n, seed=seed,
                                    target_seed=target_seed, **solve_kwargs)
            if not ok:
                failed.append(label)
        report[rel] = {"epsilon": info["epsilon"], "cost_scale": info["cost_scale"],
                       "failed": failed}
        if not failed and chosen is None:
            chosen = rel
    return chosen, report


def median_bandwidth(samples, max_points=2048, seed=99):
    """Median-heuristic sigma: median ||z_i - z_j||_2 over at most ``max_points`` of ``samples``, i < j."""
    from scipy.spatial.distance import pdist

    Z = np.asarray(samples, float)
    if len(Z) > int(max_points):
        idx = np.random.RandomState(int(seed)).choice(len(Z), size=int(max_points),
                                                      replace=False)
        Z = Z[np.sort(idx)]
    sigma = float(np.median(pdist(Z, metric="euclidean")))
    return sigma


def resolve_mmd_bandwidth(target_samples, bandwidth=None, max_points=2048, seed=99):
    """Kernel bandwidth with its provenance: ``bandwidth`` as given, else the median heuristic on the target samples."""
    if bandwidth is not None:
        sig = float(bandwidth)
        if not (np.isfinite(sig) and sig > 0):
            raise ValueError(f"absolute bandwidth must be finite and > 0, got {bandwidth!r}")
        return {"bandwidth": sig, "bandwidth_mode": "absolute", "max_points": None,
                "bandwidth_seed": None, "kernel": "gaussian_l2"}
    Y = np.asarray(target_samples, float)
    return {"bandwidth": median_bandwidth(Y, max_points=max_points, seed=seed),
            "bandwidth_mode": "median_target", "max_points": int(min(len(Y), int(max_points))),
            "bandwidth_seed": int(seed), "kernel": "gaussian_l2"}


def _gaussian_kernel(A, B, bandwidth):
    """k(a, b) = exp(-||a - b||^2 / (2 sigma^2)) -> (len(A), len(B))."""
    from scipy.spatial.distance import cdist

    d2 = cdist(np.asarray(A, float), np.asarray(B, float), metric="sqeuclidean")
    return np.exp(-0.5 * d2 / (float(bandwidth) ** 2))


def mmd_from_samples(X, Y, bandwidth=None, biased=True):
    """MMD between drawn clouds X (n, d) and Y (m, d); ``bandwidth=None`` -> median heuristic on ``Y``; biased V-statistic by default."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    n, m = len(X), len(Y)
    sigma = median_bandwidth(Y) if bandwidth is None else float(bandwidth)
    if not (np.isfinite(sigma) and sigma > 0):
        raise ValueError(f"bandwidth must be finite and > 0, got {bandwidth!r}")
    Kxx, Kyy = _gaussian_kernel(X, X, sigma), _gaussian_kernel(Y, Y, sigma)
    kxy = _gaussian_kernel(X, Y, sigma).mean()
    if biased:
        mmd2 = Kxx.mean() + Kyy.mean() - 2.0 * kxy
    else:
        mmd2 = ((Kxx.sum() - n) / (n * (n - 1)) + (Kyy.sum() - m) / (m * (m - 1)) - 2.0 * kxy)
    return float(np.sqrt(max(mmd2, 0.0)))


def mmd(means, covs, weights, target_sample_fn, bandwidth=None, n=2000, seed=0, target_seed=0,
        biased=True, q_samples=None):
    """MMD between n samples of the mixture (or ``q_samples``) and n exact samples of pi; nan for a non-finite fit."""
    if q_samples is None:
        means = np.asarray(means, float)
        covs = np.asarray(covs, float)
        if not (np.isfinite(means).all() and np.isfinite(covs).all()
                and np.isfinite(np.asarray(weights, float)).all()):
            return np.nan
        Xq = sample_gmm(means, covs, weights, n, seed)
    else:
        Xq = np.asarray(q_samples, float)
        if not np.isfinite(Xq).all():
            return np.nan
    Xp = np.asarray(target_sample_fn(n, target_seed), float)
    return mmd_from_samples(Xq, Xp, bandwidth=bandwidth, biased=biased)


def mmd_floor(target_sample_fn, bandwidth=None, n=2000, seeds=(0, 1), biased=True):
    """:func:`mmd` between two independent draws from pi: the floor a perfect fit scores."""
    Xa = np.asarray(target_sample_fn(n, seeds[0]), float)
    Xb = np.asarray(target_sample_fn(n, seeds[1]), float)
    return mmd_from_samples(Xa, Xb, bandwidth=bandwidth, biased=biased)


def reverse_kl_trace(res, target, key, z):
    """Reverse KL at every logged iteration on one shared ``(key, z)`` -> (L,)."""
    import jax.numpy as jnp

    from gmmvi.jaxvi import common

    _, logp = common.resolve_target(target)
    means = jnp.asarray(res["means_log"])
    chols = jnp.linalg.cholesky(jnp.asarray(res["covs_log"]))
    logw = jnp.log(jnp.asarray(res["weights_log"]))
    return np.asarray(
        common.full_kls(logp, means, chols, logw, jnp.arange(means.shape[0]), key, z)
    )


def trace(res, metric_fn, seeds=None, skip_nonfinite=True):
    """Map ``metric_fn(means, covs, weights[, seed]) -> float`` over the logged trajectory -> (L,).

    ``skip_nonfinite`` turns a snapshot with any non-finite entry into nan instead of scoring the survivors.
    """
    ml, cl, wl = res["means_log"], res["covs_log"], res["weights_log"]
    out = np.empty(len(ml))
    for i in range(len(ml)):
        if skip_nonfinite and not (
            np.isfinite(ml[i]).all() and np.isfinite(cl[i]).all() and np.isfinite(wl[i]).all()
        ):
            out[i] = np.nan
            continue
        if seeds is None:
            out[i] = metric_fn(ml[i], cl[i], wl[i])
        else:
            out[i] = np.mean([metric_fn(ml[i], cl[i], wl[i], s) for s in seeds])
    return out
