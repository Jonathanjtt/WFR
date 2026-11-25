"""JAX IBW / NGD for the isotropic uniform-weight mixture family.

    ibw:  eps <- (1 - step)^2 eps,          m <- m - lr*N grad_m
    ngd:  eps <- eps / (1 + eps*step),      m <- m - lr*N eps_new grad_m

with ``step = (2 N lr / d) grad_eps``.  NGD freezes the iterate the first time any
``1 + eps*step`` would be non-positive.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from gmmvi.jaxvi import common


def _grads(means, eps, score_fn, noise):
    """Monte-Carlo gradients wrt (means, eps) -> (K, d), (K,); ``noise`` (B, d) is shared."""
    K, d = means.shape
    B = noise.shape[0]
    samples = common.igmm_sample_each(means, eps, noise)
    flat = samples.reshape(K * B, d)

    glv = common.igmm_score(flat, means, eps).reshape(K, B, d)
    glp = score_fn(flat).reshape(K, B, d)
    centered = samples - means[:, None, :]

    grad_means = (glv - glp).mean(axis=1) / K
    e_pi = (glp * centered).sum(-1).mean(-1)
    e_vg = (glv * centered).sum(-1).mean(-1)
    grad_eps = (e_vg - e_pi) / (2.0 * K * eps)
    return grad_means, grad_eps


@partial(jax.jit, static_argnums=(4, 5))
def _run(init_means, init_eps, noise_seq, lr_seq, method, score_fn):
    """Scan the optimiser; the trajectories are pre-update snapshots."""
    K, d = init_means.shape
    N = K

    def step(carry, inp):
        noise, lr = inp
        means, eps, alive = carry
        snap = (means, eps)

        grad_means, grad_eps = _grads(means, eps, score_fn, noise)
        step_e = (2.0 * N * lr / d) * grad_eps

        if method == "ibw":
            new_eps = (1.0 - step_e) ** 2 * eps
            new_means = means - lr * N * grad_means
            ok = jnp.array(True)
        else:
            denom = 1.0 + eps * step_e
            ok = jnp.all(denom > 0.0)
            safe_denom = jnp.where(denom > 0.0, denom, 1.0)
            new_eps = eps / safe_denom
            new_means = means - lr * N * (new_eps[:, None] * grad_means)

        do = alive & ok
        means = jnp.where(do, new_means, means)
        eps = jnp.where(do, new_eps, eps)
        return (means, eps, alive & ok), snap

    init = (init_means, init_eps, jnp.array(True))
    (fm, fe, _), (means_traj, eps_traj) = jax.lax.scan(step, init, (noise_seq, lr_seq))
    return means_traj, eps_traj, fm, fe


def lr_sqrt_schedule(n_iterations, burst_iters=20, late_scale=0.01):
    """VI-MIG ``lr_sqrt`` schedule: ``1 - sqrt(t/T)`` for ``burst_iters``, then scaled by ``late_scale``."""
    t = np.arange(n_iterations, dtype=float)
    base = 1.0 - np.sqrt(t / float(n_iterations))
    return jnp.asarray(np.where(t <= burst_iters, base, base * late_scale))


def fit(target, n_components=5, method="ibw", n_iterations=1000, learning_rate=0.1,
        scale=1.0, s=2.0, BG=10, BKL=1000, compute_kl=None, compute_kl_trace=True,
        scheduler=False, schedule_burst=20, schedule_late_scale=0.01, lr_seq=None,
        init_means=None, init_covs=None, seed=0):
    """Fit a mixture of isotropic Gaussians to ``target`` with IBW / NGD."""
    if compute_kl is None:
        compute_kl = max(1, n_iterations // 10)

    score_fn, logp_fn = common.resolve_target(target)
    d = target.dim
    key = jax.random.PRNGKey(seed)

    k_means, k_noise, k_ci, k_z = jax.random.split(key, 4)
    if init_means is None:
        init_means = s * jax.random.normal(k_means, (n_components, d))
    else:
        init_means = jnp.asarray(init_means, dtype=float)
        n_components = init_means.shape[0]
    if init_covs is None:
        init_eps = jnp.full((n_components,), float(scale))
    else:
        init_eps = jnp.asarray(init_covs, dtype=float)[:, 0, 0]

    noise_seq = jax.random.normal(k_noise, (n_iterations, BG, d))

    if lr_seq is not None:
        lr_seq = jnp.asarray(lr_seq, dtype=float)
    elif scheduler:
        lr_seq = lr_sqrt_schedule(n_iterations, schedule_burst, schedule_late_scale)
    else:
        lr_seq = jnp.full((n_iterations,), float(learning_rate))
    means_traj, eps_traj, fm, fe = _run(init_means, init_eps, noise_seq, lr_seq, method, score_fn)

    logged = common.logged_iterations(n_iterations, compute_kl)
    logged_idx = jnp.asarray(logged)
    kls = None
    if compute_kl_trace:
        ci = jax.random.randint(k_ci, (BKL,), 0, n_components)
        z = jax.random.normal(k_z, (BKL, d))
        kls = np.asarray(common.isotropic_kls(logp_fn, means_traj, eps_traj, logged_idx, ci, z))

    fm = np.asarray(fm)
    fe = np.asarray(fe)

    mt = np.asarray(means_traj)
    et = np.asarray(eps_traj)
    lg = np.asarray(logged)
    return {
        "means": fm,
        "epsilons": fe,
        "covariances": fe[:, None, None] * np.eye(d),
        "weights": np.full(n_components, 1.0 / n_components),
        "kls": kls,
        "elbo": None if kls is None else -kls,
        "elbo_iters": lg,
        "means_traj": mt,
        "eps_traj": et,
        "means_log": mt[lg],
        "covs_log": et[lg][:, :, None, None] * np.eye(d),
        "weights_log": np.broadcast_to(1.0 / n_components, (lg.size, n_components)).copy(),
    }
