"""JAX GMBBVI: adaptive exponential integration on a full-covariance, free-weight mixture.

Gradient-free: the target enters only through ``log pi``.  Each component takes an
exponential-integrator covariance step ``C_k <- L_k exp(-dt E_k) L_k^T`` with the adaptive
``dt = min(dt_max * eta(n), beta / max_k ||E_k||)``.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from gmmvi.jaxvi import common


def _eta_schedule(n_iterations, eta_min):
    """Cosine-decay step scheduler -> (n_iterations,)."""
    N = n_iterations
    out = np.ones(N)
    for n in range(N):
        if n <= N / 2:
            out[n] = 1.0
        else:
            out[n] = eta_min + 0.5 * (1.0 - eta_min) * (1.0 + np.cos(2.0 * np.pi * (n / N - 0.5)))
    return out


@partial(jax.jit, static_argnums=(5, 6, 7, 8))
def _run(init_means, init_covs, init_logw, noise_seq, eta_seq,
         dt_max, beta, fix_weights, logp_fn):
    """Scan the GMBBVI integrator; the trajectories are pre-update snapshots."""
    K, d = init_means.shape

    def step(carry, inp):
        means, covs, logw = carry
        z, eta = inp
        Ls = jnp.linalg.cholesky(covs)
        snap = (means, Ls, logw)

        theta = means[:, None, :] + jnp.einsum("kij,kbj->kbi", Ls, z)
        J = z.shape[1]
        flat = theta.reshape(K * J, d)
        log_rho = common.gmm_log_prob(flat, means, Ls, logw)
        f = (log_rho - logp_fn(flat)).reshape(K, J)
        fbar = f.mean(axis=1)
        g = f - fbar[:, None]

        b = (z * g[:, :, None]).mean(axis=1)
        E = jnp.einsum("kbi,kbj,kb->kij", z, z, g) / J
        E = common.sym(E)
        eigval, eigvec = jnp.linalg.eigh(E)

        normE = jnp.abs(eigval).max()
        dt = jnp.minimum(dt_max * eta, beta / (normE + 1e-12))

        new_means = means - dt * jnp.einsum("kij,kj->ki", Ls, b)
        expE = jnp.einsum("kij,kj,klj->kil", eigvec, jnp.exp(-dt * eigval), eigvec)
        new_covs = common.sym(jnp.einsum("kij,kjl,kml->kim", Ls, expE, Ls))

        if fix_weights:
            new_logw = logw
        else:
            u = logw - dt * (fbar - jnp.exp(logw) @ fbar)
            new_logw = u - jax.scipy.special.logsumexp(u)
        return (new_means, new_covs, new_logw), (snap[0], snap[1], snap[2], dt)

    init = (init_means, init_covs, init_logw)
    (fm, fc, flw), (means_traj, chol_traj, logw_traj, dt_traj) = \
        jax.lax.scan(step, init, (noise_seq, eta_seq))
    return means_traj, chol_traj, logw_traj, fm, fc, flw, dt_traj


def fit(target, n_components=5, n_iterations=500, dt_max=0.1, beta=0.9, eta_min=0.1,
        J=None, BKL=1000, compute_kl=None, compute_kl_trace=True, scale=1.0, s=2.0,
        init_means=None, init_covs=None, seed=0, fix_weights=False):
    """Fit a full-covariance GMM to ``target`` with GMBBVI (``J`` samples per component)."""
    if compute_kl is None:
        compute_kl = max(1, n_iterations // 20)
    d = target.dim
    J = int(J) if J is not None else 4 * d
    _, logp_fn = common.resolve_target(target)
    key = jax.random.PRNGKey(seed)
    k_means, k_noise, k_ci, k_z = jax.random.split(key, 4)

    if init_means is None:
        init_means = s * jax.random.normal(k_means, (n_components, d))
    else:
        init_means = jnp.asarray(init_means, dtype=float)
        n_components = init_means.shape[0]
    if init_covs is None:
        init_covs = scale * jnp.broadcast_to(jnp.eye(d), (n_components, d, d))
    else:
        init_covs = jnp.asarray(init_covs, dtype=float)
    init_logw = jnp.full((n_components,), -jnp.log(n_components))

    eta_seq = jnp.asarray(_eta_schedule(n_iterations, eta_min))
    noise_seq = jax.random.normal(k_noise, (n_iterations, n_components, J, d))

    means_traj, chol_traj, logw_traj, fm, fc, flw, dt_traj = _run(
        init_means, init_covs, init_logw, noise_seq, eta_seq,
        float(dt_max), float(beta), bool(fix_weights), logp_fn)

    logged = common.logged_iterations(n_iterations, compute_kl)
    logged_idx = jnp.asarray(logged)
    kls = None
    if compute_kl_trace:
        z = jax.random.normal(k_z, (BKL, d))
        kls = np.asarray(common.full_kls(logp_fn, means_traj, chol_traj, logw_traj,
                                         logged_idx, k_ci, z))

    chol_log = chol_traj[logged_idx]
    covs_log = np.asarray(jnp.einsum("lkij,lkmj->lkim", chol_log, chol_log))
    return {
        "means": np.asarray(fm),
        "covariances": np.asarray(fc),
        "weights": np.asarray(jnp.exp(flw)),
        "epsilons": None,
        "kls": kls,
        "elbo": None if kls is None else -kls,
        "elbo_iters": np.asarray(logged),
        "dts": np.asarray(dt_traj),
        "means_log": np.asarray(means_traj[logged_idx]),
        "covs_log": covs_log,
        "weights_log": np.asarray(jnp.exp(logw_traj[logged_idx])),
    }
