"""JAX SEMTRUX (fixed K): Stein natural gradient, KL trust-region component update,
improvement-based per-component stepsize and a direct natural-gradient weight update
(Arenz et al., arXiv:2209.11533), run as one ``jax.jit`` + ``lax.scan``.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from gmmvi.jaxvi import common

_LOG2PI = float(np.log(2.0 * np.pi))
_NEG = -1e30
_BRACKET_ITERS = 64


def _chol_caches(chols, d):
    """``(L_k^{-1}, Sigma_k^{-1}, 0.5 log|Sigma_k|)`` from the Cholesky factors."""
    inv_chol = jax.vmap(lambda L: jax.scipy.linalg.solve_triangular(L, jnp.eye(d), lower=True))(chols)
    precision = jnp.einsum("kij,kil->kjl", inv_chol, inv_chol)
    half_logdet = jnp.sum(jnp.log(jnp.diagonal(chols, axis1=1, axis2=2)), axis=1)
    return inv_chol, precision, half_logdet


def _component_log_densities(X, means, inv_chol, half_logdet, d):
    """log N(x | m_k, C_k) for X (M, d) -> (K, M), weights excluded."""
    diff = X[None, :, :] - means[:, None, :]
    y = jnp.einsum("kij,kmj->kmi", inv_chol, diff)
    mahala = -0.5 * jnp.sum(y * y, axis=2)
    log_const = -half_logdet - 0.5 * d * _LOG2PI
    return mahala + log_const[:, None]


def _stein_ng(samples, means, precision, cld, background, logw, target_grads, self_normalized):
    """Stein estimate of ``(H_neg, g_neg) = (-E[grad^2 R], -E[grad R])`` per component, R = log p - log q."""
    K, M = cld.shape
    weighted = logw[:, None] + cld
    logq = jax.scipy.special.logsumexp(weighted, axis=0)
    resp = jnp.exp(weighted - logq[None, :])
    diff = samples[None, :, :] - means[:, None, :]
    prec_diff = jnp.einsum("kde,kme->kmd", precision, diff)
    grad_logq = -jnp.einsum("km,kmd->md", resp, prec_diff)
    log_ratio_grads = target_grads - grad_logq

    lw = cld - background[None, :]
    if self_normalized:
        iw = jax.nn.softmax(lw, axis=1)
    else:
        shift = jnp.max(lw, axis=1, keepdims=True)
        iw = jnp.exp(lw - shift) * (jnp.exp(shift) / M)
    wg = iw[:, :, None] * log_ratio_grads[None, :, :]
    exp_grad = wg.sum(axis=1)
    exp_hess = jnp.einsum("kmd,kme->kde", wg, prec_diff)
    exp_hess = common.sym(exp_hess)
    return -exp_hess, -exp_grad


def _kl_step_single(eta, old_prec, old_lin, old_inv_chol, reward_lin, reward_quad,
                    kl_const, old_mean, d):
    """KL(new||old) and ``(new_mean, L_new^{-1})`` for multiplier ``eta``; a non-PD precision gives kl = inf."""
    new_prec = old_prec + reward_quad / eta
    new_lin = old_lin + reward_lin / eta
    chol = jnp.linalg.cholesky(new_prec)
    bad = jnp.logical_or(jnp.logical_not(jnp.all(jnp.isfinite(chol))),
                         jnp.any(jnp.diagonal(chol) <= 0.0))
    chol_safe = jnp.where(bad, jnp.eye(d), chol)
    inv_chol_prec = jax.scipy.linalg.solve_triangular(chol_safe, jnp.eye(d), lower=True)
    new_mean = inv_chol_prec.T @ (inv_chol_prec @ new_lin)
    new_logdet_cov = -2.0 * jnp.sum(jnp.log(jnp.diagonal(chol_safe)))
    trace_term = jnp.sum((inv_chol_prec @ old_inv_chol.T) ** 2)
    diff = old_mean - new_mean
    mahala = jnp.sum((old_inv_chol @ diff) ** 2)
    kl = 0.5 * (kl_const - new_logdet_cov + trace_term + mahala)
    kl = jnp.where(bad, jnp.inf, kl)
    return kl, new_mean, inv_chol_prec


def _bracket_single(kl_bound, last_eta, old_prec, old_lin, old_inv_chol, reward_lin, reward_quad,
                    kl_const, old_mean, d):
    """Bisection in log(eta) for the largest step with KL(new||old) <= kl_bound -> (lower, upper)."""
    def kl_of(eta):
        return _kl_step_single(eta, old_prec, old_lin, old_inv_chol, reward_lin, reward_quad,
                               kl_const, old_mean, d)[0]
    lower0 = jnp.where(last_eta < 0, -20.0, jnp.maximum(0.0, jnp.log(jnp.abs(last_eta)) - 3.0))
    upper0 = jnp.where(last_eta < 0, 80.0, jnp.log(jnp.abs(last_eta)) + 3.0)
    log_eta0 = 0.5 * (lower0 + upper0)

    def body(_, c):
        log_eta, lower, upper, us, done = c
        width = jnp.minimum(jnp.exp(upper) - jnp.exp(log_eta), jnp.exp(log_eta) - jnp.exp(lower))
        width_break = width < 0.1
        kl = kl_of(jnp.exp(log_eta))
        kl_match = jnp.abs(kl_bound - kl) < 0.1 * kl_bound
        below = kl_bound > kl
        nu = jnp.where(kl_match, log_eta, jnp.where(below, log_eta, upper))
        nl = jnp.where(kl_match, log_eta, jnp.where(below, lower, log_eta))
        nus = jnp.logical_or(us, jnp.logical_and(below, jnp.logical_not(kl_match)))
        nle = 0.5 * (nu + nl)
        active = jnp.logical_and(jnp.logical_not(done), jnp.logical_not(width_break))
        return (jnp.where(active, nle, log_eta),
                jnp.where(active, nl, lower),
                jnp.where(active, nu, upper),
                jnp.where(active, nus, us),
                jnp.logical_or(done, jnp.logical_or(width_break, jnp.logical_and(active, kl_match))))

    log_eta, lower, upper, us, done = jax.lax.fori_loop(
        0, _BRACKET_ITERS, body, (log_eta0, lower0, upper0, jnp.array(False), jnp.array(False)))
    lower = jnp.where(us, upper, lower)
    return jnp.exp(lower), jnp.exp(upper)


def _trust_region_update(means, chols, inv_chol, precision, H_neg, g_neg, stepsizes,
                         last_log_etas, temperature, d):
    """KL-trust-region component update; rejected updates keep the old component."""
    old_logdet = 2.0 * jnp.sum(jnp.log(jnp.diagonal(chols, axis1=1, axis2=2)), axis=1)
    kl_const = old_logdet - d
    old_lin = jnp.einsum("kij,kj->ki", precision, means)
    reward_quad = H_neg
    reward_lin = jnp.einsum("kij,kj->ki", reward_quad, means) - g_neg

    nl, nu = jax.vmap(partial(_bracket_single, d=d))(
        stepsizes, last_log_etas, precision, old_lin, inv_chol, reward_lin, reward_quad,
        kl_const, means)
    eta = jnp.maximum(nl, temperature)
    accept_bracket = (nl == nu)

    kl, new_mean, inv_chol_prec = jax.vmap(partial(_kl_step_single, d=d))(
        eta, precision, old_lin, inv_chol, reward_lin, reward_quad, kl_const, means)
    new_cov = jnp.einsum("kij,kil->kjl", inv_chol_prec, inv_chol_prec)
    new_chol = jnp.linalg.cholesky(new_cov)
    chol_ok = jnp.logical_and(jnp.all(jnp.isfinite(new_chol.reshape(new_chol.shape[0], -1)), axis=1),
                              jnp.all(jnp.diagonal(new_chol, axis1=1, axis2=2) > 0.0, axis=1))
    success = accept_bracket & jnp.isfinite(kl) & chol_ok

    new_means = jnp.where(success[:, None], new_mean, means)
    new_chols = jnp.where(success[:, None, None], new_chol, chols)
    new_etas = jnp.where(success, eta, -1.0)
    return new_means, new_chols, new_etas


def _weight_update(samples, cld_new, background, logw, target_lnpdfs, temperature,
                   weight_stepsize, self_normalized):
    """Direct natural-gradient weight update -> (new_logw, component_rewards)."""
    weighted = logw[:, None] + cld_new
    logq = jax.scipy.special.logsumexp(weighted, axis=0)
    log_ratios = target_lnpdfs - temperature * logq
    lw = cld_new - background[None, :]
    if self_normalized:
        iw = jax.nn.softmax(lw, axis=1)
        expected_log_ratios = iw @ log_ratios
    else:
        shift = jnp.max(lw, axis=1, keepdims=True)
        w = jnp.exp(lw - shift)
        expected_log_ratios = (jnp.exp(shift[:, 0]) / lw.shape[1]) * (w @ log_ratios)

    component_rewards = temperature * logw + expected_log_ratios
    unnorm = logw + (weight_stepsize / temperature) * expected_log_ratios
    nlw = unnorm - jax.scipy.special.logsumexp(unnorm)
    nlw = jnp.maximum(nlw, -69.07)
    nlw = nlw - jax.scipy.special.logsumexp(nlw)
    return nlw, component_rewards


@partial(jax.jit, static_argnums=tuple(range(4, 16)))
def _run(init_means, init_chols, init_logw, noise_seq, J, d, temperature, self_normalized,
         init_stepsize, min_stepsize, max_stepsize, inc_factor, dec_factor, weight_stepsize,
         score_fn, logp_fn):
    """Scan the fixed-K SEMTRUX integrator; the trajectories are pre-update snapshots."""
    K = init_means.shape[0]
    log_invK = -jnp.log(K)

    def step(carry, z):
        means, chols, logw, stepsizes, last_etas, rew_prev, rew_curr = carry
        inv_chol, precision, half_logdet = _chol_caches(chols, d)

        improved = rew_curr > rew_prev
        stepsizes = jnp.where(improved,
                              jnp.minimum(inc_factor * stepsizes, max_stepsize),
                              jnp.maximum(dec_factor * stepsizes, min_stepsize))

        samples = (means[:, None, :] + jnp.einsum("kij,kbj->kbi", chols, z)).reshape(K * J, d)
        target_lnpdfs = logp_fn(samples).reshape(K * J)
        target_grads = score_fn(samples).reshape(K * J, d)

        cld = _component_log_densities(samples, means, inv_chol, half_logdet, d)
        background = jax.scipy.special.logsumexp(log_invK + cld, axis=0)

        H_neg, g_neg = _stein_ng(samples, means, precision, cld, background, logw,
                                 target_grads, self_normalized)
        new_means, new_chols, new_etas = _trust_region_update(
            means, chols, inv_chol, precision, H_neg, g_neg, stepsizes, last_etas, temperature, d)

        inv_chol_n, _, half_logdet_n = _chol_caches(new_chols, d)
        cld_new = _component_log_densities(samples, new_means, inv_chol_n, half_logdet_n, d)
        new_logw, comp_rewards = _weight_update(samples, cld_new, background, logw, target_lnpdfs,
                                                temperature, weight_stepsize, self_normalized)

        snap = (means, chols, logw)
        carry2 = (new_means, new_chols, new_logw, stepsizes, new_etas, rew_curr, comp_rewards)
        return carry2, snap

    init = (init_means, init_chols, init_logw,
            jnp.full((K,), init_stepsize), -jnp.ones((K,)),
            jnp.full((K,), _NEG), jnp.full((K,), _NEG))
    (fm, fc, flw, *_), (means_traj, chol_traj, logw_traj) = jax.lax.scan(step, init, noise_seq)
    return means_traj, chol_traj, logw_traj, fm, fc, flw


def fit(target, n_components=5, n_iterations=300, init_means=None, init_covs=None,
        init_weights=None, scale=1.0, s=2.0, temperature=1.0, J=None, self_normalized_iw=True,
        init_stepsize=1.0, min_stepsize=1e-3, max_stepsize=1.0, stepsize_inc_factor=1.15,
        stepsize_dec_factor=0.85, weight_stepsize=1.0, BKL=1000, compute_kl=None,
        compute_kl_trace=True, seed=0, **kwargs):
    """Fit a fixed-K full-covariance GMM to ``target`` with SEMTRUX."""
    if compute_kl is None:
        compute_kl = max(1, n_iterations // 10)

    score_fn, logp_fn = common.resolve_target(target)
    d = target.dim
    J = int(J) if J is not None else 4 * d
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
    if init_weights is None:
        init_logw = jnp.full((n_components,), -jnp.log(n_components))
    else:
        w = jnp.asarray(init_weights, dtype=float)
        init_logw = jnp.log(w) - jax.scipy.special.logsumexp(jnp.log(w))
    K = n_components
    init_chols = jnp.linalg.cholesky(init_covs)

    noise_seq = jax.random.normal(k_noise, (n_iterations, K, J, d))
    means_traj, chol_traj, logw_traj, fm, fc, flw = _run(
        init_means, init_chols, init_logw, noise_seq, J, d, float(temperature),
        bool(self_normalized_iw), float(init_stepsize), float(min_stepsize), float(max_stepsize),
        float(stepsize_inc_factor), float(stepsize_dec_factor), float(weight_stepsize),
        score_fn, logp_fn)

    logged = common.logged_iterations(n_iterations, compute_kl)
    logged_idx = jnp.asarray(logged)
    kls = None
    if compute_kl_trace:
        z = jax.random.normal(k_z, (BKL, d))
        kls = np.asarray(common.full_kls(logp_fn, means_traj, chol_traj, logw_traj,
                                         logged_idx, k_ci, z))

    fc_cov = np.asarray(jnp.einsum("kij,klj->kil", fc, fc))
    chol_log = chol_traj[logged_idx]
    covs_log = np.asarray(jnp.einsum("lkij,lkmj->lkim", chol_log, chol_log))
    return {
        "means": np.asarray(fm),
        "covariances": fc_cov,
        "weights": np.asarray(jnp.exp(flw)),
        "epsilons": None,
        "kls": kls,
        "elbo": None if kls is None else -kls,
        "elbo_iters": np.asarray(logged),
        "means_log": np.asarray(means_traj[logged_idx]),
        "covs_log": covs_log,
        "weights_log": np.asarray(jnp.exp(logw_traj[logged_idx])),
    }
