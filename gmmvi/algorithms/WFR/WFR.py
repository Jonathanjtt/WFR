from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from gmmvi.jaxvi import common
from gmmvi.algorithms.WFR import WFR_warmup_schedule as hv


def _kl_cov_prox(A, tau, mu_minus, mu_plus):
    """KL proximal covariance update -> ``(Cnew, B)`` with ``B @ B.T == Cnew``."""
    A = common.sym(A)
    a, U = jnp.linalg.eigh(A)
    pos = a > 0.0
    safe_a = jnp.where(pos, a, 1.0)
    chi = jnp.where(pos, jnp.clip((1.0 + tau) / safe_a, mu_minus, mu_plus), mu_plus)
    return common.sym((U * chi) @ U.T), U * jnp.sqrt(chi)


def _clip_spd(S, mu_minus, mu_plus):
    """Clip the eigenvalues of symmetric ``S`` to ``[mu_minus, mu_plus]``."""
    S = common.sym(S)
    w, V = jnp.linalg.eigh(S)
    w = jnp.clip(w, mu_minus, mu_plus)
    return common.sym((V * w) @ V.T)


def _stats_once(means, chols, precision, logw, z, score_fn, logp_fn):
    """One target-score batch -> ``(g, H, ell, s_post, prec_dx)``.

    ``g_k = E[grad log(pi/rho)]``, ``H_k = Hbar_k - C_k^{-1}`` and ``ell_k = E[log pi - log rho]``,
    where ``Hbar_k`` is the Stein estimate of ``E[grad^2 log(pi/rho)]``.
    """
    K, J, d = z.shape
    theta = means[:, None, :] + jnp.einsum("kij,kbj->kbi", chols, z)
    flat = theta.reshape(K * J, d)

    s_post = score_fn(flat).reshape(K, J, d)
    s_gm = common.gmm_score(flat, means, precision, logw, chols).reshape(K, J, d)
    dx = theta - means[:, None, :]
    prec_dx = jnp.einsum("kde,kbe->kbd", precision, dx)
    den = max(J - 1, 1)
    df = s_post - s_gm
    g = df.mean(axis=1)
    Hbar = common.sym(jnp.einsum("kbi,kbj->kij", df - g[:, None, :], prec_dx) / den)
    H = Hbar - precision

    lp_post = logp_fn(flat).reshape(K, J)
    lp_gm = common.gmm_log_prob(flat, means, chols, logw).reshape(K, J)
    ell = (lp_post - lp_gm).mean(axis=1)
    return g, H, ell, s_post, prec_dx


def _w_update_one(m, C, gk, Hk, eta_w, mu_minus, mu_plus, I):
    """W step -> ``(m, C, B)`` with ``B @ B.T == C``, from one eigendecomposition."""
    mW = m + eta_w * gk
    M = I + eta_w * Hk
    Ctil = common.sym(M @ C @ M)
    lam, V = jnp.linalg.eigh(Ctil)
    root_lam = jnp.sqrt(jnp.clip(lam * lam + 4.0 * eta_w * lam, 0.0, None))
    cnew_lam = jnp.clip(0.5 * (lam + 2.0 * eta_w + root_lam), mu_minus, mu_plus)
    return mW, common.sym((V * cnew_lam) @ V.T), V * jnp.sqrt(cnew_lam)


def _fr_update_one(m, C, P, gk, Hk, dt_fr, mu_minus, mu_plus):
    """FR step -> ``(m, C, B)`` with ``B @ B.T == C``."""
    mFR = m + dt_fr * (C @ gk)
    A = P - dt_fr * Hk
    C_fr, B_fr = _kl_cov_prox(A, dt_fr, mu_minus, mu_plus)
    return mFR, C_fr, B_fr


def _fr_bregman_residual(chols, H, precision):
    """Whitened FR residual ``E_fr,k = -L_k^T (H_k + P_k) L_k``. (K, d, d)."""
    Hbar = common.sym(H + precision)
    E_fr = -jnp.einsum("kji,kjl,klm->kim", chols, Hbar, chols)
    return common.sym(E_fr)


def _fr_subset_weight_update(logw, ell_plus, fr_mask, dt_fr):
    """Replicator update on the FR subset, renormalised to that subset's own mass.

    W components keep their log-weights exactly; an empty FR subset changes nothing.
    """
    dtype = logw.dtype
    neg_inf = jnp.asarray(-jnp.inf, dtype)
    F = fr_mask
    have_fr = jnp.any(F)

    log_mass = jax.scipy.special.logsumexp(jnp.where(F, logw, neg_inf))
    log_num = jnp.where(F, logw + dt_fr * ell_plus, neg_inf)
    lse = jax.scipy.special.logsumexp(log_num)

    log_mass_s = jnp.where(have_fr, log_mass, jnp.zeros((), dtype))
    lse_s = jnp.where(have_fr, lse, jnp.zeros((), dtype))
    num_s = jnp.where(F, logw + dt_fr * ell_plus, jnp.zeros((), dtype))
    logw_fr = log_mass_s + (num_s - lse_s)

    return jnp.where(F & have_fr, logw_fr, logw)


def _manual_fr_mask(it, n_w_steps, K):
    """All-W while ``it < n_w_steps``, all-FR afterwards. (K,) bool."""
    return jnp.full((K,), jnp.logical_not(it < n_w_steps))


_RUN_STATIC_ARGNAMES = ("eta_mode", "fr_step_mode", "auto_mode", "require_pos_clearance",
                        "score_fn", "logp_fn", "compute_kl", "n_logged")


@partial(jax.jit, static_argnames=_RUN_STATIC_ARGNAMES)
def _run(init_means, init_covs, init_logw, z_seq,
         b_fr, fr_step_eps, dt_weight,
         eta_max, eta_fixed, dt_fr, mu_minus, mu_plus,
         eta_mode, fr_step_mode, auto_mode, n_w_steps_manual, max_w_steps,
         require_pos_clearance, nonconvex_z_floor,
         score_fn, logp_fn, compute_kl, n_logged):
    d = init_means.shape[1]
    K = init_means.shape[0]
    n_iterations = z_seq.shape[0]
    dtype = init_means.dtype
    I = jnp.eye(d, dtype=dtype)
    eta_max = jnp.asarray(eta_max, dtype)
    eta_fixed = jnp.asarray(eta_fixed, dtype)
    dt_fr = jnp.asarray(dt_fr, dtype)
    mu_minus = jnp.asarray(mu_minus, dtype)
    mu_plus = jnp.asarray(mu_plus, dtype)

    def step(carry, z):
        means, covs, logw, fr_mask, w_step_counts, it, snaps = carry
        chols = jnp.linalg.cholesky(common.sym(covs))
        precision = jax.vmap(lambda L: jax.scipy.linalg.cho_solve((L, True), I))(chols)
        g, H, ell, s_post, prec_dx = _stats_once(
            means, chols, precision, logw, z, score_fn, logp_fn)

        if fr_step_mode == "adaptive":
            E_fr = _fr_bregman_residual(chols, H, precision)
            rho_E_fr_k = jnp.max(jnp.abs(jnp.linalg.eigvalsh(E_fr)), axis=-1)
            dt_fr_k = jnp.minimum(dt_fr, b_fr / jnp.maximum(rho_E_fr_k, fr_step_eps))
            fr_means, fr_covs, fr_facs = jax.vmap(
                lambda m, C, P, gk, Hk, dtk: _fr_update_one(m, C, P, gk, Hk, dtk,
                                                            mu_minus, mu_plus))(
                means, covs, precision, g, H, dt_fr_k)
        else:
            dt_fr_k = jnp.full((K,), dt_fr)
            fr_means, fr_covs, fr_facs = jax.vmap(
                lambda m, C, P, gk, Hk: _fr_update_one(m, C, P, gk, Hk, dt_fr,
                                                       mu_minus, mu_plus))(
                means, covs, precision, g, H)
        fr_cap_k = dt_fr_k < dt_fr - 10.0 * jnp.finfo(dtype).eps

        def _gated(_):
            HV = hv.stein_target_hessian(s_post, prec_dx)
            if eta_mode == "fixed":
                eta_k = jnp.full((K,), eta_fixed)
            else:
                eta_k = hv.adaptive_w_step(hv.spectral_radius(HV), eta_max)
            w_means, w_covs, w_facs = jax.vmap(
                lambda m, C, gk, Hk, e: _w_update_one(m, C, gk, Hk, e, mu_minus, mu_plus, I))(
                means, covs, g, H, eta_k)
            if auto_mode:
                use_m = hv.component_hv_log_gate_decision(
                    hv.whitened_scale_eigs_from_factor(chols, HV),
                    hv.whitened_scale_eigs_from_factor(w_facs, HV),
                    hv.whitened_scale_eigs_from_factor(fr_facs, HV),
                    fr_mask, w_step_counts, max_w_steps,
                    require_pos_clearance=require_pos_clearance,
                    nonconvex_z_floor=nonconvex_z_floor)
            else:
                use_m = fr_mask | _manual_fr_mask(it, n_w_steps_manual, K)
            return (jnp.where(use_m[:, None], fr_means, w_means),
                    jnp.where(use_m[:, None, None], fr_covs, w_covs),
                    use_m)

        def _all_fr(_):
            return fr_means, fr_covs, fr_mask

        new_means, new_covs, new_fr_mask = jax.lax.cond(jnp.all(fr_mask), _all_fr, _gated, None)
        new_w_step_counts = w_step_counts + jnp.logical_not(new_fr_mask).astype(jnp.int32)

        if fr_step_mode == "adaptive":
            new_logw = _fr_subset_weight_update(logw, ell, new_fr_mask, dt_weight)
        else:
            new_logw = _fr_subset_weight_update(logw, ell, new_fr_mask, dt_fr)

        new_snaps = common.write_snapshot(
            snaps, (means, chols, logw),
            common.snapshot_slot(it, compute_kl, n_iterations, n_logged))
        carry_new = (new_means, new_covs, new_logw, new_fr_mask, new_w_step_counts, it + 1,
                     new_snaps)
        return carry_new, (dt_fr_k, fr_cap_k)

    init = (init_means, init_covs, init_logw,
            jnp.zeros((K,), bool),
            jnp.zeros((K,), jnp.int32),
            jnp.zeros((), jnp.int32),
            common.snapshot_buffers(n_logged, (init_means, init_covs, init_logw)))
    (fm, fc, flw, _mask, _counts, _it, snaps), (dt_fr_traj, fr_cap_traj) = jax.lax.scan(
        step, init, z_seq)
    means_log, chol_log, logw_log = (s[:n_logged] for s in snaps)
    return means_log, chol_log, logw_log, dt_fr_traj, fr_cap_traj, fm, fc, flw


def fit(target, init_means, init_covs, n_iterations=500,
        dt_fr=0.5, n_w_steps="auto", fr_step_mode="fixed", b_fr=0.5, fr_step_eps=1e-12,
        dt_weight=1.0, eta_mode="adaptive", eta_max=0.25, eta_w=None,
        require_pos_clearance=True, nonconvex_z_floor=None, max_w_steps=10,
        J=None, BKL=1000, compute_kl=None, compute_kl_trace=True, dim=None, seed=0,
        mu_minus=1.168e-4, mu_plus=5.131):
    """Fit a full-covariance GMM, switching each component from W to FR.

    ``n_w_steps``: ``"auto"`` (the log(1/z) crossover gate, with at most ``max_w_steps`` W updates
    per component) or an integer N (every component takes W for N iterations, then FR).
    ``eta_mode``: ``"adaptive"`` (the curvature-based W step, capped at ``eta_max``) or ``"fixed"``
    (``eta_w`` for every component). ``fr_step_mode``: ``"fixed"`` (``dt_fr``) or ``"adaptive"``
    (``dt_fr`` is a ceiling, each component is capped at ``b_fr / ||E_fr,k||_2``, and the weights
    use ``dt_weight``). ``nonconvex_z_floor`` is the concavity-clearance threshold (``None`` ->
    1e-5); ``require_pos_clearance=False`` drops the clearance. ``compute_kl_trace=False`` skips
    the reverse-KL trace.
    """
    n_iterations = int(n_iterations)
    dt_fr = float(dt_fr)
    b_fr = float(b_fr)
    fr_step_eps = float(fr_step_eps)
    dt_weight = float(dt_weight)
    eta_max = float(eta_max)
    eta_fixed = float(eta_w) if eta_mode == "fixed" else -1.0
    auto_mode = n_w_steps == "auto"
    n_w_steps_manual = 0 if auto_mode else int(n_w_steps)
    if nonconvex_z_floor is None:
        nonconvex_z_floor = hv.NONCONVEX_Z_FLOOR
    nonconvex_z_floor = float(nonconvex_z_floor)
    if compute_kl is None:
        compute_kl = max(1, n_iterations // 20)
    compute_kl = int(compute_kl)
    mu_minus = float(mu_minus)
    mu_plus = float(mu_plus)

    d = common.target_dim(target, dim)
    J = int(J) if J is not None else 4 * d
    score_fn, logp_fn = common.resolve_target(target)
    _, k_z, k_ci, k_zkl = jax.random.split(jax.random.PRNGKey(seed), 4)

    init_means = jnp.asarray(init_means, dtype=float)
    n_components = init_means.shape[0]
    init_covs = jnp.asarray(init_covs, dtype=float)
    init_logw = jnp.full((n_components,), -jnp.log(n_components))
    init_covs = jax.vmap(lambda C: _clip_spd(C, mu_minus, mu_plus))(init_covs)
    z_seq = jax.random.normal(k_z, (n_iterations, n_components, J, d))
    logged = common.logged_iterations(n_iterations, compute_kl)

    means_log, chol_log, logw_log, dt_fr_traj, fr_cap_traj, fm, fc, flw = _run(
        init_means, init_covs, init_logw, z_seq,
        b_fr, fr_step_eps, dt_weight,
        eta_max=eta_max, eta_fixed=eta_fixed, dt_fr=dt_fr,
        mu_minus=mu_minus, mu_plus=mu_plus,
        eta_mode=str(eta_mode), fr_step_mode=str(fr_step_mode),
        auto_mode=bool(auto_mode), n_w_steps_manual=n_w_steps_manual,
        max_w_steps=int(max_w_steps),
        require_pos_clearance=bool(require_pos_clearance),
        nonconvex_z_floor=nonconvex_z_floor,
        score_fn=score_fn, logp_fn=logp_fn,
        compute_kl=compute_kl, n_logged=len(logged))

    logged_idx = jnp.asarray(logged)
    kls = None
    if compute_kl_trace:
        z = jax.random.normal(k_zkl, (BKL, d))
        kls = np.asarray(common.full_kls(logp_fn, means_log, chol_log, logw_log,
                                         jnp.arange(len(logged)), k_ci, z))
    covs_log = np.asarray(jnp.einsum("lkij,lkmj->lkim", chol_log, chol_log))

    return {
        "means": np.asarray(fm),
        "covariances": np.asarray(fc),
        "weights": np.asarray(jnp.exp(flw)),
        "kls": kls,
        "elbo": None if kls is None else -kls,
        "elbo_iters": np.asarray(logged),
        "means_log": np.asarray(means_log),
        "covs_log": covs_log,
        "weights_log": np.asarray(jnp.exp(logw_log)),
        "dt_fr_eff_per_component_log": np.asarray(dt_fr_traj[logged_idx]),
        "fr_cap_active_per_component_log": np.asarray(fr_cap_traj[logged_idx]),
        "eta_mode": eta_mode,
        "eta_max": eta_max,
        "eta_w": eta_fixed if eta_mode == "fixed" else None,
        "dt_fr": dt_fr,
        "fr_step_mode": str(fr_step_mode),
        "b_fr": b_fr,
        "dt_weight": dt_weight,
    }
