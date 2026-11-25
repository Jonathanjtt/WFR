"""Decision helpers for :mod:`gmmvi.algorithms.WFR.WFR`: the adaptive W step and the
componentwise log(1/z) crossover W -> FR switch."""

import jax
import jax.numpy as jnp

from gmmvi.jaxvi import common

__all__ = ["CLOCK_Z_FLOOR", "NONCONVEX_Z_FLOOR",
           "whitened_scale_eigs", "whitened_scale_eigs_from_factor",
           "hv_log_clock", "component_hv_log_gate_decision",
           "spectral_radius", "stein_target_hessian", "adaptive_w_step"]

CLOCK_Z_FLOOR = 1e-10
NONCONVEX_Z_FLOOR = 1e-5


def _psd_sqrt(S):
    """Principal square root of a symmetric PSD matrix (eigenvalues clipped at 0 first)."""
    w, V = jnp.linalg.eigh(common.sym(S))
    w = jnp.clip(w, 0.0, None)
    return (V * jnp.sqrt(w)) @ V.T


def spectral_radius(A):
    """Batched ``max |eig(Sym(A_k))|``. (K, d, d) -> (K,)."""
    return jax.vmap(lambda M: jnp.max(jnp.abs(jnp.linalg.eigvalsh(common.sym(M)))))(A)


def stein_target_hessian(s_post, prec_dx):
    """Stein estimate of the target-only curvature ``HV_k = -E_{q_k}[grad^2 log pi]``. (K, d, d).

    ``s_post`` is the target score batch and ``prec_dx = C_k^{-1}(theta - m_k)``, both (K, J, d).
    """
    J = s_post.shape[1]
    s_c = s_post - s_post.mean(axis=1, keepdims=True)
    den = max(J - 1, 1)
    return -common.sym(jnp.einsum("kbi,kbj->kij", s_c, prec_dx) / den)


def adaptive_w_step(rho_hv, eta_max):
    """``eta_k = 1 / (2 max(rho(HV_k), 1/(2 eta_max)))``, capped at ``eta_max``. (K,)."""
    eta = 0.5 / jnp.maximum(rho_hv, 0.5 / eta_max)
    return jnp.clip(eta, None, eta_max)


def whitened_scale_eigs(covs, curvature):
    """Whitened scales ``z_i = eig_i(C^{1/2} curvature C^{1/2})``. (K, d, d) x 2 -> (K, d)."""
    def one(Ck, Hk):
        Ch = _psd_sqrt(Ck)
        return jnp.linalg.eigvalsh(common.sym(Ch @ common.sym(Hk) @ Ch))
    return jax.vmap(one)(covs, curvature)


def whitened_scale_eigs_from_factor(factors, curvature):
    """The same scales from a factor ``B`` with ``B B^T = C``: ``eig(B^T H B)``. (K, d).

    The order ``B.T @ H @ B`` matters: ``B`` is not symmetric in general.
    """
    def one(Bk, Hk):
        return jnp.linalg.eigvalsh(common.sym(Bk.T @ common.sym(Hk) @ Bk))
    return jax.vmap(one)(factors, curvature)


def hv_log_clock(z):
    """``T(z) = log(1/z)`` where ``z > CLOCK_Z_FLOOR``, else 0 (the direction is left out)."""
    T = jnp.log(1.0 / jnp.maximum(z, CLOCK_Z_FLOOR))
    return jnp.where(z > CLOCK_Z_FLOOR, T, 0.0)


def component_hv_log_gate_decision(z_cur, z_w, z_fr, previous_fr_mask, w_step_counts,
                                   max_w_steps, require_pos_clearance=True,
                                   nonconvex_z_floor=None):
    """One evaluation of the componentwise W -> FR switch.

    ``z_cur`` / ``z_w`` / ``z_fr`` (K, d) are the whitened scales of the current covariance and of
    the virtual W and FR proposals, all under the same ``H_V``. A component switches when its clock
    is exhausted (``T_cur <= 0``) or one FR step lowers it at least as much as one W step (subject
    to the concavity clearance ``min z_cur >= -nonconvex_z_floor``), or once it has taken
    ``max_w_steps`` W updates. FR is absorbing.

    Returns the new FR mask, (K,) bool.
    """
    witness_floor = NONCONVEX_Z_FLOOR if nonconvex_z_floor is None else nonconvex_z_floor
    T_cur = jnp.max(hv_log_clock(z_cur), axis=-1)
    T_w = jnp.max(hv_log_clock(z_w), axis=-1)
    T_fr = jnp.max(hv_log_clock(z_fr), axis=-1)
    warm_done = T_cur <= 0.0
    gain_w = T_cur - T_w
    gain_fr = T_cur - T_fr
    warm_cross = gain_w - gain_fr <= 0.0
    switch_signal = warm_done | warm_cross
    if require_pos_clearance:
        switch_signal = switch_signal & (jnp.min(z_cur, axis=-1) >= -witness_floor)

    unswitched = jnp.logical_not(previous_fr_mask)
    switch_by_gate = unswitched & switch_signal
    forced_by_cap = unswitched & (w_step_counts >= max_w_steps)
    return previous_fr_mask | switch_by_gate | forced_by_cap
