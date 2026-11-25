"""Shared JAX building blocks: mixture densities, target adapter, reverse-KL traces and snapshot buffers."""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

_LOG2PI = jnp.log(2.0 * jnp.pi)


class _ResolvedTargetFn:
    """Hashable wrapper so two ``resolve_target`` calls on one target give cache-equal static jit arguments."""

    __slots__ = ("_fn", "_reshape_to_batch")

    def __init__(self, fn, reshape_to_batch):
        self._fn = fn
        self._reshape_to_batch = reshape_to_batch

    def __call__(self, x):
        y = jnp.asarray(self._fn(x))
        return y.reshape(x.shape[0]) if self._reshape_to_batch else y

    def __eq__(self, other):
        return (isinstance(other, _ResolvedTargetFn)
                and self._fn is other._fn
                and self._reshape_to_batch == other._reshape_to_batch)

    def __hash__(self):
        return hash((id(self._fn), self._reshape_to_batch))

    def __repr__(self):
        kind = "logp" if self._reshape_to_batch else "score"
        return f"<resolved {kind} {self._fn!r}>"


def resolve_target(target):
    """``(score_fn, logp_fn)`` JAX callables from a ``Target`` (its ``_grad`` / ``_logp``) or a bare pair."""
    if isinstance(target, (tuple, list)) and len(target) == 2:
        score_fn, logp_fn = target
    else:
        score_fn, logp_fn = target._grad, target._logp

    return _ResolvedTargetFn(score_fn, False), _ResolvedTargetFn(logp_fn, True)


def target_dim(target, dim=None):
    """``dim`` if given, else ``target.dim``."""
    if dim is not None:
        return int(dim)
    if hasattr(target, "dim"):
        return int(target.dim)
    raise TypeError("a (score_fn, logp_fn) target carries no dimension; pass dim=<d> to fit()")


def igmm_log_components(x, means, eps):
    """log[(1/K) N(x; m_k, eps_k I)] for x (M, d), means (K, d), eps (K,) -> (M, K)."""
    K, d = means.shape
    diff = x[:, None, :] - means[None, :, :]
    quad = jnp.sum(diff * diff, axis=-1) / eps[None, :]
    log_norm = -0.5 * (d * _LOG2PI + d * jnp.log(eps))
    return log_norm[None, :] - 0.5 * quad - jnp.log(K)


def igmm_log_prob(x, means, eps):
    """log rho(x) for the isotropic uniform-weight mixture, x (M, d) -> (M,)."""
    return logsumexp(igmm_log_components(x, means, eps), axis=1)


def igmm_score(x, means, eps):
    """grad_x log rho(x), x (M, d) -> (M, d)."""
    logc = igmm_log_components(x, means, eps)
    r = jax.nn.softmax(logc, axis=1)
    diff = x[:, None, :] - means[None, :, :]
    comp_grad = -diff / eps[None, :, None]
    return jnp.sum(r[:, :, None] * comp_grad, axis=1)


def igmm_sample_each(means, eps, noise):
    """Per-component samples: means (K, d), eps (K,), shared noise (B, d) -> (K, B, d)."""
    return means[:, None, :] + jnp.sqrt(eps)[:, None, None] * noise[None, :, :]


def isotropic_kls(logp_fn, means_traj, eps_traj, logged_idx, ci, z):
    """Reverse KL at the logged snapshots of an isotropic-mixture trajectory -> (L,); ``ci`` / ``z`` are shared."""
    Ms = means_traj[logged_idx]
    Es = eps_traj[logged_idx]

    def sample_one(M, E):
        return M[ci] + jnp.sqrt(E[ci])[:, None] * z

    xs = jax.vmap(sample_one)(Ms, Es)
    logq = jax.vmap(lambda x, M, E: igmm_log_prob(x, M, E))(xs, Ms, Es)
    logpi = jax.lax.map(logp_fn, xs)
    return (logq - logpi).mean(axis=1)


def gmm_log_components(x, means, chols, log_weights):
    """log[w_k N(x; m_k, C_k)], C_k = L_k L_k^T; x (M, d) -> (M, K)."""
    K, d = means.shape
    diff = x[:, None, :] - means[None, :, :]
    logdet = 2.0 * jnp.log(jnp.diagonal(chols, axis1=1, axis2=2)).sum(axis=1)

    def quad_one(L, dmk):
        y = jax.scipy.linalg.solve_triangular(L, dmk.T, lower=True)
        return jnp.sum(y * y, axis=0)

    quad = jax.vmap(quad_one, in_axes=(0, 1))(chols, diff)
    log_norm = log_weights - 0.5 * (d * _LOG2PI + logdet)
    return log_norm[None, :] - 0.5 * quad.T


def gmm_log_prob(x, means, chols, log_weights):
    return logsumexp(gmm_log_components(x, means, chols, log_weights), axis=1)


def gmm_score(x, means, precision, log_weights, chols):
    """grad_x log rho(x) for the full-covariance mixture; precision (K, d, d) = C_k^{-1}."""
    logc = gmm_log_components(x, means, chols, log_weights)
    r = jax.nn.softmax(logc, axis=1)
    diff = x[:, None, :] - means[None, :, :]
    prec_diff = jnp.einsum("kde,mke->mkd", precision, diff)
    return -jnp.einsum("mk,mkd->md", r, prec_diff)


def full_kls(logp_fn, means_traj, chol_traj, logw_traj, logged_idx, key, z):
    """Reverse KL at the logged snapshots of a full-covariance trajectory -> (L,); components drawn per snapshot from its weights."""
    Ms = means_traj[logged_idx]
    Ls = chol_traj[logged_idx]
    LW = logw_traj[logged_idx]
    nL, BKL = Ms.shape[0], z.shape[0]
    keys = jax.random.split(key, nL)

    def sample_one(M, L, w, kk):
        ci = jax.random.categorical(kk, w, shape=(BKL,))
        return M[ci] + jnp.einsum("bij,bj->bi", L[ci], z)

    xs = jax.vmap(sample_one)(Ms, Ls, LW, keys)
    logq = jax.vmap(lambda x, M, L, w: gmm_log_prob(x, M, L, w))(xs, Ms, Ls, LW)
    logpi = jax.lax.map(logp_fn, xs)
    return (logq - logpi).mean(axis=1)


def snapshot_slot(it, compute_kl, n_iterations, n_logged):
    """Row of the snapshot buffer iteration ``it`` writes to: ``it // compute_kl`` on the grid, ``n_logged - 1`` for an
    off-grid final iteration, else the scratch row ``n_logged``."""
    on_grid = (it % compute_kl) == 0
    is_last = it == (n_iterations - 1)
    slot = jnp.where(on_grid, it // compute_kl, n_logged - 1)
    return jnp.where(on_grid | is_last, slot, n_logged)


def write_snapshot(buffers, values, slot):
    """Write each ``values[i]`` into row ``slot`` of ``buffers[i]``. Returns the new buffers."""
    return tuple(jax.lax.dynamic_update_index_in_dim(b, v, slot, 0)
                 for b, v in zip(buffers, values))


def snapshot_buffers(n_logged, values):
    """Zero buffers holding ``n_logged`` snapshots of each of ``values``, plus one scratch row."""
    return tuple(jnp.zeros((n_logged + 1,) + v.shape, v.dtype) for v in values)


def logged_iterations(n_iterations, compute_kl):
    """Iteration indices at which the NumPy optimisers log a KL (pre-update snapshots)."""
    return [it for it in range(n_iterations)
            if it % compute_kl == 0 or it == n_iterations - 1]


def sym(A):
    """Symmetrise the last two axes."""
    return 0.5 * (A + jnp.swapaxes(A, -1, -2))
