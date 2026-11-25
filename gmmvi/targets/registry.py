"""Target registry: build a target and the inputs the metrics need to score it, by name."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

__all__ = ["TargetSpec", "build", "available"]


@dataclass
class TargetSpec:
    name: str
    dim: int
    target: object
    log_prob_np: Callable
    grid_pts: Optional[np.ndarray] = None
    ref_density: Optional[np.ndarray] = None
    grid_shape: Optional[tuple] = None
    lims: Optional[tuple] = None
    mode_means: Optional[np.ndarray] = None
    mode_precisions: Optional[np.ndarray] = None
    sample_fn: Optional[Callable] = None
    meta: dict = field(default_factory=dict)

    @property
    def has_grid(self) -> bool:
        return self.grid_pts is not None

    @property
    def can_sample(self) -> bool:
        """True when EXACT target samples are available (needed for W2 / forward KL)."""
        return self.sample_fn is not None


def _grid_2d(lims, n):
    x0, x1, y0, y1 = lims
    gx = np.linspace(x0, x1, n)
    gy = np.linspace(y0, y1, n)
    GX, GY = np.meshgrid(gx, gy)
    return np.stack([GX.ravel(), GY.ravel()], 1), GX.shape


def _reference_on_grid(target, pts):
    """Target density on the grid, normalised to sum to 1."""
    r = np.exp(np.asarray(target.log_prob(pts[:, None])).reshape(-1))
    return r / r.sum()


def _build_MoG40(kappa=1.0, grid_n=300, lim=50.0, n_modes=40, loc=40.0, scale=1.0,
                 mode_seed=0):
    """``mode_seed`` is the PRNG seed that places the mode means; 0 is the published layout."""
    import jax
    import jax.numpy as jnp

    from gmmvi.targets.MoG40 import MoG40
    from gmmvi.targets.target import Target

    g = MoG40(dim=2, num_components=n_modes, loc_scaling=loc, scale_scaling=scale,
              seed=int(mode_seed), kappa=kappa)
    logp = lambda t: g.log_prob(t)
    target = Target(grad_log_density=jax.jit(jax.vmap(jax.grad(logp))), dim=2,
                    log_density=jax.jit(jax.vmap(logp)))

    lims = (-lim, lim, -lim, lim)
    pts, shape = _grid_2d(lims, grid_n)

    prec = None
    if kappa != 1.0:
        lams = np.asarray(jnp.exp(jnp.linspace(0.5, -0.5, 2) * jnp.log(kappa)))
        prec = np.broadcast_to(np.diag(1.0 / lams), (n_modes, 2, 2)).copy()

    def sample_fn(n, seed=0):
        return np.asarray(g.sample(jax.random.PRNGKey(int(seed)), (int(n),)), dtype=float)

    return TargetSpec(
        name=(f"MoG40_kappa{kappa:g}" if kappa != 1.0 else "MoG40")
             + (f"_modeseed{int(mode_seed)}" if int(mode_seed) != 0 else ""),
        dim=2, target=target, sample_fn=sample_fn,
        log_prob_np=lambda X: np.asarray(target.log_prob(np.asarray(X, float)), float).reshape(-1),
        grid_pts=pts, ref_density=_reference_on_grid(target, pts), grid_shape=shape,
        lims=lims, mode_means=np.asarray(g.means), mode_precisions=prec,
        meta=dict(kappa=kappa, n_modes=n_modes, mode_seed=int(mode_seed)),
    )


def _build_posterior_2d(name, grid_n=160):
    from gmmvi.targets import posterior_2d

    target = posterior_2d.make_named_target(name)
    lims = posterior_2d.LIMS[name]
    pts, shape = _grid_2d(lims, grid_n)
    return TargetSpec(
        name=name, dim=2, target=target, sample_fn=posterior_2d.SAMPLERS.get(name),
        log_prob_np=lambda X: np.asarray(target.log_prob(np.asarray(X, float)), float).reshape(-1),
        grid_pts=pts, ref_density=_reference_on_grid(target, pts), grid_shape=shape,
        lims=lims, meta=dict(case=name),
    )


def _build_breast_cancer(**kw):
    import jax

    jax.config.update("jax_enable_x64", True)

    from gmmvi.targets.breast_cancer import BreastCancer
    from gmmvi.targets.target import Target

    bc = BreastCancer(**kw)
    d = int(bc.num_dimensions)
    logp = lambda t: bc.log_prob(t)
    score = jax.jit(jax.vmap(jax.grad(logp)))
    value = jax.jit(jax.vmap(logp))
    target = Target(grad_log_density=score, dim=d, log_density=value)
    return TargetSpec(
        name="breast_cancer", dim=d, target=target,
        log_prob_np=lambda X: np.asarray(value(np.asarray(X, float).reshape(-1, d)),
                                         float).reshape(-1),
        meta=dict(problem=bc),
    )


def _build_nice_digits():
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    from gmmvi.targets.nice import NiceDigits
    from gmmvi.targets.target import Target

    flow = NiceDigits()
    dim = flow.dim
    _logp = jax.jit(jax.vmap(lambda x: flow.log_prob(x[None])[0]))
    _score = jax.jit(jax.vmap(jax.grad(lambda x: flow.log_prob(x[None])[0])))
    target = Target(grad_log_density=_score, dim=dim, log_density=_logp)

    def sample_fn(n, seed=0):
        return np.asarray(flow.sample(jax.random.PRNGKey(int(seed)), (int(n),)), dtype=float)

    return TargetSpec(
        name="nice_digits", dim=dim, target=target, sample_fn=sample_fn,
        log_prob_np=lambda X: np.asarray(_logp(jnp.asarray(X, dtype=float)), float).reshape(-1),
        meta=dict(flow=flow, im_size=flow.im_size, dataset=flow.dataset),
    )


def _build_darcy_symmetry(**kw):
    import jax

    jax.config.update("jax_enable_x64", True)

    from gmmvi.targets.darcy_symmetry import DarcySymmetry
    from gmmvi.targets.target import Target

    prob = DarcySymmetry(**kw)
    target = Target(grad_log_density=prob.score, dim=prob.dim, log_density=prob.log_prob,
                    theta_true=prob.theta_ref)
    return TargetSpec(
        name="darcy_symmetry", dim=prob.dim, target=target,
        log_prob_np=lambda X: np.asarray(prob.log_prob(np.asarray(X, float)), float).reshape(-1),
        meta=dict(problem=prob),
    )


BUILDERS = {
    "MoG40": _build_MoG40,
    "MoG40_kappa50": lambda **kw: _build_MoG40(kappa=50.0, **kw),
    "MoG40_kappa500": lambda **kw: _build_MoG40(kappa=500.0, **kw),
    "MoG40_kappa1000": lambda **kw: _build_MoG40(kappa=1000.0, **kw),
    "Gaussian": lambda **kw: _build_posterior_2d("Gaussian", **kw),
    "Circle": lambda **kw: _build_posterior_2d("Circle", **kw),
    "Rosenbrock": lambda **kw: _build_posterior_2d("Rosenbrock", **kw),
    "breast_cancer": _build_breast_cancer,
    "nice_digits": _build_nice_digits,
    "darcy_symmetry": _build_darcy_symmetry,
}


def available():
    """Names accepted by :func:`build`."""
    return sorted(BUILDERS)


def build(name, **overrides):
    """Build the named target and its metric inputs.  ``overrides`` reach the builder."""
    return BUILDERS[name](**overrides)
