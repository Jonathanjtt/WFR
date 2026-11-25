"""2-D least-squares targets log pi = -1/2 ||F(t)||^2: Gaussian, Circle and Rosenbrock."""

import jax
import jax.numpy as jnp
import numpy as np

from gmmvi.targets.target import Target

__all__ = ["NAMES", "LIMS", "RESIDUALS", "F_gaussian", "F_circle", "F_rosenbrock",
           "make_target", "make_named_target", "SAMPLERS", "sample_circle",
           "sample_rosenbrock"]

_Amat = jnp.array([[1.0, 1.0], [1.0, 2.0]])
_s10 = jnp.sqrt(10.0)


def F_gaussian(t):
    return jnp.array([0.0, 1.0]) - _Amat @ t


def F_circle(t):
    return (jnp.array([1.0]) - jnp.array([t[0]**2 + t[1]**2])) / 0.3


def F_rosenbrock(t):
    return (jnp.array([0.0, 1.0]) - jnp.array([10.0*(t[1]-t[0]**2), t[0]])) / _s10


NAMES = ["Gaussian", "Circle", "Rosenbrock"]

RESIDUALS = {"Gaussian": F_gaussian, "Circle": F_circle, "Rosenbrock": F_rosenbrock}

LIMS = {
    "Gaussian": (-7, 5, -4, 5),
    "Circle": (-3, 3, -3, 3),
    "Rosenbrock": (-4, 4, -2, 15),
}


def make_target(F):
    """Wrap log pi = -1/2||F||^2 in a Target with a JAX-autodiff score."""
    log_pi = lambda t: -0.5 * jnp.sum(F(t) ** 2)
    return Target(grad_log_density=jax.jit(jax.vmap(jax.grad(log_pi))), dim=2,
                  log_density=jax.jit(jax.vmap(log_pi)))


def make_named_target(name):
    """Target for one of :data:`NAMES` ("Gaussian", "Circle", "Rosenbrock")."""
    return make_target(RESIDUALS[name])


def sample_circle(n, seed=0):
    """Exact draws from Circle: uniform angle, r^2 ~ N(1, 0.3^2) truncated at 0."""
    from scipy.special import ndtr, ndtri

    rng = np.random.RandomState(int(seed))
    lo = ndtr((0.0 - 1.0) / 0.3)
    u = 1.0 + 0.3 * ndtri(lo + (1.0 - lo) * rng.uniform(size=int(n)))
    r = np.sqrt(np.maximum(u, 0.0))
    theta = rng.uniform(0.0, 2.0 * np.pi, size=int(n))
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)


def sample_rosenbrock(n, seed=0):
    """Exact draws from Rosenbrock: t0 ~ N(1, 10), t1 | t0 ~ N(t0^2, 1/10)."""
    rng = np.random.RandomState(int(seed))
    t0 = 1.0 + np.sqrt(10.0) * rng.standard_normal(int(n))
    t1 = t0 ** 2 + (1.0 / np.sqrt(10.0)) * rng.standard_normal(int(n))
    return np.stack([t0, t1], axis=1)


SAMPLERS = {"Circle": sample_circle, "Rosenbrock": sample_rosenbrock}
