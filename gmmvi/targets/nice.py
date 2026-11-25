import os
import pickle

import numpy as np
import jax
import jax.numpy as jnp

from gmmvi.targets.base_target import Target

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

_N_STEPS = 4
_DENSE_IDX = (0, 2, 4, 6, 8)


def _load_params(dataset: str, dim: int):
    """Load ``(logscale, nets)`` of a pretrained NICE flow from its plain-NumPy pickle."""
    im_size = int(round(np.sqrt(dim)))
    path = os.path.join(_DATA_DIR, f"params_nice_{dataset}_{im_size}x{im_size}_np.pkl")
    with open(path, "rb") as fh:
        params = pickle.load(fh)["params"]
    logscale = jnp.asarray(params["logscale"], dtype=jnp.float64 if jax.config.jax_enable_x64 else jnp.float32)
    nets = []
    for i in range(_N_STEPS):
        net = params[f"nets_{i}"]
        nets.append({idx: (jnp.asarray(net[f"layers_{idx}"]["kernel"]),
                           jnp.asarray(net[f"layers_{idx}"]["bias"])) for idx in _DENSE_IDX})
    return logscale, nets


def _net_apply(net, xa):
    h = xa
    for idx in _DENSE_IDX:
        W, b = net[idx]
        h = h @ W + b
        if idx != _DENSE_IDX[-1]:
            h = jax.nn.relu(h)
    return h


class NiceTarget(Target):
    """Pretrained NICE-flow log-density; ``dim`` must match ``dataset``."""

    def __init__(self, dim: int, dataset: str = "mnist", log_Z: float = 0.0, can_sample: bool = True,
                 sample_bounds=None) -> None:
        super().__init__(dim=dim, log_Z=log_Z, can_sample=can_sample)
        self.dataset = dataset
        self.im_size = int(round(np.sqrt(dim)))
        self.split = (dim + 1) // 2
        self.perm = jnp.array(list(reversed(range(dim))))
        self.logscale, self.nets = _load_params(dataset, dim)

    def _forward(self, x):
        """Data -> latent, ``z = f(x)``; ``x`` is (B, dim)."""
        s = self.split
        for net in self.nets:
            xs = x[:, self.perm]
            xa, xb = xs[:, :s], xs[:, s:]
            x = jnp.concatenate([xa, xb + _net_apply(net, xa)], axis=-1)
        return x

    def _reverse(self, y):
        """Latent -> data, ``x = f^{-1}(y)``; ``y`` is (B, dim)."""
        s = self.split
        for net in reversed(self.nets):
            ya, yb = y[:, :s], y[:, s:]
            xb = yb - _net_apply(net, ya)
            y = jnp.concatenate([ya, xb], axis=-1)[:, self.perm]
        return y

    def log_prob(self, x):
        batched = x.ndim == 2
        if not batched:
            x = x[None,]
        z = self._forward(x)
        zs = z * jnp.exp(self.logscale)[None, :]
        log_prob = (-0.5 * jnp.sum(zs ** 2, axis=-1)
                    - 0.5 * self.dim * jnp.log(2.0 * jnp.pi)
                    + jnp.sum(self.logscale))
        if not batched:
            log_prob = jnp.squeeze(log_prob, axis=0)
        return log_prob

    def sample(self, seed, sample_shape=()):
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        scalar = len(sample_shape) == 0
        n = 1
        for s in sample_shape:
            n *= s
        zs = jax.random.normal(seed, (n, self.dim))
        z = zs / jnp.exp(self.logscale)[None, :]
        x = self._reverse(z)
        if scalar:
            return x[0]
        return x.reshape(tuple(sample_shape) + (self.dim,))


class NiceDigits(NiceTarget):
    """NICE flow trained on MNIST digits down-scaled to 14x14 -> ``dim = 196``."""

    def __init__(self, log_Z: float = 0.0, can_sample: bool = True, sample_bounds=None) -> None:
        super().__init__(dim=196, dataset="mnist", log_Z=log_Z, can_sample=can_sample,
                         sample_bounds=sample_bounds)

