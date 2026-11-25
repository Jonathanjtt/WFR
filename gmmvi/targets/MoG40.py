"""MoG40 target (fab-jax): 40 equal-weight Gaussians; ``kappa`` > 1 makes each mode anisotropic."""
import jax
import jax.numpy as jnp

from gmmvi.targets.base_target import Target

_LOG2PI = jnp.log(2.0 * jnp.pi)


class MoG40(Target):
    def __init__(
            self,
            dim: int = 2, num_components: int = 40, loc_scaling: float = 40,
            scale_scaling: float = 1.0, seed: int = 0, sample_bounds=None, can_sample=True, log_Z=0,
            kappa: float = 1.0
    ) -> None:
        super().__init__(dim, log_Z, can_sample)

        self.seed = seed
        self.n_mixes = num_components

        key = jax.random.PRNGKey(seed)
        self.log_weights = jnp.full((num_components,), -jnp.log(num_components))
        self.means = jax.random.uniform(
            shape=(num_components, dim), key=key, minval=-1.0, maxval=1.0) * loc_scaling
        self.scales = jnp.ones(shape=(num_components, dim)) * scale_scaling

        self.kappa = float(kappa)
        if self.kappa != 1.0:
            lams = jnp.exp(jnp.linspace(0.5, -0.5, dim) * jnp.log(self.kappa))
            k_rot = jax.random.fold_in(key, 1)
            gauss = jax.random.normal(k_rot, (num_components, dim, dim))
            Q, R = jnp.linalg.qr(gauss)
            Q = Q * jnp.sign(jnp.diagonal(R, axis1=1, axis2=2))[:, None, :]
            self.covariances = (scale_scaling ** 2) * jnp.einsum("kij,j,klj->kil", Q, lams, Q)
            self.chols = jnp.linalg.cholesky(self.covariances)
            self._half_logdets = jnp.sum(jnp.log(jnp.diagonal(self.chols, axis1=1, axis2=2)),
                                         axis=1)
        else:
            self.covariances = (scale_scaling ** 2) * jnp.broadcast_to(
                jnp.eye(dim), (num_components, dim, dim))
            self.chols = scale_scaling * jnp.broadcast_to(jnp.eye(dim),
                                                          (num_components, dim, dim))

    def _component_log_prob(self, x):
        d = x[:, None, :] - self.means[None, :, :]
        if self.kappa == 1.0:
            z = d / self.scales[None, :, :]
            return -0.5 * jnp.sum(z ** 2 + 2.0 * jnp.log(self.scales)[None] + _LOG2PI, axis=-1)
        z = jax.vmap(lambda Lk, dk: jax.scipy.linalg.solve_triangular(Lk, dk.T, lower=True).T,
                     in_axes=(0, 1), out_axes=1)(self.chols, d)
        return (-0.5 * jnp.sum(z ** 2 + _LOG2PI, axis=-1)
                - self._half_logdets[None, :])

    def log_prob(self, x):
        batched = x.ndim == 2
        if not batched:
            x = x[None,]

        comp = self._component_log_prob(x)
        log_prob = jax.scipy.special.logsumexp(self.log_weights[None] + comp, axis=-1)

        if not batched:
            log_prob = jnp.squeeze(log_prob, axis=0)

        return log_prob

    def sample(self, seed, sample_shape=()):
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        k_cat, k_norm = jax.random.split(seed)
        n = 1
        for s in sample_shape:
            n *= s
        comp = jax.random.categorical(k_cat, self.log_weights, shape=(n,))
        eps = jax.random.normal(k_norm, (n, self.dim))
        if self.kappa == 1.0:
            out = self.means[comp] + eps * self.scales[comp]
        else:
            out = self.means[comp] + jnp.einsum("nij,nj->ni", self.chols[comp], eps)
        return out.reshape(tuple(sample_shape) + (self.dim,))

    def entropy(self, samples=None):
        """Base-K entropy of the nearest-component histogram of ``samples``."""
        idx = jnp.argmax(self._component_log_prob(samples), axis=1)
        unique_elements, counts = jnp.unique(idx, return_counts=True)
        mode_dist = counts / samples.shape[0]
        entropy = -jnp.sum(mode_dist * (jnp.log(mode_dist) / jnp.log(self.n_mixes)))
        return entropy
