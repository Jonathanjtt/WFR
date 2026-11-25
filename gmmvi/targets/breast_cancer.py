import os

import jax
import jax.numpy as jnp
import numpy as np

from gmmvi.targets.base_target import Target

_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "breast_cancer.data")


class BreastCancer(Target):
    def __init__(self, log_Z=None, can_sample=False, sample_bounds=None,
                 split=True, test_frac=0.5, seed=42, scale_eps=1e-8):
        super().__init__(dim=31, log_Z=log_Z, can_sample=can_sample)
        data = np.loadtxt(_DATA_PATH)
        data = jnp.array(data, dtype=jnp.float32)
        X_raw = data[:, 2:]
        y = data[:, 1]

        if split:
            n = int(X_raw.shape[0])
            perm = np.random.RandomState(seed).permutation(n)
            n_test = int(round(test_frac * n))
            test_idx = jnp.asarray(perm[:n_test])
            train_idx = jnp.asarray(perm[n_test:])
            X_train_raw, y_train = X_raw[train_idx], y[train_idx]
            X_test_raw, y_test = X_raw[test_idx], y[test_idx]
        else:
            X_train_raw, y_train = X_raw, y
            X_test_raw, y_test = X_raw, y

        train_std = jnp.std(X_train_raw, 0) + scale_eps
        X_train_scaled = X_train_raw / train_std[jnp.newaxis, :]
        X_test_scaled = X_test_raw / train_std[jnp.newaxis, :]
        X_train = jnp.hstack((jnp.ones((len(X_train_scaled), 1)), X_train_scaled))
        X_test = jnp.hstack((jnp.ones((len(X_test_scaled), 1)), X_test_scaled))

        self.data = jnp.array(X_train, dtype=jnp.float32)
        self.labels = jnp.array(jnp.expand_dims(y_train.astype(jnp.float32), 1))
        self.X_train = self.data
        self.y_train = self.labels
        self.X_test = jnp.array(X_test, dtype=jnp.float32)
        self.y_test = jnp.array(jnp.expand_dims(y_test.astype(jnp.float32), 1))
        self.train_std = train_std

        self.num_dimensions = self.data.shape[1]
        self._prior_std_const = jnp.array(10., dtype=jnp.float32)
        self.prior_mean_const = jnp.array(0., dtype=jnp.float32)
        self.const_term = jnp.array(0.5 * jnp.log(2. * jnp.pi), dtype=jnp.float32)

    def log_prob(self, x):
        def _log_prob(x):
            features = -jnp.matmul(self.data, x.transpose())
            log_likelihood = jnp.sum(jnp.where(self.labels == 1, jax.nn.log_sigmoid(features),
                                               jax.nn.log_sigmoid(features) - features), axis=0)
            log_prior = jnp.sum(-jnp.log(self._prior_std_const) - self.const_term - 0.5 * jnp.square(
                (x - self.prior_mean_const) / self._prior_std_const), axis=1)
            return log_likelihood + log_prior

        batched = x.ndim == 2
        if not batched:
            x = x[None,]

        log_probs = _log_prob(x)

        if not batched:
            log_probs = jnp.squeeze(log_probs, axis=0)

        return log_probs


def sample_gmm(means, covs, weights, num_samples=1000, seed=0):
    """Draw ``num_samples`` theta from the GMM ``sum_k w_k N(m_k, C_k)``."""
    means = np.asarray(means, dtype=np.float64)
    covs = np.asarray(covs, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()

    rng = np.random.RandomState(seed)
    comp = rng.choice(len(weights), size=num_samples, p=weights)
    chols = np.linalg.cholesky(covs)
    z = rng.standard_normal((num_samples, means.shape[1]))
    return means[comp] + np.einsum("sij,sj->si", chols[comp], z)


def posterior_predictive_prob(X, theta_samples):
    """Monte-Carlo mean of sigmoid(x^T theta_s) over ``theta_samples`` -> (n,)."""
    X = np.asarray(X, dtype=np.float64)
    theta_samples = np.asarray(theta_samples, dtype=np.float64)
    logits = X @ theta_samples.T
    probs = 1.0 / (1.0 + np.exp(-logits))
    return probs.mean(axis=1)


def posterior_predictive_accuracy(X, y, theta_samples):
    """``(probs, acc)``; the likelihood is sigmoid(-x^T theta), so ``probs < 0.5`` predicts y=1."""
    probs = posterior_predictive_prob(X, theta_samples)
    y = np.asarray(y).reshape(-1)
    y_hat = (probs < 0.5).astype(int)
    acc = float((y_hat == y).mean())
    return probs, acc
