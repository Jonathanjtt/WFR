import numpy as np


class Target:
    """Target pi for mixture VI: ``dim``, the score and the unnormalised ``log_prob``."""

    theta_true = None
    dim = None

    def __init__(self, grad_log_density=None, dim=None, log_density=None, theta_true=None):
        self._grad = grad_log_density
        self._logp = log_density
        if dim is not None:
            self.dim = dim
        if theta_true is not None:
            self.theta_true = np.asarray(theta_true)

    def gradient_log_density(self, x):
        return np.asarray(self._grad(x))

    def log_prob(self, x):
        x = np.asarray(x)
        lp = np.asarray(self._logp(x.reshape(-1, self.dim)))
        return lp.reshape(x.shape[:-1])

    def compute_KL(self, vgmm, noise=None, component_indices=None, B=1000):
        """Monte-Carlo (reverse) KL up to the additive log-normaliser."""
        samples = vgmm.sample(B, noise, component_indices)
        log_pi = np.asarray(self.log_prob(samples[:, None])).reshape(-1)
        log_q = np.asarray(vgmm.log_prob(samples[:, None])).reshape(-1)
        return (log_q - log_pi).mean()
