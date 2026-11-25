"""Minimal base class for the fab-jax-style targets (MoG40, breast_cancer, nice)."""
import abc


class Target(abc.ABC):
    def __init__(self, dim: int, log_Z=None, can_sample: bool = False) -> None:
        self.dim = dim
        self.log_Z = log_Z
        self.can_sample = can_sample

    @abc.abstractmethod
    def log_prob(self, x):
        ...
