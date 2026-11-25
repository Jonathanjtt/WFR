"""Darcy-flow inverse problem with x-symmetrised observations (bimodal posterior).

Plain-JAX port of ``GMBBVI-Darcy-Symmetry.jl`` from PKU-CMEGroup/InverseProblems.jl.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import cg

__all__ = ["DarcySymmetry"]


def compute_seq_pairs(n_kl):
    """First ``n_kl`` frequency pairs (l1, l2) != (0, 0), stably sorted by l1^2 + l2^2."""
    trunc = int(np.sqrt(2.0 * n_kl)) + 1
    pairs = [(l1, l2) for l1 in range(trunc + 1) for l2 in range(trunc + 1)
             if not (l1 == 0 and l2 == 0)]
    mag = np.array([l1 * l1 + l2 * l2 for l1, l2 in pairs])
    order = np.argsort(mag, kind="stable")
    return np.array(pairs)[order[:n_kl]]


def _kl_basis(xx, pairs, tau, d_kl):
    """sqrt(lambda_l) * phi_l on the tensor grid ``xx`` -> (N_KL, N, N), indexed [l, ix, iy]."""
    X = xx[:, None]
    Y = xx[None, :]
    out = []
    for l1, l2 in pairs:
        lam = (np.pi ** 2 * (l1 ** 2 + l2 ** 2) + tau ** 2) ** (-d_kl)
        if l1 == 0 and l2 == 0:
            phi = np.ones((xx.size, xx.size))
        elif l1 == 0:
            phi = np.sqrt(2.0) * np.cos(np.pi * l2 * Y) * np.ones_like(X)
        elif l2 == 0:
            phi = np.sqrt(2.0) * np.cos(np.pi * l1 * X) * np.ones_like(Y)
        else:
            phi = 2.0 * np.cos(np.pi * l1 * X) * np.cos(np.pi * l2 * Y)
        out.append(np.sqrt(lam) * phi)
    return np.stack(out, axis=0)


def _source_2d(xx):
    """f(x, y) = f(y): 1000 / 2000 / 3000 on y <= 4/6 / (4/6, 5/6] / > 5/6  -> (N, N), [ix, iy]."""
    y = xx[None, :]
    f = np.where(y <= 4.0 / 6.0, 1000.0, np.where(y <= 5.0 / 6.0, 2000.0, 3000.0))
    return np.broadcast_to(f, (xx.size, xx.size)).copy()


class DarcySymmetry:
    """Symmetric-observation Darcy posterior as batched, jitted JAX callables.

    ``sigma0`` is the inference prior sd; ``sigma0_truth`` (default ``sigma0``) the sd of the truth.
    """

    def __init__(self, N=81, L=1.0, N_KL=64, N_theta=64, obs_dnx=5, obs_dny=5,
                 d_kl=2.0, tau=3.0, sigma0=5.0, sigma_eta=0.25, sigma0_truth=None,
                 seed_truth=111, seed_noise=123, refine_factor=3,
                 cg_tol=1e-9, cg_maxiter=4000, batch_chunk=1024):
        self.N, self.L, self.N_KL, self.N_theta = N, float(L), N_KL, N_theta
        self.d_kl, self.tau, self.sigma0, self.sigma_eta = d_kl, tau, sigma0, sigma_eta
        self.dim = N_theta
        self.cg_tol, self.cg_maxiter = float(cg_tol), int(cg_maxiter)
        self.batch_chunk = int(batch_chunk)

        self.xx = np.linspace(0.0, L, N)
        self.dx = self.xx[1] - self.xx[0]
        self.pairs = compute_seq_pairs(N_KL)
        self.basis = _kl_basis(self.xx, self.pairs, tau, d_kl)
        self.f_2d = _source_2d(self.xx)
        self.mirror_signs = np.where(self.pairs[:, 0] % 2 == 0, 1.0, -1.0)

        self.x_locs = np.arange(obs_dnx, N - obs_dnx - 1 + 1, obs_dnx)
        self.y_locs = np.arange(obs_dny, N - obs_dny - 1 + 1, obs_dny)
        nxo = len(self.x_locs)
        self._n_keep = (nxo + 1) // 2
        self.n_obs = self._n_keep * len(self.y_locs)

        rng = np.random.RandomState(seed_truth)
        self.sigma0_truth = sigma0 if sigma0_truth is None else float(sigma0_truth)
        self.theta_ref = rng.normal(0.0, self.sigma0_truth, N_KL)
        N_ref = refine_factor * (N - 1) + 1
        xx_ref = np.linspace(0.0, L, N_ref)
        basis_ref = _kl_basis(xx_ref, self.pairs, tau, d_kl)
        self.logkappa_ref_2d = np.einsum("l,lxy->xy", self.theta_ref, basis_ref)
        x_locs_ref = self.x_locs * refine_factor
        y_locs_ref = self.y_locs * refine_factor

        h_ref = np.asarray(self._solve(jnp.exp(jnp.asarray(self.logkappa_ref_2d)),
                                       jnp.asarray(_source_2d(xx_ref)),
                                       xx_ref[1] - xx_ref[0], tol=1e-12, maxiter=40_000))
        self.y_noiseless = np.asarray(self._symmetrise(h_ref[np.ix_(x_locs_ref, y_locs_ref)]))
        self.y_obs = self.y_noiseless + np.random.RandomState(seed_noise).normal(
            0.0, sigma_eta, self.n_obs)

        self._basis_j = jnp.asarray(self.basis)
        self._f_j = jnp.asarray(self.f_2d)
        self._y_j = jnp.asarray(self.y_obs)
        self._phi_single = jax.jit(self._phi_r)
        self._grad_single = jax.jit(jax.grad(self._phi_r))
        self._logp_batch = jax.jit(self._chunked(jax.vmap(lambda t: -self._phi_r(t))))
        self._score_batch = jax.jit(self._chunked(jax.vmap(jax.grad(lambda t: -self._phi_r(t)))))

    @staticmethod
    def _apply_op(h_int, kappa_2d, dx):
        """Matrix-free 5-point operator of ``solve_Darcy_2D`` on the interior (N-2, N-2)."""
        h = jnp.pad(h_int, 1)
        kc = kappa_2d[1:-1, 1:-1]
        kxp = 0.5 * (kc + kappa_2d[2:, 1:-1])
        kxm = 0.5 * (kc + kappa_2d[:-2, 1:-1])
        kyp = 0.5 * (kc + kappa_2d[1:-1, 2:])
        kym = 0.5 * (kc + kappa_2d[1:-1, :-2])
        hc = h[1:-1, 1:-1]
        return (kxp * (hc - h[2:, 1:-1]) + kxm * (hc - h[:-2, 1:-1])
                + kyp * (hc - h[1:-1, 2:]) + kym * (hc - h[1:-1, :-2])) / dx ** 2

    def _solve(self, kappa_2d, f_2d, dx, tol=None, maxiter=None):
        """CG solve of the SPD interior system, Jacobi-preconditioned -> interior h (N-2, N-2)."""
        kc = kappa_2d[1:-1, 1:-1]
        diag = (0.5 * (kc + kappa_2d[2:, 1:-1]) + 0.5 * (kc + kappa_2d[:-2, 1:-1])
                + 0.5 * (kc + kappa_2d[1:-1, 2:]) + 0.5 * (kc + kappa_2d[1:-1, :-2])) / dx ** 2
        b = f_2d[1:-1, 1:-1]
        A = partial(self._apply_op, kappa_2d=kappa_2d, dx=dx)
        h, _ = cg(A, b, tol=self.cg_tol if tol is None else tol,
                  maxiter=self.cg_maxiter if maxiter is None else maxiter,
                  M=lambda r: r / diag)
        return h

    def _symmetrise(self, obs_2d):
        """Average each sensor column with its x-mirror partner -> (n_keep, n_y) flattened."""
        top = obs_2d[: self._n_keep, :]
        bot = obs_2d[::-1][: self._n_keep, :]
        return (0.5 * (top + bot)).ravel()

    def _forward(self, theta):
        """G(theta): symmetrised sensor pressures, (N_theta,) -> (n_obs,).  Pure JAX."""
        kappa = jnp.exp(jnp.einsum("l,lxy->xy", theta, self._basis_j))
        h_int = self._solve(kappa, self._f_j, self.dx)
        h = jnp.pad(h_int, 1)
        return self._symmetrise(h[self.x_locs][:, self.y_locs])

    def _phi_r(self, theta):
        """Phi_R(theta) = ||y - G||^2 / (2 s_eta^2) + ||theta||^2 / (2 s_0^2)  (scalar, JAX)."""
        r = (self._y_j - self._forward(theta)) / self.sigma_eta
        return 0.5 * jnp.sum(r ** 2) + 0.5 * jnp.sum((theta / self.sigma0) ** 2)

    def _chunked(self, fn_vmapped):
        """Apply a vmapped (B, d) function in ``batch_chunk`` slices via lax.map (bounds memory)."""
        chunk = self.batch_chunk

        def apply(X):
            B = X.shape[0]
            if B <= chunk:
                return fn_vmapped(X)
            n_pad = (-B) % chunk
            Xp = jnp.concatenate([X, jnp.broadcast_to(X[:1], (n_pad,) + X.shape[1:])], 0)
            out = jax.lax.map(fn_vmapped, Xp.reshape(-1, chunk, X.shape[1]))
            return out.reshape((B + n_pad,) + out.shape[2:])[:B]

        return apply

    def log_prob(self, X):
        """Unnormalised log-posterior, batched: (B, d) -> (B,).  JAX in, JAX out."""
        return self._logp_batch(jnp.asarray(X, dtype=jnp.float64))

    def score(self, X):
        """grad log pi = -grad Phi_R, batched: (B, d) -> (B, d)."""
        return self._score_batch(jnp.asarray(X, dtype=jnp.float64))

    def phi_r(self, theta):
        """Scalar potential at one point (numpy float)."""
        return float(self._phi_single(jnp.asarray(theta, dtype=jnp.float64)))

    def forward(self, theta):
        """Symmetrised observation vector G(theta) -> numpy (n_obs,)."""
        return np.asarray(self._forward(jnp.asarray(theta, dtype=jnp.float64)))

    def log_kappa_2d(self, theta):
        """log-permeability field of ``theta`` on the standard mesh -> numpy (N, N), [ix, iy]."""
        return np.einsum("l,lxy->xy", np.asarray(theta), self.basis)

    def pressure_2d(self, theta):
        """Full pressure field of ``theta`` (boundary ring included) -> numpy (N, N)."""
        kappa = jnp.exp(jnp.asarray(self.log_kappa_2d(theta)))
        return np.asarray(jnp.pad(self._solve(kappa, self._f_j, self.dx), 1))

    def mirror(self, theta):
        """The exact analytic mirror ``s * theta`` (``pi(mirror(theta)) = pi(theta)``)."""
        return np.asarray(theta) * self.mirror_signs
