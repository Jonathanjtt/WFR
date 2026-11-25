"""WALNUTS-R2P (Bou-Rabee, Carpenter, Kleppe, Liu) as a sampling baseline.

A sampler, not a variational method: it returns draws.  The sampler is upstream's, vendored
unmodified under :mod:`._vendor`; this module adapts the target, counts ``lpFun`` calls
(warmup included) and seeds the numpy global stream upstream draws from.
"""

from __future__ import annotations

import numpy as np

from ._vendor import integrators as _igr
from ._vendor import walnuts_core as _wal

__all__ = ["fit", "lp_fun"]


def lp_fun(target, dim=None, counter=None, fused=False):
    """GMMVI ``Target`` -> upstream's ``lpFun(q) -> [lp, grad]`` at one point.

    ``counter`` is incremented per call.  ``fused=True`` takes value and gradient off one
    jitted ``value_and_grad`` of the target's JAX log-density (same numbers, fewer
    dispatches); it falls back to the split path when that cannot be traced.
    """
    d = int(target.dim if dim is None else dim)

    logp = getattr(target, "_logp", None) if fused else None
    _value_and_grad = None
    if logp is not None:
        import jax
        import jax.numpy as jnp

        try:
            _value_and_grad = jax.jit(jax.value_and_grad(
                lambda z: jnp.reshape(logp(jnp.reshape(z, (1, d))), ())))
            _value_and_grad(jnp.zeros(d))
        except Exception:
            _value_and_grad = None

    if _value_and_grad is not None:

        def lpFun(q):
            lp, g = _value_and_grad(jnp.asarray(q, dtype=float).reshape(d))
            if counter is not None:
                counter[0] += 1
            return [float(lp), np.asarray(g, dtype=float).reshape(d)]

        return lpFun

    def lpFun(q):
        x = np.asarray(q, dtype=float).reshape(1, d)
        lp = float(np.asarray(target.log_prob(x), dtype=float).reshape(-1)[0])
        g = np.asarray(target.gradient_log_density(x), dtype=float).reshape(d)
        if counter is not None:
            counter[0] += 1
        return [lp, g]

    return lpFun


def fit(target, n_iterations=2000, warmup_iterations=1000, seed=0, q0=None,
        H0=0.2, step_size_rand_scale=0.2, delta0=0.05, max_nuts_depth=10,
        adapt_H=True, adapt_H_target=0.8,
        adapt_delta=True, adapt_delta_target=0.6, adapt_delta_quantile=0.9,
        min_c=0, max_c=10, R2P_prob0=2.0 / 3.0,
        thin=1, dim=None, store_diagnostics=True, fused=False):
    """Run one WALNUTS-R2P chain on ``target``.

    ``n_iterations`` is the total transition count, ``warmup_iterations`` of them discarded.
    ``means`` / ``covariances`` / ``weights`` come back empty: the result is sample-only.
    ``evals_cumulative[i]`` is the pair count after transition ``i`` (index 0 is ``q0``).
    """
    d = int(target.dim if dim is None else dim)
    n_iterations = int(n_iterations)
    warmup_iterations = int(warmup_iterations)
    q0 = np.zeros(d) if q0 is None else np.asarray(q0, dtype=float).reshape(d)

    counter = [0]
    warm_counter = [0]
    lpFun = lp_fun(target, dim=d, counter=counter, fused=bool(fused))

    aux = _igr.integratorAuxPar(minC=int(min_c), maxC=int(max_c), R2Pprob0=float(R2P_prob0))

    calls = [0]
    evals_cum = []

    def generated(q):
        if calls[0] == warmup_iterations:
            warm_counter[0] = counter[0]
        calls[0] += 1
        evals_cum.append(counter[0])
        return np.asarray(q, dtype=float)

    state = np.random.get_state()
    try:
        np.random.seed(int(seed))
        samples, diagnostics = _wal.WALNUTS(
            lpFun, q0=q0, generated=generated, integrator=_igr.adaptLeapFrogR2P,
            H0=float(H0), stepSizeRandScale=float(step_size_rand_scale),
            delta0=float(delta0), numIter=n_iterations, warmupIter=warmup_iterations,
            M=int(max_nuts_depth), igrAux=aux,
            adaptH=bool(adapt_H), adaptHtarget=float(adapt_H_target),
            adaptDelta=bool(adapt_delta), adaptDeltaTarget=float(adapt_delta_target),
            adaptDeltaQuantile=float(adapt_delta_quantile))
    finally:
        np.random.set_state(state)

    draws = np.asarray(samples, dtype=float).T[warmup_iterations + 1:]
    thin = max(1, int(thin))
    if thin > 1:
        draws = draws[::thin]

    n_warm = int(warm_counter[0])
    return {
        "draws": draws,
        "means": np.zeros((0, d)), "covariances": np.zeros((0, d, d)),
        "weights": np.zeros(0),
        "q0": q0,
        "n_target_evals": int(counter[0]),
        "n_target_evals_warmup": n_warm,
        "n_target_evals_sampling": int(counter[0]) - n_warm,
        "n_integrator_evals": float(diagnostics[:, 6].sum() + diagnostics[:, 7].sum()),
        "H_final": float(diagnostics[-1, 15]),
        "delta_final": float(diagnostics[-1, 18]),
        "mean_tree_depth": float(diagnostics[:, 1].mean()),
        "mean_energy_error": float(diagnostics[:, 17].mean()),
        "n_draws": int(len(draws)),
        "evals_cumulative": np.asarray(evals_cum, dtype=np.int64),
        "thin": thin,
        "n_iterations": n_iterations,
        **({"diagnostics": np.asarray(diagnostics, dtype=float)} if store_diagnostics else {}),
    }


def draws_at_budget(res, budget, include_warmup=True):
    """The draws produced by the time the chain had spent ``budget`` value-gradient pairs.

    ``include_warmup=True`` charges the adaptation to the budget.  Returns ``(draws, info)``;
    ``draws`` is empty when the budget ran out during warmup.
    """
    cum = np.asarray(res["evals_cumulative"], dtype=np.int64)
    warm = int(res["n_iterations"]) - int(res["n_draws"]) * int(res.get("thin", 1))
    budget = int(budget)

    last = int(np.searchsorted(cum, budget, side="right")) - 1
    spent = int(cum[last]) if last >= 0 else 0

    n_kept = max(0, last - warm)
    draws = np.asarray(res["draws"])[:n_kept] if n_kept else np.asarray(res["draws"])[:0]
    return draws, dict(
        budget=budget, pairs_spent=spent, transitions=max(0, last),
        n_draws=len(draws), warmup_iterations=warm,
        warmup_pairs=int(res["n_target_evals_warmup"]),
        exhausted_by_warmup=bool(last < warm),
        budget_exceeds_chain=bool(int(res["n_target_evals"]) <= budget),
    )
