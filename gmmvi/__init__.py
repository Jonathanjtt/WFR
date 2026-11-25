"""GMMVI: GMM variational inference benchmarks (WFR / IBW / GMBBVI / SEMTRUX).

Package layout::

    gmmvi/metrics.py       post-hoc evaluation metrics shared by every benchmark
    gmmvi/algorithms/      the four optimisers + the sampler baseline
        gmbbvi/            GMBBVI baseline
        ibw/               IBW (inverse Bures-Wasserstein)
        semtrux/           SEMTRUX baseline
        walnuts/           WALNUTS sampler baseline (vendored upstream code)
        WFR/               WFR driver + its W->FR schedule
    gmmvi/jaxvi/           shared JAX helpers (target resolution, mixture log-prob, KL traces)
    gmmvi/targets/         target distributions + their data files
    gmmvi/benchmark.py     shared init / method loops / result IO
    gmmvi/plotting.py      reusable figure code

Nothing is re-exported at the top level on purpose: importing this package must not pull in
jax, so ``gmmvi.metrics`` stays usable in a plain-numpy test run.
"""
