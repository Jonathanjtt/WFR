"""Vendored subset of WALNUTSpy (bob-carpenter/walnuts, MIT license, (c) 2025 Bob Carpenter).

Upstream is https://github.com/bob-carpenter/walnuts, directory ``WALNUTSpy`` -- the
"flexible and highly instrumented" sampler used for the experiments in

    Nawaf Bou-Rabee, Bob Carpenter, Tore Selland Kleppe and Sifan Liu.
    The Within-Orbit Adaptive Step-Length No-U-Turn Sampler.

Only the **R2P path** is vendored, i.e. what ``WALNUTSpy/example.py`` runs:
``WALNUTS(..., integrator=adaptLeapFrogR2P)``.  The math is upstream's, byte for byte;
``tests/test_walnuts_vendor_parity.py`` re-downloads the upstream files and asserts that
the kept regions are unchanged.

What was kept
-------------
=========================  ==================================================
this package               upstream
=========================  ==================================================
``constants.py``           ``WALNUTSpy/constants.py``          (whole file)
``p2quantile.py``          ``WALNUTSpy/P2quantile.py``         (whole file)
``integrators.py``         ``WALNUTSpy/adaptiveIntegrators.py`` lines 1-46 and
                           361-476: ``integratorReturn``, ``integratorAuxPar``
                           and ``adaptLeapFrogR2P``
``walnuts_core.py``        ``WALNUTSpy/WALNUTS.py``            (whole file)
=========================  ==================================================

The four deviations, all mechanical, none numerical:

1. flat imports (``import adaptiveIntegrators as ai``) become relative ones, since
   upstream is a directory of scripts rather than a package;
2. the unused ``import matplotlib.pyplot as plt`` in ``WALNUTS.py`` is dropped, so
   importing the sampler does not pull in a plotting backend;
3. the other integrators (``fixedLeapFrog``, ``adaptLeapFrogD``, ``adaptYoshidaD``,
   ``adaptLeapFrogFlowD``, ``adaptImplicitMidpointD``, ``adaptRescaledLeapFrogD``) are
   not vendored -- so ``fixedLeapFrog``, which turns WALNUTS into plain NUTS, is NOT
   available as an ablation here;
4. consequently the ``integrator=`` default of ``WALNUTS()`` moves from
   ``ai.fixedLeapFrog`` to ``ai.adaptLeapFrogR2P``.  :mod:`gmmvi.algorithms.walnuts.sampler`
   passes it explicitly regardless.

Note the RNG: upstream draws from the numpy GLOBAL legacy stream (``np.random``), both in
``WALNUTS`` and inside ``adaptLeapFrogR2P``.  ``sampler.fit`` seeds and restores it around
the call rather than editing this code.
"""
