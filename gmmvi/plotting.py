"""Reusable figures and tables for the benchmarks; every function takes computed numbers and returns Axes or a string."""

from __future__ import annotations

import numpy as np

__all__ = ["LABELS", "COLORS", "METRICS", "trace_panel", "density_panel", "density_row",
           "summary_table", "metric_bars", "method_color"]

LABELS = {"gmbbvi": "GMBBVI", "semtrux": "SEMTRUX", "ibw": "IBW", "WFR": "WFR",
          "walnuts": "WALNUTS", "walnuts_long": "WALNUTS (long)",
          "WFR_W_only": "W-only", "WFR_FR_only": "FR-only"}


COLORS = {"ibw": "tab:blue", "WFR": "tab:red", "gmbbvi": "tab:orange", "semtrux": "tab:green",
          "walnuts": "tab:purple", "walnuts_long": "tab:brown",
          "WFR_W_only": "#7f0000", "WFR_FR_only": "#ff8080"}


def _label(method):
    return LABELS.get(method, method)


def method_color(method, index=0):
    """The pinned :data:`COLORS` entry for ``method``, else the ``index``-th cycle colour."""
    if method in COLORS:
        return COLORS[method]
    import matplotlib.pyplot as plt

    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color") or [None]
    return cycle[index % len(cycle)]


class _Metric:
    """How one metric column is typeset: header, ``better`` ("up" / "down" / None), its ``std`` key and ``fmt``."""

    __slots__ = ("header", "better", "std", "fmt")

    def __init__(self, header, better, std=None, fmt=None):
        self.header = header
        self.better = better
        self.std = std
        self.fmt = fmt


METRICS = {
    "emc": _Metric("EMC", "up", std="emc_std"),
    "ess": _Metric(r"ESS / $N$", "up", std="ess_std"),
    "neg_elbo": _Metric("Negative ELBO", "down", std="neg_elbo_std"),
    "reverse_kl": _Metric("Reverse KL", "down", std="reverse_kl_std"),
    "kl_shared": _Metric("Reverse KL (shared)", "down", std="kl_shared_std"),
    "reverse_kl_shared": _Metric("Reverse KL (+const)", "down", std="reverse_kl_shared_std"),
    "neg_elbo_shared": _Metric(r"$-$ELBO (+const)", "down", std="neg_elbo_shared_std"),
    "ess_mcstd": _Metric("ESS metric-seed sd", None),
    "entropic_w2": _Metric(r"Entropic $W_2$", "down", std="entropic_w2_std"),
    "mmd": _Metric("MMD", "down", std="mmd_std"),
    "tv": _Metric("TV", "down", std="tv_std"),
    "coverage": _Metric("Mode coverage", "up", fmt="{:.0f}", std="coverage_std"),
    "modes": _Metric("Modes found", "up", fmt="{:.1f}", std="modes_std"),
    "weight_entropy": _Metric("Weight entropy", "up", std="weight_entropy_std"),
    "test_acc": _Metric("Test accuracy", "up", std="test_acc_std"),
    "runtime_s": _Metric("Runtime (s)", "down", fmt="{:.1f}"),
    "n_target_evals": _Metric("Target evals", "down", fmt="{:.0f}"),
    "n_grad_evals": _Metric("Target value-gradient pair evaluations", "down", fmt="{:.0f}"),
}


def _spread(Y, spread="std", ddof=1):
    """Column-wise ``(mean, lower, upper)`` of a ``(n_seeds, n_points)`` block, skipping non-finite entries."""
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[None, :]
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(np.where(np.isfinite(Y), Y, np.nan), axis=0)
        n = np.isfinite(Y).sum(axis=0)
        if spread == "minmax":
            lo = np.nanmin(np.where(np.isfinite(Y), Y, np.nan), axis=0)
            hi = np.nanmax(np.where(np.isfinite(Y), Y, np.nan), axis=0)
            return mean, lo, hi
        sd = np.nanstd(np.where(np.isfinite(Y), Y, np.nan), axis=0, ddof=ddof)
        sd = np.where(n > 1, sd, 0.0)
        if spread == "sem":
            sd = sd / np.sqrt(np.maximum(n, 1))
    return mean, mean - sd, mean + sd


def trace_panel(traces, iters=None, *, ax=None, ylabel="", title="", logy=True,
                figsize=(4.2, 3.4), marker="-o", ms=3, legend=True,
                spread="std", ddof=1, band_alpha=0.18):
    """One metric vs iteration, one line per method; a 2-D ``(n_seeds, n_points)`` block draws a mean with a spread band.

    ``logy``: ``True``, ``False``, ``"symlog"`` or ``"auto"`` (log when every point is positive, else symlog).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    allv = []
    for i, (method, tr) in enumerate(traces.items()):
        x, y = tr if isinstance(tr, tuple) else (iters, tr)
        y = np.asarray(y, float)
        multi = y.ndim == 2 and y.shape[0] > 1
        mean, lo, hi = _spread(y, spread, ddof) if y.ndim == 2 else (y, None, None)
        x = np.arange(mean.size) if x is None else np.asarray(x)
        (line,) = ax.plot(x, mean, marker, ms=ms, label=_label(method),
                          color=method_color(method, i))
        if multi:
            ax.fill_between(x, lo, hi, color=line.get_color(), alpha=band_alpha, linewidth=0)
            allv.append(np.concatenate([lo[np.isfinite(lo)], hi[np.isfinite(hi)]]))
        allv.append(mean[np.isfinite(mean)])
    allv = np.concatenate([a for a in allv if a.size]) if allv else np.array([])
    if logy in ("auto", "symlog") and allv.size:
        vmin, vmax = float(allv.min()), float(allv.max())
        if logy == "auto" and vmin > 0:
            ax.set_yscale("log")
            ax.set_ylim(vmin / 1.5, vmax * 1.5)
        else:
            pos = allv[allv > 0]
            lt = max(float(pos.min()) if pos.size else 1e-3, 1e-9)
            ax.set_yscale("symlog", linthresh=lt)
            ax.set_ylim((vmin * 1.5 if vmin < 0 else -lt), vmax * 1.5)
    elif logy is True:
        ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    if legend:
        ax.legend(fontsize=8)
    return ax


def _contour_panel(spec, ax, means, covs, *, reference, cmap, n_sigma, levels,
                   init_means=None, init_covs=None):
    """Target log-density contours, mode means as crosses, fitted components as ``n_sigma`` ellipses, init in red."""
    from matplotlib.patches import Ellipse

    X, Y = (np.asarray(spec.grid_pts)[:, i].reshape(spec.grid_shape) for i in (0, 1))
    Z = np.asarray(spec.log_prob_np(spec.grid_pts), float).reshape(spec.grid_shape)
    ax.set_facecolor("white")
    ax.contour(X, Y, Z, levels=levels, cmap=cmap, linewidths=0.5)
    if spec.mode_means is not None:
        mm = np.asarray(spec.mode_means)
        ax.scatter(mm[:, 0], mm[:, 1], marker="x", s=22, c="black", linewidths=0.9, zorder=3)
    if not reference and means is not None:
        for m, C in zip(np.asarray(means), np.asarray(covs)):
            lam, V = np.linalg.eigh(C)
            angle = np.degrees(np.arctan2(V[1, 0], V[0, 0]))
            w, h = 2.0 * n_sigma * np.sqrt(np.maximum(lam, 0.0))
            ax.add_patch(Ellipse(m, w, h, angle=angle, facecolor="tab:blue", alpha=0.30,
                                 edgecolor="tab:blue", lw=0.8, zorder=2))
    if init_means is not None:
        im = np.asarray(init_means, float)
        if init_covs is not None:
            for m, C in zip(im, np.asarray(init_covs, float)):
                lam, V = np.linalg.eigh(C)
                angle = np.degrees(np.arctan2(V[1, 0], V[0, 0]))
                w, h = 2.0 * n_sigma * np.sqrt(np.maximum(lam, 0.0))
                ax.add_patch(Ellipse(m, w, h, angle=angle, facecolor="none",
                                     edgecolor="tab:red", lw=0.7, alpha=0.75, zorder=4))
        ax.scatter(im[:, 0], im[:, 1], marker="o", s=7, c="tab:red", linewidths=0,
                   alpha=0.9, zorder=5, label=f"init (K={len(im)})")
        ax.legend(loc="lower left", fontsize=5.5, frameon=True, framealpha=0.8,
                  handlelength=1.0, handletextpad=0.4, borderpad=0.25, borderaxespad=0.25)


def density_panel(spec, means=None, covs=None, weights=None, *, ax=None, title=None,
                  show_means=True, reference=False, figsize=(2.7, 2.75), cmap="viridis",
                  ylabel=None, mean_kw=None, style="image", n_sigma=3.0, levels=40,
                  init_means=None, init_covs=None):
    """The fitted (or reference) 2-D density on the target's grid; ``style`` is ``"image"`` or ``"contour"``."""
    import matplotlib.pyplot as plt

    from gmmvi import metrics

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if style == "contour":
        _contour_panel(spec, ax, means, covs, reference=reference, cmap=cmap,
                       n_sigma=n_sigma, levels=levels,
                       init_means=init_means, init_covs=init_covs)
    else:
        dens = spec.ref_density if reference else metrics.grid_density(spec.grid_pts, means,
                                                                       covs, weights)
        ax.imshow(np.asarray(dens).reshape(spec.grid_shape), origin="lower", extent=spec.lims,
                  aspect="auto", cmap=cmap)
        if show_means and not reference and means is not None:
            kw = dict(s=6, c="red")
            kw.update(mean_kw or {})
            m = np.asarray(means)
            ax.scatter(m[:, 0], m[:, 1], **kw)
    x0, x1, y0, y1 = spec.lims
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    if title:
        ax.set_title(title, fontsize=10, pad=3)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def density_row(spec, results, *, methods=None, titles=None, include_reference=True,
                panel_size=2.7, ylabel=None, snapshot=None, style="image", n_sigma=3.0,
                levels=40, init_means=None, init_covs=None):
    """Reference + one fitted-density panel per method in one row; ``snapshot`` selects a logged iteration."""
    import matplotlib.pyplot as plt

    methods = list(methods if methods is not None else results)
    n = len(methods) + int(include_reference)
    fig, axes = plt.subplots(1, n, figsize=(panel_size * n, panel_size + 0.05))
    axes = np.atleast_1d(axes)
    i = 0
    if include_reference:
        density_panel(spec, reference=True, ax=axes[0], title="reference", ylabel=ylabel,
                      style=style, n_sigma=n_sigma, levels=levels,
                      init_means=init_means, init_covs=init_covs)
        i = 1
    for ax, m in zip(axes[i:], methods):
        r = results[m]
        if snapshot is None:
            mm, cc, ww = r.means, r.covariances, r.weights
        else:
            mm, cc, ww = (r.means_log[snapshot], r.covs_log[snapshot],
                          r.weights_log[snapshot])
        t = titles.get(m) if titles else _label(m)
        density_panel(spec, mm, cc, ww, ax=ax, title=t, style=style, n_sigma=n_sigma,
                      levels=levels)
    fig.tight_layout(pad=0.3)
    return fig, axes


def summary_table(rows, *, columns=None, floatfmt="{:.4f}", stdfmt=None, width=None):
    """Render ``{method: {metric: value}}`` as a fixed-width text table (``mean +/- std`` where a ``*_std`` key exists)."""
    methods = list(rows)
    columns = list(columns if columns is not None
                   else dict.fromkeys(c for r in rows.values() for c in r))
    stdfmt = stdfmt or floatfmt

    def cell(m, c):
        v = rows[m].get(c)
        if v is None or (not isinstance(v, (int, np.integer)) and not np.isfinite(float(v))):
            return "-" if v is None else "nan"
        if isinstance(v, (int, np.integer)):
            return f"{v:d}"
        spec = METRICS.get(c)
        body = (spec.fmt if spec and spec.fmt else floatfmt).format(float(v))
        sd = rows[m].get(spec.std) if spec and spec.std else None
        if sd is not None and np.isfinite(float(sd)):
            body += " +/- " + stdfmt.format(float(sd))
        return body

    cells = {(m, c): cell(m, c) for m in methods for c in columns}
    w = max((len(_label(m)) for m in methods), default=6)
    cw = {c: max([len(c)] + [len(cells[(m, c)]) for m in methods]) + 2 if width is None
          else width for c in columns}
    head = "method".ljust(w) + "".join(c.rjust(cw[c]) for c in columns)
    out = [head, "-" * len(head)]
    for m in methods:
        out.append(_label(m).ljust(w) + "".join(cells[(m, c)].rjust(cw[c]) for c in columns))
    return "\n".join(out)


def metric_bars(rows, columns, *, methods=None, axes=None, figsize=None, spread_label=None,
                annotate_n=True, logy=False):
    """Final-value bar chart with error bars, one panel per metric; ``annotate_n`` marks means over fewer than all seeds."""
    import matplotlib.pyplot as plt

    methods = list(methods if methods is not None else rows)
    columns = list(columns)
    if axes is None:
        figsize = figsize or (3.1 * len(columns), 3.2)
        _, axes = plt.subplots(1, len(columns), figsize=figsize, squeeze=False)
        axes = list(axes[0])
    axes = list(np.atleast_1d(axes))

    colors = [method_color(m, i) for i, m in enumerate(methods)]
    for ax, c in zip(axes, columns):
        spec = METRICS.get(c)
        vals = np.array([float(rows[m].get(c, np.nan)) for m in methods], float)
        errs = np.array([float(rows[m].get(spec.std, np.nan)) if spec and spec.std else np.nan
                         for m in methods], float)
        errs = np.where(np.isfinite(errs), errs, 0.0)
        x = np.arange(len(methods))
        ax.bar(x, np.nan_to_num(vals, nan=0.0), yerr=errs, capsize=4,
               color=colors, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([_label(m) for m in methods], rotation=20, ha="right", fontsize=8)
        ax.set_title((spec.header if spec else c) +
                     ({"up": r" $\uparrow$", "down": r" $\downarrow$"}
                      .get(spec.better if spec else None, "")), fontsize=10)
        if logy:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
        if annotate_n:
            for i, m in enumerate(methods):
                n, N = rows[m].get(c + "_n"), rows[m].get("n_seeds")
                if n is not None and N and int(n) < int(N):
                    ax.annotate(f"{int(n)}/{int(N)}", (i, 0), xytext=(0, 4),
                                textcoords="offset points", ha="center", fontsize=7,
                                color="crimson")
        if spread_label:
            ax.set_xlabel(spread_label, fontsize=7)
    return axes
