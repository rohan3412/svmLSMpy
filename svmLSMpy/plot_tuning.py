"""Hyperparameter-tuning visualizations for the SVM-LSM grid / Nelder-Mead search.

The CV score is encoded as COLOUR, and the whole (C, gamma) plane is filled with the score of
its NEAREST evaluated point ("Voronoi" logic, via scipy NearestNDInterpolator) so the irregular
point set (log grid + resolved scale/auto + Nelder-Mead refine points) reads as a continuous map:
  SVC -> 2D  filled Voronoi heatmap over (log C, log gamma)
  SVR -> 3D  Voronoi response SURFACE, height z = CV score over (log C, log gamma)
Marker encodes provenance: grid = circle, refine = triangle, scale/auto = square (labelled, drawn
at their resolved numeric gamma). The best point is highlighted. Each figure is written as a static
PNG (matplotlib) and a self-contained interactive HTML (plotly), in TWO versions - log-scaled and
linear C/gamma axes (linear compresses the low end, C spans 1e-3..1e3, but is provided on request).
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.interpolate import NearestNDInterpolator

_MARKERS = [("o", "grid"), ("s", "scale/auto"), ("^", "refine")]
_PLOTLY_MARKER = {"o": "circle", "s": "square", "^": "triangle-up"}
_PLOTLY_MARKER_3D = {"o": "circle", "s": "square", "^": "diamond"}


def plot_tuning(evaluations, task, output_folder, suffix=""):
    """Render the tuning surface from the search's `evaluations`
    ({**params, 'avg_score', 'source', 'gamma_numeric', 'gamma_special'} dicts)."""
    rows = [e for e in evaluations
            if e.get("gamma_numeric") is not None and np.isfinite(e["gamma_numeric"])]
    if not rows:
        print("No plottable (C, gamma) evaluations; skipping tuning plot.")
        return

    C = np.array([float(e["C"]) for e in rows])
    G = np.array([float(e["gamma_numeric"]) for e in rows])
    score = np.array([float(e["avg_score"]) for e in rows])
    is_grid = np.array([e.get("source", "grid") == "grid" for e in rows])
    is_special = np.array([bool(e.get("gamma_special", False)) for e in rows])
    labels = [str(e["gamma"]) if e.get("gamma_special") else "" for e in rows]
    eps = np.array([float(e.get("epsilon", 0.1)) for e in rows])
    best_i = int(np.argmax(score)) if task.greater_is_better else int(np.argmin(score))

    grp = {
        "o": is_grid & ~is_special,   # numeric grid points
        "s": is_grid & is_special,    # scale/auto (resolved to numeric gamma)
        "^": ~is_grid,                # Nelder-Mead refine points
    }

    metric = task.scorer.__name__

    for version in ("log", "linear"):
        if version == "log":
            u, v = np.log10(C), np.log10(G)
            ulab, vlab = "log10(C)", "log10(gamma)"
        else:
            u, v = C, G
            ulab, vlab = "C", "gamma"
        title = f"{task.name} hyperparameter search ({version} axes)"
        png = output_folder / f"hyperparameter_tuning{suffix}_{version}.png"
        html = output_folder / f"hyperparameter_tuning{suffix}_{version}.html"

        if task.kind == "svr":
            _surface_png(u, v, eps, score, grp, labels, best_i, metric, title, ulab, vlab, png)
            _surface_html(u, v, eps, score, grp, labels, best_i, metric, title, ulab, vlab, html)
        else:
            _fill_png(u, v, score, grp, labels, best_i, metric, title, ulab, vlab, png)
            _fill_html(u, v, score, grp, labels, best_i, metric, title, ulab, vlab, html)
        print(f"Hyperparameter tuning plots saved to {png} and {html}")


def _voronoi_field(u, v, score, res):
    """Nearest-neighbour (Voronoi) fill of the (u, v) plane by score, on a res x res mesh.
    Returns mesh_u, mesh_v (1D), UU, VV, Z (2D, shape [res, res])."""
    def bounds(a):
        lo, hi = float(np.min(a)), float(np.max(a))
        if hi == lo:
            lo, hi = lo - 0.5, hi + 0.5
        pad = 0.05 * (hi - lo)
        return lo - pad, hi + pad
    umin, umax = bounds(u)
    vmin, vmax = bounds(v)
    mesh_u = np.linspace(umin, umax, res)
    mesh_v = np.linspace(vmin, vmax, res)
    UU, VV = np.meshgrid(mesh_u, mesh_v)
    interp = NearestNDInterpolator(np.column_stack([u, v]), score)
    Z = interp(UU, VV)
    return mesh_u, mesh_v, UU, VV, Z


def _legend_handles(grp):
    h = [Line2D([0], [0], marker=mk, linestyle="none", markerfacecolor="lightgray",
                markeredgecolor="k", markersize=9, label=name)
         for mk, name in _MARKERS if grp[mk].any()]
    h.append(Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none",
                    markeredgecolor="red", markersize=12, markeredgewidth=2, label="best"))
    return h


def _hover(score, labels, eps, metric, with_eps):
    out = []
    for i, s in enumerate(score):
        t = f"{metric}={s:.4f}"
        if labels[i]:
            t += f"<br>gamma={labels[i]}"
        if with_eps:
            t += f"<br>epsilon={eps[i]:.4g}"
        out.append(t)
    return out


# --- 2D SVC: filled Voronoi heatmap ------------------------------------------------------------
def _fill_png(u, v, score, grp, labels, best_i, metric, title, ulab, vlab, path):
    mesh_u, mesh_v, _, _, Z = _voronoi_field(u, v, score, res=300)
    vmin, vmax = float(score.min()), float(score.max())
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(Z, extent=[mesh_u[0], mesh_u[-1], mesh_v[0], mesh_v[-1]], origin="lower",
                   aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    for mk, _name in _MARKERS:
        m = grp[mk]
        if m.any():
            ax.scatter(u[m], v[m], c=score[m], cmap="viridis", vmin=vmin, vmax=vmax,
                       s=100, marker=mk, edgecolors="white", linewidths=0.8)
    for i, lab in enumerate(labels):
        if lab:
            ax.annotate(lab, (u[i], v[i]), textcoords="offset points", xytext=(6, 4), fontsize=8,
                        bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.6))
    ax.scatter(u[best_i], v[best_i], facecolors="none", edgecolors="red", s=340, linewidths=2.0)
    ax.set_xlabel(ulab); ax.set_ylabel(vlab); ax.set_title(title)
    ax.legend(handles=_legend_handles(grp), loc="best")
    fig.colorbar(im, ax=ax, label=metric)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _fill_html(u, v, score, grp, labels, best_i, metric, title, ulab, vlab, path):
    import plotly.graph_objects as go
    mesh_u, mesh_v, _, _, Z = _voronoi_field(u, v, score, res=200)
    vmin, vmax = float(score.min()), float(score.max())
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=mesh_u, y=mesh_v, z=Z, colorscale="Viridis", zmin=vmin, zmax=vmax,
                             colorbar=dict(title=metric)))
    hov = _hover(score, labels, None, metric, with_eps=False)
    for mk, name in _MARKERS:
        m = grp[mk]
        if m.any():
            fig.add_trace(go.Scatter(x=u[m], y=v[m], mode="markers", name=name,
                          text=[t for t, k in zip(hov, m) if k],
                          marker=dict(size=11, symbol=_PLOTLY_MARKER[mk], color=score[m],
                                      colorscale="Viridis", cmin=vmin, cmax=vmax,
                                      line=dict(width=1, color="white"))))
    fig.add_trace(go.Scatter(x=[u[best_i]], y=[v[best_i]], mode="markers", name="best",
                             marker=dict(size=16, color="red", symbol="x")))
    fig.update_layout(title=title, xaxis_title=ulab, yaxis_title=vlab)
    fig.write_html(str(path), include_plotlyjs=True)


# --- 3D SVR: Voronoi response surface, z = score -----------------------------------------------
def _surface_png(u, v, eps, score, grp, labels, best_i, metric, title, ulab, vlab, path):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
    mesh_u, mesh_v, UU, VV, Zs = _voronoi_field(u, v, score, res=140)
    norm = Normalize(vmin=float(score.min()), vmax=float(score.max()))
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(UU, VV, Zs, facecolors=cm.viridis(norm(Zs)), linewidth=0,
                    antialiased=False, shade=False)
    for mk, _name in _MARKERS:
        m = grp[mk]
        if m.any():
            ax.scatter(u[m], v[m], score[m], color=cm.viridis(norm(score[m])), marker=mk,
                       s=55, edgecolors="white", linewidths=0.6, depthshade=False)
    for i, lab in enumerate(labels):
        if lab:
            ax.text(u[i], v[i], score[i], lab, fontsize=8)
    ax.scatter(u[best_i], v[best_i], score[best_i], c="red", marker="*", s=320, depthshade=False)
    ax.set_xlabel(ulab); ax.set_ylabel(vlab); ax.set_zlabel(metric)
    ax.set_title(f"{title}\n(best epsilon={eps[best_i]:.4g})")
    sm = cm.ScalarMappable(norm=norm, cmap="viridis"); sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1, label=metric)
    ax.legend(handles=_legend_handles(grp), loc="best")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _surface_html(u, v, eps, score, grp, labels, best_i, metric, title, ulab, vlab, path):
    import plotly.graph_objects as go
    mesh_u, mesh_v, _, _, Zs = _voronoi_field(u, v, score, res=140)
    vmin, vmax = float(score.min()), float(score.max())
    fig = go.Figure()
    fig.add_trace(go.Surface(x=mesh_u, y=mesh_v, z=Zs, surfacecolor=Zs, colorscale="Viridis",
                             cmin=vmin, cmax=vmax, colorbar=dict(title=metric), opacity=0.9))
    hov = _hover(score, labels, eps, metric, with_eps=True)
    for mk, name in _MARKERS:
        m = grp[mk]
        if m.any():
            fig.add_trace(go.Scatter3d(x=u[m], y=v[m], z=score[m], mode="markers", name=name,
                          text=[t for t, k in zip(hov, m) if k],
                          marker=dict(size=5, symbol=_PLOTLY_MARKER_3D[mk], color=score[m],
                                      colorscale="Viridis", cmin=vmin, cmax=vmax,
                                      line=dict(width=1, color="white"))))
    fig.add_trace(go.Scatter3d(x=[u[best_i]], y=[v[best_i]], z=[score[best_i]], mode="markers",
                               name="best", marker=dict(size=9, color="red", symbol="x")))
    fig.update_layout(title=f"{title} (best epsilon={eps[best_i]:.4g})",
                      scene=dict(xaxis_title=ulab, yaxis_title=vlab, zaxis_title=metric))
    fig.write_html(str(path), include_plotlyjs=True)
