"""Hyperparameter-tuning visualizations for the SVM-LSM grid / Nelder-Mead search.

The CV score is encoded as COLOUR, and the whole (C, gamma) plane is filled with the score of
its NEAREST evaluated point ("Voronoi" logic, via scipy NearestNDInterpolator) so the irregular
point set (log grid + Nelder-Mead refine points) reads as a continuous map. Two renderings:
  PNG  -> 2D filled Voronoi heatmap over (log C, log gamma), points drawn ON TOP (no occlusion)
  HTML -> interactive 3D Voronoi response surface (z = CV score), semi-transparent so the points
          (lifted slightly above it) stay visible while rotating.
Marker encodes provenance: grid = circle, refine (Nelder-Mead) = triangle; the best point is a red
ring/marker. The evaluated points always sit on top of / above the fill so none are hidden.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import NearestNDInterpolator

_MARKERS = [("o", "grid"), ("^", "refine")]
_PLOTLY_MARKER = {"o": "circle", "^": "triangle-up"}
_PLOTLY_MARKER_3D = {"o": "circle", "^": "diamond"}


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
    best_i = int(np.argmax(score)) if task.greater_is_better else int(np.argmin(score))

    grp = {
        "o": is_grid,                 # numeric grid points
        "^": ~is_grid,                # Nelder-Mead refine points
    }

    metric = task.scorer.__name__

    u, v = np.log10(C), np.log10(G)
    ulab, vlab = "log10(C)", "log10(gamma)"
    title = f"{task.name} hyperparameter search"
    png = output_folder / f"hyperparameter_tuning{suffix}.png"
    html = output_folder / f"hyperparameter_tuning{suffix}.html"

    _fill_png(u, v, score, grp, best_i, metric, title, ulab, vlab, png)
    _surface_html(u, v, score, grp, best_i, metric, title, ulab, vlab, html)
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


def _hover(score, metric):
    out = []
    for i, s in enumerate(score):
        t = f"{metric}={s:.4f}"
        out.append(t)
    return out


# --- 2D filled Voronoi heatmap (PNG), points drawn on top -----------------------------------
def _fill_png(u, v, score, grp, best_i, metric, title, ulab, vlab, path):
    mesh_u, mesh_v, _, _, Z = _voronoi_field(u, v, score, res=300)
    vmin, vmax = float(score.min()), float(score.max())
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(Z, extent=[mesh_u[0], mesh_u[-1], mesh_v[0], mesh_v[-1]], origin="lower",
                   aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    for mk, _name in _MARKERS:
        m = grp[mk]
        if m.any():
            ax.scatter(u[m], v[m], c=score[m], cmap="viridis", vmin=vmin, vmax=vmax,
                       s=100, marker=mk, edgecolors="white", linewidths=0.9)
    ax.scatter(u[best_i], v[best_i], facecolors="none", edgecolors="red", s=340, linewidths=2.0)
    ax.set_xlabel(ulab); ax.set_ylabel(vlab); ax.set_title(title)
    ax.legend(handles=_legend_handles(grp), loc="best")
    fig.colorbar(im, ax=ax, label=metric)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _surface_html(u, v, score, grp, best_i, metric, title, ulab, vlab, path):
    import plotly.graph_objects as go
    mesh_u, mesh_v, _, _, Zs = _voronoi_field(u, v, score, res=140)
    vmin, vmax = float(score.min()), float(score.max())
    # Lift markers slightly above the surface so they are not occluded by it while rotating.
    lift = 0.02 * (vmax - vmin) if vmax > vmin else 0.02
    fig = go.Figure()
    fig.add_trace(go.Surface(x=mesh_u, y=mesh_v, z=Zs, surfacecolor=Zs, colorscale="Viridis",
                             cmin=vmin, cmax=vmax, colorbar=dict(title=metric), opacity=0.7))
    hov = _hover(score, metric)
    for mk, name in _MARKERS:
        m = grp[mk]
        if m.any():
            fig.add_trace(go.Scatter3d(x=u[m], y=v[m], z=score[m] + lift, mode="markers", name=name,
                          text=[t for t, k in zip(hov, m) if k],
                          marker=dict(size=5, symbol=_PLOTLY_MARKER_3D[mk], color=score[m],
                                      colorscale="Viridis", cmin=vmin, cmax=vmax,
                                      line=dict(width=1, color="white"))))
    fig.add_trace(go.Scatter3d(x=[u[best_i]], y=[v[best_i]], z=[score[best_i] + lift],
                               mode="markers", name="best", marker=dict(size=9, color="red", symbol="x")))
    fig.update_layout(title=title,
                      scene=dict(xaxis_title=ulab, yaxis_title=vlab, zaxis_title=metric))
    fig.write_html(str(path), include_plotlyjs=True)
