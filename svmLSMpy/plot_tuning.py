"""Hyperparameter-tuning visualizations for the SVM-LSM grid / Nelder-Mead search.

The CV score is always encoded as COLOUR (the extra dimension), so no axis is spent on it:
  SVC -> 2D  (log10 C, log10 gamma)            colour = CV score
  SVR -> 3D  (log10 C, log10 gamma, epsilon)   colour = CV score
Grid points and Nelder-Mead refine points are drawn with different markers; the best point
is highlighted. Each figure is written as a static PNG (matplotlib) and a self-contained
interactive HTML (plotly), alongside the other output-folder artifacts.
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_tuning(evaluations, task, output_folder, suffix=""):
    """Render the tuning surface from the search's `evaluations`
    ({**params, 'avg_score', 'source'} dicts). String gammas ('scale'/'auto') have no
    numeric axis position and are dropped from the plot."""
    rows = [e for e in evaluations if not isinstance(e.get("gamma"), str)]
    if not rows:
        print("No numeric (C, gamma) evaluations to plot; skipping tuning plot.")
        return

    logC = np.array([np.log10(float(e["C"])) for e in rows])
    logG = np.array([np.log10(float(e["gamma"])) for e in rows])
    score = np.array([float(e["avg_score"]) for e in rows])
    is_grid = np.array([e.get("source", "grid") == "grid" for e in rows])
    best_i = int(np.argmax(score)) if task.greater_is_better else int(np.argmin(score))

    metric = task.scorer.__name__
    title = f"{task.name} hyperparameter search"
    png_path = output_folder / f"hyperparameter_tuning{suffix}.png"
    html_path = output_folder / f"hyperparameter_tuning{suffix}.html"

    if task.kind == "svr":
        eps = np.array([float(e.get("epsilon", 0.1)) for e in rows])
        _svr_png(logC, logG, eps, score, is_grid, best_i, metric, title, png_path)
        _svr_html(logC, logG, eps, score, is_grid, best_i, metric, title, html_path)
    else:
        _svc_png(logC, logG, score, is_grid, best_i, metric, title, png_path)
        _svc_html(logC, logG, score, is_grid, best_i, metric, title, html_path)

    print(f"Hyperparameter tuning plots saved to {png_path} and {html_path}")


def _svc_png(logC, logG, score, is_grid, best_i, metric, title, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    vmin, vmax = float(score.min()), float(score.max())
    p = ax.scatter(logC[is_grid], logG[is_grid], c=score[is_grid], cmap="viridis",
                   vmin=vmin, vmax=vmax, s=90, marker="o", edgecolors="k",
                   linewidths=0.3, label="grid")
    if (~is_grid).any():
        ax.scatter(logC[~is_grid], logG[~is_grid], c=score[~is_grid], cmap="viridis",
                   vmin=vmin, vmax=vmax, s=110, marker="^", edgecolors="k",
                   linewidths=0.3, label="refine")
    ax.scatter(logC[best_i], logG[best_i], facecolors="none", edgecolors="red",
               s=320, linewidths=2.0, label="best")
    ax.set_xlabel("log10(C)")
    ax.set_ylabel("log10(gamma)")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.colorbar(p, ax=ax, label=metric)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _svr_png(logC, logG, eps, score, is_grid, best_i, metric, title, path):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    vmin, vmax = float(score.min()), float(score.max())
    p = ax.scatter(logC[is_grid], logG[is_grid], eps[is_grid], c=score[is_grid],
                   cmap="viridis", vmin=vmin, vmax=vmax, s=45, marker="o", label="grid")
    if (~is_grid).any():
        ax.scatter(logC[~is_grid], logG[~is_grid], eps[~is_grid], c=score[~is_grid],
                   cmap="viridis", vmin=vmin, vmax=vmax, s=55, marker="^", label="refine")
    ax.scatter(logC[best_i], logG[best_i], eps[best_i], c="red", marker="*",
               s=320, label="best")
    ax.set_xlabel("log10(C)")
    ax.set_ylabel("log10(gamma)")
    ax.set_zlabel("epsilon")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.colorbar(p, ax=ax, shrink=0.6, pad=0.1, label=metric)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _svc_html(logC, logG, score, is_grid, best_i, metric, title, path):
    import plotly.graph_objects as go
    fig = go.Figure()
    _add_scatter2d(fig, go, logC[is_grid], logG[is_grid], score[is_grid], "grid", "circle", metric, True)
    if (~is_grid).any():
        _add_scatter2d(fig, go, logC[~is_grid], logG[~is_grid], score[~is_grid], "refine", "triangle-up", metric, False)
    fig.add_trace(go.Scatter(
        x=[logC[best_i]], y=[logG[best_i]], mode="markers", name="best",
        marker=dict(size=16, color="red", symbol="x")))
    fig.update_layout(title=title, xaxis_title="log10(C)", yaxis_title="log10(gamma)")
    fig.write_html(str(path), include_plotlyjs=True)


def _svr_html(logC, logG, eps, score, is_grid, best_i, metric, title, path):
    import plotly.graph_objects as go
    fig = go.Figure()
    _add_scatter3d(fig, go, logC[is_grid], logG[is_grid], eps[is_grid], score[is_grid], "grid", "circle", metric, True)
    if (~is_grid).any():
        _add_scatter3d(fig, go, logC[~is_grid], logG[~is_grid], eps[~is_grid], score[~is_grid], "refine", "diamond", metric, False)
    fig.add_trace(go.Scatter3d(
        x=[logC[best_i]], y=[logG[best_i]], z=[eps[best_i]], mode="markers", name="best",
        marker=dict(size=9, color="red", symbol="x")))
    fig.update_layout(title=title, scene=dict(
        xaxis_title="log10(C)", yaxis_title="log10(gamma)", zaxis_title="epsilon"))
    fig.write_html(str(path), include_plotlyjs=True)


def _add_scatter2d(fig, go, x, y, score, name, symbol, metric, showscale):
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers", name=name,
        text=[f"{metric}={s:.4f}" for s in score],
        marker=dict(size=11, symbol=symbol, color=score, colorscale="Viridis",
                    showscale=showscale, colorbar=dict(title=metric),
                    line=dict(width=0.5, color="black"))))


def _add_scatter3d(fig, go, x, y, z, score, name, symbol, metric, showscale):
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z, mode="markers", name=name,
        text=[f"{metric}={s:.4f}" for s in score],
        marker=dict(size=5, symbol=symbol, color=score, colorscale="Viridis",
                    showscale=showscale, colorbar=dict(title=metric))))
