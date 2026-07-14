import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_correlation_matrix(behaviors, task, covariate_names, covariates, lesion_volumes, output_folder):
    """
    Diagnostic correlation matrix between the behaviour score and the covariates
    (including lesion volume), computed before any covariate regression. For
    multiclass SVC, the behaviour column is one-hot expanded into per-class 0/1
    indicators so every cell is a valid Pearson/point-biserial correlation.

    Returns the saved PNG path, or None if there are no covariates to check.
    """
    if covariates is None:
        return None

    data = {}
    if task.map_strategy == "ovr":
        for c in np.unique(behaviors):
            data[f"behavior_class{int(c)}"] = (behaviors == c).astype(int)
    else:
        data["behavior"] = behaviors

    for i, name in enumerate(covariate_names):
        data[name] = covariates[:, i]

    if "lesion_volume" not in data:
        data["lesion_volume"] = lesion_volumes.reshape(-1)

    df = pd.DataFrame(data)
    corr = df.corr()

    print("\nBehaviour/covariate correlation matrix:")
    print(corr.round(3))

    fig, ax = plt.subplots(figsize=(1.2 * len(corr.columns) + 2, 1.2 * len(corr.columns) + 2))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="bwr")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Behaviour / Covariate Correlation Matrix")
    fig.tight_layout()

    output_path = output_folder / "behavior_covariate_correlation_matrix.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path
