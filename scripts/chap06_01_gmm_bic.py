"""06_01 (実際は06_02) GMM BIC."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.colors import LogNorm
from sklearn.mixture import GaussianMixture

if TYPE_CHECKING:
    import numpy as np


def plot_gmm_bic(  # noqa: D103, PLR0913
    scatter_x: pl.Series,
    scatter_y: pl.Series,
    x_label: str,
    y_label: str,
    ax: plt.Axes,
    cont_x: pl.DataFrame = None,
    cont_y: pl.DataFrame = None,
    cont_z: pl.DataFrame = None,
) -> None:
    if cont_x is not None or cont_y is not None or cont_z is not None:
        ax.contour(
            cont_x,
            cont_y,
            cont_z,
            norm=LogNorm(vmin=1.0, vmax=7.1),
            levels=10 ** pl.linear_space(0, 0.85, 8, eager=True),
        )
    ax.scatter(scatter_x, scatter_y, c="blue")
    ax.set_xlim(-4, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def calc_bic(  # noqa: D103
    x: np.ndarray,
    n_components: int,
    covariance_type: str,
) -> float:
    model = GaussianMixture(n_components=n_components, covariance_type=covariance_type)
    model.fit(x)
    return model.bic(x)


x = pl.read_csv("dataset/sample_dataset_gmm.csv")
index = x.get_column("")
x = x.drop("")
normalized_x = x.with_columns((pl.all() - pl.all().mean()) / pl.all().std())
normalized_x_np = normalized_x.to_numpy()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# プロット
plot_gmm_bic(
    normalized_x.to_series(0),
    normalized_x.to_series(1),
    x.columns[0],
    x.columns[1],
    axes[0],
)

# GMM モデリング
num_of_gaussian_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 正規分布の数
covariance_types = ["full", "tied", "diag", "spherical"]  # 分散共分散行列の種類
# "full" : 各正規分布がそれぞれ、一般的な分散共分散行列をもつ
# "tied" : すべての正規分布が同じ一般的な分散共分散行列をもつ
# "diag" : 各正規分布がそれぞれ、共分散が全て0の分散共分散行列をもつ
# "spherical" : 各正規分布がそれぞれ、共分散が全て0で分散が同じ値の分散共分散行列をもつ

# BIC によるクラスタ数と分散共分散行列の最適化
bics = pl.DataFrame(
    {
        f"{covariance_type}": [
            calc_bic(normalized_x_np, num_of_gaussian, covariance_type)
            for num_of_gaussian in num_of_gaussian_list
        ]
        for covariance_type in covariance_types
    }
)
# BICをプロット
ax = axes[1]
colors = ["blue", "red", "green", "black"]
for bic_series, color in zip(bics.iter_columns(), colors, strict=True):
    ax.scatter(num_of_gaussian_list, bic_series, c=color, label=bic_series.name)
ax.set_xlabel("number of Gaussian")
ax.set_ylabel("BIC")
ax.legend()

# BIC が最小となる正規分布の数と分散共分散行列の種類
unpivot_bics = (
    bics.insert_column(0, pl.Series("num_of_gaussian", num_of_gaussian_list))
    .unpivot(index="num_of_gaussian", variable_name="covariance_type", value_name="bic")
    .sort("bic")
)
optimal_num_of_gaussian = unpivot_bics.item(0, "num_of_gaussian")
optimal_covariance_type = unpivot_bics.item(0, "covariance_type")
print("最適化された正規分布の数 :", optimal_num_of_gaussian)
print("最適化された分散共分散行列の種類 :", optimal_covariance_type)

model = GaussianMixture(
    n_components=optimal_num_of_gaussian, covariance_type=optimal_covariance_type
)
model.fit(normalized_x_np)

# クラスターへの割り当て
cluster_numbers = pl.Series("cluster numbers", model.predict(normalized_x_np))
index.to_frame().hstack([cluster_numbers]).write_csv(
    f"result/cluster_numbers_gmm_{optimal_num_of_gaussian}_{optimal_covariance_type}.csv",
    quote_style="never",
)
cluster_probabilities = pl.DataFrame(model.predict_proba(normalized_x_np))
cluster_probabilities.columns = [f"{i}" for i in range(cluster_probabilities.width)]
cluster_probabilities.clone().insert_column(0, index).write_csv(
    f"result/cluster_probabilities_gmm_{optimal_num_of_gaussian}_{optimal_covariance_type}.csv",
    quote_style="never",
)

# プロット
x_axis = pl.linear_space(-4.0, 3.0, 50, eager=True).rename("x")
y_axis = pl.linear_space(-3.0, 3.0, 50, eager=True).rename("y")
X = pl.DataFrame([x_axis.rename(f"x{i}") for i in range(y_axis.len())]).transpose(
    column_names=[f"x{i}" for i in range(y_axis.len())]
)
Y = pl.DataFrame([y_axis.rename(f"y{i}") for i in range(x_axis.len())])
XX = pl.concat(
    [
        X.select(pl.concat_list(pl.all()).explode().alias("x1")),
        Y.select(pl.concat_list(pl.all()).explode().alias("x2")),
    ],
    how="horizontal",
)
Z = (
    pl.DataFrame(-model.score_samples(XX.to_numpy()))
    .select(pl.all().reshape(X.shape).arr.to_struct())
    .unnest("column_0")
)
# プロット
plot_gmm_bic(
    normalized_x.to_series(0),
    normalized_x.to_series(1),
    x.columns[0],
    x.columns[1],
    axes[2],
    X,
    Y,
    Z,
)

plt.show()
