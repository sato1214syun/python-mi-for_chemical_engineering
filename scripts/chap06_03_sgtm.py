"""06_03 sparse-GTM."""

import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from dcekit.generative_model import GTM  # type: ignore[import-untyped]


def plot_gmm_bic(  # noqa: D103, PLR0913
    scatter_x: pl.Series,
    scatter_y: pl.Series,
    x_label: str,
    y_label: str,
    ax: plt.Axes,
    xlim: tuple[float, float] | None = (-1.1, 1.1),
    ylim: tuple[float, float] | None = (-1.1, 1.1),
    *,
    c: pl.Series | str = None,
    aspect: str = "equal",
) -> None:
    ax.scatter(scatter_x, scatter_y, c=c)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_aspect(aspect)


dataset = pl.read_csv("dataset/selected_descriptors_with_boiling_point.csv")
index = dataset.get_column("")

dataset = dataset.drop("")
y = dataset.to_series(0)  # 目的変数
x = dataset.drop(y.name)  # 説明変数
normalized_x = x.with_columns((pl.all() - pl.all().mean()) / pl.all().std())
normalized_x_np = normalized_x.to_numpy()

# construct SGTM model
shape_of_map = [30, 30]  # ２次元平面上のグリッド点の数 (k x k)
shape_of_rbf_centers = [5, 5]  # RBF の数 (q x q)
variance_of_rbfs = 4  # RBF の分散 (sigma^2)
lambda_in_em_algorithm = 0.001  # 正則化項 (lambda)
number_of_iterations = 300  # EM アルゴリズムにおける繰り返し回数
display_flag = True  # EM アルゴリズムにおける進捗を表示する (True) かしない (False) か
model = GTM(
    shape_of_map,
    shape_of_rbf_centers,
    variance_of_rbfs,
    lambda_in_em_algorithm,
    number_of_iterations,
    display_flag,
    sparse_flag=True,
)
model.fit(normalized_x_np)

if model.success_flag is False:
    sys.exit("SGTM model fitting failed.")

# calculate responsibilities
means, modes = model.means_modes(normalized_x_np)

means_df = pl.DataFrame(means, schema=["t1 (mean)", "t2 (mean)"])
modes_df = pl.DataFrame(modes, schema=["t1 (mode)", "t2 (mode)"])
means_df.clone().insert_column(0, index).write_csv(
    f"result/sgtm_means_{shape_of_map[0]}_{shape_of_map[1]}_{shape_of_rbf_centers[0]}_{shape_of_rbf_centers[1]}_{variance_of_rbfs}_{lambda_in_em_algorithm}_{number_of_iterations}.csv"
)
modes_df.clone().insert_column(0, index).write_csv(
    f"result/sgtm_modes_{shape_of_map[0]}_{shape_of_map[1]}_{shape_of_rbf_centers[0]}_{shape_of_rbf_centers[1]}_{variance_of_rbfs}_{lambda_in_em_algorithm}_{number_of_iterations}.csv"
)
means_x = means_df.to_series(0)
means_y = means_df.to_series(1)
modes_x = modes_df.to_series(0)
modes_y = modes_df.to_series(1)

# plot the mean of responsibilities
y_categorical = y.cast(pl.Utf8).cast(pl.Categorical).to_physical()
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
plot_gmm_bic(means_x, means_y, means_x.name, means_y.name, axes[0, 0])
plot_gmm_bic(means_x, means_y, means_x.name, means_y.name, axes[0, 1], c=y_categorical)
plot_gmm_bic(modes_x, modes_y, modes_x.name, modes_y.name, axes[1, 0])
plot_gmm_bic(modes_x, modes_y, modes_x.name, modes_y.name, axes[1, 1], c=y_categorical)

# integration of clusters based on Bayesian information criterion (BIC)
clustering_results = pl.DataFrame()
cluster_numbers = pl.DataFrame()
mixing_coef_length = len(model.mixing_coefficients.copy())
for i in range(mixing_coef_length):
    # 最尤値を取得
    likelihood = model.likelihood(normalized_x_np)
    # coef行列で0でない値の箇所を取得
    mixing_coef = pl.Series(model.mixing_coefficients)
    mixing_coef_wt_null = mixing_coef.replace(0, None)
    mixing_coef_wo_zero_length = mixing_coef_wt_null.drop_nulls().len()
    # responsibilityを再計算
    responsibilities_arg_max_in_row = pl.Series(
        f"{i}", model.responsibility(normalized_x_np).argmax(axis=1)
    )
    cluster_numbers = cluster_numbers.hstack([responsibilities_arg_max_in_row])
    # BIC計算 -2*logL(y|最尤推定値θ) + d*log(n): Lは尤度関数、dはθの次元数、nはデータ数
    bic = -2 * likelihood + mixing_coef_wo_zero_length * np.log(
        normalized_x_np.shape[0]
    )
    clustering_results = clustering_results.vstack(
        pl.DataFrame(
            [
                [
                    mixing_coef_wo_zero_length,
                    bic,
                    responsibilities_arg_max_in_row.unique().len(),
                ]
            ],
            schema=["number_of_clusters", "BIC", "max_responsibility"],
        )
    )

    if mixing_coef_wo_zero_length == 1:
        break

    # 0(nullに変換済み)以外の最小値をnullに更新
    min_coef = mixing_coef_wt_null.min()
    mixing_coef_wt_null[mixing_coef_wt_null.arg_min()] = None
    mixing_coef_wo_zero = mixing_coef_wt_null.drop_nulls()
    # mixing_coefficientsを更新
    model.mixing_coefficients = (
        (mixing_coef_wt_null + min_coef / mixing_coef_wo_zero.len())
        .fill_null(0)
        .to_numpy()
    )

plot_gmm_bic(
    clustering_results.get_column("number_of_clusters"),
    clustering_results.get_column("BIC"),
    "number of clusters",
    "BIC",
    axes[2, 0],
    None,
    None,
    c="blue",
    aspect="auto",
)

number = clustering_results.get_column("BIC").arg_min()
print(
    "最適化されたクラスター数 : "
    f"{clustering_results.item(number, 'number_of_clusters')}"
)

clusters = cluster_numbers.to_series(number).cast(pl.Int64).rename("cluster numbers")
clusters.to_frame().insert_column(0, index).write_csv(
    f"result/cluster_numbers_sgtm_{shape_of_map[0]}_{shape_of_map[1]}_{shape_of_rbf_centers[0]}_{shape_of_rbf_centers[1]}_{variance_of_rbfs}_{lambda_in_em_algorithm}_{number_of_iterations}.csv",
    quote_style="never",
)

# plot the mean of responsibilities with cluster information
plot_gmm_bic(
    means_x,
    means_y,
    "t1 (mean)",
    "t2 (mean)",
    axes[0, 2],
    c=clusters,
)

# plot the mode of responsibilities with cluster information
plot_gmm_bic(
    modes_x,
    modes_y,
    "t1 (mode)",
    "t2 (mode)",
    axes[1, 2],
    c=clusters,
)
fig.delaxes(axes[2, 1])
fig.delaxes(axes[2, 2])
plt.show()
