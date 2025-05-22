"""06_03 sparse-GTM k3e-error."""

import itertools
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from dcekit.generative_model import GTM  # type: ignore[import-untyped]
from dcekit.validation import k3nerror


def calc_k3n_error(  # noqa: D103, PLR0913
    current_iter: int,
    total_iter: int,
    normalized_x: pl.DataFrame,
    map_shape: tuple[int, int],
    rbf_centers_shape: tuple[int, int],
    rbfs_grid_variance: float,
    em_algorithm_grid_lambda: float,
    k_in_k3n_error: int,
    *,
    show_progress: bool = False,
    sparse_flag: bool = True,
) -> None:
    print(f"progress: {current_iter} / {total_iter} ({current_iter / total_iter:.2%})")
    print(
        f"map_grid:{map_shape}\n"
        f"rbf_centers_grid_shape:{rbf_centers_shape}\n"
        f"rbfs_grid_variance:{rbfs_grid_variance}\n"
        f"em_algorithm_grid_lambda:{em_algorithm_grid_lambda}\n"
    )
    # construct GTM model
    model = GTM(
        map_shape,
        rbf_centers_shape,
        rbfs_grid_variance,
        em_algorithm_grid_lambda,
        number_of_iterations,
        show_progress,
        sparse_flag=sparse_flag,
    )
    model.fit(normalized_x)

    if model.success_flag is False:
        return 10e100

    # calculate of responsibilities
    responsibilities = model.responsibility(normalized_x)
    # calculate the mean of responsibilities
    means = responsibilities.dot(model.map_grids)
    # calculate k3n-error
    return k3nerror(normalized_x, means, k_in_k3n_error) + k3nerror(
        means, normalized_x, k_in_k3n_error
    )


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
map_shapes = pl.DataFrame().with_columns(
    pl.int_range(30, 31).alias("x"),
    pl.int_range(30, 31).alias("y"),
)
rbf_center_shapes = pl.DataFrame().with_columns(
    pl.int_range(2, 22, 2).alias("x"),
    pl.int_range(2, 22, 2).alias("y"),
)
rbf_variances = 2 ** pl.int_range(-5, 4, 2, eager=True).rename("rbf_variance").cast(
    pl.Float64
)
lambdas_in_em_algorithm = 2 ** pl.int_range(-4, 0, eager=True).rename("lambda").cast(
    pl.Float64
)

# grid search
calc_iter_length = (
    map_shapes.height
    * rbf_center_shapes.height
    * rbf_variances.len()
    * lambdas_in_em_algorithm.len()
)

# show_progress = True  # noqa: ERA001
number_of_iterations = 300  # EM アルゴリズムにおける繰り返し回数
k_in_k3n_error = 10  # k3n-error における k
params_and_k3n_error_list = [
    [
        map_shape,
        rbf_centers_shape,
        rbfs_grid_variance,
        em_algorithm_grid_lambda,
        calc_k3n_error(
            i,
            calc_iter_length,
            normalized_x_np,
            map_shape,
            rbf_centers_shape,
            rbfs_grid_variance,
            em_algorithm_grid_lambda,
            k_in_k3n_error,
            sparse_flag=True,
        ),
    ]
    for i, (
        map_shape,
        rbf_centers_shape,
        rbfs_grid_variance,
        em_algorithm_grid_lambda,
    ) in enumerate(
        itertools.product(
            map_shapes.iter_rows(),
            rbf_center_shapes.iter_rows(),
            rbf_variances,
            lambdas_in_em_algorithm,
        ),
        start=1,
    )
]

params_and_k3n_error = pl.DataFrame(
    params_and_k3n_error_list,
    orient="row",
    schema={
        "map_shape_grid": pl.List(pl.Int64),
        "rbf_centers_grid_shape": pl.List(pl.Int64),
        "rbfs_grid_variance": pl.Float64,
        "em_algorithm_grid_lambda": pl.Float64,
        "k3n_error": pl.Float64,
    },
)

# optimized GTM
optimized_hyperparameters = params_and_k3n_error.filter(
    pl.col("k3n_error") == pl.col("k3n_error").min()
).row(0)
map_shape = optimized_hyperparameters[0]
rbf_center_shape = optimized_hyperparameters[1]
rbf_variance = optimized_hyperparameters[2]
lambda_in_em_algorithm = optimized_hyperparameters[3]
print("k3n-error で最適化されたハイパーパラメータ")
print(f"２次元平面上のグリッド点の数 (k x k): {map_shape}")
print(f"RBF の数 (q x q): {rbf_center_shape}")
print(f"RBF の分散 (sigma^2): {rbf_variance}")
print(f"正則化項 (lambda): {lambda_in_em_algorithm}")

# construct SGTM model
model = GTM(
    map_shape,
    rbf_center_shape,
    rbf_variance,
    lambda_in_em_algorithm,
    number_of_iterations,
    display_flag=False,
    sparse_flag=True,
)
model.fit(normalized_x_np)

if model.success_flag is False:
    sys.exit("SGTM model fitting failed.")

# calculate responsibilities
responsibilities = model.responsibility(normalized_x_np)
means, modes = model.means_modes(normalized_x_np)

means_df = pl.DataFrame(means, schema={"t1_mean": pl.Float64, "t2_mean": pl.Float64})
modes_df = pl.DataFrame(modes, schema={"t1_mode": pl.Float64, "t2_mode": pl.Float64})
means_df.insert_column(0, index).write_csv(
    f"result/sgtm_means_{map_shape[0]}_{map_shape[1]}_{rbf_center_shape[0]}_{rbf_center_shape[1]}_{rbf_variance}_{lambda_in_em_algorithm}_{number_of_iterations}.csv"
)
modes_df.insert_column(0, index).write_csv(
    f"result/sgtm_modes_{map_shape[0]}_{map_shape[1]}_{rbf_center_shape[0]}_{rbf_center_shape[1]}_{rbf_variance}_{lambda_in_em_algorithm}_{number_of_iterations}.csv"
)

means_df = means_df.drop("")
modes_df = modes_df.drop("")
means_x = means_df.to_series(0)
means_y = means_df.to_series(1)
modes_x = modes_df.to_series(0)
modes_y = modes_df.to_series(1)

# plot the mean of responsibilities
y_categorical = y.cast(pl.Utf8).cast(pl.Categorical).to_physical()
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
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
    f"result/cluster_numbers_sgtm_{map_shape[0]}_{map_shape[1]}_{rbf_center_shape[0]}_{rbf_center_shape[1]}_{rbf_variance}_{lambda_in_em_algorithm}_{number_of_iterations}.csv",
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
