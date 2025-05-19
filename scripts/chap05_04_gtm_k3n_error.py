"""05_04 generative topographic mapping (GTM)."""

# %%
import itertools
import sys

import matplotlib.pyplot as plt
import polars as pl
from dcekit.generative_model import GTM  # type: ignore[import-untyped]
from dcekit.validation import k3nerror
from traitlets import Bool  # type: ignore[import-untyped]


def calc_k3n_error(  # noqa: D103, PLR0913
    current_iter: int,
    total_iter: int,
    normalized_x: pl.DataFrame,
    map_shape_grid: int,
    rbf_centers_grid_shape: int,
    rbfs_grid_variance: float,
    em_algorithm_grid_lambda: float,
    *,
    show_progress: bool = False,
) -> None:
    print(f"progress: {current_iter} / {total_iter} ({current_iter / total_iter:.2%})")
    print(
        f"map_grid:{map_shape_grid}\n"
        f"rbf_centers_grid_shape:{rbf_centers_grid_shape}\n"
        f"rbfs_grid_variance:{rbfs_grid_variance}\n"
        f"em_algorithm_grid_lambda:{em_algorithm_grid_lambda}\n"
    )
    # construct GTM model
    model = GTM(
        [map_shape_grid, map_shape_grid],
        [rbf_centers_grid_shape, rbf_centers_grid_shape],
        rbfs_grid_variance,
        em_algorithm_grid_lambda,
        number_of_iterations,
        show_progress,
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


def set_plot(  # noqa: D103, PLR0913
    x: pl.Series,
    y: pl.Series,
    x_label: str,
    y_label: str,
    ax: plt.Axes,
    *,
    c: pl.Series = None,
    color_bar: Bool = False,
) -> None:
    sc = ax.scatter(x, y, c=c)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(-1.1, 1.1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_aspect("equal")
    if color_bar:
        plt.colorbar(sc, ax=ax)


# %%
# load dataset
dataset = pl.read_csv("../dataset/selected_descriptors_with_boiling_point.csv").drop("")
y = dataset.to_series(0)  # 目的変数
x = dataset.drop(y.name)  # 説明変数
normalized_x = x.with_columns((pl.all() - pl.all().mean()) / pl.all().std())

map_shape_candidates = pl.int_range(30, 31, eager=True)  # k の候補
rbf_centers_shape_candidates = pl.int_range(2, 22, 2, eager=True)  # q の候補
# sigma^2 の候補
rbf_variance_candidates = 2 ** pl.int_range(-5, 4, 2, eager=True).cast(pl.Float64)
# 正則化項 lambda の候補
lambda_candidates_in_em_algorithm = pl.Series([0.0]).append(
    2 ** pl.int_range(-4, 0, eager=True).cast(pl.Float64)
)
# grid search
calc_iter_length = (
    map_shape_candidates.len()
    * rbf_centers_shape_candidates.len()
    * rbf_variance_candidates.len()
    * lambda_candidates_in_em_algorithm.len()
)
# show_progress = True  # EM アルゴリズムにおける進捗を表示する (True) かしない (False) か  # noqa: ERA001
number_of_iterations = 300  # EM アルゴリズムにおける繰り返し回数
k_in_k3n_error = 10  # k3n-error における k
params_and_k3n_error_list = [
    [
        map_shape_grid,
        rbf_centers_grid_shape,
        rbfs_grid_variance,
        em_algorithm_grid_lambda,
        calc_k3n_error(
            i,
            calc_iter_length,
            normalized_x,
            map_shape_grid,
            rbf_centers_grid_shape,
            rbfs_grid_variance,
            em_algorithm_grid_lambda,
        ),
    ]
    for i, (
        map_shape_grid,
        rbf_centers_grid_shape,
        rbfs_grid_variance,
        em_algorithm_grid_lambda,
    ) in enumerate(
        itertools.product(
            map_shape_candidates,
            rbf_centers_shape_candidates,
            rbf_variance_candidates,
            lambda_candidates_in_em_algorithm,
        ),
        start=1,
    )
]
# %%
params_and_k3n_error = pl.DataFrame(
    params_and_k3n_error_list,
    orient="row",
    schema={
        "map_shape_grid": pl.Int64,
        "rbf_centers_grid_shape": pl.Int64,
        "rbfs_grid_variance": pl.Float64,
        "em_algorithm_grid_lambda": pl.Float64,
        "k3n_error": pl.Float64,
    },
)

# optimized GTM
optimized_hyperparameters = params_and_k3n_error.filter(
    pl.col("k3n_error") == pl.col("k3n_error").min()
).row(0)
shape_of_map = [optimized_hyperparameters[0], optimized_hyperparameters[0]]
shape_of_rbf_centers = [optimized_hyperparameters[1], optimized_hyperparameters[1]]
variance_of_rbfs = optimized_hyperparameters[2]
lambda_in_em_algorithm = optimized_hyperparameters[3]
print("k3n-error で最適化されたハイパーパラメータ")
print(f"２次元平面上のグリッド点の数 (k x k): {shape_of_map}")
print(f"RBF の数 (q x q): {shape_of_rbf_centers}")
print(f"RBF の分散 (sigma^2): {variance_of_rbfs}")
print(f"正則化項 (lambda): {lambda_in_em_algorithm}")

# %%
# construct GTM model with optimized hyperparameters
model = GTM(
    shape_of_map,
    shape_of_rbf_centers,
    variance_of_rbfs,
    lambda_in_em_algorithm,
    number_of_iterations,
    display_flag=False,
)
model.fit(normalized_x)

# check if the model was successfully trained
if model.success_flag is False:
    print("GTM モデルの学習に失敗しました")
    sys.exit(1)
# %%
# calculate responsibilities
responsibilities = model.responsibility(normalized_x)
means, modes = model.means_modes(normalized_x)

means_df = pl.DataFrame(means, schema={"t1_mean": pl.Float64, "t2_mean": pl.Float64})
modes_df = pl.DataFrame(modes, schema={"t1_mode": pl.Float64, "t2_mode": pl.Float64})
means_df.write_csv(
    f"../result/gtm_means_{shape_of_map[0]}_{shape_of_map[1]}_{shape_of_rbf_centers[0]}_{shape_of_rbf_centers[1]}_{variance_of_rbfs}_{lambda_in_em_algorithm}_{number_of_iterations}_0.csv"
)
modes_df.write_csv(
    f"../result/gtm_modes_{shape_of_map[0]}_{shape_of_map[1]}_{shape_of_rbf_centers[0]}_{shape_of_rbf_centers[1]}_{variance_of_rbfs}_{lambda_in_em_algorithm}_{number_of_iterations}_0.csv"
)
y_categorical = y.cast(pl.Utf8).cast(pl.Categorical).to_physical()
# %%
# plot the mean of responsibilities
fig, ax = plt.subplots(2, 2)
ax1 = ax[0, 0]
set_plot(
    means_df.to_series(0),
    means_df.to_series(1),
    x_label="t1 (mean)",
    y_label="t2 (mean)",
    ax=ax1,
)
# plot the mode of responsibilities
ax2 = ax[1, 0]
set_plot(
    modes_df.to_series(0),
    modes_df.to_series(1),
    x_label="t1 (mode)",
    y_label="t2 (mode)",
    ax=ax2,
)

try:
    # plot the mean of responsibilities
    ax3 = ax[0, 1]
    set_plot(
        means_df.to_series(0),
        means_df.to_series(1),
        c=y_categorical,
        x_label="t1 (mean)",
        y_label="t2 (mean)",
        ax=ax3,
        color_bar=True,
    )

    # plot the mode of responsibilities
    ax4 = ax[1, 1]
    set_plot(
        modes_df.to_series(0),
        modes_df.to_series(1),
        c=y_categorical,
        x_label="t1 (mode)",
        y_label="t2 (mode)",
        ax=ax4,
        color_bar=True,
    )
except NameError:
    print("y がありません")

plt.show()

# %%
