"""05_04 generative topographic mapping (GTM)."""

import sys

import matplotlib.pyplot as plt
import polars as pl
from dcekit.generative_model import GTM  # type: ignore[import-untyped]
from traitlets import Bool  # type: ignore[import-untyped]


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


# load dataset
dataset = pl.read_csv("dataset/selected_descriptors_with_boiling_point.csv")
index = dataset.to_series(0)
dataset = dataset.drop(index.name)
y = dataset.to_series(0)  # 目的変数
x = dataset.drop(y.name)  # 説明変数
normalized_x = x.with_columns((pl.all() - pl.all().mean()) / pl.all().std())

map_shape = pl.Series([30, 30])
rbf_centers_shape = pl.Series([5, 5])
rbf_variance = 4  # sigma^2 の候補
lambda_in_em_algorithm = 0.001  # 正則化項 lambda の候補
iter_num = 300  # EM アルゴリズムにおける繰り返し回数

# construct GTM model with optimized hyperparameters
model = GTM(
    map_shape,
    rbf_centers_shape,
    rbf_variance,
    lambda_in_em_algorithm,
    iter_num,
    display_flag=False,
)
model.fit(normalized_x)

# check if the model was successfully trained
if model.success_flag is False:
    print("GTM モデルの学習に失敗しました")
    sys.exit(1)

# calculate responsibilities
responsibilities = model.responsibility(normalized_x)
means, modes = model.means_modes(normalized_x)

means_df = pl.DataFrame(means, schema={"t1_mean": pl.Float64, "t2_mean": pl.Float64})
modes_df = pl.DataFrame(modes, schema={"t1_mode": pl.Float64, "t2_mode": pl.Float64})
means_df.insert_column(0, index).write_csv(
    f"result/gtm_means_{map_shape[0]}_{map_shape[1]}_{rbf_centers_shape[0]}_{rbf_centers_shape[1]}_{rbf_variance}_{lambda_in_em_algorithm}_{iter_num}.csv"
)
modes_df.insert_column(0, index).write_csv(
    f"result/gtm_modes_{map_shape[0]}_{map_shape[1]}_{rbf_centers_shape[0]}_{rbf_centers_shape[1]}_{rbf_variance}_{lambda_in_em_algorithm}_{iter_num}.csv"
)
y_categorical = y.cast(pl.Utf8).cast(pl.Categorical).to_physical()

means_df=means_df.drop("")
modes_df=modes_df.drop("")

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
