"""06_01 GMM."""

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.colors import LogNorm
from sklearn.mixture import GaussianMixture

x = pl.read_csv("dataset/sample_dataset_gmm.csv")
index = x.get_column("")
x = x.drop("")
normalized_x = x.with_columns((pl.all() - pl.all().mean()) / pl.all().std())

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# プロット
ax = axes[0]
ax.scatter(normalized_x.to_series(0), normalized_x.to_series(1), c="blue")
ax.set_xlim(-4, 3)
ax.set_ylim(-3, 3)
ax.set_xlabel(x.columns[0])
ax.set_ylabel(x.columns[1])

# GMM モデリング
num_of_gaussian = 2  # 正規分布の数
covariance_type = "spherical"  # 分散共分散行列の種類
# "full" : 各正規分布がそれぞれ、一般的な分散共分散行列をもつ
# "tied" : すべての正規分布が同じ一般的な分散共分散行列をもつ
# "diag" : 各正規分布がそれぞれ、共分散が全て0の分散共分散行列をもつ
# "spherical" : 各正規分布がそれぞれ、共分散が全て0で分散が同じ値の分散共分散行列をもつ
model = GaussianMixture(n_components=num_of_gaussian, covariance_type=covariance_type)
model.fit(normalized_x)

# クラスターへの割り当て
cluster_numbers = pl.Series("cluster_numbers", model.predict(normalized_x))
index.to_frame().hstack([cluster_numbers]).write_csv(
    f"result/cluster_numbers_gmm_{num_of_gaussian}_{covariance_type}0.csv"
)
cluster_probabilities = pl.DataFrame(model.predict_proba(normalized_x))
cluster_probabilities.clone().insert_column(0, index).write_csv(
    f"result/cluster_probabilities_gmm_{num_of_gaussian}_{covariance_type}0.csv"
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
    pl.DataFrame(-model.score_samples(XX))
    .select(pl.all().reshape(X.shape).arr.to_struct())
    .unnest("column_0")
)
CS = plt.contour(
    X,
    Y,
    Z,
    norm=LogNorm(vmin=1.0, vmax=7.1),
    levels=pl.linear_space(0, 0.85, 8, eager=True).log10(),
)

ax1 = axes[1]
ax1.scatter(normalized_x.to_series(0), normalized_x.to_series(1), c="blue")
ax1.set_xlim(-4, 3)
ax1.set_ylim(-3, 3)
ax1.set_xlabel(x.columns[0])
ax1.set_ylabel(x.columns[1])

plt.show()
