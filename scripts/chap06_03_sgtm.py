"""06_03 s-GTM."""

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
    *,
    c: pl.Series = None,
) -> None:
    ax.scatter(scatter_x, scatter_y, c=c)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_aspect("equal")


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
means_df.write_csv(
    f"result/sgtm_means_{shape_of_map[0]}_{shape_of_map[1]}_{shape_of_rbf_centers[0]}_{shape_of_rbf_centers[1]}_{variance_of_rbfs}_{lambda_in_em_algorithm}_{number_of_iterations}_0.csv"
)
modes_df.write_csv(
    f"result/sgtm_modes_{shape_of_map[0]}_{shape_of_map[1]}_{shape_of_rbf_centers[0]}_{shape_of_rbf_centers[1]}_{variance_of_rbfs}_{lambda_in_em_algorithm}_{number_of_iterations}_0.csv"
)
means_x = means_df.to_series(0)
means_y = means_df.to_series(1)
modes_x = modes_df.to_series(0)
modes_y = modes_df.to_series(1)

fig, axes = plt.subplots(4, 2)
plot_gmm_bic(means_x, means_y, means_x.name, means_y.name, axes[0, 0])
# plot the mean of responsibilities
y_categorical = y.cast(pl.Utf8).cast(pl.Categorical).to_physical()
plot_gmm_bic(means_x, means_y, means_x.name, means_y.name, axes[0, 1], c=y_categorical)
plot_gmm_bic(modes_x, modes_y, modes_x.name, modes_y.name, axes[1, 0])
plot_gmm_bic(modes_x, modes_y, modes_x.name, modes_y.name, axes[1, 1], c=y_categorical)

# integration of clusters based on Bayesian information criterion (BIC)
responsibilities = model.responsibility(normalized_x_np)
clustering_results = np.empty([3, responsibilities.shape[1]])
cluster_numbers = np.empty(responsibilities.shape)
for i in range(len(model.mixing_coefficients.copy())):
    likelihood = model.likelihood(normalized_x_np)
    non0_indexes = np.where(model.mixing_coefficients != 0)[0]

    responsibilities = model.responsibility(normalized_x_np)
    cluster_numbers[:, i] = responsibilities.argmax(axis=1)

    bic = -2 * likelihood + len(non0_indexes) * np.log(normalized_x_np.shape[0])
    clustering_results[:, i] = np.array(
        [len(non0_indexes), bic, len(np.unique(responsibilities.argmax(axis=1)))]
    )

    if len(non0_indexes) == 1:
        break

    non0_mixing_coefficient = model.mixing_coefficients[non0_indexes]
    model.mixing_coefficients[non0_indexes[non0_mixing_coefficient.argmin()]] = 0
    non0_indexes = np.delete(non0_indexes, non0_mixing_coefficient.argmin())
    model.mixing_coefficients[non0_indexes] = model.mixing_coefficients[
        non0_indexes
    ] + min(non0_mixing_coefficient) / len(non0_indexes)

clustering_results = np.delete(
    clustering_results, np.arange(i, responsibilities.shape[1]), axis=1
)
cluster_numbers = np.delete(
    cluster_numbers, np.arange(i, responsibilities.shape[1]), axis=1
)

plt.scatter(clustering_results[0, :], clustering_results[1, :], c="blue")
plt.xlabel("number of clusters")
plt.ylabel("BIC")
plt.show()

number = np.where(clustering_results[1, :] == min(clustering_results[1, :]))[0][0]
print(f"最適化されたクラスター数 : {int(clustering_results[0, number])}")
clusters = cluster_numbers[:, number].astype("int64")
clusters = pl.DataFrame(clusters, index=x.index, columns=["cluster numbers"])
clusters.to_csv(
    f"cluster_numbers_sgtm_{shape_of_map[0]}_{shape_of_map[1]}_{shape_of_rbf_centers[0]}_{shape_of_rbf_centers[1]}_{variance_of_rbfs}_{lambda_in_em_algorithm}_{number_of_iterations}.csv"
)

# plot the mean of responsibilities with cluster information
fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(means[:, 0], means[:, 1], c=clusters.iloc[:, 0])
plt.ylim(-1.1, 1.1)
plt.xlim(-1.1, 1.1)
plt.xlabel("t1 (mean)")
plt.ylabel("t2 (mean)")
ax.set_aspect("equal")
plt.show()

# plot the mode of responsibilities with cluster information
fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(modes[:, 0], modes[:, 1], c=clusters.iloc[:, 0])
plt.ylim(-1.1, 1.1)
plt.xlim(-1.1, 1.1)
plt.xlabel("t1 (mode)")
plt.ylabel("t2 (mode)")
ax.set_aspect("equal")


plt.show()
