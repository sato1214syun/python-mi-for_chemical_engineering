"""04_04 相関係数で特徴量のクラスタリング."""

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering

x = pl.read_csv("dataset/x_selected_same_value.csv").drop("")
print("同じ値をもつサンプルの割合で削除後の特徴量の数 :", x.width)

# 相関係数を求めて、絶対値に変換
abs_x_corr_df = (
    x.corr(ddof=1)
    .with_columns(pl.all().abs())
    .insert_column(0, pl.Series("", x.columns))
)
# unpivot
unpivot_df = abs_x_corr_df.unpivot(
    index="", variable_name="variable", value_name="abs_corr"
).with_columns(
    (1 / pl.col("abs_corr")).alias("distance_in_x"),
    # 対角成分を10の10乗にする
    pl.when(pl.col("").eq(pl.col("variable")))
    .then(10**10)
    .otherwise(pl.col("abs_corr"))
    .alias("abs_corr"),
)

# 相関係数に基づくクラスタリング
corr_threshold = 0.95
clustering = AgglomerativeClustering(
    n_clusters=None,
    metric="precomputed",
    compute_full_tree=True,
    distance_threshold=1 / corr_threshold,
    linkage="complete",
)

clustering.fit(
    pl.DataFrame([[1] * abs_x_corr_df.height] * abs_x_corr_df.width, orient="col")  # type: ignore[arg-type]
    / abs_x_corr_df.drop("")
)
cluster_numbers_df = pl.DataFrame(
    clustering.labels_, schema=["cluster_number"]
).insert_column(0, pl.Series("", x.columns))
cluster_numbers_df.write_csv(
    "result/cluster_numbers_correlation.csv", quote_style="never"
)
print(
    f"相関係数に基づいてクラスタリングした後の特徴量クラスターの数: {cluster_numbers_df.n_unique('cluster_number')}"  # noqa: E501
)

# クラスターごとに一つの特徴量を選択
selected_variables = (
    cluster_numbers_df.unique(
        subset="cluster_number", keep="first", maintain_order=True
    )
    .sort("cluster_number")
    .get_column("")
).to_list()
deleted_variables = cluster_numbers_df.filter(
    pl.col("").is_in(selected_variables).not_()
).get_column("")
x_selected = x.select(selected_variables).insert_column(
    0, pl.Series("", [f"sample_{i + 1}" for i in range(x.height)])
)
x_selected.write_csv(
    "result/x_selected_correlation_clustering.csv", quote_style="never"
)

similarity_matrix = (
    abs_x_corr_df.select(["", *deleted_variables.to_list()])
    .filter(pl.col("").is_in(selected_variables))
    .sort(pl.col("").cast(pl.Enum(selected_variables)))
)
col_list = similarity_matrix.columns
col_list[0] = ""
similarity_matrix.columns = col_list

# ヒートマップ
plt.rcParams["font.size"] = 12
sns.heatmap(
    similarity_matrix.to_pandas().set_index(""),
    vmax=1,
    vmin=0,
    cmap="seismic",
    xticklabels=1,
    yticklabels=1,
)
plt.xlim([0, similarity_matrix.shape[1]])
plt.ylim([0, similarity_matrix.shape[0]])
plt.show()

similarity_matrix.write_csv(
    "result/similarity_matrix_correlation_clustering.csv", quote_style="never"
)

# クラスターごとに特徴量を平均化
x_averaged = pl.DataFrame()
# クラスタ番号でgroup_byするためにtransposeする
x_t = (
    x.transpose(
        include_header=True,
        header_name="variable",
        column_names=[f"sample_{i + 1}" for i in range(x.height)],
    )
    .insert_column(0, cluster_numbers_df.get_column("cluster_number"))
    .sort("cluster_number")
)
for cluster_number, clustered_t_x in x_t.group_by(
    "cluster_number", maintain_order=True
):
    # transposeを戻す
    variables = clustered_t_x.get_column("variable")
    clustered_x = clustered_t_x.drop(["cluster_number", "variable"]).transpose(
        column_names=variables
    )
    # clustered_xの列数が1の場合はそのままx_averagedに格納
    # clustered_xの列数が2以上の場合は、各変数列毎にデータを正規化したあとに、
    # 行(サンプル)毎の平均をとったものをx_averagedに格納
    if clustered_x.width == 1:
        x_averaged.hstack(clustered_x, in_place=True)
    else:
        normalized_c_x = clustered_x.with_columns(
            (pl.all() - pl.all().mean()) / pl.all().std()
        )
        x_averaged.hstack(
            [
                normalized_c_x.mean_horizontal().rename(
                    f"mean_in_cluster_{cluster_number[0]}"
                )
            ],
            in_place=True,
        )

x_averaged.write_csv(
    "result/x_averaged_correlation_clustering.csv", quote_style="never"
)  # 保存
