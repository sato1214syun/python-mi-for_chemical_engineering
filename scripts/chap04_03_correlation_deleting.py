"""04_03 相関係数で特徴量削除."""

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

x = pl.read_csv("dataset/x_selected_same_value.csv").drop("")
print("同じ値をもつサンプルの割合で削除後の特徴量の数 :", x.width)

# 相関係数を求めて、絶対値に変換
abs_x_corr_df = (
    x.corr(ddof=1)
    .with_columns(pl.all().abs())
    .insert_column(0, pl.Series("", x.columns))
)
# unpivotしてから対角成分を0にする
unpivot_df = abs_x_corr_df.unpivot(
    index="", variable_name="variable", value_name="abs_corr"
).with_columns(
    pl.when(pl.col("").eq(pl.col("variable")))
    .then(0)
    .otherwise(pl.col("abs_corr"))
    .alias("abs_corr")
)

corr_threshold = 0.95
deleted_variables = []
for _ in x.columns:
    biggest_corr_index = unpivot_df["abs_corr"].arg_max()
    # サンプルコードと合わせるためにvar1と2を逆にしている
    var2, var1, biggest_corr = unpivot_df.row(biggest_corr_index)
    if biggest_corr < corr_threshold:
        break
    sum_of_var1_corr = abs_x_corr_df.get_column(var1).sum()
    sum_of_var2_corr = abs_x_corr_df.get_column(var2).sum()
    delete_var = var1 if sum_of_var1_corr >= sum_of_var2_corr else var2
    deleted_variables.append(delete_var)
    unpivot_df = unpivot_df.with_columns(
        pl.when(pl.col("").eq(delete_var) | pl.col("variable").eq(delete_var))
        .then(0)
        .otherwise(pl.col("abs_corr"))
        .alias("abs_corr")
    )

x_selected = x.drop(deleted_variables)
print("相関係数で削除後の特徴量の数 :", x_selected.width)

x_selected.insert_column(
    0, pl.Series("", [f"sample_{i + 1}" for i in range(x_selected.height)])
).write_csv("result/x_selected_correlation_deleting.csv", quote_style="never")

similarity_matrix = abs_x_corr_df.select(["", *deleted_variables]).filter(
    pl.col("").is_in(x_selected.columns)
)

# ヒートマップ
sns.heatmap(
    similarity_matrix.to_pandas().set_index(""),
    vmax=1,
    vmin=0,
    cmap="seismic",
    xticklabels=1,
    yticklabels=1,
)
plt.xlim([0, similarity_matrix.width])
plt.ylim([0, similarity_matrix.height])
plt.show()

similarity_matrix.write_csv(
    "dataset/similarity_matrix_correlation_deleting.csv", quote_style="never"
)
