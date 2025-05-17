"""04_02 同じ値を多くもつ特徴量の削除."""
import polars as pl

# 同じ値をもつサンプルの割合で特徴量を削除するためのしきい値
threshold_dup_value_rate = 0.8

x = pl.read_csv("dataset/descriptors_with_logS.csv").drop("", "logS")

print("最初の特徴量の数 :", x.width)
# 同じ値の割合が、threshold_dup_value_rate 以上の特徴量を削除
dup_val_rates = (
    x.select(
        pl.all()
        .value_counts(sort=True, normalize=True, name="dup_rate")
        .first()
        .struct.field("dup_rate")
        .name.keep()
    )
    .with_columns(pl.all() < threshold_dup_value_rate)
    .row(0)
)
x_selected = x[:, dup_val_rates]
print("削除後の特徴量の数 :", x_selected.width)

x_selected.insert_column(
    0, pl.Series("", [f"sample_{i + 1}" for i in range(x_selected.height)])
).write_csv("result/x_selected_same_value.csv", quote_style="never")  # 保存
