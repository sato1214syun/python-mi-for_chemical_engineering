"""03_07 (iGMR)."""

import polars as pl
from dcekit.generative_model import GMR  # type: ignore[import-untyped]


def calc_interpolated_values(
    model: GMR,
    x_row: list[float],
    normalized_x_row: list[float],
    x: pl.DataFrame,
) -> list[float]:
    """補完値の計算と挿入."""
    if None not in x_row:
        return x_row
    row_series = pl.Series(x_row)
    normalized_row_series = pl.Series(normalized_x_row)
    is_null_series = row_series.is_null()
    is_val_series = row_series.is_not_null()
    mode_of_estimated_mean, *_ = model.predict(
        normalized_row_series.filter(is_val_series).to_frame().transpose(),
        is_val_series.arg_true(),
        is_null_series.arg_true(),
    )

    interpolated_vals = pl.Series(mode_of_estimated_mean[0]) * pl.Series(
        x.std().row(0)
    ).filter(is_null_series) + pl.Series(x.mean().row(0)).filter(is_null_series)

    # Noneをinterpolated_valueに置換
    # for i, j in enumerate(is_null_series.arg_true()):
    return row_series.scatter(is_null_series.arg_true(), interpolated_vals).to_list()


# load dataset
x_wt_nulls = pl.read_csv("dataset/iris_with_nan.csv").drop("")
x = x_wt_nulls.drop_nulls()
# 正規化
normalized_x_wt_nulls = x_wt_nulls.with_columns(
    (pl.all() - pl.all().drop_nulls().mean()) / pl.all().drop_nulls().std()
)
normalized_x = x.with_columns((pl.all() - pl.all().mean()) / pl.all().std())
# 元/正規化データのrowデータ(リスト)のDataFrameを作成
rows_of_ori_and_normalized_x_df = x_wt_nulls.select(
    pl.concat_list(pl.all()).alias("x_row")
).hstack(
    normalized_x_wt_nulls.select(pl.concat_list(pl.all()).alias("normalized_x_row"))
)

max_num_of_compo = 20
covariance_types = ["full", "diag", "tied", "spherical"]
for i in range(iterations := 10):
    print(f"{i + 1} / {iterations}")
    # グリッドサーチで最適(BICが一番小さい)パラメータを取得.
    bic_values = pl.DataFrame(
        {
            "num_of_compo": [
                num_of_compo
                for _ in covariance_types
                for num_of_compo in range(max_num_of_compo)
            ],
            "cov_type": [
                cov_type
                for cov_type in covariance_types
                for _ in range(max_num_of_compo)
            ],
            "BIC": [
                GMR(num_of_compo + 1, cov_type, random_state=0)
                .fit(normalized_x)
                .bic(normalized_x)
                for cov_type in covariance_types
                for num_of_compo in range(max_num_of_compo)
            ],
        }
    ).sort(pl.col("BIC"))

    # GMM
    model = GMR(
        n_components=bic_values.item(0, "num_of_compo") + 1,
        covariance_type=bic_values.item(0, "cov_type"),
        random_state=0,
    )
    model.fit(normalized_x)

    # interpolation(補完).
    x = (
        rows_of_ori_and_normalized_x_df.select(
            pl.struct(pl.col("x_row"), pl.col("normalized_x_row"))
            .map_elements(
                lambda row_dict, model=model, x=x: calc_interpolated_values(  # type: ignore[misc]
                    model,
                    row_dict["x_row"],
                    row_dict["normalized_x_row"],
                    x,
                ),
                return_dtype=pl.List(pl.Float64),
            )
            .alias("x_row")
        )
        .with_columns(pl.col("x_row").list.to_struct(fields=x.columns))
        .unnest("x_row")
    )

    # 正規化データを更新
    normalized_x = x.with_columns((pl.all() - pl.all().mean()) / pl.all().std())
    # 元/正規化データのrowデータ(リスト)のDataFrameを更新
    next_normalized_x_rows = normalized_x.select(
        pl.concat_list(pl.all()).alias("normalized_x_row")
    ).get_column("normalized_x_row")
    rows_of_ori_and_normalized_x_df.replace_column(1, next_normalized_x_rows)

# save interpolated dataset
x.insert_column(
    0, pl.Series("", [f"sample_{i + 1}" for i in range(x.height)])
).write_csv("result/interpolated_dataset.csv", quote_style="never")
