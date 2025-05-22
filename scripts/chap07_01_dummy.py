"""07_01 ダミー変数の作成."""

import pandas as pd
import polars as pl
import polars.selectors as cs

x = pl.read_csv("dataset/iris.csv")  # データセットの読み込み
index = x.to_series(0)
x = x.drop(index.name)  # インデックスを削除
num_data = x.select(cs.numeric())
str_data = x.select(cs.string())

# polars 1.30.0時点では、to_dummiesの挙動がpandasと異なるようなので、pandasを使用
# https://github.com/pola-rs/polars/issues/7639
# また、to_dummiesの出力をbool値で出力するオプションを付けるか、ここで議論されている
# https://github.com/pola-rs/polars/issues/22629
x_with_dummy = num_data.hstack(pl.from_pandas(pd.get_dummies(str_data.to_pandas())))
x_with_dummy.insert_column(0, index).write_csv("result/x_with_dummy_variables_0.csv")
