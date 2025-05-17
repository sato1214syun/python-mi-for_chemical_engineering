"""05_03_k3_nerror."""

import matplotlib.pyplot as plt
import polars as pl
from dcekit.validation import k3nerror  # type: ignore[import-untyped]
from sklearn.manifold import TSNE


def load_dataset(file_path: str) -> pl.DataFrame:
    """データセットを読み込む."""
    return pl.read_csv(file_path).drop("")


def pre_processing_dataset(
    dataset: pl.DataFrame,
) -> pl.DataFrame:
    """データセットの前処理."""
    return dataset.with_columns(pl.nth(0).replace(None, 999).replace(999, None))


# データセットの読み込み
dataset = load_dataset("dataset/selected_descriptors_with_boiling_point.csv")
dataset = pre_processing_dataset(dataset)
y = dataset.to_series(0)  # 目的変数
x = dataset.drop(y.name)  # 説明変数
normalized_x = x.with_columns((pl.all() - pl.all().mean()) / pl.all().std())

# k3n-error を用いた perplexity の最適化
n_components = 2
k_in_k3n_error = 10  # k3n-error の k
# t-SNE の perplexity の候補
candidates_of_perplexity = pl.int_range(5, 105, 5, eager=True)
k3n_errors = pl.Series("k3n_error", dtype=pl.Float64)
for i, perplexity in enumerate(candidates_of_perplexity):
    print(i + 1, "/", len(candidates_of_perplexity))

    t = pl.DataFrame(
        TSNE(
            n_components, perplexity=perplexity, init="pca", random_state=10
        ).fit_transform(
            normalized_x.to_numpy()  # pl.DataFrameだとサンプルの結果と差が出る
        )
    )
    scaled_t = t.with_columns((pl.all() - pl.all().mean()) / pl.all().std())
    k3n_errors.append(
        pl.Series(
            [
                k3nerror(normalized_x, scaled_t, k_in_k3n_error)
                + k3nerror(scaled_t, normalized_x, k_in_k3n_error)
            ]
        ),
    )

plt.scatter(candidates_of_perplexity, k3n_errors, c="blue")
plt.xlabel("perplexity")
plt.ylabel("k3n-error")
plt.show()

optimal_perplexity = candidates_of_perplexity[k3n_errors.arg_min()]
print("k3n-error による perplexity の最適値 :", optimal_perplexity)

# t-SNE
t = pl.DataFrame(
    TSNE(
        n_components, perplexity=optimal_perplexity, init="pca", random_state=10
    ).fit_transform(normalized_x.to_numpy())  # pl.DataFrameだとサンプルの結果と差が出る
)
t.columns = ["t1", "t2"]
index = pl.Series("", [f"sample_{i + 1}" for i in range(t.height)])
t.insert_column(0, index).write_csv(
    f"result/tsne_score_perplexity_{optimal_perplexity}.csv", quote_style="never"
)

# t1 と t2 の散布図
plt.scatter(t.to_series(0), t.to_series(1), c="blue")
plt.xlabel("t1")
plt.ylabel("t2")
plt.show()
