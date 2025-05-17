"""04_06 GAWLSPLS."""

from __future__ import annotations

import random
from itertools import batched, cycle
from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl
from deap import base, creator, tools  # type: ignore[import-untyped]
from sklearn import model_selection
from sklearn.cross_decomposition import PLSRegression

if TYPE_CHECKING:
    from polars._typing import PythonLiteral

# シードの固定
random.seed(100)
# random.seed()  # noqa: ERA001

# データの読み込みと正規化
dataset = pl.read_csv("dataset/sample_spectra_dataset_with_y.csv").drop("")
y_train = dataset.to_series(0)  # 目的変数
x_train = dataset.drop(y_train.name)  # 説明変数
# 正規化
normalized_y_train = (y_train - y_train.mean()) / y_train.std()
normalized_x_train = x_train.with_columns((pl.all() - pl.all().mean()) / pl.all().std())

# GAWLSPLS
creator.create(
    "FitnessMax", base.Fitness, weights=(1.0,)
)  # for minimization, set weights as (-1.0,)
creator.create("Individual", list, fitness=creator.FitnessMax)

# toolboxに関数を登録
toolbox = base.Toolbox()


def create_individual(min_boundary: pl.Series, max_boundary: pl.Series) -> list[float]:
    """個体の生成用関数."""
    index = []
    for min_val, max_val in zip(min_boundary, max_boundary, strict=False):
        index.append(random.uniform(min_val, max_val))  # noqa: S311
    return index


NUM_OF_RANGE = 5  # 選択する領域の数
MAX_WIDTH_IN_A_AREA = 20  # 選択する領域の幅の最大値
# 個体の生成
min_boundary = pl.Series([0] * NUM_OF_RANGE * 2)
cycle_iter = cycle([x_train.width, MAX_WIDTH_IN_A_AREA])
max_boundary = pl.Series([next(cycle_iter) for _ in range(NUM_OF_RANGE * 2)])
toolbox.register("create_individual", create_individual, min_boundary, max_boundary)
toolbox.register(
    "individual", tools.initIterate, creator.Individual, toolbox.create_individual
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def eval_one_max(  # noqa: PLR0913
    individual: creator.Individual,
    fold_number: int,
    max_number_of_components: int,
    normalized_x_train: pl.DataFrame,
    normalized_y_train: pl.Series,
    y_train: pl.Series,
) -> tuple[PythonLiteral | None, ...]:
    """個体の評価関数."""
    # [開始波長のインデックス, 波長範囲のリスト]を取り出し、小数点は切り捨て
    selected_wl_indices = (
        pl.DataFrame(individual, ["wl_idx_and_range"])
        .select(
            pl.col("wl_idx_and_range")
            .floor()  # 小数点切り捨て
            .cast(pl.Int64)
            .reshape((NUM_OF_RANGE, 2))  # [波長開始index, 波長範囲]の配列に並べ替え
            .arr.to_struct(["start_wl_idx", "wl_range"])  # structに変換
        )
        .unnest("wl_idx_and_range")  # structを別の列に展開
        .with_columns(
            # 波長終了indexを計算
            (pl.col("start_wl_idx") + pl.col("wl_range")).alias("end__wl_idx")
        )
        .drop("wl_range")
        .select(
            # 波長(特徴量・遺伝子)のindexを全て抽出
            pl.int_ranges(
                "start_wl_idx",
                pl.min_horizontal(
                    pl.col("end__wl_idx"), pl.lit(normalized_x_train.width)
                ).alias("wl_indices"),
            )
            .explode()
            .drop_nulls()  # 波長の範囲が0の時にNoneになるので削除
        )
        .to_series(0)
        .to_list()
    )

    if not selected_wl_indices:
        return (-999,)

    selected_normalized_x_train = normalized_x_train.select(
        # 波長(特徴量)インデックスの重複に対応する
        [
            pl.col(normalized_x_train.columns[idx]).alias(f"{i}")
            for i, idx in enumerate(selected_wl_indices)
        ]
    )

    # cross-validation
    min_pls_component = min(
        np.linalg.matrix_rank(selected_normalized_x_train), max_number_of_components
    )
    estimated_y_train_cv = pl.DataFrame(
        {
            f"{pls_component}": model_selection.cross_val_predict(
                PLSRegression(n_components=pls_component),
                selected_normalized_x_train,  # type: ignore[arg-type]
                normalized_y_train,
                cv=fold_number,
            ).flatten()
            for pls_component in range(1, min_pls_component + 1)
        }
    )

    # クロスバリデーションの結果を元のスケールに戻す
    estimated_y_train_cv = estimated_y_train_cv.with_columns(
        pl.all() * y_train.std() + y_train.mean()
    )
    # r2を計算
    r2_cv_all = estimated_y_train_cv.select(
        (
            pl.lit(1)
            - ((pl.all() - y_train) ** 2).sum()
            / ((y_train - y_train.mean()) ** 2).sum()
        ).name.prefix("r2_")
    )
    return (r2_cv_all.max_horizontal().max(),)


# 評価関数を登録
toolbox.register(
    "evaluate",
    eval_one_max,
    fold_number=5,  # クロスバリデーションのfold数
    max_number_of_components=10,  # PLS の最大成分数
    normalized_x_train=normalized_x_train,
    normalized_y_train=normalized_y_train,
    y_train=y_train,
)
# その他の関数はGAPLSと同じ
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


def update_fitness(individual: creator.Individual) -> creator.Individual:
    """個体の適合度を更新する."""
    individual.fitness.values = toolbox.evaluate(individual)
    return individual


def mate(individual1: creator.Individual, individual2: creator.Individual) -> None:
    """個体の交叉."""
    toolbox.mate(individual1, individual2)
    del individual1.fitness.values, individual2.fitness.values


def mutate(individual: creator.Individual) -> None:
    """個体の突然変異."""
    toolbox.mutate(individual)


print("Start of evolution")

NUM_OF_POPULATION = 100  # GA の個体数
NUM_OF_GENERATION = 150  # GA の世代数
CROSSOVER_PROBABILITY = 0.5
MUTATION_PROBABILITY = 0.2

population = cast("list[creator.Individual]", toolbox.population(n=NUM_OF_POPULATION))
for generation in range(NUM_OF_GENERATION + 1):
    print(f"-- Generation {generation} --")
    # 1世代目以降は変異させる
    if generation > 0:
        # トーナメント選択で、individual 3個体の中1個体を選択。population数分繰り返す
        offspring = toolbox.select(population, k=len(population))
        offspring = [toolbox.clone(child) for child in offspring]  # deepcopy

        # 親世代を変異させて、子世代を作成
        # 変異済みの判定はfitness.valuesを削除してnot fitness.valid = Trueで行う
        # 2個ずつ個体を取り出して、交叉を行う
        [
            mate(*children)  # type: ignore[func-returns-value]
            for children in batched(offspring, 2, strict=False)
            if len(children) != 1 and random.random() < CROSSOVER_PROBABILITY  # noqa: S311
        ]
        # 突然変異
        [mutate(child) for child in offspring if random.random() < MUTATION_PROBABILITY]  # type: ignore[func-returns-value]  # noqa: S311
        # 個体群を新世代に置き換え
        population = offspring

    num_of_updated_children = len(
        [child for child in population if not child.fitness.valid]
    )
    # 各個体の適合度を更新
    population = [
        child if child.fitness.valid else update_fitness(child) for child in population
    ]

    fits: list[float] = [individual.fitness.values[0] for individual in population]
    length = len(population)
    mean = sum(fits) / length
    sum2 = sum(fit**2 for fit in fits)
    std = abs(sum2 / length - mean**2) ** 0.5

    print(f"\tEvaluated {num_of_updated_children} individuals")
    print(f"\tMin {min(fits)}")
    print(f"\tMax {max(fits)}")
    print(f"\tAvg {mean}")
    print(f"\tStd {std}")

print("-- End of (successful) evolution --")

best_individual = tools.selBest(population, 1)[0]
selected_var_indices = (
    pl.DataFrame(best_individual, ["wl_idx_and_range"])
    .select(
        pl.col("wl_idx_and_range")
        .floor()  # 小数点切り捨て
        .cast(pl.Int64)
        .reshape((NUM_OF_RANGE, 2))  # [波長開始index, 波長範囲]の配列に並べ替え
        .arr.to_struct(["start_wl_idx", "wl_range"])  # structに変換
    )
    .unnest("wl_idx_and_range")  # structを別の列に展開
    .with_columns(
        # 波長終了indexを計算
        (pl.col("start_wl_idx") + pl.col("wl_range")).alias("end__wl_idx")
    )
    .drop("wl_range")
    .select(
        # 波長(特徴量・遺伝子)のindexを全て抽出
        pl.int_ranges(
            "start_wl_idx",
            pl.min_horizontal(
                pl.col("end__wl_idx"), pl.lit(normalized_x_train.width)
            ).alias("wl_indices"),
        )
        .explode()
        .drop_nulls()  # 波長の範囲が0の時にNoneになるので削除
    )
    .to_series(0)
    .to_list()
)

selected_descriptors = x_train.select(pl.nth(selected_var_indices))
selected_descriptors.with_row_index("").write_csv(
    "dataset/gawlspls_selected_x.csv", quote_style="never"
)
