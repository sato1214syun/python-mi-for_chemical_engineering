"""jupyter上でmultiprocessingを使用してGAPLSを実行するスクリプト."""

import random
from itertools import batched
from multiprocessing import Pool  # 処理速度向上のために追加
from typing import cast

import numpy as np
import polars as pl
from deap import base, creator, tools  # type: ignore[import-untyped]
from sklearn import model_selection
from sklearn.cross_decomposition import PLSRegression

# シードの固定
random.seed(100)
# random.seed()  # noqa: ERA001

# データの読み込みと正規化
dataset = pl.read_csv("dataset/selected_descriptors_with_boiling_point.csv").drop("")
y_train = dataset.select(dataset.columns[0])  # 目的変数
x_train = dataset.drop(dataset.columns[0])  # 説明変数
# 正規化
normalized_x_train = x_train.with_columns((pl.all() - pl.all().mean()) / pl.all().std())
normalized_y_train = y_train.with_columns((pl.all() - pl.all().mean()) / pl.all().std())

# GAPLSの設定
# deapのcreatorを使用してFitnessMaxとIndividualのクラスを作成
creator.create(  # for minimization, set weights as (-1.0,)
    "FitnessMax", base.Fitness, weights=(1.0,)
)
creator.create("Individual", list, fitness=creator.FitnessMax)

# toolboxに関数を登録
toolbox = base.Toolbox()


# 個体の生成用関数の登録
def create_individual(min_boundary: pl.Series, max_boundary: pl.Series) -> list:
    """個体の生成用関数."""
    return [
        random.uniform(min_boundary[i], max_boundary[i])  # noqa: S311
        for i in range(len(min_boundary))
    ]


# 評価関数の登録
def eval_one_max(  # noqa: PLR0913
    individual: creator.Individual,
    fold_number: int,
    max_number_of_components: int,
    threshold_of_variable_selection: float,
    normalized_x_train: pl.DataFrame,
    normalized_y_train: pl.DataFrame,
    y_train: pl.DataFrame,
) -> float:
    """個体の評価関数."""
    selected_normalized_x_train = normalized_x_train.select(
        [
            col
            for i, col in enumerate(normalized_x_train.columns)
            if individual[i] > threshold_of_variable_selection
        ]
    )
    # 選択された変数の数が0の場合は、適合度を-999にする
    if not len(selected_normalized_x_train.columns):
        individual.fitness.values = (-999,)
        return individual

    # cross-validation
    min_pls_component = min(
        np.linalg.matrix_rank(selected_normalized_x_train), max_number_of_components
    )
    estimated_y_train_cv_df = y_train.clone().rename({"boiling_point": "y_train"})
    for pls_component in range(1, min_pls_component + 1):
        cross_val_predicts_np = model_selection.cross_val_predict(
            PLSRegression(n_components=pls_component),
            selected_normalized_x_train,  # type: ignore[arg-type]
            normalized_y_train,
            cv=fold_number,
        )
        estimated_y_train_cv_df = estimated_y_train_cv_df.with_columns(
            pl.Series(f"{pls_component}", cross_val_predicts_np.flatten())
        )

    # クロスバリデーションの結果を元のスケールに戻す
    estimated_y_train_cv_df = estimated_y_train_cv_df.with_columns(
        pl.exclude("y_train") * pl.col("y_train").std() + pl.col("y_train").mean()
    )
    # r2を計算
    r2_cv_all = estimated_y_train_cv_df.select(
        pl.exclude("y_train")
        .sub(pl.col("y_train"))
        .pow(2)
        .sum()
        .truediv(pl.col("y_train").sub(pl.col("y_train").mean()).pow(2).sum())
        .mul(-1)
        .add(1)
        .name.prefix("r2_")
    )
    individual.fitness.values = (r2_cv_all.max_horizontal().max(),)
    return individual


def mate(individual1: creator.Individual, individual2: creator.Individual) -> None:
    """個体の交叉."""
    toolbox.mate(individual1, individual2)
    del individual1.fitness.values, individual2.fitness.values


def mutate(individual: creator.Individual) -> None:
    """個体の突然変異."""
    toolbox.mutate(individual)
    del individual.fitness.values


def main() -> None:
    """メイン関数."""
    # 個体の生成
    min_boundary = pl.Series([0.0] * x_train.width)
    max_boundary = pl.Series([1.0] * x_train.width)
    pool = Pool()  # CPU のコア数に合わせて変更
    toolbox.register("map", pool.map)
    toolbox.register("starmap", pool.starmap)
    toolbox.register("create_individual", create_individual, min_boundary, max_boundary)
    toolbox.register(
        "individual",
        tools.initIterate,
        creator.Individual,  # container
        toolbox.create_individual,  # generator
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register(
        "evaluate",
        eval_one_max,
        fold_number=5,  # クロスバリデーションのfold数
        max_number_of_components=10,  # PLS の最大成分数
        threshold_of_variable_selection=(  # 染色体の0, 1を分ける閾値
            threshold_of_variable_selection := 0.5
        ),
        normalized_x_train=normalized_x_train,
        normalized_y_train=normalized_y_train,
        y_train=y_train,
    )

    # 2点交叉(Two-point crossover): 2個体のランダムで選択した遺伝子を入れ替える
    toolbox.register("mate", tools.cxTwoPoint)
    # 突然変異: indpbの確率で、個体の遺伝子を反転させる(特徴量の選択有無を反転)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    # 個体の選別方法: トーナメント選択で3個体の中から1個体を選択.
    toolbox.register("select", tools.selTournament, tournsize=3)

    print("Start of evolution")

    crossover_probability = 0.5
    mutation_probability = 0.2
    num_of_population = 100  # GAの個体数
    number_of_generation = 150  # GAの世代数
    population = cast(
        "list[creator.Individual]", toolbox.population(n=num_of_population)
    )
    for generation in range(number_of_generation + 1):
        print(f"-- Generation {generation} --")
        # 1世代目以降は変異させる
        if generation > 0:
            # トーナメント選択で、individual 3個体の中1個体を選択。population回繰り返す
            offspring = toolbox.select(population, k=len(population))
            offspring = [toolbox.clone(child) for child in offspring]  # deepcopy

            # 親世代を変異させて、子世代を作成
            # 変異済みの判定はfitness.valuesを削除してnot fitness.valid = Trueで行う
            # 2個ずつ個体を取り出して、交叉を行う
            [
                mate(*children)  # type: ignore[func-returns-value]
                for children in batched(offspring, 2, strict=False)
                if len(children) != 1 and random.random() < crossover_probability  # noqa: S311
            ]
            # 突然変異
            [
                mutate(child)  # type: ignore[func-returns-value]
                for child in offspring
                if random.random() < mutation_probability  # noqa: S311
            ]
            # 個体群を新世代に置き換え
            population = offspring

        num_of_updated_children = len(
            [child for child in population if not child.fitness.valid]
        )
        # 各個体の適合度を更新
        population_iter = toolbox.map(toolbox.evaluate, population)
        population = list(population_iter)
        fits = [individual.fitness.values[0] for individual in population_iter]
        mean = sum(fits) / num_of_population
        sum2 = sum(fit**2 for fit in fits)
        std = abs(sum2 / num_of_population - mean**2) ** 0.5

        print(f"\tEvaluated {num_of_updated_children} individuals")
        print(f"\tMin {min(fits)}")
        print(f"\tMax {max(fits)}")
        print(f"\tAvg {mean}")
        print(f"\tStd {std}")

    print("-- End of (successful) evolution --")

    best_individual = tools.selBest(population, 1)[0]
    best_individual_series = pl.Series(best_individual)
    selected_descriptors = x_train.select(
        col
        for i, col in enumerate(x_train.columns)
        if best_individual_series[i] > threshold_of_variable_selection
    )
    selected_descriptors.insert_column(
        0,
        pl.Series("", [f"sample_{i + 1}" for i in range(selected_descriptors.height)]),
    ).write_csv("dataset/gapls_selected_x.csv")


if __name__ == "__main__":
    main()
