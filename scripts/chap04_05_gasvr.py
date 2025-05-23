"""04_05 GASVR."""

import random
from itertools import batched
from typing import cast

import polars as pl
from deap import base, creator, tools  # type: ignore[import-untyped]
from sklearn import model_selection, svm

# シードの固定
random.seed(100)
# random.seed()  # noqa: ERA001

# データの読み込みと正規化
dataset = pl.read_csv("dataset/selected_descriptors_with_boiling_point.csv").drop("")
y_train = dataset.to_series(0)  # 目的変数
x_train = dataset.drop(y_train.name)  # 説明変数
# 正規化
normalized_y_train = (y_train - y_train.mean()) / y_train.std()
normalized_x_train = x_train.with_columns((pl.all() - pl.all().mean()) / pl.all().std())


# GASVRの設定
# deapのcreatorを使用してFitnessMaxとIndividualのクラスを作成
creator.create(  # for minimization, set weights as (-1.0,)
    "FitnessMax", base.Fitness, weights=(1.0,)
)
creator.create("Individual", list, fitness=creator.FitnessMax)

# toolboxに登録する関数の準備
toolbox = base.Toolbox()
# 個体の生成用関数
# 末尾3つはSVRのハイパーパラメータ
# C, epsilon, gamma の範囲
min_boundary = pl.Series([0.0] * (x_train.width)).append(
    pl.Series([-5, -10, -20]).cast(pl.Float64)
)
max_boundary = pl.Series([1.0] * (x_train.width)).append(
    pl.Series([10, 0, 10]).cast(pl.Float64)
)


def create_individual(min_boundary: pl.Series, max_boundary: pl.Series) -> list:
    """個体の生成用関数."""
    return [
        random.uniform(min_boundary[i], max_boundary[i])  # noqa: S311
        for i in range(len(min_boundary))
    ]


# toolboxに関数を登録
toolbox.register("create_individual", create_individual, min_boundary, max_boundary)
toolbox.register(
    "individual",
    tools.initIterate,
    creator.Individual,  # container
    toolbox.create_individual,  # generator
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# 評価関数
def eval_one_max(  # noqa: PLR0913
    individual: creator.Individual,
    fold_number: int,
    threshold_of_variable_selection: float,
    normalized_x_train: pl.DataFrame,
    normalized_y_train: pl.Series,
    y_train: pl.Series,
) -> tuple[float]:
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
        return (-999,)

    # cross-validation
    rounded_hyper_prams = pl.Series(individual).round()[-3:]
    model_in_cv = svm.SVR(
        kernel="rbf",
        C=2 ** rounded_hyper_prams.item(0),
        epsilon=2 ** rounded_hyper_prams.item(1),
        gamma=2 ** rounded_hyper_prams.item(2),
    )
    cross_val_predicts_np = model_selection.cross_val_predict(
        model_in_cv,
        selected_normalized_x_train,  # type: ignore[arg-type]
        normalized_y_train,
        cv=fold_number,
    )
    estimated_y_train_in_cv = pl.Series("cv_predict", cross_val_predicts_np.flatten())
    # クロスバリデーションの結果を元のスケールに戻す
    estimated_y_train_in_cv = estimated_y_train_in_cv * y_train.std() + y_train.mean()
    # r2を計算
    r2_cv = (
        1
        - (estimated_y_train_in_cv - y_train).pow(2).sum()
        / (y_train - y_train.mean()).pow(2).sum()
    )

    return (r2_cv,)


# 評価関数を登録
toolbox.register(
    "evaluate",
    eval_one_max,
    fold_number=(fold_number := 5),  # クロスバリデーションのfold数
    threshold_of_variable_selection=(  # 染色体の0, 1を分ける閾値
        threshold_of_variable_selection := 0.5
    ),
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


# 最適化開始
print("Start of evolution")

NUM_OF_POPULATION = 100  # GAの個体数
NUM_OF_GENERATION = 150  # GAの世代数
MAX_NUMBER_OF_COMPONENTS = 10  # PLS の最大成分数
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

    print(f"\tEvaluated {num_of_updated_children} individuals")

    fits: list[float] = [individual.fitness.values[0] for individual in population]
    length = len(population)
    mean = sum(fits) / length
    sum2 = sum(fit**2 for fit in fits)
    std = abs(sum2 / length - mean**2) ** 0.5

    print(f"\tMin {min(fits)}")
    print(f"\tMax {max(fits)}")
    print(f"\tAvg {mean}")
    print(f"\tStd {std}")

print("-- End of (successful) evolution --")

# 最適化結果のを保存
best_individual = tools.selBest(population, 1)[0]
best_individual_series = pl.Series(best_individual)
selected_descriptors = x_train.select(
    col
    for i, col in enumerate(x_train.columns)
    if best_individual_series[i] > threshold_of_variable_selection
)
selected_descriptors.insert_column(
    0, pl.Series("", [f"sample_{i + 1}" for i in range(selected_descriptors.height)])
).write_csv("result/gasvr_selected_x.csv", quote_style="never")
# ハイパーパラメーターを保存
selected_hyperparameters = (
    best_individual_series.tail(3)
    .rename("hyperparameters of SVR (log2)")
    .round(0)
    .to_frame()
    .insert_column(0, pl.Series("", ["C", "epsilon", "gamma"]))
)
selected_hyperparameters.write_csv(
    "dataset/gasvr_selected_hyperparameters.csv", quote_style="never"
)
