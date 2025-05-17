"""04_07 GAVDSSVR."""

import random
from itertools import batched, cycle
from typing import cast, overload

import numpy as np
import pandas as pd
import polars as pl
from deap import base, creator, tools  # type: ignore[import-untyped]
from sklearn import model_selection, svm

# シードの固定
random.seed(100)

# for minimization, set weights as (-1.0,)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


def normalize[T: pl.Series | pl.DataFrame | np.ndarray | pd.DataFrame](dataset: T) -> T:
    """正規化."""
    if isinstance(dataset, pl.DataFrame):
        return dataset.with_columns((pl.all() - pl.all().mean()) / pl.all().std())
    if isinstance(dataset, pl.Series):
        return (dataset - dataset.mean()) / dataset.std()
    if isinstance(dataset, (np.ndarray, pd.DataFrame)):
        return (dataset - dataset.mean(axis=0)) / dataset.std(axis=0, ddof=1)
    msg = (
        f"Unsupported type {type(dataset)}. "
        "Expected pl.DataFrame, pl.Series, np.ndarray, or pd.DataFrame."
    )
    raise TypeError(msg)


def create_individual(min_boundary: pl.Series, max_boundary: pl.Series) -> list[float]:
    """個体の生成用関数."""
    index = []
    for min_val, max_val in zip(min_boundary, max_boundary, strict=False):
        index.append(random.uniform(min_val, max_val))  # noqa: S311
    return index


def extract_var_indices(
    individual: creator.Individual,
    number_of_areas: int,
    var_start_indices: pl.Series,
    var_end_indices: pl.Series,
) -> list[int]:
    """individualの各特徴量のr2を計算."""
    return (
        # ハイパーパラメータの列を削除しておく
        pl.DataFrame(individual[: NUM_OF_RANGE * 2], ["idx_and_range"])
        .select(
            pl.col("idx_and_range")
            .floor()  # 小数点切り捨て
            .cast(pl.Int64)
            .reshape((number_of_areas, 2))  # [開始index, 範囲]の配列に並べ替え
            .arr.to_struct(["start_idx", "range"])  # structに変換
        )
        .unnest("idx_and_range")  # structを別の列に展開
        .with_columns(
            # 終了indexを計算
            (pl.col("start_idx") + pl.col("range")).alias("end_idx"),
            # 変数のインデックスを取得
            pl.col("start_idx")
            .map_elements(
                lambda x: ((var_start_indices <= x) & (var_end_indices >= x)).index_of(
                    element=True
                ),
                return_dtype=pl.Int64,
            )
            .alias("target_var_idx_in_start_indices"),
        )
        .drop("range")
        .with_columns(
            # 該当する変数の元々の開始、終了indexを取得
            pl.col("target_var_idx_in_start_indices")
            .map_elements(lambda x: var_start_indices.item(x), return_dtype=pl.Int64)
            .alias("target_var_start_idx"),
            pl.col("target_var_idx_in_start_indices")
            .map_elements(lambda x: var_end_indices.item(x), return_dtype=pl.Int64)
            .alias("target_var_end_idx"),
        )
        .with_columns(
            # 個体の特徴量選択範囲が、各変数枚の範囲を超えていないかチェック
            pl.when(pl.col("end_idx").gt(pl.col("target_var_end_idx")))
            .then(pl.col("target_var_end_idx"))
            .otherwise(pl.col("end_idx"))
            .alias("end_idx"),
        )
        .select(
            # 各範囲のindexを全て抽出し、1次元に並べる
            pl.int_ranges("start_idx", "end_idx")
            .explode()
            .drop_nulls(),  # 範囲が0の時にNoneになるので削除
        )
        .to_series(0)
        .unique(maintain_order=True)
        .to_list()
    )


def cross_validation(  # noqa: PLR0913
    selected_normalized_x_train: pl.DataFrame,
    normalized_y_train: pl.Series,
    fold_number: int,
    c_exp: float,
    epsilon_exp: float,
    gamma_exp: float,
) -> pl.Series:
    """クロスバリデーション."""
    model_in_cv = svm.SVR(
        kernel="rbf",
        C=2 ** round(c_exp),
        epsilon=2 ** round(epsilon_exp),
        gamma=2 ** round(gamma_exp),
    )
    cross_val_predicts_np = model_selection.cross_val_predict(
        model_in_cv, selected_normalized_x_train, normalized_y_train, cv=fold_number
    )

    estimated_y_train_in_cv = pl.Series("cv_predict", cross_val_predicts_np.flatten())

    # クロスバリデーションの結果を元のスケールに戻す
    return estimated_y_train_in_cv * y_train.std() + y_train.mean()


@overload
def calc_r2(y_train: pl.Series, estimated_y_train_cv: pl.DataFrame) -> pl.DataFrame: ...


@overload
def calc_r2(y_train: pl.Series, estimated_y_train_cv: pl.Series) -> float: ...


@overload
def calc_r2(y_train: pl.Series, estimated_y_train_cv: np.ndarray) -> np.ndarray: ...


@overload
def calc_r2(y_train: pl.Series, estimated_y_train_cv: pd.DataFrame) -> pd.DataFrame: ...


def calc_r2(y_train, estimated_y_train_cv):
    """r2を計算."""
    # r2を計算
    if isinstance(estimated_y_train_cv, pl.DataFrame):
        return estimated_y_train_cv.select(
            (
                pl.lit(1)
                - ((pl.all() - y_train) ** 2).sum()
                / ((y_train - y_train.mean()) ** 2).sum()
            ).name.prefix("r2_")
        )
    if isinstance(estimated_y_train_cv, pl.Series):
        return (
            1
            - ((estimated_y_train_cv - y_train) ** 2).sum()
            / ((y_train - y_train.mean()) ** 2).sum()
        )
    if isinstance(estimated_y_train_cv, np.ndarray):
        return (
            1
            - ((estimated_y_train_cv - y_train.to_numpy()) ** 2).sum()
            / ((y_train - y_train.mean()) ** 2).sum()
        )
    if isinstance(estimated_y_train_cv, pd.DataFrame):
        return (
            1
            - ((estimated_y_train_cv - y_train.to_pandas()) ** 2).sum()
            / ((y_train - y_train.mean()) ** 2).sum()
        )

    msg = (
        f"Unsupported type {type(estimated_y_train_cv)}. "
        "Expected pl.DataFrame or pl.Series."
    )
    raise TypeError(msg)


def eval_one_max(  # noqa: PLR0913
    individual: creator.Individual,
    fold_number: int,
    number_of_areas: int,
    var_start_indices: pl.Series,
    var_end_indices: pl.Series,
    normalized_x_train: pl.DataFrame,
    normalized_y_train: pl.Series,
    y_train: pl.Series,
) -> tuple[float, ...]:
    """個体の評価関数."""
    selected_var_indices = extract_var_indices(
        individual, number_of_areas, var_start_indices, var_end_indices
    )

    if not selected_var_indices:
        return (-999,)

    selected_normalized_x_train = normalized_x_train.select(
        # 波長(特徴量)インデックスの重複に対応する
        [
            pl.col(normalized_x_train.columns[idx]).alias(f"{i}")
            for i, idx in enumerate(selected_var_indices)
        ]
    )

    c_exp, epsilon_exp, gamma_exp = individual[-3:]
    estimated_y_train_cv = cross_validation(
        selected_normalized_x_train,
        normalized_y_train,
        fold_number,
        c_exp,
        epsilon_exp,
        gamma_exp,
    )

    r2_cv_all = calc_r2(y_train, estimated_y_train_cv)

    return (r2_cv_all,)


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


def load_dataset(file_path: str) -> pl.DataFrame:
    """データセットを読み込む."""
    return pl.read_csv(file_path).drop("")


def preprocessing_dataset(
    dataset: pl.DataFrame,
    max_dynamics_considered: int = 50,  # 最大いくつまでの時間遅れを考慮するか
    dynamics_span: int = 1,  # 時間遅れの間隔
) -> pl.DataFrame:
    """データセットの前処理."""
    dynamics_length = dataset.height - max_dynamics_considered
    dynamics_dataset_wt_null = dataset.select(
        [
            pl.nth(0).slice(max_dynamics_considered, dynamics_length),
            *[
                pl.col(c)
                .slice(
                    max_dynamics_considered - (i * dynamics_span),
                    dynamics_length,
                )
                .alias(f"{c}_{i}")
                for c in dataset.columns[1:]
                for i in range(max_dynamics_considered // dynamics_span + 1)
            ],
        ]
    )
    return dynamics_dataset_wt_null.with_row_index(
        "", max_dynamics_considered
    ).drop_nulls()


def generate_func_registered_toolbox(  # noqa: PLR0913
    number_of_areas: int,
    min_boundary: pl.Series,
    max_boundary: pl.Series,
    var_start_indices: pl.Series,
    var_end_indices: pl.Series,
    normalized_x_train: pl.DataFrame,
    normalized_y_train: pl.Series,
    y_train: pl.Series,
) -> base.Toolbox:
    """toolboxに関数を登録する."""
    toolbox = base.Toolbox()
    # 個体初期化用の関数を登録
    toolbox.register("create_individual", create_individual, min_boundary, max_boundary)
    toolbox.register(
        "individual", tools.initIterate, creator.Individual, toolbox.create_individual
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 評価関数を登録
    toolbox.register(
        "evaluate",
        eval_one_max,
        fold_number=5,  # クロスバリデーションのfold数
        number_of_areas=number_of_areas,
        var_start_indices=var_start_indices,
        var_end_indices=var_end_indices,
        normalized_x_train=normalized_x_train,
        normalized_y_train=normalized_y_train,
        y_train=y_train,
    )
    # その他の関数はGAPLSと同じ
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox


def mate_and_mutate(
    population: list[creator.Individual],
    toolbox: base.Toolbox,
    crossover_probability: float,
    mutation_probability: float,
) -> list[creator.Individual]:
    """個体の交叉と突然変異を実施."""
    # トーナメント選択で、individual 3個体の中1個体を選択。population数分繰り返す
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
    [mutate(child) for child in offspring if random.random() < mutation_probability]  # noqa: S311
    # 個体群を新世代に置き換え
    return offspring


# データ読み込み
dataset = load_dataset("dataset/debutanizer_y_measurement_span_10.csv")
dataset = dataset.with_columns(pl.nth(0).replace(None, 999).replace(999, None))

# 前処理
MAX_CONSIDERING_DYNAMICS = 50  # 最大いくつまでの時間遅れを考慮するか
DYNAMICS_SPAN = 1  # # 時間遅れの間隔
dynamics_dataset_wo_null_wt_index = preprocessing_dataset(
    dataset, MAX_CONSIDERING_DYNAMICS, DYNAMICS_SPAN
)
valid_indices = dynamics_dataset_wo_null_wt_index.to_series(0)  # 有効なインデックス
y_train = dynamics_dataset_wo_null_wt_index.to_series(1)  # 目的変数
x_train = dynamics_dataset_wo_null_wt_index.drop(
    [valid_indices.name, y_train.name]
)  # 説明変数

# 正規化して、目的変数と説明変数に分割
normalized_dynamics_dataset_wo_null_wt_index = normalize(
    dynamics_dataset_wo_null_wt_index
)
normalized_y_train = normalized_dynamics_dataset_wo_null_wt_index.to_series(1)
normalized_x_train = normalized_dynamics_dataset_wo_null_wt_index.drop(pl.nth(0, 1))

# GAの準備
# toolboxに関数を登録
NUM_OF_RANGE = 5  # 選択する領域の数
MAX_WIDTH_IN_A_AREA = 20  # 選択する領域の幅の最大値
SVR_C_2_RANGE = (-5, 10)  # SVR の C の範囲 (2 の何乗か)
SVR_EPSILON_2_RANGE = (-10, 0)  # SVR の epsilon の範囲 (2 の何乗か)
SVR_GAMMA_2_RANGE = (-20, 10)  # SVR の gamma の範囲 (2 の何乗か)
cycle_iter = cycle([x_train.width, MAX_WIDTH_IN_A_AREA])
# 個体製作事の
min_boundary = pl.Series([0] * NUM_OF_RANGE * 2).append(
    pl.Series([r[0] for r in [SVR_C_2_RANGE, SVR_EPSILON_2_RANGE, SVR_GAMMA_2_RANGE]])
)
max_boundary = pl.Series([next(cycle_iter) for _ in range(NUM_OF_RANGE * 2)]).append(
    pl.Series([r[1] for r in [SVR_C_2_RANGE, SVR_EPSILON_2_RANGE, SVR_GAMMA_2_RANGE]])
)
var_start_indices = pl.int_range(
    0, normalized_x_train.width, MAX_CONSIDERING_DYNAMICS + 1, eager=True
)
var_end_indices = var_start_indices.slice(1).append(
    pl.Series([normalized_x_train.width])
)
toolbox = generate_func_registered_toolbox(
    NUM_OF_RANGE,
    min_boundary,
    max_boundary,
    var_start_indices,
    var_end_indices,
    normalized_x_train,
    normalized_y_train,
    y_train,
)

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
        population = mate_and_mutate(
            population, toolbox, CROSSOVER_PROBABILITY, MUTATION_PROBABILITY
        )

    num_of_updated_children = len(
        [child for child in population if not child.fitness.valid]
    )
    # 各個体の適合度を更新
    population = [
        child if child.fitness.valid else update_fitness(child) for child in population
    ]

    fits: list[float] = [individual.fitness.values[0] for individual in population]  # noqa: PD011
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
selected_var_indices = extract_var_indices(
    best_individual, NUM_OF_RANGE, var_start_indices, var_end_indices
)

selected_descriptors = x_train.select(pl.nth(selected_var_indices))
selected_descriptors.insert_column(0, valid_indices).write_csv(
    "result/gavdssvr_selected_x.csv", quote_style="never"
)

selected_hyperparameters = pd.DataFrame(
    np.round(best_individual[-3:]),
    index=["C", "epsilon", "gamma"],
    columns=["hyperparameters of SVR (log2)"],
)
selected_hyperparameters.to_csv("result/gavdssvr_selected_hyperparameters.csv")  # 保存
