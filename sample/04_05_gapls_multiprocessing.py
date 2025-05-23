"""@author: Hiromasa Kaneko."""  # noqa: N999

import random
from multiprocessing import Pool  # 処理速度向上のために追加
from time import perf_counter

import numpy as np
import pandas as pd
from deap import base, creator, tools  # type: ignore[import-untyped]
from sklearn import model_selection
from sklearn.cross_decomposition import PLSRegression

# シードの固定
random.seed(100)

# 設定 ここから
number_of_population = 100  # GA の個体数
number_of_generation = 150  # GA の世代数
fold_number = 5  # クロスバリデーションの fold 数
max_number_of_components = 10  # PLS の最大成分数
threshold_of_variable_selection = 0.5  # 染色体の 0, 1 を分ける閾値
# 設定 ここまで

probability_of_crossover = 0.5
probability_of_mutation = 0.2

dataset = pd.read_csv(
    "dataset/selected_descriptors_with_boiling_point.csv", index_col=0
)  # データセットの読み込み

y_train = dataset.iloc[:, 0]  # 目的変数
x_train = dataset.iloc[:, 1:]  # 説明変数

# autoscaling
autoscaled_x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
autoscaled_y_train = (y_train - y_train.mean()) / y_train.std(ddof=1)

# GAPLS
creator.create(
    "FitnessMax", base.Fitness, weights=(1.0,)
)  # for minimization, set weights as (-1.0,)
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
min_boundary = np.zeros(x_train.shape[1])
max_boundary = np.ones(x_train.shape[1]) * 1.0


def create_ind_uniform(  # noqa: D103
    min_boundary: np.ndarray, max_boundary: np.ndarray
) -> list[float]:
    index = []
    for min_val, max_val in zip(min_boundary, max_boundary, strict=False):
        index.append(random.uniform(min_val, max_val))  # noqa: S311
    return index


def eval_one_max(individual: list[np.ndarray]) -> tuple[int | np.ndarray]:  # noqa: D103
    individual_array = np.array(individual)
    selected_x_variable_numbers = np.where(
        individual_array > threshold_of_variable_selection
    )[0]
    selected_autoscaled_x_train = autoscaled_x_train.iloc[
        :, selected_x_variable_numbers
    ]
    if len(selected_x_variable_numbers):
        # cross-validation
        pls_components = np.arange(
            1,
            min(
                np.linalg.matrix_rank(selected_autoscaled_x_train) + 1,
                max_number_of_components + 1,
            ),
            1,
        )
        r2_cv_all = []
        for pls_component in pls_components:
            model_in_cv = PLSRegression(n_components=pls_component)
            estimated_y_train_in_cv = np.ndarray.flatten(
                model_selection.cross_val_predict(
                    model_in_cv,
                    selected_autoscaled_x_train,
                    autoscaled_y_train,
                    cv=fold_number,
                )
            )
            estimated_y_train_in_cv = (
                estimated_y_train_in_cv * y_train.std(ddof=1) + y_train.mean()  # type: ignore[assignment]
            )
            r2_cv_all.append(
                1
                - sum((y_train - estimated_y_train_in_cv) ** 2)
                / sum((y_train - y_train.mean()) ** 2)
            )
        value = np.max(r2_cv_all)
    else:
        value = -999

    return (value,)


if __name__ == "__main__":
    start = perf_counter()

    pool = Pool()  # CPU のコア数に合わせて変更
    toolbox.register("map", pool.map)
    toolbox.register("create_ind", create_ind_uniform, min_boundary, max_boundary)
    toolbox.register(
        "individual", tools.initIterate, creator.Individual, toolbox.create_ind
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_one_max)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=number_of_population)

    print("Start of evolution")

    fitnesses = list(toolbox.map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses, strict=False):
        ind.fitness.values = fit

    print(f"  Evaluated {len(pop)} individuals")

    for generation in range(number_of_generation):
        print(f"-- Generation {generation + 1} --")

        offspring = toolbox.select(pop, len(pop))
        offspring = list(toolbox.map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2], strict=False):
            if random.random() < probability_of_crossover:  # noqa: S311
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < probability_of_mutation:  # noqa: S311
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)  # type: ignore[assignment]
        for ind, fit in zip(invalid_ind, fitnesses, strict=False):
            ind.fitness.values = fit

        print(f"  Evaluated {len(invalid_ind)} individuals")

        pop[:] = offspring
        fits = [ind.fitness.values[0] for ind in pop]  # noqa: PD011

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean**2) ** 0.5

        print(f"  Min {min(fits)}")
        print(f"  Max {max(fits)}")
        print(f"  Avg {mean}")
        print(f"  Std {std}")

    print("-- End of (successful) evolution --")

    best_individual = tools.selBest(pop, 1)[0]
    best_individual_array = np.array(best_individual)
    selected_x_variable_numbers = np.where(
        best_individual_array > threshold_of_variable_selection
    )[0]
    selected_descriptors = x_train.iloc[:, selected_x_variable_numbers]
    selected_descriptors.to_csv("dataset/gapls_selected_x.csv")  # 保存

    end = perf_counter()
    minutes, seconds = divmod(end - start, 60)
    print(f"処理時間: {minutes}分{seconds}秒")
