"""@author: Hiromasa Kaneko."""  # noqa: N999

import random

import numpy as np
import pandas as pd
from deap import base, creator, tools  # type: ignore[import-untyped]
from sklearn import model_selection
from sklearn.cross_decomposition import PLSRegression

# 設定 ここから
max_dynamics_considered = 50  # 最大いくつまでの時間遅れを考慮するか
dynamics_span = 1  # 時間遅れの間隔
number_of_areas = 5  # 選択する領域の数
max_width_of_areas = 20  # 選択する領域の幅の最大値
number_of_population = 100  # GA の個体数
number_of_generation = 150  # GA の世代数
fold_number = 5  # クロスバリデーションの fold 数
max_number_of_components = 10  # PLS の最大成分数
# 設定 ここまで

probability_of_crossover = 0.5
probability_of_mutation = 0.2

# load and pre-process dataset
dataset = pd.read_csv("debutanizer_y_measurement_span_10.csv", index_col=0)
dataset = dataset.fillna(999)
original_indexes = dataset.index
original_x_variable_names = dataset.columns[1:]
x_variable_names = []
dataset = np.array(dataset)
if max_dynamics_considered:
    indexes = original_indexes[-(dataset.shape[0] - max_dynamics_considered) :]
    dataset_with_dynamics = np.empty((dataset.shape[0] - max_dynamics_considered, 0))
    dataset_with_dynamics = np.append(
        dataset_with_dynamics, dataset[max_dynamics_considered:, 0:1], axis=1
    )
    for x_variable_number in range(dataset.shape[1] - 1):
        x_variable_names.append(original_x_variable_names[x_variable_number] + f"_{0}")
        dataset_with_dynamics = np.append(
            dataset_with_dynamics,
            dataset[
                max_dynamics_considered:, x_variable_number + 1 : x_variable_number + 2
            ],
            axis=1,
        )
        for time_delay_number in range(
            int(np.floor(max_dynamics_considered / dynamics_span))
        ):
            x_variable_names.append(
                original_x_variable_names[x_variable_number]
                + f"_{time_delay_number + 1}"
            )
            dataset_with_dynamics = np.append(
                dataset_with_dynamics,
                dataset[
                    max_dynamics_considered
                    - (time_delay_number + 1) * dynamics_span : -(time_delay_number + 1)
                    * dynamics_span,
                    x_variable_number + 1 : x_variable_number + 2,
                ],
                axis=1,
            )
else:
    dataset_with_dynamics = dataset.copy()
    indexes = original_indexes

x_train_with_999 = dataset_with_dynamics[:, 1:]
y_train_with_999 = dataset_with_dynamics[:, 0]
meaning_numbers = np.where(y_train_with_999 != 999)[0]
meaning_indexes = indexes[meaning_numbers]
x_train = x_train_with_999[meaning_numbers, :]
y_train = y_train_with_999[meaning_numbers]
x_train = pd.DataFrame(x_train, index=meaning_indexes, columns=x_variable_names)

# autoscaling
autoscaled_x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
autoscaled_y_train = (y_train - y_train.mean()) / y_train.std(ddof=1)

# GAVDSPLS
creator.create(
    "FitnessMax", base.Fitness, weights=(1.0,)
)  # for minimization, set weights as (-1.0,)
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
min_boundary = np.zeros(number_of_areas * 2)
max_boundary = np.ones(number_of_areas * 2) * x_train.shape[1]
max_boundary[np.arange(1, number_of_areas * 2, 2)] = max_width_of_areas


def create_ind_uniform(min_boundary, max_boundary):
    index = []
    for min, max in zip(min_boundary, max_boundary, strict=False):
        index.append(random.uniform(min, max))
    return index


toolbox.register("create_ind", create_ind_uniform, min_boundary, max_boundary)
toolbox.register(
    "individual", tools.initIterate, creator.Individual, toolbox.create_ind
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evalOneMax(individual):
    individual_array = np.array(np.floor(individual), dtype=int)
    first_number_of_process_variables = np.arange(
        0, autoscaled_x_train.shape[1], max_dynamics_considered + 1
    )
    selected_x_variable_numbers = np.zeros(0, dtype=int)
    for area_number in range(number_of_areas):
        check_of_two_process_variables_selected = (
            first_number_of_process_variables - individual_array[2 * area_number]
        ) * (
            first_number_of_process_variables
            - individual_array[2 * area_number]
            - individual_array[2 * area_number + 1]
        )
        flag = np.where(check_of_two_process_variables_selected < 0)[0]
        if len(flag) > 0:
            individual_array[2 * area_number + 1] = (
                first_number_of_process_variables[flag[0]]
                - individual_array[2 * area_number]
            )
        flag = np.where(
            first_number_of_process_variables
            - individual_array[2 * area_number]
            - individual_array[2 * area_number + 1]
            == 0
        )[0]
        if len(flag) > 0:
            individual_array[2 * area_number + 1] = (
                first_number_of_process_variables[flag[0]]
                - individual_array[2 * area_number]
            )

        if (
            individual_array[2 * area_number] + individual_array[2 * area_number + 1]
            <= autoscaled_x_train.shape[1]
        ):
            selected_x_variable_numbers = np.r_[
                selected_x_variable_numbers,
                np.arange(
                    individual_array[2 * area_number],
                    individual_array[2 * area_number]
                    + individual_array[2 * area_number + 1],
                ),
            ]
        else:
            selected_x_variable_numbers = np.r_[
                selected_x_variable_numbers,
                np.arange(
                    individual_array[2 * area_number], autoscaled_x_train.shape[1]
                ),
            ]

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
                estimated_y_train_in_cv * y_train.std(ddof=1) + y_train.mean()
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


toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# random.seed(100)
random.seed()
pop = toolbox.population(n=number_of_population)

print("Start of evolution")

fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses, strict=False):
    ind.fitness.values = fit

print("  Evaluated %i individuals" % len(pop))

for generation in range(number_of_generation):
    print(f"-- Generation {generation + 1} --")

    offspring = toolbox.select(pop, len(pop))
    offspring = list(map(toolbox.clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2], strict=False):
        if random.random() < probability_of_crossover:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < probability_of_mutation:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses, strict=False):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(invalid_ind))

    pop[:] = offspring
    fits = [ind.fitness.values[0] for ind in pop]

    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x * x for x in fits)
    std = abs(sum2 / length - mean**2) ** 0.5

    print("  Min %s" % min(fits))
    print("  Max %s" % max(fits))
    print("  Avg %s" % mean)
    print("  Std %s" % std)

print("-- End of (successful) evolution --")

best_individual = tools.selBest(pop, 1)[0]
best_individual_array = np.array(np.floor(best_individual), dtype=int)
first_number_of_process_variables = np.arange(
    0, autoscaled_x_train.shape[1], max_dynamics_considered
)
selected_x_variable_numbers = np.zeros(0, dtype=int)
for area_number in range(number_of_areas):
    check_of_two_process_variables_selected = (
        first_number_of_process_variables - best_individual_array[2 * area_number]
    ) * (
        first_number_of_process_variables
        - best_individual_array[2 * area_number]
        - best_individual_array[2 * area_number + 1]
    )
    flag = np.where(check_of_two_process_variables_selected < 0)[0]
    if len(flag) > 0:
        best_individual_array[2 * area_number + 1] = (
            first_number_of_process_variables[flag[0]]
            - best_individual_array[2 * area_number]
        )
    flag = np.where(
        first_number_of_process_variables
        - best_individual_array[2 * area_number]
        - best_individual_array[2 * area_number + 1]
        == 0
    )[0]
    if len(flag) > 0:
        best_individual_array[2 * area_number + 1] = (
            first_number_of_process_variables[flag[0]]
            - best_individual_array[2 * area_number]
        )

    if (
        best_individual_array[2 * area_number]
        + best_individual_array[2 * area_number + 1]
        <= autoscaled_x_train.shape[1]
    ):
        selected_x_variable_numbers = np.r_[
            selected_x_variable_numbers,
            np.arange(
                best_individual_array[2 * area_number],
                best_individual_array[2 * area_number]
                + best_individual_array[2 * area_number + 1],
            ),
        ]
    else:
        selected_x_variable_numbers = np.r_[
            selected_x_variable_numbers,
            np.arange(
                best_individual_array[2 * area_number], autoscaled_x_train.shape[1]
            ),
        ]

selected_descriptors = x_train.iloc[:, selected_x_variable_numbers]
selected_descriptors.to_csv("gavdspls_selected_x.csv")  # 保存
