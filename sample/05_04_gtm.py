"""@author: Hiromasa Kaneko."""  # noqa: N999
# Demonstration of GTM

import matplotlib.pyplot as plt
import pandas as pd
from dcekit.generative_model import GTM  # type: ignore[import-untyped]

# settings
shape_of_map = [30, 30]  # ２次元平面上のグリッド点の数 (k x k)
shape_of_rbf_centers = [5, 5]  # RBF の数 (q x q)
variance_of_rbfs = 4  # RBF の分散 (sigma^2)
lambda_in_em_algorithm = 0.001  # 正則化項 (λ)
number_of_iterations = 300  # EM アルゴリズムにおける繰り返し回数
display_flag = True  # EM アルゴリズムにおける進捗を表示する (True) かしない (False) か

# load dataset
dataset = pd.read_csv(
    "dataset/selected_descriptors_with_boiling_point.csv", index_col=0
)  # データセットの読み込み

y = dataset.iloc[:, 0]  # 目的変数
x = dataset.iloc[:, 1:]  # 説明変数
autoscaled_x = (x - x.mean()) / x.std()  # オートスケーリング

# construct GTM model
model = GTM(
    shape_of_map,
    shape_of_rbf_centers,
    variance_of_rbfs,
    lambda_in_em_algorithm,
    number_of_iterations,
    display_flag,
)
model.fit(autoscaled_x)

if model.success_flag:
    # calculate responsibilities
    responsibilities = model.responsibility(autoscaled_x)
    means, modes = model.means_modes(autoscaled_x)

    means_pd = pd.DataFrame(means, index=x.index, columns=["t1 (mean)", "t2 (mean)"])
    modes_pd = pd.DataFrame(modes, index=x.index, columns=["t1 (mode)", "t2 (mode)"])
    means_pd.to_csv(
        f"result/gtm_means_{shape_of_map[0]}_{shape_of_map[1]}_{shape_of_rbf_centers[0]}_{shape_of_rbf_centers[1]}_{variance_of_rbfs}_{lambda_in_em_algorithm}_{number_of_iterations}.csv"
    )
    modes_pd.to_csv(
        f"result/gtm_modes_{shape_of_map[0]}_{shape_of_map[1]}_{shape_of_rbf_centers[0]}_{shape_of_rbf_centers[1]}_{variance_of_rbfs}_{lambda_in_em_algorithm}_{number_of_iterations}.csv"
    )

    plt.rcParams["font.size"] = 18
    # plot the mean of responsibilities
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(means[:, 0], means[:, 1])
    plt.ylim(-1.1, 1.1)
    plt.xlim(-1.1, 1.1)
    plt.xlabel("t1 (mean)")
    plt.ylabel("t2 (mean)")
    ax.set_aspect("equal")
    plt.show()

    # plot the mean of responsibilities
    try:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if y.dtype == "O":
            plt.scatter(means[:, 0], means[:, 1], c=pd.factorize(y)[0])
        else:
            plt.scatter(means[:, 0], means[:, 1], c=y)
        plt.ylim(-1.1, 1.1)
        plt.xlim(-1.1, 1.1)
        plt.xlabel("t1 (mean)")
        plt.ylabel("t2 (mean)")
        plt.colorbar()
        ax.set_aspect("equal")
        plt.show()
    except NameError:
        print("y がありません")

    # plot the mode of responsibilities
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(modes[:, 0], modes[:, 1])
    plt.ylim(-1.1, 1.1)
    plt.xlim(-1.1, 1.1)
    plt.xlabel("t1 (mode)")
    plt.ylabel("t2 (mode)")
    ax.set_aspect("equal")
    plt.show()

    # plot the mode of responsibilities
    try:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if y.dtype == "O":
            plt.scatter(modes[:, 0], modes[:, 1], c=pd.factorize(y)[0])
        else:
            plt.scatter(modes[:, 0], modes[:, 1], c=y)
        plt.ylim(-1.1, 1.1)
        plt.xlim(-1.1, 1.1)
        plt.xlabel("t1 (mode)")
        plt.ylabel("t2 (mode)")
        plt.colorbar()
        ax.set_aspect("equal")
        plt.show()
    except NameError:
        print("y がありません")
