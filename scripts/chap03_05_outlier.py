"""03_05_outlier."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# 設定 ここから
type_of_samples = 0  # 仮想サンプルの種類 0:正規乱数、1:時系列
window_length = 61  # SG 法における窓枠の数
poly_order = 2  # SG 法における多項式の次数
deriv = 0  # SG 法における微分次数 (0 は微分なし)
number_of_samples = 300  # 仮想サンプルの数
noise_rate = 8  # SN比

rng = np.random.default_rng(seed=10)

if type_of_samples == 0:
    outliers = [-20, 6, -6, 25]  # 外れ値
    outlier_indices = [100, 150, 200, 250]  # 外れ値のインデックス
    x = rng.standard_normal(number_of_samples)
elif type_of_samples == 1:
    outliers = [1, 3, 10, -2]  # %外れ値
    outlier_indices = [80, 150, 200, 250]  # 外れ値のインデックス
    x = np.sin(np.arange(number_of_samples) * np.pi / 50)
    noise = rng.standard_normal(number_of_samples)
    noise = noise * (x.var() / noise_rate) ** 0.5
    x += noise
x[outlier_indices] = outliers  # 外れ値の追加

fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(7, 5))

fig1 = ax[0, 0]
fig1.plot(x, "b.")  # プロット
fig1.plot(
    outlier_indices, x[outlier_indices], "r.", label="original outliers"
)  # プロット
fig1.set_xlabel("sample number")  # x 軸の名前
fig1.set_ylabel("x or y")  # y 軸の名前
fig1.legend()


# 3 sigma method
upper_3_sigma = x.mean() + 3 * x.std()
lower_3_sigma = x.mean() - 3 * x.std()
fig2 = ax[0, 1]
fig2.plot(x, "b.")
fig2.plot(outlier_indices, x[outlier_indices], "r.", label="original outliers")
fig2.plot([0, len(x)], [upper_3_sigma, upper_3_sigma], "k-")
fig2.plot([0, len(x)], [lower_3_sigma, lower_3_sigma], "k-")
fig2.set_xlabel("sample number")  # x 軸の名前
fig2.set_ylabel("x or y")  # y 軸の名前
fig2.set_title("3 sigma method")
fig2.legend()

# Hampel identifier
upper_hampel = np.median(x) + 3 * 1.4826 * np.median(np.absolute(x - np.median(x)))
lower_hampel = np.median(x) - 3 * 1.4826 * np.median(np.absolute(x - np.median(x)))
fig3 = ax[1, 0]
fig3.plot(x, "b.")
fig3.plot(outlier_indices, x[outlier_indices], "r.", label="original outliers")
fig3.plot([0, len(x)], [upper_hampel, upper_hampel], "k-")
fig3.plot([0, len(x)], [lower_hampel, lower_hampel], "k-")
fig3.set_xlabel("sample number")  # x 軸の名前
fig3.set_ylabel("x or y")  # y 軸の名前
fig3.set_title("Hampel identifier")
fig3.legend()

# SG method + Hampel identifier
preprocessed_x = savgol_filter(
    x, window_length=window_length, polyorder=poly_order, deriv=deriv
)  # SG 法
x_diff = x - preprocessed_x
upper_sg_hampel = (
    preprocessed_x
    + np.median(x_diff)
    + 3 * 1.4826 * np.median(np.absolute(x_diff - np.median(x_diff)))
)
lower_sg_hampel = (
    preprocessed_x
    + np.median(x_diff)
    - 3 * 1.4826 * np.median(np.absolute(x_diff - np.median(x_diff)))
)
fig4 = ax[1, 1]
fig4.plot(x, "b.")
fig4.plot(outlier_indices, x[outlier_indices], "r.", label="original outliers")
fig4.plot(range(len(x)), upper_sg_hampel, "k-")
fig4.plot(range(len(x)), lower_sg_hampel, "k-")
fig4.set_xlabel("sample number")  # x 軸の名前
fig4.set_ylabel("x or y")  # y 軸の名前
fig4.set_title("SG method + Hampel identifier")
fig4.legend()

plt.subplots_adjust(hspace=0.5)
plt.show()  # 以上の設定で描画
