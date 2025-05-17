"""03_04_sg."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.signal import savgol_filter

# SG法の設定値
window_length = 21  # 窓枠の数
poly_order = 2  # 多項式の次数
deriv = 0  # 微分次数 (0 は微分なし)
plot_spectra_number = 12  # 表示するスペクトルのサンプル番号 (0, 1, ..., 227)

csv_path = Path("dataset/sample_spectra_dataset.csv")
preprocessed_csv_path0 = Path(
    f"dataset/preprocessed_sample_spectra_dataset_w{window_length}_p{poly_order}_d{deriv}.csv"
)
preprocessed_csv_path1 = Path(
    f"dataset/preprocessed_sample_spectra_dataset_w{window_length}_p{poly_order}_d{1}.csv"
)
preprocessed_csv_path2 = Path(
    f"dataset/preprocessed_sample_spectra_dataset_w{window_length}_p{poly_order}_d{2}.csv"
)

# データセットの読み込み
x = pl.read_csv(csv_path).drop("")
# SG 法
sg_x0 = savgol_filter(
    x.to_numpy(), window_length=window_length, polyorder=poly_order, deriv=deriv
)
sg_x1 = savgol_filter(
    x.to_numpy(), window_length=window_length, polyorder=poly_order, deriv=1
)
sg_x2 = savgol_filter(
    x.to_numpy(), window_length=window_length, polyorder=poly_order, deriv=2
)
# pl.DataFrameに変換してcsvとして保存
preprocessed_x0 = pl.DataFrame(sg_x0, schema=x.schema)  # type: ignore[assignment]
preprocessed_x1 = pl.DataFrame(sg_x1, schema=x.schema)  # type: ignore[assignment]
preprocessed_x2 = pl.DataFrame(sg_x2, schema=x.schema)  # type: ignore[assignment]
preprocessed_x0.with_row_index("").write_csv(
    preprocessed_csv_path0, quote_style="never"
)
preprocessed_x1.with_row_index("").write_csv(
    preprocessed_csv_path1, quote_style="never"
)
preprocessed_x2.with_row_index("").write_csv(
    preprocessed_csv_path2, quote_style="never"
)

# プロット
fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(7, 7))
wave_lengths = np.array(x.columns, dtype="float64")
fig1 = ax[0, 0]
fig1.plot(wave_lengths, x.row(plot_spectra_number), "b-", label="original")
fig1.set_xlabel("wavelength [nm]")
fig1.set_ylabel("Absorbance")
fig1.set_xlim(wave_lengths[0] - 1, wave_lengths[-1] + 1)
fig1.legend()

fig2 = ax[0, 1]
fig2.plot(
    wave_lengths,
    preprocessed_x0.row(plot_spectra_number),
    "b-",
    label="preprocessed",
)
fig2.set_xlabel("wavelength [nm]")
fig2.set_ylabel("Absorbance")
fig2.set_xlim(wave_lengths[0] - 1, wave_lengths[-1] + 1)
fig2.legend()

fig3 = ax[1, 0]
fig3.plot(
    wave_lengths,
    preprocessed_x1.row(plot_spectra_number),
    "b-",
    label="preprocessed",
)
fig3.set_xlabel("wavelength [nm]")
fig3.set_ylabel("Absorbance")
fig3.set_xlim(wave_lengths[0] - 1, wave_lengths[-1] + 1)
fig3.legend()

fig4 = ax[1, 1]
fig4.plot(
    wave_lengths,
    preprocessed_x2.row(plot_spectra_number),
    "b-",
    label="preprocessed",
)
fig4.set_xlabel("wavelength [nm]")
fig4.set_ylabel("Absorbance")
fig4.set_xlim(wave_lengths[0] - 1, wave_lengths[-1] + 1)
fig4.legend()

plt.subplots_adjust(wspace=0.4, hspace=0.3)
plt.show()
