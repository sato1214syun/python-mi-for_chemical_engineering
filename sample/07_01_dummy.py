"""@author: Hiromasa Kaneko."""  # noqa: N999

import numpy as np
import pandas as pd

x = pd.read_csv("dataset/iris.csv", index_col=0)  # データセットの読み込み
number_variable_numbers = np.where(x.dtypes != object)[0]  # noqa: E721
str_variable_numbers = np.where(x.dtypes == object)[0]  # noqa: E721

if len(number_variable_numbers) != 0:
    x_with_dummy_variables = x.iloc[:, number_variable_numbers].copy()
else:
    x_with_dummy_variables = pd.DataFrame([])

if len(str_variable_numbers) != 0:
    category_variables = x.iloc[:, str_variable_numbers].copy()
    dummy_variables = pd.get_dummies(category_variables)
    x_with_dummy_variables = pd.concat(
        [x_with_dummy_variables, dummy_variables], axis=1
    )
    x_with_dummy_variables.to_csv("result/x_with_dummy_variables.csv")
