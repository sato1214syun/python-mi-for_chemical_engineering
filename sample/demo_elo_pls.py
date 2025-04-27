"""@author: Hiromasa Kaneko."""

import numpy as np
import pandas as pd
from dcekit.learning import (  # type: ignore[import-untyped]
    ensemble_outlier_sample_detection,
)
from sklearn.cross_decomposition import PLSRegression  # type: ignore[import-untyped]
from sklearn.model_selection import GridSearchCV, KFold  # type: ignore[import-untyped]

# Demonstration of Ensemble Learning Outlier sample detection (ELO)
# https://datachemeng.com/ensembleoutliersampledetection/
# https://www.sciencedirect.com/science/article/abs/pii/S0169743917305919

number_of_submodels = 100
max_iteration_number = 30
fold_number = 2
max_pls_component_number = 30

dataset = pd.read_csv("numerical_simulation_data.csv", index_col=0)
y = dataset.iloc[:, 0]
x = dataset.iloc[:, 1:]

# PLS
pls_components = np.arange(1, min(max_pls_component_number, x.shape[1]) + 1)
cross_validation = KFold(n_splits=fold_number, random_state=9, shuffle=True)
cv_regressor = GridSearchCV(
    PLSRegression(), {"n_components": pls_components}, cv=cross_validation
)
outlier_sample_flags = ensemble_outlier_sample_detection(
    cv_regressor,
    x,
    y,
    cv_flag=True,
    n_estimators=number_of_submodels,
    iteration=max_iteration_number,
    autoscaling_flag=True,
    random_state=0,
)

outlier_sample_flags = pd.DataFrame(outlier_sample_flags)
outlier_sample_flags.columns = ["TRUE if outlier samples"]
outlier_sample_flags.to_csv("outlier_sample_detection_results.csv")
