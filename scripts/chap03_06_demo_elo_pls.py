"""03_06 demo_elo_pls."""

import numpy as np
import polars as pl
from dcekit.learning import (  # type: ignore[import-untyped]
    ensemble_outlier_sample_detection,
)
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV, KFold

# Demonstration of Ensemble Learning Outlier sample detection (ELO)
# https://datachemeng.com/ensembleoutliersampledetection/
# https://www.sciencedirect.com/science/article/abs/pii/S0169743917305919

dataset = pl.read_csv("dataset/numerical_simulation_data.csv").drop("")
y = dataset.get_column("y").to_numpy()
x = dataset.drop("y").to_numpy()

# PLS
max_pls_component_number = 30
min_pls_components = np.arange(1, min(max_pls_component_number, x.shape[1]) + 1)
cross_validation = KFold(n_splits=2, random_state=9, shuffle=True)
cv_regressor = GridSearchCV(
    PLSRegression(), {"n_components": min_pls_components}, cv=cross_validation
)
number_of_sub_models = 100
max_iteration_number = 30
outlier_sample_flags = ensemble_outlier_sample_detection(
    cv_regressor,
    x,
    y,
    cv_flag=True,
    n_estimators=number_of_sub_models,
    iteration=max_iteration_number,
    autoscaling_flag=True,
    random_state=0,
)

outlier_sample_flags = pl.DataFrame(outlier_sample_flags)
outlier_sample_flags.columns = ["TRUE if outlier samples"]
outlier_sample_flags.write_csv(
    "dataset/outlier_sample_detection_results.csv", quote_style="never"
)
