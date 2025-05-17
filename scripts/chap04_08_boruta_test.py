"""04_08 Boruta feature selection."""

import polars as pl
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

# load X and y
# NOTE BorutaPy accepts numpy arrays only, hence the .values attribute
X = pl.read_csv("dataset/boruta/test_X.csv").drop("")
y = pl.read_csv("dataset/boruta/test_y.csv", has_header=False).to_series(1)

# define random forest classifier, with utilizing all cores and
# sampling in proportion to y labels
rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=5)

# define Boruta feature selection method
feat_selector = BorutaPy(rf, n_estimators="auto", verbose=2, random_state=1)

# find all relevant features - 5 features should be selected
feat_selector.fit(X.to_numpy(), y.to_numpy())

# check selected features - first 5 features are selected
print(f"selected features: {feat_selector.support_}")

# check ranking of features
print(f"feature ranking: {feat_selector.ranking_}")

# call transform() on X to filter it down to selected features
X_filtered = feat_selector.transform(X)
X_filtered.write_csv("result/boruta_X_filtered.csv")
