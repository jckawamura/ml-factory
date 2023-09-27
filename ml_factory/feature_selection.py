from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold

def selector():
    return BorutaPy(
        RandomForestClassifier(class_weight="balanced", max_depth=5),
        n_estimators="auto",
    )
