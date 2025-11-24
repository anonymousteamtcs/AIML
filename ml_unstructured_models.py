# ml_structured_models.py
# Utility functions for training many ML algorithms on structured data
# and computing common metrics.

from typing import Dict, Any, Tuple

import numpy as np

from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.model_selection import train_test_split

# CLASSIFIERS
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.naive_bayes import (
    GaussianNB,
    MultinomialNB,
    BernoulliNB,
)
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)

# REGRESSORS
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
    BaggingRegressor,
    StackingRegressor,
)
from sklearn.neighbors import KNeighborsRegressor

# UNSUPERVISED / ANOMALY
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FactorAnalysis, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope

# Optional: external libs (install if needed)
try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = None
    XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier = None
    LGBMRegressor = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:
    CatBoostClassifier = None
    CatBoostRegressor = None


# -------------------------
# Helper: metric calculators
# -------------------------

def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "weighted",
) -> Dict[str, float]:
    """
    Compute common classification metrics.
    average: 'binary', 'micro', 'macro', 'weighted', 'samples' (for multiclass use macro/weighted).
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    return metrics


def evaluate_regressor(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute common regression metrics.
    """
    mse = mean_squared_error(y_true, y_pred)
    metrics = {
        "r2": r2_score(y_true, y_pred),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mean_absolute_error(y_true, y_pred),
    }
    return metrics


# -------------------------
# Model builder functions
# -------------------------

def get_classifier(name: str, **kwargs) -> ClassifierMixin:
    """
    Return an untrained classifier instance by name.
    You can extend hyperparameters via **kwargs.
    """
    n = name.lower()

    if n == "logistic_regression":
        return LogisticRegression(max_iter=1000, **kwargs)
    if n == "knn":
        return KNeighborsClassifier(**kwargs)
    if n == "svm" or n == "svc":
        return SVC(probability=True, **kwargs)
    if n == "decision_tree":
        return DecisionTreeClassifier(**kwargs)
    if n == "random_forest":
        return RandomForestClassifier(**kwargs)
    if n == "extra_trees":
        return ExtraTreesClassifier(**kwargs)
    if n == "gradient_boosting":
        return GradientBoostingClassifier(**kwargs)
    if n == "naive_bayes_gaussian":
        return GaussianNB(**kwargs)
    if n == "naive_bayes_multinomial":
        return MultinomialNB(**kwargs)
    if n == "naive_bayes_bernoulli":
        return BernoulliNB(**kwargs)
    if n == "lda":
        return LinearDiscriminantAnalysis(**kwargs)
    if n == "qda":
        return QuadraticDiscriminantAnalysis(**kwargs)

    # Gradient boosting family
    if n == "xgboost":
        if XGBClassifier is None:
            raise ImportError("XGBoost is not installed.")
        return XGBClassifier(eval_metric="logloss", use_label_encoder=False, **kwargs)
    if n == "lightgbm":
        if LGBMClassifier is None:
            raise ImportError("LightGBM is not installed.")
        return LGBMClassifier(**kwargs)
    if n == "catboost":
        if CatBoostClassifier is None:
            raise ImportError("CatBoost is not installed.")
        return CatBoostClassifier(verbose=False, **kwargs)

    # Ensemble wrappers (generic)
    if n == "bagging":
        base = kwargs.pop("base_estimator", DecisionTreeClassifier())
        return BaggingClassifier(estimator=base, **kwargs)
    if n == "voting":
        estimators = kwargs.pop("estimators")
        voting_type = kwargs.pop("voting", "hard")
        return VotingClassifier(estimators=estimators, voting=voting_type, **kwargs)
    if n == "stacking":
        estimators = kwargs.pop("estimators")
        final_estimator = kwargs.pop("final_estimator", LogisticRegression(max_iter=1000))
        return StackingClassifier(estimators=estimators, final_estimator=final_estimator, **kwargs)

    raise ValueError(f"Unknown classifier name: {name}")


def get_regressor(name: str, **kwargs) -> RegressorMixin:
    """
    Return an untrained regressor instance by name.
    """
    n = name.lower()

    if n == "linear_regression":
        return LinearRegression(**kwargs)
    if n == "ridge":
        return Ridge(**kwargs)
    if n == "lasso":
        return Lasso(**kwargs)
    if n == "elastic_net":
        return ElasticNet(**kwargs)
    if n == "polynomial_regression":
        degree = kwargs.pop("degree", 2)
        base_model = kwargs.pop("base_model", LinearRegression())
        return Pipeline(
            [
                ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                ("model", base_model),
            ]
        )
    if n == "svr":
        return SVR(**kwargs)
    if n == "decision_tree":
        return DecisionTreeRegressor(**kwargs)
    if n == "random_forest":
        return RandomForestRegressor(**kwargs)
    if n == "extra_trees":
        return ExtraTreesRegressor(**kwargs)
    if n == "gradient_boosting":
        return GradientBoostingRegressor(**kwargs)
    if n == "knn":
        return KNeighborsRegressor(**kwargs)

    if n == "xgboost":
        if XGBRegressor is None:
            raise ImportError("XGBoost is not installed.")
        return XGBRegressor(**kwargs)
    if n == "lightgbm":
        if LGBMRegressor is None:
            raise ImportError("LightGBM is not installed.")
        return LGBMRegressor(**kwargs)
    if n == "catboost":
        if CatBoostRegressor is None:
            raise ImportError("CatBoost is not installed.")
        return CatBoostRegressor(verbose=False, **kwargs)

    # Ensemble wrappers
    if n == "bagging":
        base = kwargs.pop("base_estimator", DecisionTreeRegressor())
        return BaggingRegressor(estimator=base, **kwargs)
    if n == "voting":
        estimators = kwargs.pop("estimators")
        return VotingRegressor(estimators=estimators, **kwargs)
    if n == "stacking":
        estimators = kwargs.pop("estimators")
        final_estimator = kwargs.pop("final_estimator", LinearRegression())
        return StackingRegressor(estimators=estimators, final_estimator=final_estimator, **kwargs)

    raise ValueError(f"Unknown regressor name: {name}")


# -------------------------
# High-level train & eval
# -------------------------

def train_and_evaluate_classifier(
    model_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    average: str = "weighted",
    **model_kwargs,
) -> Tuple[ClassifierMixin, Dict[str, float]]:
    """
    Train a classifier and return (fitted_model, metrics_dict).
    Metrics: accuracy, precision, recall, f1.
    """
    clf = get_classifier(model_name, **model_kwargs)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    metrics = evaluate_classifier(y_test, y_pred, average=average)
    return clf, metrics


def train_and_evaluate_regressor(
    model_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    **model_kwargs,
) -> Tuple[RegressorMixin, Dict[str, float]]:
    """
    Train a regressor and return (fitted_model, metrics_dict).
    Metrics: r2, mse, rmse, mae.
    """
    reg = get_regressor(model_name, **model_kwargs)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    metrics = evaluate_regressor(y_test, y_pred)
    return reg, metrics


# -------------------------
# Unsupervised helpers
# -------------------------

def get_clusterer(name: str, **kwargs):
    n = name.lower()
    if n == "kmeans":
        return KMeans(**kwargs)
    if n == "hierarchical":
        return AgglomerativeClustering(**kwargs)
    if n == "dbscan":
        return DBSCAN(**kwargs)
    if n == "gmm":
        return GaussianMixture(**kwargs)
    raise ValueError(f"Unknown clusterer name: {name}")


def get_dim_reducer(name: str, **kwargs):
    n = name.lower()
    if n == "pca":
        return PCA(**kwargs)
    if n == "tsne":
        return TSNE(**kwargs)
    if n == "umap":
        # UMAP is external; user must install umap-learn
        try:
            import umap
        except ImportError:
            raise ImportError("umap-learn is not installed.")
        return umap.UMAP(**kwargs)
    if n == "factor_analysis":
        return FactorAnalysis(**kwargs)
    if n == "svd":
        return TruncatedSVD(**kwargs)
    raise ValueError(f"Unknown dim reducer name: {name}")


def get_anomaly_detector(name: str, **kwargs):
    n = name.lower()
    if n == "isolation_forest":
        return IsolationForest(**kwargs)
    if n == "lof":
        return LocalOutlierFactor(novelty=True, **kwargs)
    if n == "one_class_svm":
        return OneClassSVM(**kwargs)
    if n == "elliptic_envelope":
        return EllipticEnvelope(**kwargs)
    raise ValueError(f"Unknown anomaly detector name: {name}")


# Example convenience usage function (optional)

def quick_classification_experiment(
    model_names,
    X,
    y,
    test_size: float = 0.2,
    random_state: int = 42,
    average: str = "weighted",
) -> Dict[str, Dict[str, float]]:
    """
    Quickly train multiple classifiers on a single train/test split
    and return a dict: {model_name: metrics_dict}.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    results = {}
    for name in model_names:
        _, metrics = train_and_evaluate_classifier(
            name, X_train, X_test, y_train, y_test, average=average
        )
        results[name] = metrics
    return results
