from ml_structured_models import train_and_evaluate_classifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model, metrics = train_and_evaluate_classifier(
    "random_forest", X_train, X_test, y_train, y_test, n_estimators=200
)

print(model)
print(metrics)  # {'accuracy': ..., 'precision': ..., 'recall': ..., 'f1': ...}
