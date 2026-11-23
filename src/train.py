
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score
)

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "Crop_recommendation.csv")

X = df.drop(columns=["label"])
y = df["label"]

# 15% ficam reservados para o teste final.
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# Dos 85% restantes, separamos 15% do total para validação.
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev,
    test_size=3/17,
    random_state=42,
    stratify=y_dev
)

models = {
    "Dummy": (
        Pipeline([("model", DummyClassifier(strategy="most_frequent"))]),
        {}
    ),
    "Random Forest": (
        Pipeline([("model", RandomForestClassifier(random_state=42))]),
        {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 15]
        }
    ),
    "SVM": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(probability=True, random_state=42))
        ]),
        {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["linear", "rbf"]
        }
    )
}

results = {}
best_name = None
best_score = -1
best_model = None

for name, (pipeline, params) in models.items():
    print(f"\n=== Treinando {name} ===")

    search = GridSearchCV(
        pipeline,
        params,
        scoring="f1_macro",
        cv=3,
        n_jobs=-1
    )

    search.fit(X_train, y_train)
    model = search.best_estimator_

    val_pred = model.predict(X_val)
    val_f1 = f1_score(y_val, val_pred, average="macro")

    results[name] = {
        "cv_f1": search.best_score_,
        "validation_f1": val_f1,
        "params": search.best_params_
    }

    print("F1 validação:", round(val_f1, 4))
    print("Melhores parâmetros:", search.best_params_)

    if val_f1 > best_score:
        best_score = val_f1
        best_name = name
        best_model = model

print("\n=== COMPARAÇÃO ===")
print(pd.DataFrame(results).T.to_string())

print("\nModelo selecionado:", best_name)

# O teste é utilizado somente após a seleção.
test_pred = best_model.predict(X_test)
test_proba = best_model.predict_proba(X_test)

print("\n=== RESULTADOS FINAIS ===")
print("Acurácia:", accuracy_score(y_test, test_pred))
print("F1 macro:", f1_score(y_test, test_pred, average="macro"))
print(classification_report(y_test, test_pred, zero_division=0))

print("AUC-ROC multiclasse:", roc_auc_score(
    y_test, test_proba,
    multi_class="ovr",
    average="macro",
    labels=best_model.classes_
))

print("\nMatriz de confusão:")
print(confusion_matrix(
    y_test, test_pred,
    labels=best_model.classes_
))

# Persistência do pipeline completo.
model_path = ROOT / "models" / "model.joblib"
model_path.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(best_model, model_path)
print("\nModelo salvo em:", model_path)
