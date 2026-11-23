
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Crop_recommendation.csv"
OUTPUT = ROOT / "artigo"

df = pd.read_csv(DATA)

X = df.drop(columns=["label"])
y = df["label"]

# Reproduzir exatamente a divisao do treinamento original
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y,
    test_size=0.15,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev,
    test_size=3 / 17,
    random_state=42,
    stratify=y_dev
)

# Mesmos modelos e parametros do experimento original
models = {
    "DummyClassifier": (
        Pipeline([
            ("model", DummyClassifier(strategy="most_frequent"))
        ]),
        {}
    ),
    "Random Forest": (
        Pipeline([
            ("model", RandomForestClassifier(random_state=42))
        ]),
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

results = []

for name, (pipeline, params) in models.items():
    print(f"\nAvaliando: {name}")

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=params,
        scoring="f1_macro",
        cv=3,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    best_idx = search.best_index_

    mean = search.cv_results_["mean_test_score"][best_idx]
    std = search.cv_results_["std_test_score"][best_idx]

    results.append({
        "Modelo": name,
        "F1_macro_media_CV": mean,
        "F1_macro_desvio_CV": std,
        "Melhores_parametros": str(search.best_params_)
    })

    print(f"F1-macro CV: {mean:.6f} +/- {std:.6f}")
    print("Parametros:", search.best_params_)

report = pd.DataFrame(results)

OUTPUT.mkdir(parents=True, exist_ok=True)
report.to_csv(
    OUTPUT / "comparacao_modelos_cv.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n=== RELATORIO FINAL ===")
print(report.to_string(index=False))
print("\nRelatorio CSV salvo com sucesso.")
