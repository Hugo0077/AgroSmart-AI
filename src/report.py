
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    f1_score
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artigo" / "figuras"
OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    ROOT / "data" / "Crop_recommendation.csv"
)

X = df.drop(columns=["label"])
y = df["label"]

# Reconstituir a mesma divisão do treinamento.
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y,
    test_size=0.15,
    random_state=42,
    stratify=y
)

model = joblib.load(ROOT / "models" / "model.joblib")
predictions = model.predict(X_test)

# Figura 4 - Matriz de confusão
fig, ax = plt.subplots(figsize=(15, 13))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    labels=model.classes_,
    xticks_rotation=90,
    cmap="Blues",
    ax=ax
)

ax.set_title("Matriz de Confusão - Random Forest")
plt.tight_layout()
plt.savefig(
    OUTPUT / "matriz_confusao.png",
    dpi=300
)
plt.close()

# Figura 5 - Importância dos atributos
importance = pd.Series(
    model.named_steps["model"].feature_importances_,
    index=X.columns
).sort_values()

plt.figure(figsize=(9, 5))
importance.plot.barh()
plt.title("Importância dos Atributos - Random Forest")
plt.xlabel("Importância relativa")
plt.tight_layout()
plt.savefig(
    OUTPUT / "importancia_atributos.png",
    dpi=300
)
plt.close()

# Relatório de métricas
report = classification_report(
    y_test,
    predictions,
    output_dict=True,
    zero_division=0
)

pd.DataFrame(report).T.to_csv(
    ROOT / "artigo" / "metricas_por_classe.csv"
)

print("Acurácia:", accuracy_score(y_test, predictions))
print("F1 macro:", f1_score(y_test, predictions, average="macro"))

print("\nImportância dos atributos:")
print(importance.sort_values(ascending=False))

print("\nFiguras e métricas salvas com sucesso!")
