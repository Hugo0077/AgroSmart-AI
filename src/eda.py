
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Crop_recommendation.csv"
OUTPUT = ROOT / "artigo" / "figuras"

OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
features = df.drop(columns=["label"])

# Figura 1: distribuição das culturas
plt.figure(figsize=(12, 7))
sns.countplot(
    data=df,
    y="label",
    order=df["label"].value_counts().index,
    color="steelblue"
)
plt.title("Distribuição das culturas agrícolas")
plt.xlabel("Quantidade de registros")
plt.ylabel("Cultura")
plt.tight_layout()
plt.savefig(OUTPUT / "distribuicao_culturas.png", dpi=300)
plt.close()

# Figura 2: correlações
plt.figure(figsize=(10, 7))
sns.heatmap(
    features.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlação entre atributos")
plt.tight_layout()
plt.savefig(OUTPUT / "correlacoes.png", dpi=300)
plt.close()

# Figura 3: distribuição e possíveis outliers
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, column in enumerate(features.columns):
    sns.boxplot(y=df[column], ax=axes[i], color="lightgreen")
    axes[i].set_title(column)

axes[-1].axis("off")
fig.suptitle("Distribuição dos atributos e possíveis outliers")
plt.tight_layout()
plt.savefig(OUTPUT / "boxplots.png", dpi=300)
plt.close()

# Quantidade de valores fora dos limites IQR
print("\n=== POSSÍVEIS OUTLIERS (IQR) ===")

for column in features.columns:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    count = ((df[column] < lower) |
             (df[column] > upper)).sum()

    print(f"{column}: {count}")

print("\nGráficos salvos em:", OUTPUT)
