
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = ROOT / "artigo" / "comparacao_modelos_cv.csv"
OUTPUT = ROOT / "artigo" / "figuras"

OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)
required = {"Modelo", "F1_macro_media_CV", "F1_macro_desvio_CV"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Colunas ausentes: {sorted(missing)}. Execute python src/train.py.")

fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(
    df["Modelo"],
    df["F1_macro_media_CV"],
    yerr=df["F1_macro_desvio_CV"],
    capsize=6,
    color=["gray", "forestgreen", "steelblue"]
)

ax.set_ylim(0, 1.05)
ax.set_ylabel("F1-macro")
ax.set_xlabel("Modelo")
ax.set_title("Comparação dos modelos - Validação cruzada")

for i, value in enumerate(df["F1_macro_media_CV"]):
    ax.text(
        i,
        value + 0.02,
        f"{value:.4f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    OUTPUT / "comparacao_modelos_cv.png",
    dpi=300
)

plt.close()

print("Gráfico salvo em:", OUTPUT / "comparacao_modelos_cv.png")
