
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "Crop_recommendation.csv"

df = pd.read_csv(DATA_PATH)

print("=== DIMENSÕES ===")
print(df.shape)

print("\n=== PRIMEIROS REGISTROS ===")
print(df.head())

print("\n=== INFORMAÇÕES DAS COLUNAS ===")
df.info()

print("\n=== VALORES AUSENTES ===")
print(df.isnull().sum())

print("\n=== REGISTROS DUPLICADOS ===")
print(df.duplicated().sum())

print("\n=== DISTRIBUIÇÃO DAS CULTURAS ===")
print(df["label"].value_counts())

print("\n=== ESTATÍSTICAS DESCRITIVAS ===")
print(df.describe())
