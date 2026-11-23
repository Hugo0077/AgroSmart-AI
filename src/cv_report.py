"""Mostra a comparação da validação cruzada gerada pelo train.py.

Não executa novamente o treinamento nem altera o modelo salvo.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artigo" / "comparacao_modelos_cv.csv"
REQUIRED = ["Modelo", "F1_macro_media_CV", "F1_macro_desvio_CV",
            "F1_macro_validacao", "Melhores_parametros"]


def load_comparison(path=REPORT):
    if not path.is_file():
        raise FileNotFoundError(
            f"Relatório não encontrado: {path}. Execute python src/train.py primeiro."
        )
    df = pd.read_csv(path)
    missing = [column for column in REQUIRED if column not in df.columns]
    if missing:
        raise ValueError(
            f"Colunas ausentes no relatório: {missing}. "
            "Execute python src/train.py para gerar o formato atualizado."
        )
    return df


if __name__ == "__main__":
    print("=== COMPARAÇÃO DOS MODELOS ===")
    print(load_comparison().to_string(index=False))
