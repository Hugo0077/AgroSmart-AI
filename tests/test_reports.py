"""Testes do contrato entre o treinamento e os relatórios para o artigo."""
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "artigo" / "comparacao_modelos_cv.csv"


def test_csv_comparacao_colunas_padronizadas():
    df = pd.read_csv(CSV)
    assert {"Modelo", "F1_macro_media_CV", "F1_macro_desvio_CV",
            "F1_macro_validacao", "Melhores_parametros"}.issubset(df.columns)
    assert set(df["Modelo"]) == {"DummyClassifier", "Random Forest", "SVM"}


def test_metricas_validas():
    df = pd.read_csv(CSV)
    assert df["F1_macro_media_CV"].between(0, 1).all()
    assert df["F1_macro_desvio_CV"].ge(0).all()
    assert df["F1_macro_validacao"].between(0, 1).all()


def test_random_forest_melhor_cv():
    df = pd.read_csv(CSV)
    winner = df.loc[df["F1_macro_media_CV"].idxmax(), "Modelo"]
    assert winner == "Random Forest"
