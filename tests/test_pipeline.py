from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'models' / 'model.joblib'
FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

@pytest.fixture(scope='module')
def model():
    assert MODEL_PATH.is_file(), 'Execute python src/train.py antes dos testes.'
    return joblib.load(MODEL_PATH)

@pytest.fixture
def example():
    return pd.DataFrame([{'N': 90, 'P': 42, 'K': 43,
                          'temperature': 20.879744, 'humidity': 82.002744,
                          'ph': 6.502985, 'rainfall': 202.935536}], columns=FEATURES)

def test_pipeline_recebe_as_sete_colunas(model):
    assert list(model.feature_names_in_) == FEATURES

def test_previsao_gera_classe_valida(model, example):
    result = model.predict(example)
    assert len(result) == 1
    assert result[0] in model.classes_

def test_probabilidades_somam_um(model, example):
    proba = model.predict_proba(example)
    assert proba.shape == (1, 22)
    assert np.isfinite(proba).all()
    assert ((proba >= 0) & (proba <= 1)).all()
    assert np.isclose(proba.sum(), 1.0)

def test_entrada_com_coluna_faltando_e_rejeitada(model, example):
    with pytest.raises(ValueError):
        model.predict(example.drop(columns=['ph']))
