from app.validation import validate_inputs

SAMPLE = {'N': 90, 'P': 42, 'K': 43, 'temperature': 20.88,
          'humidity': 82.0, 'ph': 6.5, 'rainfall': 202.94}

def test_exemplo_valido():
    errors, warnings = validate_inputs(SAMPLE)
    assert errors == []
    assert warnings == []

def test_ph_invalido():
    assert validate_inputs({**SAMPLE, 'ph': 15})[0]

def test_umidade_invalida():
    assert validate_inputs({**SAMPLE, 'humidity': 101})[0]

def test_valor_fora_da_faixa_gera_aviso():
    errors, warnings = validate_inputs({**SAMPLE, 'rainfall': 500})
    assert errors == []
    assert any('rainfall' in w for w in warnings)

def test_coluna_faltando():
    data = dict(SAMPLE)
    data.pop('ph')
    assert validate_inputs(data)[0]

def test_nan_invalido():
    assert validate_inputs({**SAMPLE, 'ph': float('nan')})[0]
