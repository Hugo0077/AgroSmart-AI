"""Validação de entradas e alertas sobre o domínio de treinamento."""
from math import isfinite

FEATURES = ('N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall')
TRAINING_RANGES = {
    'N': (0.0, 140.0), 'P': (5.0, 145.0), 'K': (5.0, 205.0),
    'temperature': (8.825675, 43.675493),
    'humidity': (14.258040, 99.981876),
    'ph': (3.504752, 9.935091),
    'rainfall': (20.211267, 298.560117),
}

def validate_inputs(values):
    """Retorna (erros, avisos); avisos não impedem a predição."""
    errors, warnings = [], []
    if set(values) != set(FEATURES):
        missing = set(FEATURES) - set(values)
        extra = set(values) - set(FEATURES)
        return ([f'Campos ausentes: {sorted(missing)}; campos inesperados: {sorted(extra)}'], [])
    for key in FEATURES:
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not isfinite(value):
            errors.append(f'{key}: informe um número finito.')
            continue
        if key in ('N', 'P', 'K', 'rainfall') and value < 0:
            errors.append(f'{key}: não pode ser negativo.')
        elif key == 'humidity' and not 0 <= value <= 100:
            errors.append('Umidade deve estar entre 0 e 100%.')
        elif key == 'ph' and not 0 <= value <= 14:
            errors.append('pH deve estar entre 0 e 14.')
        else:
            low, high = TRAINING_RANGES[key]
            if not low <= value <= high:
                warnings.append(f'{key}: {value:g} (faixa observada: {low:g} a {high:g})')
    return errors, warnings
