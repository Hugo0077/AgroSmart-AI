# AgroSmart AI

Protótipo acadêmico de classificação de culturas agrícolas a partir de sete variáveis de solo/clima, treinado sobre o **Crop Recommendation Dataset** (Atharva Ingle, Kaggle). **Não é recomendação agronômica validada para uso real.** As unidades de N, P, K e o período de referência da precipitação precisam de validação na fonte.

## Requisitos

- Python 3.11 (ambiente utilizado no desenvolvimento)
- Dependências de `requirements.txt`
- Dataset `Crop_recommendation.csv` em `data/`
- Pipeline treinado em `models/model.joblib`, gerado com `src/train.py`

## Instalação e execução (Windows / PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/inspect_data.py
python src/eda.py
python src/train.py
python src/report.py
python -m pytest tests/ -v
python -m streamlit run app/main.py
```

Abrir `http://localhost:8501` caso o navegador não abra automaticamente.

> Atenção: `src/train.py` e `src/report.py` são os scripts de treinamento e avaliação produzidos durante o trabalho; mantenha-os no repositório. `report.py` pressupõe a mesma divisão estratificada de teste (15%, `random_state=42`) usada em `train.py`.

## Conjunto de dados

Fonte: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

Formato esperado: 2.200 linhas, 7 entradas (`N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`) e rótulo `label` com 22 culturas. Consultar a licença e atribuição indicadas na página do fornecedor antes de redistribuir o CSV.

## Modelagem e resultados (execução de 23/11/2026)

- Partição estratificada: 70% treino, 15% validação e 15% teste; semente 42.
- Baseline `DummyClassifier`, `RandomForestClassifier` e `SVC`.
- Seleção com F1-macro na validação; busca com `GridSearchCV` (3 folds) no treino.
- Random Forest selecionado: `n_estimators=100`, `max_depth=15`.
- Teste (330 exemplos): acurácia **0,993939**, F1-macro **0,993933**, AUC-ROC OvR **1,0**; 2 erros.
- Importância global via redução de impureza: precipitação (0,236077), umidade (0,206864), K (0,167713), P (0,149242), N (0,112458), temperatura (0,072392), pH (0,055254).

**Limitações:** o desempenho foi medido em uma separação do mesmo dataset, não em amostras agrícolas externas. Probabilidades de classificação não são garantias de rendimento ou adequação real; a calibração não foi medida. Não utilizar a ferramenta como recomendação profissional.

## Testes

`python -m pytest tests/ -v` inclui verificações do pipeline persistido e das regras de validação. Os testes do pipeline exigem `models/model.joblib` gerado anteriormente.

## Estrutura esperada

```text
AgroSmart-AI/
  app/main.py
  app/validation.py
  data/Crop_recommendation.csv
  src/inspect_data.py
  src/eda.py
  src/train.py
  src/report.py
  models/model.joblib
  tests/test_pipeline.py
  tests/test_validation.py
  notebooks/
  artigo/figuras/
  requirements.txt
  README.md
```

## Reprodutibilidade e apresentação

Os resultados e figuras devem ser obtidos executando os scripts e registrados no artigo SBC. Inclua no artigo a declaração de uso de ferramentas de IA generativa, como exige a disciplina. O vídeo deve demonstrar a entrada dos sete campos, a previsão de arroz para um registro de exemplo, a visualização de probabilidades e a explicação de limitações. Não confundir teste funcional com estimativa de generalização.
