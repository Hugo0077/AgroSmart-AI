# 🌱 AgroSmart AI

**Sistema Inteligente de Recomendação de Cultivos Agrícolas Utilizando Aprendizado de Máquina**

O AgroSmart AI é um protótipo acadêmico desenvolvido como trabalho final da disciplina de Inteligência Artificial, com o objetivo de classificar culturas agrícolas com base nas características do solo e nas condições climáticas.

O sistema utiliza algoritmos de aprendizado de máquina supervisionado para analisar sete atributos e estimar a cultura agrícola mais compatível com os padrões presentes no conjunto de dados.

**Aviso:** trata-se de um sistema experimental. As previsões não constituem recomendações agronômicas profissionais e não garantem produtividade ou viabilidade de cultivo em condições reais.

## 1. Tecnologias utilizadas

- Python 3.11
- pandas 3.1.1
- scikit-learn 1.9.1
- Streamlit 1.67.0
- joblib 1.6.0
- NumPy
- Matplotlib
- Seaborn
- pytest
- Git e GitHub

As dependências utilizadas estão registradas no arquivo `requirements.txt`.

## 2. Conjunto de dados

**Nome:** Crop Recommendation Dataset  
**Autor da publicação:** Atharva Ingle  
**Plataforma:** Kaggle  
**Fonte:** https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

O dataset contém:

- 2.200 registros.
- 7 atributos de entrada.
- 22 culturas agrícolas.
- 100 registros para cada cultura.
- Nenhum valor ausente.
- Nenhum registro completamente duplicado.

### Atributos

| Atributo | Descrição |
|---|---|
| N | Nitrogênio do solo |
| P | Fósforo do solo |
| K | Potássio do solo |
| temperature | Temperatura em °C |
| humidity | Umidade relativa (%) |
| ph | pH do solo |
| rainfall | Precipitação em mm |
| label | Cultura agrícola — variável-alvo |

As unidades de N, P e K, assim como o período de referência da precipitação, precisam de confirmação documental.

**Licença do dataset:** confirmar na página original do Kaggle e registrar a versão consultada antes da entrega. A cópia do CSV não é redistribuída neste repositório.

O CSV original não é incluído no repositório. Para reproduzir os experimentos, deve ser baixado diretamente da fonte.

## 3. Instalação

### Requisitos

- Python 3.11
- Git
- Conexão com a internet para instalar dependências
- Dataset original para reproduzir o treinamento
- A aplicação online não precisa do CSV original, pois utiliza o pipeline salvo

Clone o repositório:

```bash
git clone https://github.com/Hugo0077/AgroSmart-AI.git
cd AgroSmart-AI
```

### Criar ambiente virtual — Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Instalar as dependências:

```powershell
python -m pip install -r requirements.txt
```

### Preparação dos dados

Baixe o dataset no Kaggle, extraia o arquivo CSV e coloque-o no seguinte caminho:

```text
data/Crop_recommendation.csv
```

## 4. Executar a aplicação

Como o repositório inclui o pipeline treinado em `models/model.joblib`, não é necessário repetir o treinamento para utilizar a aplicação.

Execute:

```powershell
python -m streamlit run app/main.py
```

Acesse:

http://localhost:8501

A interface permite informar os sete atributos e apresenta a cultura prevista, sua probabilidade estimada, as três culturas mais prováveis e a importância global dos atributos do modelo.

O sistema também valida entradas e apresenta avisos quando os dados informados estão fora das faixas observadas no treinamento.

## 5. Reproduzir os experimentos

### Análise exploratória

```powershell
python src/inspect_data.py
python src/eda.py
```

A análise inclui estatísticas descritivas, distribuição das culturas, correlações e identificação de possíveis outliers pelo método IQR.

Os registros extremos foram mantidos inicialmente, pois diferenças entre culturas podem produzir valores considerados extremos na distribuição geral.

### Treinamento

```powershell
python src/train.py
```

O treinamento utiliza:

- 70% dos registros para treinamento.
- 15% para validação.
- 15% para teste.
- Divisão estratificada.
- Semente aleatória 42.
- Busca de hiperparâmetros com GridSearchCV e três partições de validação cruzada.
- F1-macro como métrica principal de comparação.

São avaliadas três abordagens:

1. DummyClassifier — baseline.
2. Random Forest.
3. Support Vector Machine (SVM).

O pré-processamento necessário é realizado dentro do pipeline, evitando ajustar transformações com informações do conjunto de teste.

O `src/train.py` grava os relatórios, mas **preserva o modelo existente** em `models/model.joblib` por padrão. Para substituir intencionalmente o modelo publicado, utilize `python src/train.py --overwrite-model`, após verificar os resultados e criar um backup. O teste final não deve orientar novos ajustes de hiperparâmetros.

### Relatórios

```powershell
python src/report.py
```

Para visualizar o CSV gerado pelo treinamento e atualizar o gráfico de comparação (sem retreinar):

```powershell
python src/cv_report.py
python src/plot_cv.py
```

## 6. Resultados experimentais

### Comparação dos modelos

| Modelo | F1-macro médio CV | Desvio-padrão CV |
|---|---:|---:|
| DummyClassifier | 0,003899 | 0,000003 |
| Random Forest | **0,992861** | **0,000918** |
| SVM | 0,986948 | 0,003223 |

O Random Forest apresentou o melhor desempenho de validação entre os modelos avaliados. Os desvios-padrão refletem três partições de validação cruzada. O relatório `artigo/comparacao_modelos_cv.csv` inclui também o F1-macro no conjunto de validação.

**Hiperparâmetros selecionados:**

- `n_estimators = 100`
- `max_depth = 15`
- `random_state = 42`

### Avaliação final do Random Forest

| Métrica | Resultado |
|---|---:|
| Acurácia | 0,993939 |
| F1-macro | 0,993933 |
| AUC-ROC multiclasse OvR | 1,000000 |
| Amostras de teste | 330 |
| Classificações corretas | 328 |
| Classificações incorretas | 2 |

Os erros observados foram:

- 1 amostra de blackgram classificada como milho.
- 1 amostra de arroz classificada como juta.

### Importância dos atributos

| Atributo | Importância |
|---|---:|
| Precipitação | 0,236077 |
| Umidade | 0,206864 |
| Potássio | 0,167713 |
| Fósforo | 0,149242 |
| Nitrogênio | 0,112458 |
| Temperatura | 0,072392 |
| pH | 0,055254 |

Essas importâncias foram calculadas por redução de impureza no Random Forest e não demonstram relações causais.

Os resultados apresentados foram obtidos experimentalmente no conjunto de dados utilizado. Não representam garantia de desempenho em propriedades agrícolas reais.

## 7. Testes automatizados

O projeto inclui testes para verificar:

- Presença dos sete atributos.
- Retorno de uma classe válida.
- Probabilidades entre 0 e 1 e soma igual a 1.
- Rejeição de colunas ausentes.
- Validação de pH e umidade.
- Tratamento de valores inválidos ou ausentes.
- Avisos para entradas fora das faixas de treinamento.

Para executar:

```powershell
python -m pytest tests/ -v
```

**Resultado registrado:** 10 testes aprovados.

Os testes também foram executados com sucesso em um segundo ambiente virtual, no qual a aplicação Streamlit iniciou e realizou uma previsão funcional.

## 8. Estrutura do projeto

```text
AgroSmart-AI/
├── app/
│   ├── main.py
│   └── validation.py
├── artigo/
│   ├── figuras/
│   ├── comparacao_modelos_cv.csv
│   └── metricas_por_classe.csv
├── data/
│   └── Crop_recommendation.csv
├── models/
│   └── model.joblib
├── notebooks/
│   └── analise_exploratoria_agrosmart.ipynb
├── src/
│   ├── inspect_data.py
│   ├── eda.py
│   ├── train.py
│   ├── report.py
│   ├── cv_report.py
│   └── plot_cv.py
├── tests/
│   ├── test_pipeline.py
│   └── test_validation.py
├── requirements.txt
├── .gitignore
└── README.md
```

O arquivo CSV permanece local e deve ser obtido pelo usuário na fonte original.

## 9. Limitações e considerações éticas

O AgroSmart AI foi desenvolvido para fins acadêmicos e apresenta limitações importantes:

- O dataset não foi validado pelos autores deste projeto em propriedades agrícolas brasileiras.
- As previsões dependem dos padrões presentes no conjunto de dados.
- Os resultados de teste não garantem generalização para regiões ou condições não representadas.
- As probabilidades estimadas não foram submetidas a uma avaliação específica de calibração.
- O sistema não considera todos os fatores relevantes para uma decisão agrícola, como custo, disponibilidade hídrica, pragas, produtividade e condições locais.
- A origem e representatividade dos dados precisam ser consideradas antes de qualquer aplicação prática.

A utilização da ferramenta deve ser restrita a demonstrações e experimentos acadêmicos, não substituindo profissionais da área agronômica.

## 10. Vídeo de demonstração

**Link:** a adicionar após a gravação.

**Aplicação pública:** https://agrosmart-ai-v2.streamlit.app

O vídeo demonstrará a aplicação em funcionamento, as entradas utilizadas, os resultados do modelo e suas limitações.

## 11. Uso de inteligência artificial generativa

Ferramentas de IA generativa foram utilizadas como apoio ao planejamento, desenvolvimento, revisão de código, depuração e organização documental.

Os resultados numéricos apresentados são provenientes dos experimentos realizados com os modelos de aprendizado de máquina.

O artigo científico incluirá uma declaração específica de uso de IA generativa conforme as exigências da disciplina.

## 12. Contexto acadêmico

Projeto desenvolvido para a disciplina de Inteligência Artificial, período 2026.2, da Universidade Federal do Maranhão (UFMA).

**Docente:** Profa. Dra. Alana Oliveira.

**Equipe:** identificar os integrantes na versão final do artigo.

**Repositório:** https://github.com/Hugo0077/AgroSmart-AI


## 13. Verificações antes da entrega

- Confirmar versão e licença do dataset diretamente na fonte.
- Executar `python -m pytest tests/ -q` após qualquer atualização.
- Conferir a publicação no Streamlit Community Cloud e testar a previsão do arroz.
- Inserir o link final do vídeo na seção 10.
- Executar `python src/train.py` em uma instalação separada para confirmar a reprodução do treinamento, preservando o modelo publicado.
