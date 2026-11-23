from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from validation import FEATURES, validate_inputs

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'models' / 'model.joblib'

CROPS_PT = {
    'apple': 'Maçã', 'banana': 'Banana', 'blackgram': 'Feijão-urd-preto',
    'chickpea': 'Grão-de-bico', 'coconut': 'Coco', 'coffee': 'Café',
    'cotton': 'Algodão', 'grapes': 'Uva', 'jute': 'Juta',
    'kidneybeans': 'Feijão-vermelho', 'lentil': 'Lentilha', 'maize': 'Milho',
    'mango': 'Manga', 'mothbeans': 'Feijão-moth', 'mungbean': 'Feijão-mungo',
    'muskmelon': 'Melão', 'orange': 'Laranja', 'papaya': 'Mamão',
    'pigeonpeas': 'Feijão-guandu', 'pomegranate': 'Romã', 'rice': 'Arroz',
    'watermelon': 'Melancia',
}

st.set_page_config(page_title='AgroSmart AI', page_icon='🌱', layout='centered')

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title('🌱 AgroSmart AI')
st.subheader('Sistema Inteligente de Recomendação de Cultivos')
st.info('Informe as características do solo e do clima para estimar uma cultura com base no conjunto de dados estudado.')
st.warning('Protótipo acadêmico. Não substitui avaliação agronômica. A fonte original não esclarece suficientemente as unidades de N, P, K nem o período da precipitação: não use o sistema para decisões de plantio reais.')

try:
    model = load_model()
except (OSError, ValueError, ImportError) as exc:
    st.error(f'Falha ao carregar o modelo salvo: {exc}')
    st.stop()

with st.form('prediction_form'):
    st.subheader('Características do solo')
    col1, col2, col3 = st.columns(3)
    with col1:
        n = st.number_input('Nitrogênio (N)', min_value=0, value=50, step=1)
    with col2:
        p = st.number_input('Fósforo (P)', min_value=0, value=50, step=1)
    with col3:
        k = st.number_input('Potássio (K)', min_value=0, value=40, step=1)
    ph = st.number_input('pH do solo', min_value=0.0, max_value=14.0, value=6.5, step=0.1)

    st.subheader('Condições climáticas')
    temperature = st.number_input('Temperatura (°C)', min_value=-50.0, max_value=60.0, value=25.0, step=0.1)
    humidity = st.number_input('Umidade relativa (%)', min_value=0.0, max_value=100.0, value=70.0, step=0.1)
    rainfall = st.number_input('Precipitação (mm; período não especificado)', min_value=0.0, value=100.0, step=0.1)
    submitted = st.form_submit_button('🌿 Analisar condições', use_container_width=True)

if submitted:
    values = dict(zip(FEATURES, [n, p, k, temperature, humidity, ph, rainfall]))
    errors, outside = validate_inputs(values)
    if errors:
        st.error('Entradas inválidas:\n\n' + '\n\n'.join(f'- {line}' for line in errors))
        st.stop()
    if outside:
        st.warning('Valores fora das faixas do dataset; a previsão pode ser menos confiável:\n\n' + '\n\n'.join(f'- {line}' for line in outside))

    data = pd.DataFrame([values], columns=FEATURES)
    try:
        prediction = model.predict(data)[0]
        probabilities = model.predict_proba(data)[0]
    except (ValueError, AttributeError) as exc:
        st.error(f'Não foi possível realizar a previsão: {exc}')
        st.stop()

    ranking = (pd.DataFrame({'Cultura': [CROPS_PT.get(label, label) for label in model.classes_],
                             'Probabilidade': probabilities})
               .sort_values('Probabilidade', ascending=False).head(3))

    st.divider()
    st.success(f'Cultura prevista: {CROPS_PT.get(prediction, prediction)}')
    st.metric('Probabilidade estimada', f'{ranking.iloc[0]["Probabilidade"]:.2%}')
    st.subheader('Três culturas mais prováveis')
    st.dataframe(ranking.style.format({'Probabilidade': '{:.2%}'}), hide_index=True, use_container_width=True)
    st.caption('As probabilidades são estimativas do modelo e não garantias de sucesso no cultivo.')

    with st.expander('Como a previsão foi realizada?'):
        st.write('O modelo Random Forest foi selecionado após comparação com um baseline e uma SVM. No teste separado, obteve 99,39% de acurácia em 330 registros da mesma base.')
        importance = pd.Series(model.named_steps['model'].feature_importances_, index=FEATURES).sort_values(ascending=False)
        st.write('Importância global dos atributos:')
        st.bar_chart(importance)
        st.caption('Importância global por redução de impureza; não explica causalmente esta previsão individual.')
