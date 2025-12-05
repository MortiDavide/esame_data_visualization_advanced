# streamlit_vgsales_app.py
# Esame Finale: Prototipo Streamlit per Game Analytics
# Requisiti: data/vgsales_clean.csv

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import altair as alt

st.set_page_config(page_title="Game Analytics — Executive Prototype", layout="wide")

@st.cache_data
def load_data(path="data/vgsales_clean.csv"):
    df = pd.read_csv(path)
    # Basic cleaning: ensure numeric scores
    df['User_Score'] = pd.to_numeric(df['User_Score'], errors='coerce')
    df['Critic_Score'] = pd.to_numeric(df['Critic_Score'], errors='coerce')
    df['Year_of_Release'] = pd.to_numeric(df['Year_of_Release'], errors='coerce')
    return df

# --- Utility functions ---

def compute_kpis(df):
    last5 = df[df['Year_of_Release'] >= (df['Year_of_Release'].max() - 4)]
    return {
        'GlobalSales_last5y': round(last5['Global_Sales'].sum(), 2),
        'AvgCriticScore': round(df['Critic_Score'].mean(), 2),
        'AvgUserScore': round(df['User_Score'].mean(), 2),
        'NGames': int(df.shape[0])
    }


def prepare_model_data(df):
    # Target: HIT = Global_Sales >= 1.0 (million)
    data = df.copy()
    data['HIT'] = (data['Global_Sales'] >= 1.0).astype(int)
    # Select features (simple, executive-friendly)
    features = ['Platform', 'Genre', 'Year_of_Release', 'Critic_Score', 'User_Score', 'Rating']
    data = data[features + ['HIT']].dropna()
    # Cut extreme years to reasonable range
    data = data[data['Year_of_Release'] >= 2000]
    return data, features


def train_model(data, features, random_state=42):
    X = data[features]
    y = data['HIT']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=random_state)

    numeric_features = ['Year_of_Release', 'Critic_Score', 'User_Score']
    categorical_features = ['Platform', 'Genre', 'Rating']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )

    clf = Pipeline(steps=[('pre', preprocessor), ('clf', RandomForestClassifier(n_estimators=200, random_state=random_state))])
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    return clf, metrics, (X_test, y_test, y_proba)


# --- App ---

df = load_data()

df['Genre'] = df['Genre'].astype(str)

st.title('Game Analytics — Executive Prototype')
st.markdown('Prototipo rapido per supportare decisioni di business sui lanci di videogiochi.')

# Sidebar: filters & data notes
with st.sidebar:
    st.header('Filtri rapidi')
    years = st.slider('Anno di rilascio', int(df['Year_of_Release'].min()), int(df['Year_of_Release'].max()), (2005, 2020))
    selected_platforms = st.multiselect('Piattaforme', options=sorted(df['Platform'].unique()), default=['PS', 'XOne'] if 'XOne' in df['Platform'].unique() else list(df['Platform'].unique())[:2])
    selected_genres = st.multiselect('Generi', options=sorted(df['Genre'].unique()), default=['Action'] if 'Action' in df['Genre'].unique() else list(df['Genre'].unique())[:2])
    show_data_notes = st.checkbox('Note sui dati', value=True)

if show_data_notes:
    st.sidebar.markdown("""
    **Note rapide sui dati:**
    - Il dataset contiene giochi dal 2000 in poi.
    - `Global_Sales` è espresso in milioni di copie.
    - Target definito come `HIT = Global_Sales >= 1.0` (≥ 1M copie).
    - Alcuni giochi mancano di `User_Score` o `Critic_Score` e sono esclusi dalle analisi che li richiedono.
    """)

# Apply filters
df_filt = df[(df['Year_of_Release'] >= years[0]) & (df['Year_of_Release'] <= years[1])]
if selected_platforms:
    df_filt = df_filt[df_filt['Platform'].isin(selected_platforms)]
if selected_genres:
    df_filt = df_filt[df_filt['Genre'].isin(selected_genres)]

# KPIs
kpis = compute_kpis(df_filt)
col1, col2, col3, col4 = st.columns(4)
col1.metric('Giochi nel period selezionato', kpis['NGames'])
col2.metric('Vendite globali ultimi 5 anni (M copie)', kpis['GlobalSales_last5y'])
col3.metric('Media Critic Score', kpis['AvgCriticScore'])
col4.metric('Media User Score', kpis['AvgUserScore'])

st.markdown('---')

# Main charts: Sales over time by platform and genre
st.subheader('1) Andamento vendite per piattaforma e genere')

sales_time = df_filt.groupby(['Year_of_Release', 'Platform'], as_index=False)['Global_Sales'].sum()
chart1 = alt.Chart(sales_time).mark_line(point=True).encode(
    x='Year_of_Release:O',
    y='Global_Sales:Q',
    color='Platform:N',
    tooltip=['Year_of_Release', 'Platform', 'Global_Sales']
).properties(height=300)

st.altair_chart(chart1, use_container_width=True)
st.write('Questo grafico mostra vendite aggregate per piattaforma nel periodo selezionato. Utile per valutare trend e investimenti in porting o piattaforme emergenti.')

# Genre share pie-like (bar)
st.subheader('Distribuzione vendite per genere (periodo selezionato)')
genre_share = df_filt.groupby('Genre', as_index=False)['Global_Sales'].sum().sort_values('Global_Sales', ascending=False)
chart2 = alt.Chart(genre_share.head(10)).mark_bar().encode(
    x='Global_Sales:Q',
    y=alt.Y('Genre:N', sort='-x'),
    tooltip=['Genre', 'Global_Sales']
).properties(height=300)
st.altair_chart(chart2, use_container_width=True)
st.write('Mostra i generi che hanno prodotto più vendite; utile per decisions su greenlighting di nuovi titoli.')

st.markdown('---')

# Reviews vs Sales
st.subheader('2) Recensioni vs Vendite — correlazione')
scatter = df_filt.dropna(subset=['Critic_Score','User_Score'])
if not scatter.empty:
    chart3 = alt.Chart(scatter).mark_circle(size=60).encode(
        x='Critic_Score:Q',
        y='Global_Sales:Q',
        color='Genre:N',
        tooltip=['Name','Platform','Year_of_Release','Critic_Score','User_Score','Global_Sales']
    ).properties(height=400)
    st.altair_chart(chart3, use_container_width=True)
    st.write('Scatter che aiuta a capire se alti punteggi critici si traducono quasi sempre in vendite maggiori. Nota: correlazione non significa causalità; considerare marketing, franchise e piattaforme.')
else:
    st.info('Nessun dato completo per Critic/User Score nel filtro selezionato.')

st.markdown('---')

# ML page
st.header('3) Prototype ML — Predizione probabilità di HIT')
st.markdown('**Target business**: HIT = vendite globali ≥ 1M copie. Questo aiuta a stimare la probabilità che un nuovo prodotto raggiunga il successo commerciale minimo richiesto per considerarlo un "HIT". Visualizziamo un modello semplice, explainable e utilizzabile da un executive.')

with st.expander('Preparazione e training del modello (click per dettagli)'):
    st.write('Alleniamo un modello RandomForest su feature standard: Platform, Genre, Year_of_Release, Critic_Score, User_Score, Rating.')
    data, features = prepare_model_data(df)
    st.write('Dimensione dataset usato per il modello:', data.shape)
    model, model_metrics, (X_test, y_test, y_proba) = train_model(data, features)
    st.write('Metriche sul test set:')
    st.write({'Accuracy': round(model_metrics['accuracy'],3), 'ROC AUC': round(model_metrics['roc_auc'],3)})
    st.write('Confusion matrix (rows: true, cols: predicted):')
    st.write(model_metrics['confusion_matrix'])

# Input form for product card
st.subheader('Simula un nuovo gioco — scheda prodotto')
with st.form('product_form'):
    colA, colB = st.columns(2)
    with colA:
        name = st.text_input('Nome (opzionale)')
        platform = st.selectbox('Platform', options=sorted(df['Platform'].unique()))
        genre = st.selectbox('Genre', options=sorted(df['Genre'].unique()))
        year = st.number_input('Anno di rilascio', min_value=2000, max_value=2030, value=2025)
    with colB:
        critic = st.number_input('Critic Score (0-100)', min_value=0.0, max_value=100.0, value=75.0)
        user = st.number_input('User Score (0-10)', min_value=0.0, max_value=10.0, value=7.5, step=0.1)
        rating = st.selectbox('Rating (ESRB)', options=sorted(df['Rating'].dropna().unique()))
    submitted = st.form_submit_button('Valuta probabilità di HIT')

if submitted:
    # Build input df
    input_df = pd.DataFrame([{ 'Platform': platform, 'Genre': genre, 'Year_of_Release': year, 'Critic_Score': critic, 'User_Score': user, 'Rating': rating }])
    proba = model.predict_proba(input_df)[0][1]
    st.metric('Probabilità stimata di HIT (≥1M copie)', f"{proba:.2%}")
    # semaforo
    if proba >= 0.7:
        st.success('Segnale VERDE — probabilmente un HIT')
    elif proba >= 0.4:
        st.warning('Segnale GIALLO — moderata probabilità')
    else:
        st.error('Segnale ROSSO — bassa probabilità')

    st.write('Esempi di reference:')
    # show example positive and negative
    pos_example = data[data['HIT']==1].sample(1).iloc[0]
    neg_example = data[data['HIT']==0].sample(1).iloc[0]
    st.write('Esempio HIT reale dal dataset:')
    st.write(pos_example.to_dict())
    st.write('Esempio NON-HIT reale dal dataset:')
    st.write(neg_example.to_dict())

st.markdown('---')

# Simple natural-language-ish data query (nice-to-have)
st.header('4) Interroga i dati (NLP-like semplice)')
st.markdown('Scrivi una richiesta semplice come: "mostrami i top 5 giochi action su PS dopo 2015". L\'interprete cercherà parole chiave (platform, genre, top, year) e restituirà la tabella filtrata.')
query = st.text_input('Inserisci una domanda (italiano semplice)')
if query:
    q = query.lower()
    # naive parsing
    res = df.copy()
    # detect platform
    for p in df['Platform'].unique():
        if str(p).lower() in q:
            res = res[res['Platform']==p]
            break
    for g in df['Genre'].unique():
        if str(g).lower() in q:
            res = res[res['Genre']==g]
            break
    # detect year
    import re
    years_found = re.findall(r"(20\d{2})", q)
    if years_found:
        y = int(years_found[0])
        res = res[res['Year_of_Release'] >= y]
    # detect top N
    topN = 10
    top_match = re.search(r"top\s*(\d+)", q)
    if top_match:
        topN = int(top_match.group(1))
    res_table = res.sort_values('Global_Sales', ascending=False).head(topN)
    st.dataframe(res_table[['Name','Platform','Year_of_Release','Genre','Global_Sales']])
    st.write(f'Resultati: {res_table.shape[0]} righe')

st.markdown('---')

st.sidebar.markdown('---')
st.sidebar.write('Versione prototipo — preparato per esame. Per domande tecniche chiedere al team.')

# Footer
st.caption('Note: questo è un prototipo. Per produzione: validare i dati, gestire leakage temporale, bilanciare classi e testare modelli più robusti e explainability (es. SHAP).')
