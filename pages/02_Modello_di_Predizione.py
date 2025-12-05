import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Videogames Sales – Analytics Dashboard",
    layout="wide",
    page_icon="🎮"
) 

# ---------------- VARIABILI ----------------
DATA_PATH = Path("data/vgsales_clean.csv")
TARGET = 'Hit'
HIT_THRESHOLD = 1.0
CATEGORICAL_FEATURES = ['Genre', 'Platform', 'Publisher', 'Developer', 'Rating']
NUMERIC_FEATURES = [
    'Publisher_Hit_Rate_Storico', 'Developer_Hit_Rate_Storico', 
    'Genre_Hit_Rate_Storico', 'Platform_Hit_Rate_Storico', 
    'Publisher_Avg_Global_Sales_Storico', 'Genre_Avg_Global_Sales_Storico'
]

# ---------------- UTILS ----------------
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    
    # pulizia dati
    df['Name'] = df['Name'].fillna('Unknown')
    df['Genre'] = df['Genre'].fillna('Unknown')
    df['Publisher'] = df['Publisher'].fillna('Unknown')
    df['Developer'] = df['Developer'].fillna('Unknown')
    df['Rating'] = df['Rating'].fillna('Unknown')
    
    year_mode = df['Year_of_Release'].mode()[0]
    df['Year_of_Release'] = df['Year_of_Release'].fillna(year_mode)
    
    df['User_Score'] = pd.to_numeric(df['User_Score'], errors='coerce')
    
    return df

def preparazione_iniziale(df, threshold):
    df['Year_of_Release'] = df['Year_of_Release'].astype(int)
    for col in ['Publisher', 'Developer', 'Rating']:
        df[col] = df[col].fillna('Sconosciuto')
    df['Hit'] = (df['Global_Sales'] >= threshold).astype(int)
    df = df.sort_values(by='Year_of_Release').reset_index(drop=True)
    df_clean = df.drop(columns=['Name', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 
                                'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count'], errors='ignore')
    return df_clean

def calculate_historical_feature(df, group_col, target_col):
    df_temp = df.copy()
    global_mean = df_temp[target_col].mean()
    df_temp[f"{group_col}_{target_col}_Storico"] = np.nan
    anni_unici = sorted(df_temp['Year_of_Release'].unique())
    for anno in anni_unici:
        df_storico = df_temp[df_temp['Year_of_Release'] < anno]
        idx_anno_corrente = df_temp[df_temp['Year_of_Release'] == anno].index
        if not df_storico.empty:
            map_values = df_storico.groupby(group_col)[target_col].mean().to_dict()
            df_temp.loc[idx_anno_corrente, f"{group_col}_{target_col}_Storico"] = \
                df_temp.loc[idx_anno_corrente, group_col].map(map_values).fillna(global_mean)
    df_temp[f"{group_col}_{target_col}_Storico"] = df_temp[f"{group_col}_{target_col}_Storico"].fillna(global_mean)
    return df_temp[f"{group_col}_{target_col}_Storico"]

def crea_feature_storiche(df):
    df_feat = df.copy()
    for col in ['Publisher', 'Developer', 'Genre', 'Platform']:
        df_feat[f"{col}_Hit_Rate_Storico"] = calculate_historical_feature(df_feat, col, TARGET)
    for col in ['Publisher', 'Genre']:
        df_feat[f"{col}_Avg_Global_Sales_Storico"] = calculate_historical_feature(df_feat, col, 'Global_Sales')
    df_feat = df_feat.drop(columns=['Global_Sales'], errors='ignore')
    return df_feat

def crea_pipeline():
    """Crea una pipeline con preprocessamento e modello"""
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES),
            ('num', 'passthrough', NUMERIC_FEATURES)
        ])
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=3, 
            random_state=42
        ))
    ])
    
    return pipeline

def addestra_modello(df_feat):
    """Addestra il modello usando la pipeline"""
    all_features = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    X_train = df_feat[all_features]
    y_train = df_feat[TARGET]
    
    pipeline = crea_pipeline()
    pipeline.fit(X_train, y_train)
    
    return pipeline

# ---------------- MAIN ----------------
# Carica dati
df = load_data()

# Usa session_state per salvare modello
if "pipeline" not in st.session_state:
    df_temp = preparazione_iniziale(df.copy(), HIT_THRESHOLD)
    df_feat = crea_feature_storiche(df_temp)
    pipeline = addestra_modello(df_feat)
    st.session_state.pipeline = pipeline
    st.session_state.df_feat = df_feat

# Accesso rapido
pipeline = st.session_state.pipeline
df_feat = st.session_state.df_feat

# ---------------- FORM ----------------
st.subheader("🎮 Simula un Nuovo Gioco")

media_hit_generale = df_feat['Hit'].mean()
media_sales_generale = df_feat.get('Global_Sales', pd.Series([1])).mean()

user_input = {}

cols = st.columns(3)
with cols[0]:
    user_input["Name"] = st.text_input("Nome del gioco", value="Nuovo Gioco")
with cols[1]:
    user_input["Year_of_Release"] = st.number_input(
        "Anno di uscita",
        min_value=int(df_feat["Year_of_Release"].min()),
        max_value=2100,
        value=2025,
        step=1
    )
with cols[2]:
    user_input["Genre"] = st.selectbox(
        "Genere",
        options=sorted(df_feat["Genre"].astype(str).unique())
    )

cols2 = st.columns(2)
with cols2[0]:
    user_input["Platform"] = st.selectbox(
        "Piattaforma",
        options=sorted(df_feat["Platform"].astype(str).unique())
    )
with cols2[1]:
    user_input["Rating"] = st.selectbox(
        "Rating",
        options=sorted(df_feat["Rating"].dropna().unique().tolist())
    )

cols3 = st.columns(2)
with cols3[0]:
    user_input["Publisher"] = st.selectbox(
        "Publisher",
        options=sorted(df_feat["Publisher"].astype(str).unique())
    )
with cols3[1]:
    user_input["Developer"] = st.selectbox(
        "Developer",
        options=sorted(df_feat["Developer"].astype(str).unique())
    )

# Feature storiche (medie generali per nuovi publisher/developer)
user_input["Publisher_Hit_Rate_Storico"] = media_hit_generale
user_input["Developer_Hit_Rate_Storico"] = media_hit_generale
user_input["Genre_Hit_Rate_Storico"] = media_hit_generale
user_input["Platform_Hit_Rate_Storico"] = media_hit_generale
user_input["Publisher_Avg_Global_Sales_Storico"] = media_sales_generale
user_input["Genre_Avg_Global_Sales_Storico"] = media_sales_generale

# ---------------- PREDIZIONE ----------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None

if st.button("Predici Probabilità Hit", type="primary"):
    # Prepara input
    input_df = pd.DataFrame([user_input])
    
    # La pipeline gestisce tutto automaticamente!
    all_features = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    input_df = input_df[all_features]
    
    # Predizione
    st.session_state.prediction = pipeline.predict_proba(input_df)[0][1]

# --- Mostra il risultato ---
if st.session_state.prediction is not None:
    proba = st.session_state.prediction
    
    st.divider()
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        st.metric("Probabilità di HIT", f"{proba:.1%}")
    
    with col_res2:
        rischio = "ALTO 🎯" if proba >= 0.5 else "BASSO ⚠️"
        st.metric("Rischio stimato", rischio)
    
    st.info("⚠️ Questa stima è indicativa, basata su dati storici.")