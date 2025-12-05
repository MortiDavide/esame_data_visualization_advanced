import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Videogames Sales – Analytics Dashboard",
    layout="wide",
    page_icon="🎮"
) 

# ---------------- VARIABILI ----------------
DATA_PATH = Path("data/vgsales_clean.csv")
TARGET = 'Hit'
ANNO_SPLIT = 2015
HIT_THRESHOLD = 1.0

# ---------------- UTILS ----------------
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Year_of_Release"] = pd.to_numeric(df["Year_of_Release"], errors="coerce")
    df["User_Score"] = pd.to_numeric(df["User_Score"], errors="coerce")
    return df

def preparazione_iniziale(df, threshold):
    df = df.dropna(subset=['Year_of_Release'])
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
            df_temp.loc[idx_anno_corrente, f"{group_col}_{target_col}_Storico"] = df_temp.loc[idx_anno_corrente, group_col].map(map_values).fillna(global_mean)
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

def addestra_modello(df_feat, anno_split):
    CATEGORICAL_FEATURES = ['Genre', 'Platform', 'Publisher', 'Developer', 'Rating']
    NUMERIC_FEATURES = [
        'Publisher_Hit_Rate_Storico', 'Developer_Hit_Rate_Storico', 
        'Genre_Hit_Rate_Storico', 'Platform_Hit_Rate_Storico', 
        'Publisher_Avg_Global_Sales_Storico', 'Genre_Avg_Global_Sales_Storico'
    ]
    df_encoded = pd.get_dummies(df_feat, columns=CATEGORICAL_FEATURES, dummy_na=False)
    FEATURES = NUMERIC_FEATURES + [col for col in df_encoded.columns if any(cat in col for cat in CATEGORICAL_FEATURES)]
    FEATURES = list(set(FEATURES))
    X_train = df_encoded[df_encoded['Year_of_Release'] < anno_split][FEATURES]
    y_train = df_encoded[df_encoded['Year_of_Release'] < anno_split][TARGET]
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    return model, FEATURES

# ---------------- MAIN ----------------
# Carica dati
df = load_data()

# Usa session_state per salvare modello e FEATURES
if "model" not in st.session_state or "FEATURES" not in st.session_state:
    df_temp = preparazione_iniziale(df.copy(), HIT_THRESHOLD)
    df_feat = crea_feature_storiche(df_temp)
    model, FEATURES = addestra_modello(df_feat, ANNO_SPLIT)
    st.session_state.model = model
    st.session_state.FEATURES = FEATURES
    st.session_state.df_feat = df_feat

# Accesso rapido
model = st.session_state.model
FEATURES = st.session_state.FEATURES
df_feat = st.session_state.df_feat

# ---------------- FORM ----------------
st.subheader("🎮 Simula un Nuovo Gioco (Publisher/Developer Nuovo)")

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
        options=df_feat["Genre"].astype(str).unique()
    )

cols2 = st.columns(2)
with cols2[0]:
    user_input["Platform"] = st.selectbox(
        "Piattaforma",
        options=df_feat["Platform"].astype(str).unique()
    )
with cols2[1]:
    user_input["Rating"] = st.selectbox(
        "Rating",
        options=df_feat["Rating"].dropna().unique().tolist()
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

user_input["Publisher_Hit_Rate_Storico"] = media_hit_generale
user_input["Developer_Hit_Rate_Storico"] = media_hit_generale
user_input["Genre_Hit_Rate_Storico"] = media_hit_generale
user_input["Platform_Hit_Rate_Storico"] = media_hit_generale
user_input["Publisher_Avg_Global_Sales_Storico"] = media_sales_generale
user_input["Genre_Avg_Global_Sales_Storico"] = media_sales_generale

# ---------------- PREDIZIONE ----------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None

if st.button("Predici Probabilità Hit"):
    input_df = pd.DataFrame([user_input])
    for col in ["Genre", "Platform", "Publisher", "Developer", "Rating"]:
        input_df[col] = input_df[col].astype('category')

    # One-hot encoding delle categorie come nel training
    input_df_encoded = pd.get_dummies(input_df, columns=["Genre", "Platform", "Publisher", "Developer", "Rating"], dummy_na=False)

    # Allinea le colonne con quelle del training
    for col in FEATURES:
        if col not in input_df_encoded.columns:
            input_df_encoded[col] = 0
    input_df_encoded = input_df_encoded[FEATURES]

    # Predizione
    st.session_state.prediction = model.predict_proba(input_df_encoded)[0][1]

# --- Mostra il risultato se esiste ---
if st.session_state.prediction is not None:
    proba = st.session_state.prediction
    col_res1, col_res2 = st.columns(2)
    col_res1.metric("Probabilità di HIT", f"{proba:.1%}")
    col_res2.metric("Rischio stimato", "ALTO" if proba >= 0.5 else "BASSO")
    st.write("Valori inseriti:")
    st.json(user_input)
    st.info("⚠️ Questa stima è indicativa, basata su dati storici.")



