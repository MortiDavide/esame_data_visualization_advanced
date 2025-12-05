import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# PAGE CONGFIG ---------------------------------
st.set_page_config(
    page_title="Videogames Sales – Analytics Dashboard",
    layout="wide",
    page_icon="🎮"
) 

# VARIABILI ---------------------------------
DATA_PATH = Path("data/vgsales_clean.csv")

SALES_COLS = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]

# UTILS ---------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    #Years_of_Release è una colonna float, quindi la converto in int
    df["Year_of_Release"] = pd.to_numeric(df["Year_of_Release"], errors="coerce")

    #User_Score è una colonna che contiene "tbd" per i giochi senza voti, quindi la converto in numerica
    df["User_Score"] = pd.to_numeric(df["User_Score"], errors="coerce")

    return df


df = load_data()

# MODELLO DI PREDIZIONE ---------------------------------   

# Definisci la soglia per l'HIT (>= 1 milione di copie globali)
HIT_THRESHOLD = 1.0 

def preparazione_iniziale(df, threshold):
    """Pulisce i dati e crea la variabile target 'Hit'."""
    print("Inizio preparazione dati...")
    
    # Pulizia e conversione Year_of_Release
    df = df.dropna(subset=['Year_of_Release'])
    df['Year_of_Release'] = df['Year_of_Release'].astype(int)
    
    # Gestione valori mancanti per le feature categoriche
    for col in ['Publisher', 'Developer', 'Rating']:
        df[col] = df[col].fillna('Sconosciuto')
        
    # Creazione della Variabile Target: Hit
    df['Hit'] = (df['Global_Sales'] >= threshold).astype(int)
    
    # Ordina i dati per anno (necessario per il lookback temporale)
    df = df.sort_values(by='Year_of_Release').reset_index(drop=True)
    
    # Rimuovi colonne non utili o che causano Data Leakage (come le vendite)
    df_clean = df.drop(columns=['Name', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count'], errors='ignore')
    
    return df_clean

df_base = preparazione_iniziale(df.copy(), HIT_THRESHOLD)


def crea_feature_storiche(df):
    """
    Crea le feature storiche (Hit Rate e Average Sales) utilizzando solo i dati 
    precedenti all'anno di rilascio di ciascun gioco per prevenire Data Leakage.
    """
    df_feat = df.copy()
    
    # Colonne per cui calcolare la Hit Rate Storica (A, B, C, D)
    hit_rate_cols = ['Publisher', 'Developer', 'Genre', 'Platform']
    
    # Colonne per cui calcolare le Average Sales Storiche (E, F)
    avg_sales_cols = ['Publisher', 'Genre']
    
    # Inizializza i placeholder (Useremo un valore NaN o la media generale per il fill-in)
    for col in hit_rate_cols:
        df_feat[f'{col}_Hit_Rate_Storico'] = np.nan
    for col in avg_sales_cols:
        df_feat[f'{col}_Avg_Global_Sales_Storico'] = np.nan

    # Calcola la media di Hit e Sales su tutto il dataset pulito
    media_hit_generale = df_feat['Hit'].mean()
    media_sales_generale = df_feat['Global_Sales'].mean()

    # Itera attraverso gli anni unici del dataset, escludendo il primo
    anni_unici = sorted(df_feat['Year_of_Release'].unique())
    
    for anno in anni_unici:
        # Dati passati (lookback): tutti i giochi rilasciati prima dell'anno corrente
        df_storico = df_feat[df_feat['Year_of_Release'] < anno]
        
        # Dati correnti: giochi rilasciati nell'anno corrente
        idx_anno_corrente = df_feat[df_feat['Year_of_Release'] == anno].index
        
        # Se non ci sono dati storici (primo anno), si riempie con la media generale
        if df_storico.empty:
            continue
            
        # 1. Calcolo Hit Rate Storico (A, B, C, D)
        for col in hit_rate_cols:
            rate_map = df_storico.groupby(col)['Hit'].mean().to_dict()
            nuovi_valori = df_feat.loc[idx_anno_corrente, col].map(rate_map).fillna(media_hit_generale)
            df_feat.loc[idx_anno_corrente, f'{col}_Hit_Rate_Storico'] = nuovi_valori
            
        # 2. Calcolo Average Global Sales Storiche (E, F)
        for col in avg_sales_cols:
            sales_map = df_storico.groupby(col)['Global_Sales'].mean().to_dict()
            nuovi_valori = df_feat.loc[idx_anno_corrente, col].map(sales_map).fillna(media_sales_generale)
            df_feat.loc[idx_anno_corrente, f'{col}_Avg_Global_Sales_Storico'] = nuovi_valori

    # Riempi i NaN rimanenti (per il primissimo anno o categorie mai viste) con la media generale
    for col in hit_rate_cols:
        df_feat[f'{col}_Hit_Rate_Storico'] = df_feat[f'{col}_Hit_Rate_Storico'].fillna(media_hit_generale)
        
    for col in avg_sales_cols:
        df_feat[f'{col}_Avg_Global_Sales_Storico'] = df_feat[f'{col}_Avg_Global_Sales_Storico'].fillna(media_sales_generale)

    print("Feature storiche create e applicate.")
    return df_feat

# Re-includiamo 'Global_Sales' nel set di dati pulito per il calcolo delle medie
df_temp = preparazione_iniziale(df.copy(), HIT_THRESHOLD)
df_feat = crea_feature_storiche(df_temp)



# 3.1. Definizione delle Feature
TARGET = 'Hit'

# Feature numeriche (quelle che abbiamo appena creato)
NUMERIC_FEATURES = [
    'Publisher_Hit_Rate_Storico', 'Developer_Hit_Rate_Storico', 
    'Genre_Hit_Rate_Storico', 'Platform_Hit_Rate_Storico', 
    'Publisher_Avg_Global_Sales_Storico', 'Genre_Avg_Global_Sales_Storico'
]

# Feature categoriche (quelle originali, verranno gestite nativamente da LightGBM)
CATEGORICAL_FEATURES = ['Genre', 'Platform', 'Publisher', 'Developer', 'Rating']

# La Year_of_Release viene usata solo per lo split, non come feature di training
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# 3.2. Split Temporale dei Dati
# Usiamo l'ultimo anno del dataset come set di test
ANNO_TEST = df_feat['Year_of_Release'].max()

X_train = df_feat[df_feat['Year_of_Release'] < ANNO_TEST][FEATURES]
y_train = df_feat[df_feat['Year_of_Release'] < ANNO_TEST][TARGET]
X_test = df_feat[df_feat['Year_of_Release'] == ANNO_TEST][FEATURES]
y_test = df_feat[df_feat['Year_of_Release'] == ANNO_TEST][TARGET]

print(f"\nTraining Set (pre {ANNO_TEST}): {X_train.shape[0]} osservazioni")
print(f"Test Set (anno {ANNO_TEST}): {X_test.shape[0]} osservazioni")

# Converti le colonne categoriche in tipo 'category' (richiesto da LightGBM)
for col in CATEGORICAL_FEATURES:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# 3.3. Addestramento del Modello (LightGBM)
print("\nAddestramento del Modello LightGBM...")

lgbm = lgb.LGBMClassifier(
    objective='binary',
    metric='auc',
    random_state=42,
    n_estimators=100,
    learning_rate=0.05,
    categorical_feature=CATEGORICAL_FEATURES,
    n_jobs=-1 # Usa tutti i core
)

lgbm.fit(X_train, y_train)

# 3.4. Valutazione del Modello
# Previsione delle probabilità (necessaria per l'AUC)
y_pred_proba = lgbm.predict_proba(X_test)[:, 1]

# Calcolo dell'Area Under the Curve (AUC)
auc_score = roc_auc_score(y_test, y_pred_proba)

print(f"--- Risultati del Modello (Anno Test: {ANNO_TEST}) ---")
print(f"Area Sotto la Curva ROC (AUC): {auc_score:.4f}")

# 3.5. Analisi dell'Importanza delle Feature
importanze = pd.Series(lgbm.feature_importances_, index=FEATURES).sort_values(ascending=False)

print("\nImportanza delle Feature:")
print(importanze)


# Creiamo un dizionario per le feature leggibili
FEATURE_LABELS = {
    "Genre": "Genere",
    "Platform": "Piattaforma",
    "Publisher": "Publisher",
    "Developer": "Developer",
    "Rating": "Rating",
    "Publisher_Hit_Rate_Storico": "Hit Rate Storico Publisher",
    "Developer_Hit_Rate_Storico": "Hit Rate Storico Developer",
    "Genre_Hit_Rate_Storico": "Hit Rate Storico Genere",
    "Platform_Hit_Rate_Storico": "Hit Rate Storico Piattaforma",
    "Publisher_Avg_Global_Sales_Storico": "Media Vendite Publisher",
    "Genre_Avg_Global_Sales_Storico": "Media Vendite Genere"
}

st.subheader("🎮 Simula un Nuovo Gioco (Publisher/Developer Nuovo)")

# Medie generali per le feature storiche
media_hit_generale = df_feat['Hit'].mean()
media_sales_generale = df_feat['Global_Sales'].mean()

user_input = {}

cols = st.columns(3)

# 1. Nome gioco
with cols[0]:
    user_input["Name"] = st.text_input("Nome del gioco", value="Nuovo Gioco")

# 2. Anno di uscita
with cols[1]:
    user_input["Year_of_Release"] = st.number_input(
        "Anno di uscita",
        min_value=int(df_feat["Year_of_Release"].min()),
        max_value=int(df_feat["Year_of_Release"].max()) + 1,
        value=int(df_feat["Year_of_Release"].max()) + 1,
        step=1,
        key="year_input"
    )

# 3. Genere
with cols[2]:
    user_input["Genre"] = st.selectbox(
        "Genere",
        options=df_feat["Genre"].astype(str).unique(),
        key="genre_input"
    )

# Nuova riga per piattaforma e rating
cols2 = st.columns(2)

# 4. Piattaforma
with cols2[0]:
    user_input["Platform"] = st.selectbox(
        "Piattaforma",
        options=df_feat["Platform"].astype(str).unique(),
        key="platform_input"
    )

# 5. Rating ESRB
with cols2[1]:
    user_input["Rating"] = st.selectbox(
        "Rating",
        options=df_feat["Rating"].dropna().unique().tolist(),
        key="rating_input"
    )

# Imposta valori medi per le feature storiche e nuovo publisher/developer
user_input["Publisher"] = "Nuovo"
user_input["Developer"] = "Nuovo"
user_input["Publisher_Hit_Rate_Storico"] = media_hit_generale
user_input["Developer_Hit_Rate_Storico"] = media_hit_generale
user_input["Genre_Hit_Rate_Storico"] = media_hit_generale
user_input["Platform_Hit_Rate_Storico"] = media_hit_generale
user_input["Publisher_Avg_Global_Sales_Storico"] = media_sales_generale
user_input["Genre_Avg_Global_Sales_Storico"] = media_sales_generale

# Bottone di predizione
if st.button("Predici Probabilità Hit"):
    # Crea DataFrame con i valori da passare al modello
    input_df = pd.DataFrame([user_input])
    
    # Assicura che le colonne categoriche siano 'category'
    for col in CATEGORICAL_FEATURES:
        input_df[col] = input_df[col].astype('category')
    
    # Calcolo probabilità
    proba = lgbm.predict_proba(input_df[FEATURES])[0][1]
    
    col_res1, col_res2 = st.columns(2)
    col_res1.metric("Probabilità di HIT", f"{proba:.1%}")
    label_hit = "ALTO" if proba >= 0.5 else "BASSO"
    col_res2.metric("Rischio stimato", label_hit)

    st.write("Valori inseriti:")
    st.json({k: v for k, v in user_input.items()})
    
    st.info(
        "⚠️ Questa stima si basa sul modello addestrato su dati storici. "
        "Per giochi nuovi senza storico, le probabilità sono indicative."
    )

