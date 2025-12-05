import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
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




# Carica il dataset
df = pd.read_csv(DATA_PATH)

# Definisci la soglia per l'HIT (>= 1 milione di copie globali)
HIT_THRESHOLD = 1.0 
TARGET = 'Hit'
ANNO_SPLIT = 2015 # Anno di split ottimizzato per la validazione

# --- 1. Preparazione Iniziale ---

def preparazione_iniziale(df, threshold):
    """Pulisce i dati e crea la variabile target 'Hit'."""
    print("1. Inizio preparazione dati...")
    
    df = df.dropna(subset=['Year_of_Release'])
    df['Year_of_Release'] = df['Year_of_Release'].astype(int)
    
    for col in ['Publisher', 'Developer', 'Rating']:
        df[col] = df[col].fillna('Sconosciuto')
        
    df['Hit'] = (df['Global_Sales'] >= threshold).astype(int)
    
    # Ordina i dati per anno
    df = df.sort_values(by='Year_of_Release').reset_index(drop=True)
    
    # Rimuovi le colonne non utili/di leakage, mantenendo Global_Sales per il calcolo storico
    df_clean = df.drop(columns=['Name', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 
                                'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count'], errors='ignore')
    return df_clean

# --- 2. Feature Engineering Storico (Funzione Riutilizzabile) ---

def calculate_historical_feature(df, group_col, target_col, feature_name, default_value):
    """Calcola la media storica (Hit Rate o Sales Avg) per una colonna categorica."""
    
    df_temp = df.copy()
    
    # 1. Calcola la media globale (fallback)
    global_mean = df_temp[target_col].mean()

    # 2. Inizializza la colonna storica
    df_temp[feature_name] = np.nan
    
    # Itera attraverso gli anni unici
    anni_unici = sorted(df_temp['Year_of_Release'].unique())
    
    for anno in anni_unici:
        # Dati storici (precedenti all'anno corrente)
        df_storico = df_temp[df_temp['Year_of_Release'] < anno]
        idx_anno_corrente = df_temp[df_temp['Year_of_Release'] == anno].index
        
        if not df_storico.empty:
            # Calcola la mappa di Target Encoding / Average Sales
            map_values = df_storico.groupby(group_col)[target_col].mean().to_dict()
            
            # Applica e gestisce le categorie nuove (fillna con la media generale)
            nuovi_valori = df_temp.loc[idx_anno_corrente, group_col].map(map_values).fillna(global_mean)
            df_temp.loc[idx_anno_corrente, feature_name] = nuovi_valori

    # 3. Riempimento finale: copre il primo anno e le categorie non viste
    df_temp[feature_name] = df_temp[feature_name].fillna(global_mean)
    
    return df_temp[feature_name]


def crea_feature_storiche(df):
    """Applica la funzione di calcolo storico per tutte le feature richieste."""
    print("2. Creazione Feature Storiche...")
    df_feat = df.copy()
    
    # A, B, C, D: Hit Rate Storico (Target: Hit)
    hit_rate_cols = ['Publisher', 'Developer', 'Genre', 'Platform']
    for col in hit_rate_cols:
        feature_name = f'{col}_Hit_Rate_Storico'
        df_feat[feature_name] = calculate_historical_feature(
            df_feat, col, TARGET, feature_name, df_feat[TARGET].mean()
        )
        
    # E, F: Average Global Sales Storiche (Target: Global_Sales)
    avg_sales_cols = ['Publisher', 'Genre']
    for col in avg_sales_cols:
        feature_name = f'{col}_Avg_Global_Sales_Storico'
        df_feat[feature_name] = calculate_historical_feature(
            df_feat, col, 'Global_Sales', feature_name, df_feat['Global_Sales'].mean()
        )
        
    # Rimuovi la colonna di vendita originale
    df_feat = df_feat.drop(columns=['Global_Sales'], errors='ignore')
    return df_feat

# --- 3. Modeling ---

def addestra_e_valuta(df_feat, anno_split):
    """Esegue One-Hot Encoding, Split Temporale, Addestramento e Valutazione."""
    print("3. Addestramento e Valutazione del Modello...")
    
    CATEGORICAL_FEATURES = ['Genre', 'Platform', 'Publisher', 'Developer', 'Rating']
    NUMERIC_FEATURES = [
        'Publisher_Hit_Rate_Storico', 'Developer_Hit_Rate_Storico', 
        'Genre_Hit_Rate_Storico', 'Platform_Hit_Rate_Storico', 
        'Publisher_Avg_Global_Sales_Storico', 'Genre_Avg_Global_Sales_Storico'
    ]
    
    # One-Hot Encoding
    df_encoded = pd.get_dummies(df_feat, columns=CATEGORICAL_FEATURES, dummy_na=False)
    
    # Definizione delle feature finali
    FEATURES = NUMERIC_FEATURES + [col for col in df_encoded.columns if any(cat in col for cat in CATEGORICAL_FEATURES)]
    FEATURES = list(set(FEATURES)) 

    # Split Temporale
    X_train = df_encoded[df_encoded['Year_of_Release'] < anno_split][FEATURES]
    y_train = df_encoded[df_encoded['Year_of_Release'] < anno_split][TARGET]
    X_test = df_encoded[df_encoded['Year_of_Release'] >= anno_split][FEATURES]
    y_test = df_encoded[df_encoded['Year_of_Release'] >= anno_split][TARGET]

    # Allinea le colonne (cruciale per OHE)
    common_cols = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    FEATURES = common_cols # Aggiorna la lista FEATURES
    
    print(f"   Training Set (pre {anno_split}): {X_train.shape[0]} oss.")
    print(f"   Test Set (dal {anno_split}): {X_test.shape[0]} oss.")

    # Modello
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Valutazione
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)

    # Importanza delle Feature
    importanze = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    
    return auc_score, importanze


# --- Esecuzione Principale ---

df_temp = preparazione_iniziale(df.copy(), HIT_THRESHOLD)
df_feat = crea_feature_storiche(df_temp)
auc_score, importanze = addestra_e_valuta(df_feat, ANNO_SPLIT)

print("\n--- Risultati Finali del Modello ---")
print(f"Area Sotto la Curva ROC (AUC): {auc_score:.4f}")
print("\nImportanza delle Feature (Top 10):")
print(importanze.head(10))