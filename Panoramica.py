import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd



# PAGE CONFIG ---------------------------------
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
    
    # pulizia dati come nel notebook
    df['Name'] = df['Name'].fillna('Unknown')
    df['Genre'] = df['Genre'].fillna('Unknown')
    df['Publisher'] = df['Publisher'].fillna('Unknown')
    df['Developer'] = df['Developer'].fillna('Unknown')
    df['Rating'] = df['Rating'].fillna('Unknown')
    
    # per l'anno metto il valore più frequente
    year_mode = df['Year_of_Release'].mode()[0]
    df['Year_of_Release'] = df['Year_of_Release'].fillna(year_mode)
    
    # converto User_Score in numero
    df['User_Score'] = pd.to_numeric(df['User_Score'], errors='coerce')
    
    return df

# --------------------------------- APP ---------------------------------
df = load_data()

# TITOLO PRINCIPALE ---------------------------------
st.title("🎮 Analisi Esplorativa delle Vendite di Videogiochi")
st.markdown("Dashboard interattiva per esplorare i dati delle vendite di videogiochi. Questa analisi è pensata per aiutare il team di analytics a capire quali piattaforme e generi funzionano meglio.")

if len(df) == 0:
    st.warning("⚠️ Nessun dato disponibile con i filtri selezionati. Prova a modificare i filtri nella sidebar.")
    st.stop()

# STATISTICHE GENERALI ---------------------------------
st.header("📊 Panoramica Generale")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Totale Giochi", f"{len(df):,}")

with col2:
    st.metric("Vendite Globali Totali", f"{df['Global_Sales'].sum():.2f}M")

with col3:
    st.metric("Numero di Generi", f"{df['Genre'].nunique()}")

with col4:
    st.metric("Numero di Piattaforme", f"{df['Platform'].nunique()}")

st.divider()





