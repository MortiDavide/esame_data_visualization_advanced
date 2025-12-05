import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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

    return df


# --------------------------------- APP ---------------------------------
df = load_data()

st.title("🎮 Videogames Sales – Analytics Dashboard")
st.caption("Dashboard esplorativa del catalogo videogiochi con dati di vendita globali")

# PANORAMICA ---------------------------------
st.subheader("📊 Panoramica del Dataset")

col1, col2, col3 = st.columns(3)

col1.metric("Numero giochi", len(df))
col2.metric("Numero piattaforme", df["Platform"].nunique())
col3.metric("Numero generi", df["Genre"].nunique())

with st.expander("Mostra prime righe del dataset"):
    st.dataframe(df.head())