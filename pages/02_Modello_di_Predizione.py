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

    #User_Score è una colonna che contiene "tbd" per i giochi senza voti, quindi la converto in numerica
    df["User_Score"] = pd.to_numeric(df["User_Score"], errors="coerce")

    return df


df = load_data()

# MODELLO DI PREDIZIONE ---------------------------------   