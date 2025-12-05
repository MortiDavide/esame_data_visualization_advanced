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

# ANALISI GENERALE ---------------------------------
st.subheader("Analisi delle Vendite per Genere e Piattaforma")
st.markdown(
    "In questa sezione analizziamo le vendite globali dei giochi suddivise per **genere** e **piattaforma**. "
    "I grafici mostrano quali generi e piattaforme hanno contribuito maggiormente al successo commerciale del catalogo.")

col1, col2 = st.columns(2)

with col1:
    genre_sales = df.groupby("Genre")["Global_Sales"].sum().sort_values()

    fig_g, ax_g = plt.subplots(figsize=(8, 7))
    genre_sales.plot(kind="barh", ax=ax_g)
    ax_g.set_title("Vendite globali per genere")
    ax_g.set_xlabel("Milioni di copie")
    st.pyplot(fig_g)

with col2:
    platform_sales = df.groupby("Platform")["Global_Sales"].sum().sort_values()

    fig_p, ax_p = plt.subplots(figsize=(8, 7))
    platform_sales.plot(kind="barh", ax=ax_p, color="purple")
    ax_p.set_title("Vendite globali per piattaforma")
    ax_p.set_xlabel("Milioni di copie")
    st.pyplot(fig_p)


# RECENSIONI E RATING ---------------------------------
st.subheader("Impatto di Recensioni e Rating sul Successo Commerciale")
st.markdown("Analizziamo quanto le valutazioni della critica e degli utenti influenzano le vendite globali di un videogioco.")

colA, colB = st.columns(2)

# Filtra Global_Sales <= 20 milioni per il plot, poiché a causa di un outlier il grafico risultava poco leggibile
df_plot = df[df["Global_Sales"] <= 20]

# --- CRITIC SCORE vs SALES ---
with colA:
    fig_cs, ax_cs = plt.subplots(figsize=(6, 5))
    sns.regplot(
        data=df_plot,
        x="Critic_Score",
        y="Global_Sales",
        scatter_kws={"alpha": 0.4},
        line_kws={"color": "red"},
        ax=ax_cs
    )
    ax_cs.set_title("Critic Score → Vendite Globali")
    ax_cs.set_xlabel("Voto Critica")
    ax_cs.set_ylabel("Vendite (milioni)")
    st.pyplot(fig_cs)

# --- USER SCORE vs SALES ---
with colB:
    fig_us, ax_us = plt.subplots(figsize=(6, 5))
    sns.regplot(
        data=df_plot,
        x="User_Score",
        y="Global_Sales",
        scatter_kws={"alpha": 0.4},
        line_kws={"color": "red"},
        ax=ax_us
    )
    ax_us.set_title("User Score → Vendite Globali")
    ax_us.set_xlabel("Voto Utenti")
    ax_us.set_ylabel("Vendite (milioni)")
    st.pyplot(fig_us)