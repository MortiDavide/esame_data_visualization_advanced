import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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


df = load_data()

# SIDEBAR ---------------------------------
with st.sidebar:
    
    st.header("🔍 Filtri Dataset")
    
    # Filtro per anno
    min_year = int(df['Year_of_Release'].min())
    max_year = int(df['Year_of_Release'].max())
    year_range = st.slider(
        "Seleziona range di anni:",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )
    
    # Filtro per piattaforme
    all_platforms = sorted(df['Platform'].unique().tolist())
    selected_platforms = st.multiselect(
        "Filtra per piattaforme:",
        options=all_platforms,
        default=all_platforms,
        help="Lascia vuoto per mostrare tutte"
    )
    
    # Filtro per generi
    all_genres = sorted(df['Genre'].unique().tolist())
    selected_genres = st.multiselect(
        "Filtra per generi:",
        options=all_genres,
        default=all_genres,
        help="Lascia vuoto per mostrare tutti"
    )

# Applica filtri al dataframe
if selected_platforms and selected_genres:
    df = df[
        (df['Year_of_Release'] >= year_range[0]) & 
        (df['Year_of_Release'] <= year_range[1]) &
        (df['Platform'].isin(selected_platforms)) &
        (df['Genre'].isin(selected_genres))
    ]

# ANALISI GENERI E PIATTAFORME ---------------------------------
st.header("🎯 Analisi delle Vendite Medie per Genere e Piattaforma")
st.markdown(
    "In questa sezione analizziamo le vendite globali dei giochi suddivise per **genere** e **piattaforma**. "
    "I grafici mostrano quali generi e piattaforme hanno contribuito maggiormente al successo commerciale.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Vendite per Genere")
    # Escludo Unknown
    df_genre = df[df['Genre'] != 'Unknown']
    genre_sales = df_genre.groupby("Genre")["Global_Sales"].mean().sort_values()
    
    fig_g, ax_g = plt.subplots(figsize=(8, 7))
    genre_sales.plot(kind="barh", ax=ax_g, color='orchid')
    ax_g.set_title("Vendite Medie per Genere")
    ax_g.set_xlabel("Milioni di copie")
    ax_g.set_ylabel("Genere")
    plt.close()
    st.pyplot(fig_g)
    
    st.info("**Action** e **Sports** sono i generi che vendono di più. Questi sono i generi più sicuri su cui puntare.")

with col2:
    st.subheader("Vendite Medie per Piattaforma")
    platform_sales = df.groupby("Platform")["Global_Sales"].mean().sort_values(ascending=False).head(15)
    
    fig_p, ax_p = plt.subplots(figsize=(8, 7))
    platform_sales.plot(kind="barh", ax=ax_p, color="lightgreen")
    ax_p.set_title("Top 15 Piattaforme per Vendite Globali")
    ax_p.set_xlabel("Milioni di copie")
    ax_p.set_ylabel("Piattaforma")
    ax_p.invert_yaxis()
    plt.close()
    st.pyplot(fig_p)
    
    st.info("**PS2** e **X360** sono le piattaforme con più vendite totali nel dataset.")

st.divider()

# RECENSIONI E RATING ---------------------------------
st.header("⭐ Impatto di Recensioni e Rating sul Successo Commerciale")
st.markdown("Analizziamo quanto le valutazioni della critica e degli utenti influenzano le vendite globali di un videogioco.")

colA, colB = st.columns(2)

# Filtra per visualizzazione migliore
df_plot = df[df["Global_Sales"] <= 20]

# --- CRITIC SCORE vs SALES ---
with colA:
    st.subheader("Voti della Critica")
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
    plt.close()
    st.pyplot(fig_cs)
    
    st.info("Sembra esserci una **leggera correlazione** tra voti alti della critica e vendite, ma non è chiarissima.")

# --- USER SCORE vs SALES ---
with colB:
    st.subheader("Voti degli Utenti")
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
    plt.close()
    st.pyplot(fig_us)
    
    st.info("Anche per i voti degli utenti la correlazione non è fortissima. Probabilmente altri fattori come il **marketing** e il **brand** contano di più.")

st.divider()

# TOP PUBLISHERS ---------------------------------
st.header("🏢 Top Publisher")
st.markdown("Vediamo quali publisher hanno pubblicato più giochi e quali sono i loro generi preferiti.")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Top 10 Publisher")
    top_publishers = df['Publisher'].value_counts().head(10)
    
    fig_pub, ax_pub = plt.subplots(figsize=(6, 6))
    top_publishers.plot(kind='barh', ax=ax_pub, color='salmon')
    ax_pub.set_title('Top 10 Publisher per Numero di Giochi')
    ax_pub.set_xlabel('Numero di Giochi')
    ax_pub.set_ylabel('Publisher')
    ax_pub.invert_yaxis()
    plt.close()
    st.pyplot(fig_pub)

with col2:
    st.subheader("Generi Preferiti dei Top 3 Publisher")
    
    top_3_publishers = df['Publisher'].value_counts().head(3).index.tolist()
    
    fig_genres, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, publisher in enumerate(top_3_publishers):
        publisher_data = df[df['Publisher'] == publisher]
        genre_counts = publisher_data['Genre'].value_counts().head(5)
        
        axes[i].bar(range(len(genre_counts)), genre_counts.values, color='teal')
        axes[i].set_title(f'{publisher}\n({len(publisher_data)} giochi)')
        axes[i].set_xlabel('Genere')
        axes[i].set_ylabel('Numero di Giochi')
        axes[i].set_xticks(range(len(genre_counts)))
        axes[i].set_xticklabels(genre_counts.index, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.close()
    st.pyplot(fig_genres)

st.info("**Electronic Arts**, **Activision** e **Namco Bandai** sono i publisher più attivi. Ognuno ha le sue specialità di genere.")

st.divider()

# DIFFERENZE REGIONALI ---------------------------------
st.header("🌍 Differenze tra Mercati Regionali")
st.markdown("I gusti sono diversi in Nord America, Europa e Giappone? Vediamo quali generi vendono di più in ogni regione.")

regional_sales = df.groupby('Genre')[['NA_Sales', 'EU_Sales', 'JP_Sales']].sum()
top_genres = df.groupby('Genre')['Global_Sales'].sum().nlargest(8).index
regional_sales_top = regional_sales.loc[top_genres]

fig_regional, axes = plt.subplots(1, 3, figsize=(16, 5))

regions = [('NA_Sales', 'Nord America', 'steelblue'), 
           ('EU_Sales', 'Europa', 'coral'), 
           ('JP_Sales', 'Giappone', 'mediumseagreen')]

for i, (col, title, color) in enumerate(regions):
    regional_sales_top[col].sort_values(ascending=True).plot(kind='barh', ax=axes[i], color=color)
    axes[i].set_title(f'Top Generi in {title}')
    axes[i].set_xlabel('Vendite (milioni)')
    axes[i].set_ylabel('Genere')

plt.tight_layout()
plt.close()
st.pyplot(fig_regional)

st.info("Si vede chiaramente che **il Giappone preferisce Role-Playing** molto di più rispetto ad America ed Europa. L'America e l'Europa invece preferiscono **Action e Shooter**. Ogni mercato ha le sue preferenze!")

st.divider()

# PROFITABILITA' GENERI ---------------------------------
st.header("💰 Generi Più Profittevoli")
st.markdown("Quali generi vendono meglio in media per gioco? Questo ci aiuta a capire quali generi sono più profittevoli.")

# Escludo Unknown dall'analisi
df_no_unknown = df[df['Genre'] != 'Unknown']

genre_profitability = df_no_unknown.groupby('Genre').agg({
    'Global_Sales': ['sum', 'mean', 'count']
}).round(2)

genre_profitability.columns = ['Vendite_Totali', 'Vendite_Medie', 'Numero_Giochi']
genre_profitability = genre_profitability.sort_values('Vendite_Medie', ascending=False)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Tabella Profittabilità")
    st.dataframe(genre_profitability, use_container_width=True)

with col2:
    st.subheader("Vendite Medie per Genere")
    fig_prof, ax_prof = plt.subplots(figsize=(8, 6))
    genre_profitability['Vendite_Medie'].plot(kind='bar', ax=ax_prof, color='darkturquoise')
    ax_prof.set_title('Vendite Medie per Gioco per Genere')
    ax_prof.set_xlabel('Genere')
    ax_prof.set_ylabel('Vendite Medie (milioni)')
    ax_prof.set_xticklabels(ax_prof.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.close()
    st.pyplot(fig_prof)

st.info("**Platform** e **Shooter** sono molto profittevoli in media. Conviene fare pochi giochi di questi generi ma di alta qualità!")

st.divider()

# GIOCHI HIT ---------------------------------
st.header("🌟 Top 20 Giochi di Successo")
st.markdown("Quali sono i giochi che hanno venduto di più? Analizziamo le loro caratteristiche.")

top_games = df.nlargest(20, 'Global_Sales')[['Name', 'Platform', 'Year_of_Release', 'Genre', 'Publisher', 'Global_Sales']]

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Classifica")
    st.dataframe(top_games.reset_index(drop=True), use_container_width=True)

with col2:
    st.subheader("Generi nei Top 20")
    genre_hits = top_games['Genre'].value_counts()
    
    fig_hits, ax_hits = plt.subplots(figsize=(6, 6))
    ax_hits.bar(range(len(genre_hits)), genre_hits.values, color='gold')
    ax_hits.set_title('Generi nei Top 20 Giochi')
    ax_hits.set_xlabel('Genere')
    ax_hits.set_ylabel('Numero di Giochi')
    ax_hits.set_xticks(range(len(genre_hits)))
    ax_hits.set_xticklabels(genre_hits.index, rotation=45, ha='right')
    plt.tight_layout()
    plt.close()
    st.pyplot(fig_hits)

st.info("I giochi di maggior successo sono soprattutto **Sports e Platform**. La maggior parte sono parte di **franchise famosi** come Mario, Pokemon, Call of Duty.")

st.divider()

# EVOLUZIONE GENERI NEL TEMPO ---------------------------------
st.header("📅 Evoluzione dei Generi nel Tempo")
st.markdown("Come sono cambiate le preferenze dei giocatori negli anni? Vediamo l'evoluzione dei generi principali.")

top_6_genres = df.groupby('Genre')['Global_Sales'].sum().nlargest(6).index.tolist()
colors = ['steelblue', 'coral', 'mediumseagreen', 'gold', 'mediumpurple', 'salmon']

# Dropdown per scegliere tra globale e regionale
view_option = st.selectbox(
    "Seleziona la vista:",
    ["Evoluzione Globale", "Nord America", "Europa", "Giappone"]
)

if view_option == "Evoluzione Globale":
    st.subheader("Evoluzione Globale")
    genre_year_sales = df[df['Genre'].isin(top_6_genres)].groupby(['Year_of_Release', 'Genre'])['Global_Sales'].sum().reset_index()
    
    fig_evo, ax_evo = plt.subplots(figsize=(14, 7))
    
    for i, genre in enumerate(top_6_genres):
        genre_data = genre_year_sales[genre_year_sales['Genre'] == genre]
        ax_evo.plot(genre_data['Year_of_Release'], genre_data['Global_Sales'], 
                    marker='o', linewidth=2, label=genre, color=colors[i])
    
    ax_evo.set_title('Evoluzione delle Vendite per Genere nel Tempo')
    ax_evo.set_xlabel('Anno')
    ax_evo.set_ylabel('Vendite Globali (milioni)')
    ax_evo.legend(loc='best')
    ax_evo.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.close()
    st.pyplot(fig_evo)
    
    st.markdown("Si vede che **Sports** ha avuto un grande picco intorno al 2008-2010. **Shooter** è cresciuto molto negli anni 2000.")

else:
    # Mappa delle regioni
    region_map = {
        "Nord America": ('NA_Sales', 'Nord America', 'steelblue'),
        "Europa": ('EU_Sales', 'Europa', 'coral'),
        "Giappone": ('JP_Sales', 'Giappone', 'mediumseagreen')
    }
    
    region_col, region_name, region_color = region_map[view_option]
    
    st.subheader(f"Evoluzione in {region_name}")
    genre_year_regional = df[df['Genre'].isin(top_6_genres)].groupby(['Year_of_Release', 'Genre'])[region_col].mean().reset_index()
    
    fig_reg, ax_reg = plt.subplots(figsize=(14, 7))
    
    for i, genre in enumerate(top_6_genres):
        genre_data = genre_year_regional[genre_year_regional['Genre'] == genre]
        ax_reg.plot(genre_data['Year_of_Release'], genre_data[region_col], 
                marker='o', linewidth=2, label=genre, color=colors[i])
    
    ax_reg.set_title(f'Vendite Medie per Genere in {region_name}')
    ax_reg.set_xlabel('Anno')
    ax_reg.set_ylabel('Vendite Medie (milioni)')
    ax_reg.legend(loc='best')
    ax_reg.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.close()
    st.pyplot(fig_reg)
    
    # Messaggio specifico per regione
    if view_option == "Giappone":
        st.info("In Giappone **Role-Playing** è molto più importante rispetto alle altre regioni. **Shooter** invece vende molto meno.")
    elif view_option == "Nord America":
        st.info("In Nord America **Action** e **Shooter** sono molto popolari. **Sports** ha avuto un grande picco negli anni 2000.")
    else:
        st.info("In Europa le preferenze sono simili al Nord America, con **Action** e **Sports** tra i generi più popolari.")

st.divider()

# CONCLUSIONI ---------------------------------
st.header("📝 Conclusioni")
st.markdown("""
Dopo questa analisi semplice, ecco cosa abbiamo capito:

- **I generi più popolari** sono Action, Sports e Shooter - sono quelli che vendono di più
- **Le piattaforme più attive** nel dataset sono DS, PS2, PS3 e X360
- **Le vendite hanno avuto un picco** intorno al 2008-2010 e poi sono diminuite
- **La maggior parte dei giochi vende poco**, solo pochi vendono tantissimo
- **Le recensioni influenzano un po' le vendite**, ma non sono il fattore principale
- **I mercati regionali sono molto diversi**: il Giappone preferisce RPG, mentre America ed Europa preferiscono Action e Shooter
- **I giochi di successo** fanno parte di franchise famosi

Questa analisi ci aiuta a capire su quali piattaforme e generi puntare per i prossimi progetti!
""")