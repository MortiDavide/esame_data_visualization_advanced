# *FORM NUOVO GIOCO ------------------------------------
cols = st.columns(3)
user_input = {}





st.subheader("🎮 Simula un Gioco Nuovo")

# Imposta valori generali per feature storiche
media_hit_generale = df_feat['Hit'].mean()
media_sales_generale = df_feat['Global_Sales'].mean()

cols = st.columns(3)
user_input = {}

for i, col_name in enumerate(FEATURES):
    if col_name in NUMERIC_FEATURES:
        default = media_hit_generale if "Hit_Rate" in col_name else media_sales_generale
        label = FEATURE_LABELS[col_name]
        with cols[i % 3]:
            user_input[col_name] = st.number_input(
                label,
                min_value=0.0,
                max_value=1.0 if "Hit_Rate" in col_name else 20.0,
                value=float(default),
                step=0.01,
                key=f"num_{col_name}"   # <---- aggiungi questo
            )

    elif col_name in CATEGORICAL_FEATURES:
        label = FEATURE_LABELS[col_name]
        options = df_feat[col_name].astype(str).unique().tolist()
        if col_name in ['Publisher', 'Developer']:
            options = ['Nuovo'] + options
        with cols[i % 3]:
            user_input[col_name] = st.selectbox(
                label,
                options,
                key=f"cat_{col_name}"  # <---- aggiungi anche qui per selectbox
            )


# Bottone di predizione
if st.button("Predici Probabilità Hit"):
    input_df = pd.DataFrame([user_input])
    
    # Converti le categoriche in 'category'
    for col in CATEGORICAL_FEATURES:
        input_df[col] = input_df[col].astype('category')
    
    # Probabilità di diventare hit
    proba = lgbm.predict_proba(input_df)[0][1]
    
    col_res1, col_res2 = st.columns(2)
    col_res1.metric("Probabilità di HIT", f"{proba:.1%}")
    label_hit = "ALTO" if proba >= 0.5 else "BASSO"
    col_res2.metric("Rischio stimato", label_hit)

    st.write("Valori inseriti:")
    st.json({FEATURE_LABELS[k]: v for k, v in user_input.items()})

    st.info(
        "⚠️ Questa stima si basa sul modello addestrato su dati storici. "
        "Per giochi nuovi senza storico, le probabilità sono indicative."
    )
