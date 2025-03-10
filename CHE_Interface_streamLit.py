import pandas as pd
import joblib
import warnings
import streamlit as st
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from PIL import Image
import base64

# Ajouter l'image en arrière-plan via CSS
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded_image = base64.b64encode(image.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp  {{
                background-image: url("data:image/png;base64,{encoded_image}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

# Vérification avec une image locale
add_bg_from_local('interface.jpg')

# Ignorer tous les avertissements
warnings.filterwarnings("ignore")

# Charger le modèle avec pickle
model_CHE_gb = joblib.load('model.pkl')

# Charger le scaler avec pickle
scaler = joblib.load('scaler.pkl')

# Initialize or retrieve the prediction history from session_state
if 'conso_NRJ' not in st.session_state:
    st.session_state.conso_NRJ = []

# Initialiser des listes pour stocker les horodatages et les valeurs
if 'timestamps' not in st.session_state:
    st.session_state.timestamps = []

# Streamlit App Title
st.markdown("""
    <style>
    .centered-title {
        text-align: center;
        color: white;
        font-size: 48px;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown("<h1 class='centered-title'>Modèle MMV Énergie_Chevrières</h1>", unsafe_allow_html=True)

# Téléchargement du fichier Excel
uploaded_file = st.sidebar.file_uploader("📂 Téléchargez votre fichier Excel", type=["xlsx"])


# Bouton pour déclencher la prédiction
if st.sidebar.button("Calcul Ratio Énergie"):
    df_CHE_testing = pd.read_excel(uploaded_file)
    
    df_CHE_testing["Date"] = pd.to_datetime(df_CHE_testing["Date"])
    df_CHE_testing.set_index("Date", inplace=True)
    st.dataframe(df_CHE_testing.round(2)) 
    df_CHE_testing= df_CHE_testing[[
    "Jus soutiré RT",
    "Jus soutiré BW",
    "T°- JAE sortie réchauffeur n°6 (ºC)",
    "JAE - Brix poids (g%g)",
    "Brix- Jus sortie 6ème effet B (%)"
    "Débit - JAE entrée évaporation",
    "Débit - Sucre bande peseuse",  
]]
    # Standardiser les valeurs d'entrée
    x_testing = scaler.transform(df_CHE_testing)

    # Prédiction avec le modèle entraîné
    gb_CHE_pred_testing = model_CHE_gb.predict(x_testing)
    df_pred = pd.DataFrame(gb_CHE_pred_testing , columns=["Prédictions"], index= variables.index)
    df_results = pd.concat([df_CHE_testing, df_pred], axis=1)

    st.markdown("<h1 style='text-align: center; color: #003366; font-size: 28px;'>📊 Prédiction & Analyse</h1>", unsafe_allow_html=True)


    # Plotting the predictions
    fig, ax = plt.subplots(figsize=(20, 10), dpi=100)
    mean = df_results["Prédictions"].mean()
    std_dev = df_results["Prédictions"].std()
    upper_limit = mean + 2 * std_dev
    lower_limit = mean - 2 * std_dev

    # Ajouter une ligne horizontale représentant l'objectif
    ax.axhline(y=objectif, color="red", linestyle="--", linewidth=2, label=f'Objectif : {objectif} kWh')

    # Identifier et marquer les points au-dessus de l'objectif
    au_dessus = df_results["Prédictions"] > objectif  # Masque booléen
    ax.scatter(df_results.index[au_dessus], df_results["Prédictions"][au_dessus], color="red", label="Au-dessus de l'objectif", zorder=3)

    ax.axhline(upper_limit, color="green", linestyle="dashed", linewidth=1, label=f"Mean + 2σ = {upper_limit:.2f}")
    ax.axhline(lower_limit, color="green", linestyle="dashed", linewidth=1, label=f"Mean - 2σ = {lower_limit:.2f}")
    ax.plot(df_results.index, df_results["Prédictions"], color="blue", label='Prédiction CB24', alpha=0.6)
    #ax.bar(df_results.index, df_results["Prédictions"], color="red", label='Prédiction CB24', alpha=0.6)
    ax.set_title("Prédiction CB24")
    ax.set_xlabel("Date")
    ax.set_ylabel("Conso NRJ (kWh/tcossette)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig,use_container_width=False)
