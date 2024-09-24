import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import streamlit as st
from datetime import datetime
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

# List of parameter names
parametres_list = ['Jus soutiré RT', 'Jus soutiré BW', 'Temp. JAE sortie réchauffeur n°6',
                   'Brix JAE', 'Brix sirop sortie evapo', 'Débit JAE entrée évaporation',
                   'Débit Sucre bande peseuse']

# Create a dictionary to hold input values from the user
parametres = {}

# Loop to create number input fields for each parameter
for parametre in parametres_list:
    parametres[parametre] = st.sidebar.number_input(f"Veuillez entrer la valeur de {parametre}:", value=0.0)

# Create a button to trigger the prediction
if st.sidebar.button("Calcul Ratio Énergie"):
    try:
        # Create a DataFrame for the user inputs
        df_CHE_testing = pd.DataFrame({
            'Jus soutiré RT': [parametres['Jus soutiré RT']],
            'Jus soutiré BW': [parametres['Jus soutiré BW']],
            'Temp. JAE sortie réchauffeur n°6': [parametres['Temp. JAE sortie réchauffeur n°6']],
            'Brix JAE': [parametres['Brix JAE']],
            'Brix sirop sortie evapo': [parametres['Brix sirop sortie evapo']],
            'Débit JAE entrée évaporation': [parametres['Débit JAE entrée évaporation']],
            'Débit Sucre bande peseuse': [parametres['Débit Sucre bande peseuse']]
        })

        # Standardize the input values
        x_testing = scaler.transform(df_CHE_testing)

        # Predict with the trained model
        gb_CHE_pred_testing = model_CHE_gb.predict(x_testing)

        nrj = gb_CHE_pred_testing[0].round(2)

        # Enregistrer la valeur et le timestamp actuel
        maintenant = datetime.now()
        st.session_state.timestamps.append(maintenant)

        # Append the prediction to session_state
        st.session_state.conso_NRJ.append(nrj)

        # Display the prediction result
        st.markdown(f"<h1 style='text-align: center; color: white; font-size: 25px;'>Prédiction du Ratio NRJ : {nrj} kWh/tcoss </h1>", unsafe_allow_html=True)

        # Create DataFrame for historical data
        historique_df = pd.DataFrame({
            'Horodatage': st.session_state.timestamps, 
            'Calcul Ratio Énergie': st.session_state.conso_NRJ
        })

        # Display stored data in a dataframe
        st.markdown("""
        <style>
        .azz-title {
            text-align: center;
            color: white;
            font-size: 40px;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown("<h1 class='azz-title'>Affichage des données enregistrées</h1>", unsafe_allow_html=True)
        st.dataframe(historique_df)

        # Plot the prediction curve
        plt.figure(figsize=(15, 6))
        plt.plot(historique_df['Horodatage'], historique_df['Calcul Ratio Énergie'], marker='o', linestyle='-', color='b')
        plt.title('Évolution des prédictions de consommation d\'énergie')
        plt.xlabel('Date Mesure')
        plt.ylabel('Prédiction du Ratio kWh/tcoss')
        plt.ylim(50, 300)
        plt.grid(True)

        # Format the date on the x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.gcf().autofmt_xdate()

        st.pyplot(plt)

    except ValueError:
        st.error("Erreur: Veuillez entrer un nombre valide pour chaque paramètre.")
    except Exception as e:
        st.error(f"Une erreur est survenue : {e}")
