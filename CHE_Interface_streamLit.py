import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_error
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from scipy.stats import variation
from scipy.stats import shapiro
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import streamlit as st
import io
from datetime import datetime
import time
import joblib
import pickle
import warnings

import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
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
model_CHE_gb  = joblib.load('model.pkl')

# Charger le scaler avec pickle
scaler  = joblib.load('scaler.pkl')

# Initialize or retrieve the prediction history from session_state
if 'conso_NRJ' not in st.session_state:
    st.session_state.conso_NRJ = []

# Initialiser des listes pour stocker les horodatages et les valeurs
if 'timestamps' not in st.session_state:
    st.session_state.timestamps = []


# Streamlit App Title
#st.title("Prédiction du Ratio kWh /tcoss")
st.markdown("""
    <style>
    .centered-title {
        text-align: center;        /* Centrer le texte */
        color: white;                /* Couleur du texte */
        font-size: 55px;           /* Taille de la police */
    }
    </style>
""", unsafe_allow_html=True)
st.markdown("<h1 class='centered-title'>Prédiction du Ratio Energie_Tereos Chevrières</h1>", unsafe_allow_html=True)

# List of parameter names
parametres_list = ['Jus soutiré RT', 'Jus soutiré BW', 'Temp. JAE sortie réchauffeur n°6',
                   'Brix JAE', 'Brix sirop sortie evapo', 'Débit JAE entrée évaporation',
                   'Débit Sucre bande peseuse']

# Create a dictionary to hold input values from the user
parametres = {}

# Loop to create number input fields for each parameter
for parametre in parametres_list:
    parametres[parametre] = st.sidebar.number_input(f"Veuillez entrer la valeur de {parametre}:", value=0.0)


# Création de colonnes pour centrer le bouton
#col1, col2, col3 = st.columns([1, 2, 1])  # Trois colonnes, la colonne du milieu est plus large

#with col2:
    # Create a button to trigger the prediction

st.sidebar.markdown("""
    <style>
    .center-button {
        display: flex;
        justify-content: center;
    }
    .center-button button {
        background-color: white;  /* Couleur de fond */
        color: white;               /* Couleur du texte */
        padding: 20px 48px;         /* Ajuster la taille du bouton */
        font-size: 23px;            /* Taille de la police */
        border: none;               /* Pas de bordure */
        border-radius: 8px;         /* Coins arrondis */
        cursor: pointer;            /* Curseur au survol */
    }
    .center-button button:hover {
        background-color: #45a049;  /* Couleur au survol */
    }
    </style>
""", unsafe_allow_html=True)

if st.sidebar.markdown('<div class="center-button"><button>Prédire le Ratio kWh /tcoss</button></div>', unsafe_allow_html=True):
#if st.sidebar.button("Prédire le Ratio kWh /tcoss", key="predict_button"):
    try:
        # Create a DataFrame for the user inputs
        df_CHE_testing = {
            'Jus soutiré RT': [parametres['Jus soutiré RT']],
            'Jus soutiré BW': [parametres['Jus soutiré BW']],
            ' Temp. JAE sortie réchauffeur n°6': [parametres['Temp. JAE sortie réchauffeur n°6']],
            'Brix JAE': [parametres['Brix JAE']],
            'Brix sirop sortie evapo': [parametres['Brix sirop sortie evapo']],
            'Débit JAE entrée évaporation': [parametres['Débit JAE entrée évaporation']],
            'Débit Sucre bande peseuse': [parametres['Débit Sucre bande peseuse']]
        }
        
        df_CHE_testing = pd.DataFrame(df_CHE_testing)
        
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
        #st.write(f"Prédiction du Ratio kWh /coss: {nrj}")


        st.markdown(f"<h1 style='text-align: center; color: black;font-size: 25px;'>Prédiction du Ratio NRJ : {nrj} kWh/coss </h1>", unsafe_allow_html=True)

        historique_df = pd.DataFrame({
        'Horodatage': st.session_state.timestamps, 'Prédiction du Ratio kWh /coss': st.session_state.conso_NRJ
        })

        # Titre de la page
        st.markdown("""
        <style>
        .azz-title {
            text-align: center;        /* Centrer le texte */
            color: white;                /* Couleur du texte */
            font-size: 40px;           /* Taille de la police */
            white-space: nowrap;       /* Empêcher le retour à la ligne */
        }
        </style>
    """, unsafe_allow_html=True)
        st.markdown("<h1 class='azz-title'>Affichage des données enregistrées</h1>", unsafe_allow_html=True)
        #st.title('Affichage des données enregistrées ')

        # Afficher le DataFrame avec st.dataframe (interactive)
        st.dataframe(historique_df)

        # Plot the prediction curve
        plt.figure(figsize=(15, 6))
        plt.plot(historique_df['Horodatage'], historique_df['Prédiction du Ratio kWh /coss'], marker='o', linestyle='-', color='b')
        plt.title('Évolution des prédictions de consommation d\'énergie')
        plt.xlabel('Date Mesure')
        plt.ylabel('Prédiction du Ratio kWh /tcoss')
        #plt.xlim(0, 6)  # Ajuste les marges de l'axe des x
        plt.ylim(50, 300)  # Ajuste les marges de l'axe des y
        plt.grid(True)
        # Format the date on the x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        #plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.gcf().autofmt_xdate()  # Rotate and format the date labels for better readability

        st.pyplot(plt)  # To display the matplotlib chart in Streamlit


    except ValueError:
        st.error("Erreur: Veuillez entrer un nombre valide pour chaque paramètre.")
    except Exception as e:
        st.error(f"Une erreur est survenue : {e}")


# Affichage du bouton dans une div centrée
st.markdown("""
<style>
    .stButton > button {
        font-weight: bold;           /* Texte en gras */
        background-color: skyblue;  /* Couleur de fond */
        color: white;                /* Couleur du texte */
        padding: 15px 32px;         /* Espacement */
        text-align: center;         /* Alignement du texte */
        text-decoration: none;       /* Pas de soulignement */
        display: inline-block;       /* Affichage en bloc */
        font-size: 18px;            /* Taille de la police */
        margin: 4px 2px;            /* Marges */
        cursor: pointer;            /* Curseur en forme de main */
        border: none;                /* Pas de bordure */
        border-radius: 8px;         /* Coins arrondis */
    }
</style>
""", unsafe_allow_html=True)
