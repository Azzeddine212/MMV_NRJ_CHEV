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

# Liste des paramètres
parametres_list = ['Jus soutiré RT', 'Jus soutiré BW', 'Temp. JAE sortie réchauffeur n°6',
                   'Brix JAE', 'Brix sirop sortie evapo', 'Débit JAE entrée évaporation',
                   'Débit Sucre bande peseuse']

# Dictionnaire pour stocker les valeurs d'entrée de l'utilisateur
parametres = {}

# Boucle pour créer des champs d'entrée pour chaque paramètre
for parametre in parametres_list:
    parametres[parametre] = st.sidebar.number_input(f"Veuillez entrer la valeur de {parametre}:", value=0.0)

# Bouton pour déclencher la prédiction
if st.sidebar.button("Calcul Ratio Énergie"):
    try:
        # Créer un DataFrame pour les entrées utilisateur
        df_CHE_testing = pd.DataFrame({
            'Jus soutiré RT': [float(parametres['Jus soutiré RT'])],
            'Jus soutiré BW': [float(parametres['Jus soutiré BW'])],
            ' Temp. JAE sortie réchauffeur n°6': [float(parametres['Temp. JAE sortie réchauffeur n°6'])],
            'Brix JAE': [float(parametres['Brix JAE'])],
            'Brix sirop sortie evapo': [float(parametres['Brix sirop sortie evapo'])],
            'Débit JAE entrée évaporation': [float(parametres['Débit JAE entrée évaporation'])],
            'Débit Sucre bande peseuse': [float(parametres['Débit Sucre bande peseuse'])]
        })
        df_CHE_testing = pd.DataFrame(df_CHE_testing)
        
        # Standardiser les valeurs d'entrée
        x_testing = scaler.transform(df_CHE_testing)

        # Prédiction avec le modèle entraîné
        gb_CHE_pred_testing = model_CHE_gb.predict(x_testing)

        # Arrondir la prédiction
        nrj = gb_CHE_pred_testing[0].round(2)

        # Enregistrer la valeur et l'horodatage
        maintenant = datetime.now() + timedelta(hours=2)
        st.session_state.timestamps.append(maintenant)
        st.session_state.conso_NRJ.append(nrj)

        # Affichage du résultat de la prédiction
        st.markdown(f"<h1 style='text-align: center; color: white; font-size: 32px;'>Prédiction du Ratio NRJ : {nrj} kWh/tcoss </h1>", unsafe_allow_html=True)

        # Créer un DataFrame pour les données historiques
        historique_df = pd.DataFrame({
            'Horodatage': st.session_state.timestamps, 
            'Calcul Ratio Énergie': st.session_state.conso_NRJ
        })

        # Afficher les données enregistrées
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
        target = 146
        # Tracer l'évolution des prédictions
        plt.figure(figsize=(15, 6))
        plt.plot(historique_df['Horodatage'], historique_df['Calcul Ratio Énergie'], marker='o', linestyle='-', color='b')
        plt.plot(historique_df['Horodatage'], [target] * len(historique_df['Horodatage']),linestyle='--', linewidth=2 ,label=f'Conso NRJ cible CB24:{target} kwh/tcoss', color='red')
        #plt.axhline(y=155, color='red', linestyle='--', linewidth=2, label='Conso NRJ cible CB24:155 kwh/tcoss')

        plt.title("Évolution des prédictions de consommation d'énergie")
        plt.xlabel('Date Mesure')
        plt.ylabel('Prédiction du Ratio kWh/tcoss')
        #plt.ylim(120, 220)
        plt.grid(True)

        # Formatage des dates sur l'axe x
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.gcf().autofmt_xdate()

        st.pyplot(plt)

    except ValueError:
        # Afficher une erreur si l'utilisateur entre une valeur invalide
        st.error("Erreur: Veuillez entrer un nombre valide pour chaque paramètre.")
    except Exception as e:
        # Afficher tout autre type d'erreur
        st.error(f"Une erreur est survenue : {e}")
