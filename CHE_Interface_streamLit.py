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
import warnings

# Ignorer tous les avertissements
warnings.filterwarnings("ignore")

# Charger le modèle avec pickle
with open('model.pkl', 'rb') as file:
    model_CHE_gb = pickle.load(file)


gb_CHE_pred_test  = model_CHE_gb .predict(x_test_CHE )
gb_CHE_pred_train  = model_CHE_gb .predict(x_train_CHE )

#print("\n")
#print("Gradient Boosting_ result")

# Calculer l'erreur quadratique moyenne (MSE)
gb_CHE_mse_test  = mean_squared_error(y_test_CHE, gb_CHE_pred_test )
#print("Mean Squared Error_test:", gb_CHE_mse_test.round(2))
gb_CHE_mse_train  = mean_squared_error(y_train_CHE, gb_CHE_pred_train )
#print("Mean Squared Error_train:", gb_CHE_mse_train.round(2))

# Calculer le coefficient de détermination sur l'ensemble de test
gb_CHE_r_squared_test = model_CHE_gb.score( x_test_CHE, y_test_CHE)
#print("Coefficient de détermination (R²_test) :", gb_CHE_r_squared_test.round(2))

# Calculer le coefficient de détermination sur l'ensemble de train
gb_CHE_r_squared_train = model_CHE_gb.score( x_train_CHE, y_train_CHE)
#print("Coefficient de détermination (R²_train) :", gb_CHE_r_squared_train.round(2))

# Calculer l'erreur absolue moyenne (MAE)
gb_CHE_mae = mean_absolute_error(y_test_CHE, gb_CHE_pred_test )
#print("Mean Absolute Error (MAE) :", gb_CHE_mae.round(2))

gb_CHE_rmse_test = gb_CHE_mse_test ** 0.5
#print("Root Mean Squared Error_Test (RMSE) :", gb_CHE_rmse_test.round(2))
gb_CHE_rmse_train = gb_CHE_mse_train ** 0.5
#print("Root Mean Squared Error_Train (RMSE) :", gb_CHE_rmse_train.round(2))


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.dates as mdates

# Assuming model_CHE_gb and scaler are already defined and trained

# Initialize or retrieve the prediction history from session_state
if 'conso_NRJ' not in st.session_state:
    st.session_state.conso_NRJ = []

# Initialiser des listes pour stocker les horodatages et les valeurs
if 'timestamps' not in st.session_state:
    st.session_state.timestamps = []

# Streamlit App Title
st.title("Prédiction du Ratio kWh /coss")

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
if st.button("Prédire le Ratio kWh /coss"):
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
        st.write(f"Prédiction du Ratio kWh /coss: {nrj}")

        historique_df = pd.DataFrame({
        'Horodatage': st.session_state.timestamps, 'Prédiction du Ratio kWh /coss': st.session_state.conso_NRJ
         })

        # Titre de la page
        st.title('Affichage des données enregistrées ')

        # Afficher le DataFrame avec st.dataframe (interactive)
        st.dataframe(historique_df)

        # Plot the prediction curve
        plt.figure(figsize=(15, 6))
        plt.plot(historique_df['Horodatage'], historique_df['Prédiction du Ratio kWh /coss'], marker='o', linestyle='-', color='b')
        plt.title('Évolution des prédictions de consommation d\'énergie')
        plt.xlabel('Date Mesure')
        plt.ylabel('Prédiction du Ratio kWh /coss')
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
