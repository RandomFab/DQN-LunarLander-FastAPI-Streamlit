import streamlit as st
import pandas as pd
import numpy as np
import sys
import requests
from pathlib import Path
import math

# Définition de la racine du projet et ajout au sys.path pour les imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.scripts.extract_logs import load_tfevents_to_df

# On utilise une session pour les appels API
if "session" not in st.session_state:
    st.session_state.session = requests.Session()

# Utilisation de la fonction pour charger les données
log_path = BASE_DIR / "logs" / "dqn_LunarLander_v1" / "best_HP_1" / "events.out.tfevents.1771422769.LAPTOP-R6T5Q982.28996.8"

@st.cache_data
def get_log_data(path):
    try:
        return load_tfevents_to_df(str(path))
    except Exception as e:
        return pd.DataFrame()

df = get_log_data(log_path)


def render_dashboard():
    st.title("📊 Tableau de Bord des Performances")
    
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("⚙️ Paramètres de simulation")
        min_angular = -2 * round(math.pi,3)
        max_angular = 2 * round(math.pi,3)
        x_speed = st.slider('Vitesse X', -10.0, 10.0, 0.0)
        y_speed = st.slider('Vitesse Y', -10.0, 10.0, 0.0)
        angular_position = st.slider('Position Angulaire', min_angular, max_angular, 0.0)
        angular_speed = st.slider('Vitesse Angulaire', -10.0, 10.0, 0.0)

    with col2:
        st.subheader("🗺️ Carte des décisions")
        
        # Génération de la grille
        x_coord = np.linspace(-1.5, 1.5, 50)
        y_coord = np.linspace(0.0, 1.5, 50)
        X, Y = np.meshgrid(x_coord, y_coord)
        x_flat = X.ravel()
        y_flat = Y.ravel()
        
        # Préparation du batch pour l'API
        # On doit envoyer une liste d'objets {"state": [...]}
        observations = []
        for x, y in zip(x_flat, y_flat):
            observations.append({"state": [float(x), float(y), float(x_speed), float(y_speed), 
                                         float(angular_position), float(angular_speed), 0.0, 0.0]})

        try:
            response = st.session_state.session.post(
                "http://localhost:8000/predict_batch", 
                json=observations,
                timeout=30
            )
            if response.status_code == 200:
                actions = response.json()["actions"]
                
                # Création d'un DataFrame pour l'affichage
                data_map = pd.DataFrame({
                    'x': x_flat,
                    'y': y_flat,
                    'action': actions
                })
                
                # Mapping des noms d'actions
                action_names = {0: "Rien", 1: "Moteur Gauche", 2: "Moteur Principal", 3: "Moteur Droit"}
                data_map['Action Nom'] = data_map['action'].map(action_names)
                
                # Affichage
                import plotly.express as px
                fig = px.scatter(data_map, x='x', y='y', color='Action Nom',
                                 color_discrete_map={
                                     "Rien": "lightgrey",
                                     "Moteur Gauche": "blue",
                                     "Moteur Principal": "red",
                                     "Moteur Droit": "green"
                                 },
                                 title="Décisions de l'IA selon la position",
                                 labels={'x': 'Position X', 'y': 'Altitude (Y)'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Erreur API ({response.status_code})")
        except Exception as e:
            st.error(f"❌ Erreur API : {e}")
    
    st.markdown("---")
    st.write("### 📈 Entraînement : Récompense Moyenne")
    if not df.empty:
        st.line_chart(data=df, x='step', y="rollout/ep_rew_mean")
    else:
        st.warning("Données d'entraînement non disponibles.")

