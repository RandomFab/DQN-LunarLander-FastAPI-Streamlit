import streamlit as st
import gymnasium as gym
import requests

def render_interface():
    st.title("🎮 Eagle-1 : Pilote Automatique")
    st.subheader("Visualisation de l'atterrissage en temps réel")

    if "session" not in st.session_state:
        st.session_state.session = requests.Session()

    col1, col2 = st.columns([3, 1])

    with col1:
        video_placeholder = st.empty()
        video_placeholder.info("Prêt pour le lancement. Appuyez sur le bouton dans la barre latérale.")

    with col2:
        st.write("### 📈 Télémétrie")
        score_metric = st.metric("Récompense Totale", "0.00")
        
        # --- AJOUT VISUEL 1 : Barre d'altitude ---
        st.write("**Altitude**")
        alt_bar = st.progress(100) # De 0 à 100%
        
        status_text = st.empty()

    if st.sidebar.button("🚀 Lancer la mission", use_container_width=True):
        env = gym.make("LunarLander-v3", render_mode="rgb_array")
        observation, info = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            try:
                response = st.session_state.session.post(
                    "http://localhost:8000/predict", 
                    json={"state": observation.tolist()},
                    timeout=30
                )
                action = response.json()["action"]
            except Exception as e:
                st.error(f"❌ Erreur API : {e}")
                break

            observation, reward, terminated, truncated, info = env.step(action=action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

            # Rendu vidéo : chaque frame pour la fluidité
            frame = env.render()
            if steps % 3 == 0:
                video_placeholder.image(frame, use_container_width=True)
            
            # Métriques légères : tous les 10 pas
            if steps % 10 == 0:
                score_metric.metric("Récompense Totale", f"{total_reward:.2f}", delta=f"{reward:.2f}")
                altitude = max(0, min(100, int(observation[1] * 66)))
                alt_bar.progress(altitude)
            
        # Mise à jour finale des métriques après la boucle
        score_metric.metric("Récompense Totale", f"{total_reward:.2f}", delta=f"{reward:.2f}")
        alt_bar.progress(0) # Le vaisseau est posé (ou crashé)

        # Résultat final
        if total_reward >= 200:
            st.balloons()
            status_text.success(f"✅ MISSION RÉUSSIE ! Score final: {total_reward:.2f}")
        else:
            status_text.error(f"💥 ÉCHEC DE LA MISSION. Score final: {total_reward:.2f}")