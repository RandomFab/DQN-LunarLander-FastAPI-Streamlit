import streamlit as st
import gymnasium as gym
import requests
import time

st.title("LunarLander")

if "session" not in st.session_state:
    st.session_state.session = requests.Session()

if st.button("Lancer l'atterrissage"):

    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    video_placeholder = st.empty()
    observation, info = env.reset()
    done = False

    score_placeholder = st.sidebar.empty()
    total_reward = 0

    while not done:

        response = st.session_state.session.post(
            "http://localhost:8000/predict", json={"state": observation.tolist()}
        )
        if response.status_code == 200:
            action = response.json()["action"]

        observation, reward, terminated, truncated, info = env.step(action=action)

        frame = env.render()
        video_placeholder.image(frame)
        total_reward += reward
        score_placeholder.metric("Récompense :", f"{total_reward:.2f}")

        if terminated or truncated:
            done = True
    
    if terminated and reward > 0:
        st.success((f" 🚀 Landing achieved with success ! Score final de {total_reward:.2f}"))
    elif terminated:
        st.error(" 💥 Crash ...Score final de {total_reward:.2f}")

