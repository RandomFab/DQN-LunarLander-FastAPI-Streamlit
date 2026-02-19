import streamlit as st
from interface import render_interface
from dashboard import render_dashboard

# Configuration de la page
st.set_page_config(
    page_title="Eagle-1 Exploration Console",
    page_icon="🚀",
    layout="wide",
)

# Sidebar Navigation
st.sidebar.title("🚀 AstroDynamics")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["🎮 Pilotage", "📊 Dashboard"])
st.sidebar.markdown("---")
st.sidebar.write("**Mission :** Eagle-1 Atterrissage Automatisé")

# Routing
if page == "🎮 Pilotage":
    render_interface()
else:
    render_dashboard()
