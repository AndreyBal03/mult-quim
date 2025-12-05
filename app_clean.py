import streamlit as st
import time
import base64
import pathlib
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import plotly.figure_factory as ff
import seaborn as sns
import os

# Import MVCS modules
from models import *
from views import *
from controllers import *
from services import *

# Initialize controllers
session_controller = SessionController()
page_controller = PageController()

# -------------------------
# Initialize Groq client
# -------------------------
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -------------------------------------------------
# MUST BE FIRST: Page config
# -------------------------------------------------
st.set_page_config(page_title="QuimioAnalytics", page_icon="🧪", layout="wide")

# -------------------------------------------------
# Enhanced Professional CSS
# -------------------------------------------------
professional_css = """
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Styling */
* {
    font-family: 'Inter', sans-serif;
}

/* Main title styling */
h1 {
    color: #B0A461 !important; 
    text-align: center;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}

/* Headers styling */
h2 {
    color: #66A3ED !important;
    font-weight: 600 !important;
    border-bottom: 2px solid #B0A461;
    padding-bottom: 8px;
    margin-top: 20px !important;
}

h3 {
    color: #4A525A !important;
    font-weight: 500 !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(176, 164, 97, 0.15) 0%, rgba(74, 82, 90, 0.1) 100%);
    backdrop-filter: blur(10px);
}

[data-testid="stSidebar"] button {
    width: 100%;
    margin: 5px 0;
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.3s ease;
}

[data-testid="stSidebar"] button:hover {
    transform: translateX(5px);
    box-shadow: 0 4px 12px rgba(176, 164, 97, 0.3);
}

/* Main content area */
.stApp > div:first-child {
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 20px;
    margin: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
}

/* Metrics styling */
[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #B0A461 !important;
}

/* Button styling */
.stButton button {
    background: linear-gradient(135deg, #B0A461 0%, #8E9E9A 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: 500;
    transition: all 0.3s ease;
}

.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(176, 164, 97, 0.4);
}

/* DataFrame styling */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
}

/* Info/Warning/Error boxes */
.stAlert {
    border-radius: 10px;
    border-left: 4px solid #B0A461;
}

/* Selectbox and multiselect */
.stSelectbox, .stMultiselect {
    border-radius: 8px;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 2px dashed #B0A461;
    border-radius: 10px;
    padding: 20px;
    background: rgba(176, 164, 97, 0.05);
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 10px 20px;
    font-weight: 500;
}

/* Expander styling */
.streamlit-expanderHeader {
    background: rgba(176, 164, 97, 0.1);
    border-radius: 8px;
    font-weight: 500;
}
</style>
"""
st.markdown(professional_css, unsafe_allow_html=True)


# -------------------------------------------------
# Convert image to Base64
# -------------------------------------------------
@st.cache_data
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return None


IMAGE_FILE_PATH = "background4.jpg"
base64_image = get_base64_image(IMAGE_FILE_PATH)

# -------------------------------------------------
# Background CSS
# -------------------------------------------------
if base64_image:
    mime = "image/" + Path(IMAGE_FILE_PATH).suffix[1:]
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:{mime};base64,{base64_image}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """
else:
    page_bg_img = """
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    </style>
    """

st.markdown(page_bg_img, unsafe_allow_html=True)

# -------------------------------------------------
# Auto Snow (run only once)
# -------------------------------------------------
if not st.session_state.snow_triggered:
    st.snow()
    st.session_state.snow_triggered = True

# -------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------
with st.sidebar:
    st.markdown("### 🧪 QuimioAnalytics")
    st.markdown("---")

    if st.button("🏠 Home", use_container_width=True):
        page_controller.set_page("Home")

    if st.button("📂 Cargar Dataset", use_container_width=True):
        page_controller.set_page("Cargar dataset")

    if st.button("🔧 Preprocesamiento", use_container_width=True):
        page_controller.set_page("Preprocesamiento de Datos")

    if st.button("📈 Análisis PCA", use_container_width=True):
        page_controller.set_page("PCA")

    if st.button("🍇 Clustering", use_container_width=True):
        page_controller.set_page("Clustering")

    if st.button("🧮 ANOVA", use_container_width=True):
        page_controller.set_page("ANOVA")

    if st.button("💬 AI Chat", use_container_width=True):
        page_controller.set_page("AI Chat")

    if st.button("📊 Dashboard", use_container_width=True):
        page_controller.set_page("Dashboard")

    st.markdown("---")
    st.markdown("##### 📊 Estado del Dataset")
    if st.session_state.df is not None:
        st.success("✅ Dataset cargado")
        st.info(f"📋 {st.session_state.df.shape[0]} filas")
        st.info(f"📊 {st.session_state.df.shape[1]} columnas")
        if st.session_state.get("standardized"):
            st.success("✅ Estandarizado")
    else:
        st.warning("⚠️ Sin dataset")

# -------------------------------------------------
# PAGE CONTROLLER
# -------------------------------------------------
current_page = page_controller.get_current_page()

if current_page == "Home":
    home_page()

elif current_page == "Cargar dataset":
    cargar_dataset()

elif current_page == "Preprocesamiento de Datos":
    preprocessing_page()

elif current_page == "PCA":
    pca_page()

elif current_page == "Clustering":
    cluster_page()

elif current_page == "ANOVA":
    anova_page()

elif current_page == "AI Chat":
    ai_chat_page()

elif current_page == "Dashboard":
    dashboard_page()
