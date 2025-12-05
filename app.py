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
# Initialize Groq client
# -------------------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -------------------------------------------------
# MUST BE FIRST: Page config
# -------------------------------------------------
st.set_page_config(page_title="QuimioAnalytics", page_icon="🧪", layout="wide")

# -------------------------------------------------
# Initialize session_state variables
# -------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

if "df" not in st.session_state:
    st.session_state.df = None

if "snow_triggered" not in st.session_state:
    st.session_state.snow_triggered = False

if "plot_color_choice" not in st.session_state:
    st.session_state.plot_color_choice = "QuimioAnalytics (Custom)"

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
# HOME PAGE
# -------------------------------------------------
def home_page():
    st.title("QuimioAnalytics")

    # Define the text for the speech bubble (Using Spanish greeting from your source)
    ASSISTANT_GREETING = "Hola, mi nombre es Heisenberg y soy tu assistente virtual. Carga tu dataset y puedo ayudarte 🙂"

    # ---------- Read local image and convert to base64 ----------
    # NOTE: This part assumes 'man.png' exists in the same directory where the script is run.
    img_path = pathlib.Path("man.png")
    img_data_uri = None
    try:
        with img_path.open("rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode("utf-8")
            img_data_uri = f"data:image/png;base64,{b64}"
    except Exception as e:
        # If reading fails, fall back to a simple st.image (so user still sees something)
        st.warning(
            f"Could not embed man.png as data URI ({e}). Falling back to st.image below."
        )

    # ---------- CSS for positioning the floating image AND the speech bubble ----------
    # The image is now just a static floating element.
    st.markdown(
        f"""
    <style>
    /* ensures the floating image sits above other content and doesn't affect layout */
    .floating-image-container {{
        position: fixed;
        top: 250px;      /* Vertical position */
        right: 18px;    /* distance from right edge */
        width: 160px;   /* max width of the container */
        z-index: 9999;
        text-align: center;
        /* cursor: pointer; -- REMOVED */
    }}
    
    /* Ensure the link wrapper takes up the whole container area and removes link styles */
    /* .floating-link is no longer used, CSS kept for robustness */
    .floating-link {{
        display: block;
        color: inherit;
        text-decoration: none;
    }}

    .floating-image-container img {{
        width: 100%;
        height: auto;
        border-radius: 0px;
        box-shadow: 0 0px 0px rgba(0,0,0,0);
        display: block;
        margin: 0;
    }}

    /* Speech Bubble Styles */
    .speech-bubble {{
        background: #eef4ff; /* Light blue/grey for a modern look */
        color: #333333;
        padding: 10px 15px;
        border-radius: 15px;
        margin-bottom: 8px; /* Space between bubble and image */
        position: relative;
        font-size: 13px; /* Slightly smaller text for the bubble */
        text-align: left;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        line-height: 1.4;
    }}

    /* Speech bubble tail (points down to the image) */
    .speech-bubble::after {{
        content: '';
        position: absolute;
        bottom: -10px; /* Position below the bubble */
        left: 50%;
        transform: translateX(-50%);
        width: 0;
        height: 0;
        border-left: 10px solid transparent;
        border-right: 10px solid transparent;
        border-top: 10px solid #eef4ff; /* Match bubble background color */
    }}

    /* Add a subtle hover effect to indicate clickability */
    .floating-image-container:hover {{
        /* REMOVED hover effect */
        /* transition: transform 0.2s ease-in-out; */
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )

    # ---------- WELCOME CARD (unchanged from your source) ----------
    st.markdown(
        """
    <div style='
        background-color: #ffffff;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
        max-width: 650px;
        margin-left: auto;
        margin-right: auto;
    '>
        <h2 style='margin-bottom: 10px;'>Bienvenido a QuimioAnalytics</h2>
        <p style='font-size: 16px; color: #444;'>
            Esta plataforma está diseñada especialmente para <strong>estudiantes de química</strong> 
            que desean explorar y analizar datos de manera intuitiva, sin necesidad de programar.
        </p>
        <p style='font-size: 1rem; color: #666; margin-top: 20px;'>
            Utiliza el panel lateral para navegar entre las diferentes herramientas de análisis.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ---------- Insert floating image (data URI) or fallback (STATIC) ----------
    if img_data_uri:
        # RENDERED WITHOUT THE <a> TAG WRAPPER
        html = f"""
            <div class="floating-image-container">
                <div class="speech-bubble">
                    {ASSISTANT_GREETING}
                </div>
                <img src="{img_data_uri}" alt="Heisenberg">
            </div>
        """
        st.markdown(html, unsafe_allow_html=True)
    else:
        # fallback: use st.image (will occupy a normal Streamlit spot but only shown if embedding failed)
        st.image("man.png", caption=ASSISTANT_GREETING, width=140)

    st.markdown("---")

    # Tabs for organization
    tab1, tab2, tab3 = st.tabs(
        ["Introducción", "Fundamentos Teóricos", "Ejemplo Guiado"]
    )

    with tab1:
        st.markdown("### Herramientas Disponibles")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                """
            <div style='padding: 20px; background: rgba(176,164,97,0.1); border-radius: 10px; text-align: center;'>
                <h3>Carga de Datos</h3>
                <p>CSV y Excel</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                """
            <div style='padding: 20px; background: rgba(153,191,240,0.1); border-radius: 10px; text-align: center;'>
                <h3>Preprocesamiento</h3>
                <p>Limpieza y escalado</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                """
            <div style='padding: 20px; background: rgba(176,164,97,0.1); border-radius: 10px; text-align: center;'>
                <h3>Análisis PCA</h3>
                <p>Reducción de dimensiones</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col4:
            st.markdown(
                """
            <div style='padding: 20px; background: rgba(153,191,240,0.1); border-radius: 10px; text-align: center;'>
                <h3>ANOVA</h3>
                <p>Comparación de grupos</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with tab2:
        st.markdown("### Fundamentos de Quimiometría")

        with st.expander("Importancia del Análisis Multivariante", expanded=False):
            st.markdown("""
            El análisis multivariante es fundamental en química analítica moderna porque permite:
            
            - Trabajar simultáneamente con múltiples variables químicas (señales espectrales, concentraciones, intensidades)
            - Identificar patrones ocultos en datos complejos
            - Reducir la dimensionalidad preservando información química relevante
            - Clasificar muestras según su composición química
            - Detectar correlaciones entre variables que no son evidentes en análisis univariados
            """)

        with st.expander("Estandarización de Datos", expanded=False):
            st.markdown("""
            ### ¿Por qué estandarizar?
            
            La estandarización transforma las variables para que tengan media cero y desviación estándar uno.
            
            #### Para el ANOVA
            
            Aunque no es estrictamente necesario para ANOVA simple, la estandarización es importante cuando:
            
            - **Comparabilidad entre variables:** Permite comparar la magnitud del efecto cuando se usan múltiples variables en escalas diferentes
            - **Homogeneidad de varianzas:** Puede ayudar a estabilizar varianzas cuando los grupos tienen desviaciones estándar desiguales
            - **Interpretación de coeficientes:** Facilita la comparación en modelos de regresión subyacentes
            - **Métodos avanzados:** Es fundamental para PCA y análisis de clusters
            
            #### Para el PCA
            
            **La estandarización es esencial para PCA.** Si las variables no se estandarizan y una tiene varianza mucho mayor, 
            esa variable controlará la primera componente principal. La estandarización evita esto haciendo que todas las 
            variables tengan el mismo peso.
            
            En términos matemáticos, las componentes principales son los autovectores de la matriz de correlación. 
            Para datos estandarizados, cada variable original tiene varianza de 1, por lo que la varianza total 
            del conjunto de datos y la suma de los autovalores son ambos iguales al número de variables.
            
            #### Para Análisis de Clusters
            
            **La estandarización es altamente recomendada** porque la mayoría de algoritmos (k-means, clustering jerárquico) 
            se basan en medidas de distancia sensibles a la magnitud de cada variable.
            
            Desde el punto de vista teórico, la distancia Euclidiana se define como la suma de diferencias al cuadrado 
            entre variables. Una variable con escala grande domina la contribución total de la distancia y determina 
            artificialmente la formación de clusters. Al estandarizar, todas las variables tienen influencia equilibrada.
            """)

        with st.expander("Análisis de Varianza (ANOVA)", expanded=False):
            st.markdown("""
            ### ¿Qué es el ANOVA?
            
            El **Análisis de la Varianza** es una técnica estadística para separar y estimar diferentes causas de variación.
            
            **Función principal:** Evaluar si las variaciones en la respuesta química (señales espectroscópicas, concentraciones, 
            absorbancias, intensidades) se deben realmente a factores experimentales y no al azar.
            
            ### Importancia en Quimiometría
            
            En quimiometría, el ANOVA es fundamental porque:
            
            1. **Identifica variables discriminantes:** Determina qué variables químicas realmente diferencian los grupos
            2. **Valida diferencias químicas:** Confirma que hay verdaderas diferencias antes del análisis multivariado
            3. **Interpretación univariada:** Relaciona directamente variables individuales (picos FAME) con fenómenos químicos
            4. **Detecta señales responsables:** Identifica qué señales químicas causan la variación en PCA
            
            ### Ecuaciones Fundamentales
            
            **Suma de cuadrados total (SST):** Variabilidad total en los datos
            $SS_T = \\sum_{i=1}^{k}\\sum_{j=1}^{n_i} (y_{ij} - \\bar{y})^2$
            
            **Suma de cuadrados entre grupos (SSB):** Variabilidad explicada por diferencias entre grupos
            $SS_A = \\sum_{i=1}^{k} n_i (\\bar{y}_i - \\bar{y})^2$
            
            **Suma de cuadrados dentro de grupos (SSE):** Variabilidad interna (error o ruido)
            $SS_E = \\sum_{i=1}^{k}\\sum_{j=1}^{n_i} (y_{ij} - \\bar{y}_i)^2$
            
            **Estadístico F:** Compara variabilidad explicada vs no explicada
            $F = \\frac{MS_A}{MS_E}$
            
            ### Interpretación
            
            - **F grande:** Las diferencias entre grupos son mayores que la variabilidad aleatoria
            - **p-valor < 0.05:** Rechazamos la hipótesis nula, hay diferencias significativas
            - **p-valor ≥ 0.05:** No hay evidencia suficiente de diferencias significativas
            """)

        with st.expander("Análisis de Componentes Principales (PCA)", expanded=False):
            st.markdown("""
            ### ¿Qué es el PCA?
            
            Es una técnica para **reducir la dimensionalidad** cuando existe correlación entre variables.
            
            La idea es encontrar componentes principales Z₁, Z₂, ..., Zₙ que sean combinaciones lineales 
            de las variables originales X₁, X₂, ..., Xₙ:
            
            $Z_1 = a_{11}X_1 + a_{12}X_2 + a_{13}X_3 + \\dots + a_{1n}X_n$
            
            Los coeficientes se eligen para que:
            1. Las nuevas variables no estén correlacionadas entre sí
            2. La primera componente (PC1) capture la mayor variación
            3. La segunda (PC2) capture la siguiente mayor variación, y así sucesivamente
            
            ### Importancia en Quimiometría
            
            PCA es central en quimiometría para explorar mezclas químicas y señales multivariadas. Permite detectar:
            
            - **Agrupamientos** en los datos
            - **Variables responsables** de la diferenciación
            - **Outliers** experimentales o químicos
            - **Relaciones entre picos** (covarianzas químicas)
            
            Los **loadings** muestran cómo cada variable contribuye químicamente a las componentes principales.
            
            ### Visualizaciones Clave
            
            **Scree Plot:** Muestra varianza explicada por cada componente. La varianza explicada por el componente i es:
            $\\frac{\\lambda_i}{\\sum_{j=1}^{n} \\lambda_j}$
            
            Ayuda a decidir cuántos componentes son necesarios para describir la estructura química.
            
            **Scores Plot:** Distribución de muestras en el espacio de componentes principales. Muestras cercanas 
            tienen patrones similares (composiciones químicas parecidas).
            
            **Loadings Plot:** Muestra contribución de cada variable a los componentes. Los loadings son elementos 
            de los vectores propios. Mayor valor absoluto indica mayor influencia.
            
            **Biplot:** Combina scores y loadings en una sola gráfica, permitiendo relacionar directamente 
            características químicas con patrones observados.
            """)

        with st.expander("Análisis de Clusters", expanded=False):
            st.markdown("""
            ### ¿Qué es el Análisis de Clusters?
            
            Es un método para **dividir objetos en clases** de manera que objetos similares queden en la misma clase.
            
            Como en PCA, los grupos no se conocen antes del análisis. Busca objetos próximos en el espacio de variables.
            
            **Distancia Euclidiana:**
            $d = \\sqrt{\\sum_{i=1}^n (x_i-y_i)^2}$
            
            **Distancia Manhattan:**
            $D_{Manhattan} = |x_1 - x_2| + |y_1 - y_2|$
            
            ### Importancia en Quimiometría
            
            El clustering es fundamental porque permite:
            
            - **Identificar patrones naturales** sin categorías predefinidas
            - **Agrupar muestras** según similitud química
            - **Detectar relaciones** no evidentes a simple vista
            - **Distinguir feedstocks** e identificar adulteraciones
            - **Evaluar calidad** de lotes y procesos
            - **Complementar PCA** asignando grupos en el espacio reducido
            - **Detectar outliers** (errores instrumentales o contaminación)
            
            ### Visualizaciones
            
            **Dendrograma:** Muestra cómo las muestras se agrupan jerárquicamente. La altura de unión indica 
            la diferencia química. Un corte horizontal permite decidir el número de clusters.
            
            **Scatter Plot en espacio PCA:** Proyecta clusters en PC1 vs PC2, mostrando separación espacial 
            y consistencia con la estructura química.
            
            ### Métrica de Calidad: Índice Silhouette
            
            Evalúa cuán bien definidas están las clases obtenidas. Mide:
            - Separación química entre grupos
            - Coherencia interna de cada grupo
            - Validación de que las diferencias son químicamente reales
            """)

    with tab3:
        st.markdown("### Ejemplo: Análisis de Datos Espectrales")

        st.markdown("""
        A continuación se presenta un flujo de trabajo típico para análisis quimiométrico de datos espectrales:
        """)

        st.markdown("#### Paso 1: Carga de Datos")
        st.info("""
        **Acción:** Navega a "Cargar Dataset" en el panel lateral.
        
        - Sube un archivo CSV o Excel con tus datos espectrales
        - Las filas representan muestras individuales
        - Las columnas representan variables químicas (longitudes de onda, picos FAME, concentraciones)
        - Asegúrate de incluir al menos una columna categórica (ej: tipo de feedstock, lote, concentración)
        """)

        st.markdown("#### Paso 2: Preprocesamiento")
        st.info("""
        **Acción:** Ve a "Preprocesamiento de Datos"
        
        1. **Limpieza de valores nulos:** Elimina filas con datos faltantes
        2. **Eliminación de columnas:** Remueve variables no relevantes (ej: identificadores, fechas)
        3. **Estandarización:** Aplica transformación Z-score a variables numéricas
        
        **Importante:** La estandarización es esencial antes de PCA y clustering.
        """)

        st.markdown("#### Paso 3: Análisis ANOVA")
        st.info("""
        **Acción:** Selecciona "ANOVA" en el panel lateral
        
        - **Variable Dependiente:** Elige una variable química numérica (ej: intensidad de pico)
        - **Variable Factor:** Selecciona la variable categórica (ej: tipo de feedstock)
        - **Interpretación:**
          - Si p < 0.05: Existen diferencias significativas entre grupos
          - Revisa el Test de Tukey para identificar qué pares de grupos difieren
          - Observa los box plots y violin plots para entender la distribución
        """)

        st.markdown("#### Paso 4: Análisis PCA")
        st.info("""
        **Acción:** Navega a "Análisis PCA"
        
        1. **Selección de variables:** Elige las columnas numéricas relevantes
        2. **Scree Plot:** Determina cuántos componentes capturan la varianza (usualmente 2-3 para >80%)
        3. **Scores Plot:** 
           - Selecciona qué componentes visualizar (PC1 vs PC2, PC1 vs PC3, etc.)
           - Identifica agrupamientos de muestras similares
           - Detecta outliers (puntos muy alejados)
        4. **Loadings:** Identifica qué variables contribuyen más a cada componente
        5. **Biplot:** Relaciona variables con la separación de muestras
        
        **Personalización disponible:**
        - Cambiar paleta de colores
        - Seleccionar diferentes pares de componentes
        - Colorear por feedstock o concentración
        """)

        st.markdown("#### Paso 5: Interpretación Química")
        st.success("""
        **Integra los resultados:**
        
        - **ANOVA:** Confirma qué variables diferencian significativamente los grupos
        - **PCA:** Visualiza la estructura multivariada y detecta patrones
        - **Loadings:** Identifica qué picos o señales causan la separación
        - **Consistencia:** Verifica que los grupos en PCA correspondan con diferencias significativas en ANOVA
        
        **Ejemplo de conclusión química:**
        "Las muestras de feedstock A y B se separan claramente en PC1 (65% varianza), 
        principalmente debido a diferencias en las variables X1 y X5 (loadings altos). 
        El ANOVA confirma que estas diferencias son estadísticamente significativas (p < 0.001)."
        """)

        st.markdown("---")
        st.markdown("#### Dataset de Ejemplo")

        # Create example dataset
        np.random.seed(42)
        n_samples = 30

        example_data = {
            "Feedstock": ["Tipo_A"] * 10 + ["Tipo_B"] * 10 + ["Tipo_C"] * 10,
            "Peak_280nm": np.concatenate(
                [
                    np.random.normal(15, 2, 10),
                    np.random.normal(25, 2, 10),
                    np.random.normal(20, 2, 10),
                ]
            ),
            "Peak_320nm": np.concatenate(
                [
                    np.random.normal(30, 3, 10),
                    np.random.normal(35, 3, 10),
                    np.random.normal(40, 3, 10),
                ]
            ),
            "Peak_450nm": np.concatenate(
                [
                    np.random.normal(50, 4, 10),
                    np.random.normal(45, 4, 10),
                    np.random.normal(55, 4, 10),
                ]
            ),
            "Peak_600nm": np.concatenate(
                [
                    np.random.normal(20, 2, 10),
                    np.random.normal(30, 2, 10),
                    np.random.normal(25, 2, 10),
                ]
            ),
        }

        example_df = pd.DataFrame(example_data)

        st.markdown("**Descarga este dataset de ejemplo para practicar:**")
        st.dataframe(example_df.head(10), use_container_width=True)

        csv_example = example_df.to_csv(index=False)
        st.download_button(
            label="Descargar Dataset de Ejemplo (CSV)",
            data=csv_example,
            file_name="ejemplo_espectros.csv",
            mime="text/csv",
        )


# -------------------------------------------------
# CARGAR DATASET
# -------------------------------------------------
def cargar_dataset():
    st.header("📂 Cargar Dataset")

    uploaded_file = st.file_uploader(
        "Arrastra o selecciona tu archivo CSV o Excel",
        type=["csv", "xlsx", "xls"],
        help="Formatos soportados: CSV, XLSX, XLS",
    )

    if uploaded_file:
        df = None

        with st.spinner("Cargando archivo..."):
            if uploaded_file.name.endswith(".csv"):
                try:
                    df = pd.read_csv(uploaded_file)
                except Exception as e:
                    st.error(f"❌ Error leyendo archivo CSV: {e}")

            elif uploaded_file.name.endswith((".xlsx", ".xls")):
                try:
                    df = pd.read_excel(uploaded_file)
                except Exception as e:
                    st.error(f"❌ Error leyendo archivo Excel: {e}")

        if df is not None:
            st.session_state.df = df
            st.success("✅ ¡Archivo cargado exitosamente!")

            # Dataset overview
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📋 Filas", df.shape[0])
            with col2:
                st.metric("📊 Columnas", df.shape[1])
            with col3:
                st.metric(
                    "💾 Tamaño", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB"
                )

            st.markdown("### 👁️ Vista Previa de los Datos")
            st.dataframe(df.head(10), use_container_width=True)

            # Statistical summary
            st.markdown("## 📊 Resumen Estadístico Básico")

            numeric_df = df.select_dtypes(include=["number"])

            if numeric_df.empty:
                st.warning(
                    "⚠️ El dataset no contiene columnas numéricas para calcular estadísticas."
                )
            else:
                tab1, tab2 = st.tabs(["📌 Tendencia Central", "📐 Dispersión"])

                with tab1:
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown("**📍 Media**")
                        st.dataframe(numeric_df.mean(), use_container_width=True)

                    with col2:
                        st.markdown("**🎯 Mediana**")
                        st.dataframe(numeric_df.median(), use_container_width=True)

                    with col3:
                        st.markdown("**🔢 Moda**")
                        st.dataframe(
                            numeric_df.mode().iloc[0], use_container_width=True
                        )

                with tab2:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**📏 Rango**")
                        st.dataframe(
                            numeric_df.max() - numeric_df.min(),
                            use_container_width=True,
                        )

                        st.markdown("**📊 Desviación Estándar**")
                        st.dataframe(numeric_df.std(), use_container_width=True)

                    with col2:
                        st.markdown("**📈 Varianza**")
                        st.dataframe(numeric_df.var(), use_container_width=True)

                        st.markdown("**📦 Rango Intercuartílico (IQR)**")
                        st.dataframe(
                            numeric_df.quantile(0.75) - numeric_df.quantile(0.25),
                            use_container_width=True,
                        )

            st.markdown("---")

            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button(
                    "➡️ Ir a Preprocesamiento de Datos",
                    type="primary",
                    use_container_width=True,
                ):
                    st.session_state.page = "Preprocesamiento de Datos"
                    st.rerun()


# -------------------------------------------------
# PREPROCESSING PAGE
# -------------------------------------------------
def preprocessing_page():
    st.header("🔧 Preprocesamiento de Datos")

    if st.session_state.df is None:
        st.warning("⚠️ Aún no has cargado un dataset.")
        if st.button("⬅️ Regresar a cargar dataset"):
            st.session_state.page = "Cargar dataset"
            st.rerun()
        return

    df_current = st.session_state.df.copy()
    current_rows, current_cols = df_current.shape

    col1, col2 = st.columns(2)
    with col1:
        st.metric("📋 Filas actuales", current_rows)
    with col2:
        st.metric("📊 Columnas actuales", current_cols)

    st.dataframe(df_current.head(), use_container_width=True)

    st.markdown("---")

    # Limpieza NaN
    with st.expander("🧹 Limpieza de Valores Nulos (NaN)", expanded=True):
        initial_nan = df_current.isnull().any(axis=1).sum()
        st.metric("Filas con valores nulos", initial_nan)

        if st.button("🗑️ Eliminar filas con NaN", key="clean_btn"):
            df_clean = df_current.dropna()
            dropped = df_current.shape[0] - df_clean.shape[0]
            st.session_state.df = df_clean
            st.success(f"✅ Se eliminaron **{dropped}** filas.")
            st.rerun()

    # Eliminación de Columnas
    with st.expander("❌ Eliminación de Columnas"):
        st.write("Selecciona las columnas que deseas eliminar:")

        cols_to_display = list(df_current.columns)
        cols_per_row = 4

        for i in range(0, len(cols_to_display), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, column in enumerate(cols_to_display[i : i + cols_per_row]):
                with cols[j]:
                    if st.button(f"❌ {column}", key=f"drop_{column}"):
                        df_updated = df_current.drop(columns=[column])
                        st.session_state.df = df_updated
                        st.success(f"Columna '{column}' eliminada.")
                        st.rerun()

    # Estandarización
    with st.expander("📏 Estandarización (Z-Score)", expanded=True):
        numerical_cols = df_current.select_dtypes(include=[np.number]).columns.tolist()

        if not numerical_cols:
            st.warning("⚠️ No hay columnas numéricas disponibles.")
        else:
            st.info(f"**Columnas numéricas detectadas:** {', '.join(numerical_cols)}")

            if st.button(
                "⚡ Estandarizar Columnas Numéricas",
                key="standardize_btn",
                type="primary",
            ):
                try:
                    # Hacer una copia para no modificar el original
                    df_to_standardize = df_current.copy()

                    # Reemplazar infinitos y eliminar NaNs
                    df_to_standardize.replace([np.inf, -np.inf], np.nan, inplace=True)
                    df_to_standardize.dropna(subset=numerical_cols, inplace=True)

                    # Estandarizar solo las columnas numéricas
                    scaler = StandardScaler()
                    df_to_standardize[numerical_cols] = scaler.fit_transform(
                        df_to_standardize[numerical_cols]
                    )

                    # Guardar el dataframe estandarizado
                    st.session_state.df = df_to_standardize
                    st.session_state.standardized = True
                    st.success("✅ ¡Estandarización completada!")
                    time.sleep(0.5)
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error durante estandarización: {e}")
                    st.error(f"Detalles: {str(e)}")

    if st.session_state.get("standardized"):
        st.success("✅ El dataset ya fue estandarizado.")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button(
                "➡️ Ir a Análisis PCA", type="primary", use_container_width=True
            ):
                st.session_state.page = "PCA"
                st.rerun()


# -------------------------------------------------
# PCA PAGE (ENHANCED)
# -------------------------------------------------
def pca_page():
    st.header("📈 Análisis de Componentes Principales (PCA)")

    with st.expander("ℹ️ ¿Qué es el PCA?", expanded=False):
        st.markdown("""
        El **Análisis de Componentes Principales (PCA)** es una técnica de reducción de dimensionalidad que:
        
        - 🎯 Transforma variables correlacionadas en componentes independientes
        - 📊 Captura la mayor varianza posible en menos dimensiones
        - 🔍 Facilita la visualización de datos multidimensionales
        - ⚡ Mejora el rendimiento de modelos de machine learning
        
        **Importante:** Los datos deben estar estandarizados antes de aplicar PCA.
        """)

    COLOR_PALETTES = {
        "QuimioAnalytics (Custom)": [
            "#B0A461",
            "#4A525A",
            "#E0D7B2",
            "#2E3339",
            "#8E9E9A",
        ],
        "Viridis (Default)": "viridis",
        "Plasma": "plasma",
        "Cividis": "cividis",
        "Inferno": "inferno",
        "Magma": "magma",
        "Cool Warm": ["#0000FF", "#87CEEB", "#FFFFFF", "#FF6347", "#FF0000"],
        "Greyscale": ["#000000", "#555555", "#AAAAAA", "#CCCCCC", "#FFFFFF"],
    }

    if st.session_state.df is None:
        st.warning("⚠️ Carga un dataset primero.")
        return

    df_pca = st.session_state.df.copy()
    numeric_cols = df_pca.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.error("❌ No hay columnas numéricas en el dataset.")
        return

    if not st.session_state.get("standardized", False):
        st.warning("⚠️ Debes estandarizar los datos antes de realizar PCA.")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        selected_columns = st.multiselect(
            "🎯 Selecciona columnas numéricas para PCA:",
            numeric_cols,
            default=numeric_cols,
        )

    with col2:
        st.session_state.plot_color_choice = st.selectbox(
            "🎨 Paleta de colores:",
            list(COLOR_PALETTES.keys()),
            index=list(COLOR_PALETTES.keys()).index(st.session_state.plot_color_choice),
        )

    if len(selected_columns) < 2:
        st.warning("⚠️ Selecciona al menos 2 columnas para aplicar PCA.")
        return

    if st.button("▶️ Aplicar PCA", key="run_pca", type="primary"):
        st.session_state.pca_ready = True
        st.session_state.pca_columns = selected_columns
        st.rerun()

    if st.session_state.get("pca_ready", False):
        columns = st.session_state.get("pca_columns", selected_columns)
        X = df_pca[columns].values
        n_components = min(X.shape)

        pca = PCA(n_components=n_components)
        pc_values = pca.fit_transform(X)

        explained = pca.explained_variance_ratio_
        cumulative = np.cumsum(explained)

        st.success("✅ PCA aplicado correctamente")

        # Tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "📊 Varianza Explicada",
                "🎯 Gráfico de Componentes",
                "🔗 Biplot",
                "🔍 Loadings",
                "📋 Datos Transformados",
            ]
        )

        with tab1:
            st.subheader("Scree Plot - Varianza Explicada")

            df_var = pd.DataFrame(
                {
                    "Componente": [f"PC{i + 1}" for i in range(n_components)],
                    "Varianza (%)": explained * 100,
                    "Acumulada (%)": cumulative * 100,
                }
            )

            palette = COLOR_PALETTES[st.session_state.plot_color_choice]

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=df_var["Componente"],
                    y=df_var["Varianza (%)"],
                    name="Varianza Individual",
                    marker_color="#B0A461",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=df_var["Componente"],
                    y=df_var["Acumulada (%)"],
                    mode="lines+markers",
                    name="Varianza Acumulada",
                    line=dict(color="#4A525A", width=3),
                    marker=dict(size=10),
                )
            )

            fig.update_layout(
                title="Varianza Explicada por Componente Principal",
                xaxis_title="Componente",
                yaxis_title="Varianza (%)",
                hovermode="x unified",
                template="plotly_white",
            )

            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(df_var, use_container_width=True)

        with tab2:
            st.subheader("Visualización de Componentes Principales")

            if n_components >= 2:
                col1, col2 = st.columns(2)

                with col1:
                    pc_x = st.selectbox(
                        "Componente X:",
                        [f"PC{i + 1}" for i in range(n_components)],
                        index=0,
                    )

                with col2:
                    pc_y = st.selectbox(
                        "Componente Y:",
                        [f"PC{i + 1}" for i in range(n_components)],
                        index=1,
                    )

                pc_x_idx = int(pc_x.replace("PC", "")) - 1
                pc_y_idx = int(pc_y.replace("PC", "")) - 1

                df_plot = pd.DataFrame(
                    pc_values[:, [pc_x_idx, pc_y_idx]], columns=[pc_x, pc_y]
                )
                df_plot["ID"] = df_pca.index.astype(str)

                fig_scatter = px.scatter(
                    df_plot,
                    x=pc_x,
                    y=pc_y,
                    color="ID",
                    title=f"PCA: {pc_x} vs {pc_y}",
                    color_discrete_sequence=palette
                    if isinstance(palette, list)
                    else None,
                )

                fig_scatter.update_traces(
                    marker=dict(size=10, line=dict(width=1, color="white"))
                )
                fig_scatter.update_layout(template="plotly_white")

                st.plotly_chart(fig_scatter, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(f"Varianza {pc_x}", f"{explained[pc_x_idx] * 100:.2f}%")
                with col2:
                    st.metric(f"Varianza {pc_y}", f"{explained[pc_y_idx] * 100:.2f}%")

        with tab3:
            st.subheader("Biplot - Scores y Loadings Combinados")

            st.markdown("""
            El **biplot** combina las puntuaciones (scores) de las muestras con las cargas (loadings) de las variables.
            Permite identificar qué variables son responsables de las agrupaciones observadas en las muestras.
            """)

            if n_components >= 2:
                col1, col2 = st.columns(2)

                with col1:
                    pc_x_bi = st.selectbox(
                        "Componente X:",
                        [f"PC{i + 1}" for i in range(n_components)],
                        index=0,
                        key="biplot_x",
                    )

                with col2:
                    pc_y_bi = st.selectbox(
                        "Componente Y:",
                        [f"PC{i + 1}" for i in range(n_components)],
                        index=1,
                        key="biplot_y",
                    )

                pc_x_idx_bi = int(pc_x_bi.replace("PC", "")) - 1
                pc_y_idx_bi = int(pc_y_bi.replace("PC", "")) - 1

                # Create biplot
                fig_biplot = go.Figure()

                # Add scores (samples)
                scores_x = pc_values[:, pc_x_idx_bi]
                scores_y = pc_values[:, pc_y_idx_bi]

                fig_biplot.add_trace(
                    go.Scatter(
                        x=scores_x,
                        y=scores_y,
                        mode="markers",
                        name="Muestras (Scores)",
                        marker=dict(
                            size=10, color="#B0A461", line=dict(width=1, color="white")
                        ),
                        text=[f"Muestra {i}" for i in df_pca.index],
                        hovertemplate="<b>%{text}</b><br>%{x:.2f}, %{y:.2f}<extra></extra>",
                    )
                )

                # Add loadings (variables) as arrows
                loadings_x = pca.components_[pc_x_idx_bi, :]
                loadings_y = pca.components_[pc_y_idx_bi, :]

                # Scale loadings for visualization
                scale_factor = (
                    0.8
                    * max(np.max(np.abs(scores_x)), np.max(np.abs(scores_y)))
                    / max(np.max(np.abs(loadings_x)), np.max(np.abs(loadings_y)))
                )

                for i, var_name in enumerate(columns):
                    fig_biplot.add_trace(
                        go.Scatter(
                            x=[0, loadings_x[i] * scale_factor],
                            y=[0, loadings_y[i] * scale_factor],
                            mode="lines+markers+text",
                            name=var_name,
                            line=dict(color="#4A525A", width=2),
                            marker=dict(size=[0, 8], color="#4A525A"),
                            text=["", var_name],
                            textposition="top center",
                            textfont=dict(size=10, color="#4A525A"),
                            hovertemplate=f"<b>{var_name}</b><br>Loading X: {loadings_x[i]:.3f}<br>Loading Y: {loadings_y[i]:.3f}<extra></extra>",
                            showlegend=False,
                        )
                    )

                fig_biplot.update_layout(
                    title=f"Biplot: {pc_x_bi} vs {pc_y_bi}",
                    xaxis_title=f"{pc_x_bi} ({explained[pc_x_idx_bi] * 100:.2f}%)",
                    yaxis_title=f"{pc_y_bi} ({explained[pc_y_idx_bi] * 100:.2f}%)",
                    template="plotly_white",
                    hovermode="closest",
                    width=800,
                    height=600,
                )

                # Add origin lines
                fig_biplot.add_hline(
                    y=0, line_dash="dash", line_color="gray", opacity=0.5
                )
                fig_biplot.add_vline(
                    x=0, line_dash="dash", line_color="gray", opacity=0.5
                )

                st.plotly_chart(fig_biplot, use_container_width=True)

                st.markdown("### Interpretación del Biplot")
                st.info("""
                **Cómo leer el biplot:**
                
                - **Puntos (Muestras):** Representan las observaciones proyectadas en el espacio de componentes principales
                - **Vectores (Variables):** Muestran la dirección y magnitud de la contribución de cada variable
                - **Ángulos entre vectores:**
                  - Ángulo pequeño (< 30°): Variables positivamente correlacionadas
                  - Ángulo cercano a 90°: Variables no correlacionadas
                  - Ángulo cercano a 180°: Variables negativamente correlacionadas
                - **Longitud del vector:** Indica cuán bien está representada la variable en estas componentes
                - **Dirección muestra-variable:** Si una muestra está en la dirección de un vector, tiene valores altos en esa variable
                """)

        with tab4:
            st.subheader("Matriz de Loadings")
            st.markdown(
                "Los **loadings** muestran la contribución de cada variable original a cada componente principal."
            )

            loadings_df = pd.DataFrame(
                pca.components_.T,
                columns=[f"PC{i + 1}" for i in range(n_components)],
                index=columns,
            )

            st.dataframe(
                loadings_df.style.background_gradient(cmap="RdYlGn", axis=None),
                use_container_width=True,
            )

            # Heatmap de loadings
            fig_heatmap = px.imshow(
                loadings_df.T,
                labels=dict(x="Variables", y="Componentes", color="Loading"),
                x=loadings_df.index,
                y=loadings_df.columns,
                color_continuous_scale="RdBu_r",
                aspect="auto",
            )
            fig_heatmap.update_layout(title="Heatmap de Loadings")
            st.plotly_chart(fig_heatmap, use_container_width=True)

        with tab4:
            st.subheader("Datos Transformados (Scores)")

            pc_df = pd.DataFrame(
                pc_values, columns=[f"PC{i + 1}" for i in range(n_components)]
            )

            st.dataframe(pc_df.head(20), use_container_width=True)

            csv = pc_df.to_csv(index=False)
            st.download_button(
                label="📥 Descargar datos PCA (CSV)",
                data=csv,
                file_name="pca_transformed.csv",
                mime="text/csv",
            )

        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button(
                "➡️ Ir a Clustering",
                key="to_clustering",
                type="primary",
                use_container_width=True,
            ):
                st.session_state.page = "Clustering"
                st.session_state.pca_ready = False
                st.rerun()


# -------------------------------------------------
# CLUSTERING PAGE (ENHANCED)
# -------------------------------------------------
def cluster_page():
    st.header("🍇 Clustering")

    with st.expander("ℹ️ ¿Qué es el Clustering?", expanded=False):
        st.markdown("""
        El **Clustering** es una técnica de aprendizaje no supervisado que agrupa datos similares en "clústeres" o grupos para identificar patrones y estructuras ocultas. 
        
        - 🎯 Su objetivo es que los puntos de datos dentro de un mismo grupo sean más parecidos entre sí que con los de otros grupos, sin tener una etiqueta previa. 
        
        """)

    # Check dataset
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ Aún no has cargado un dataset.")
        return

    df = st.session_state.df.copy()

    COLOR_PALETTES = {
        "QuimioAnalytics (Custom)": [
            "#B0A461",
            "#4A525A",
            "#E0D7B2",
            "#2E3339",
            "#8E9E9A",
        ],
        "Viridis (Default)": "viridis",
        "Plasma": "plasma",
        "Cividis": "cividis",
        "Inferno": "inferno",
        "Magma": "magma",
        "Cool Warm": ["#0000FF", "#87CEEB", "#FFFFFF", "#FF6347", "#FF0000"],
        "Greyscale": ["#000000", "#555555", "#AAAAAA", "#CCCCCC", "#FFFFFF"],
    }

    palette = COLOR_PALETTES.get(
        st.session_state.get("plot_color_choice", "QuimioAnalytics (Custom)")
    )

    st.markdown("---")

    # ================================================
    # 🔢 NUMERICAL COLUMNS
    # ================================================
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numerical_cols:
        st.error("❌ No hay columnas numéricas para clustering.")
        return

    # ================================================
    # 🎯 PCA Projection (Consistent with PCA Page)
    # ================================================
    st.subheader("📉 Proyección PCA para Visualización de Clústeres")

    if st.session_state.get("pca_ready", False):
        try:
            # Use the PCA computed in pca_page
            columns = st.session_state.pca_columns
            X = df[columns].values
            pca = PCA(n_components=2)
            proj = pca.fit_transform(X)

            df["PC1"] = proj[:, 0]
            df["PC2"] = proj[:, 1]

            st.success("🎯 Usando PCA seleccionado en la página anterior.")
        except:
            st.error("El PCA previo no fue compatible. Se recalculará.")
            pca = PCA(n_components=2)
            proj = pca.fit_transform(df[numerical_cols])
            df["PC1"] = proj[:, 0]
            df["PC2"] = proj[:, 1]
    else:
        # Compute PCA from scratch for visualization
        pca = PCA(n_components=2)
        proj = pca.fit_transform(df[numerical_cols])
        df["PC1"] = proj[:, 0]
        df["PC2"] = proj[:, 1]

    # ================================================
    # 🔷 K-MEANS SECTION
    # ================================================
    with st.expander("📌 K-Means Clustering", expanded=True):
        k = st.slider("Número de clusters (k)", 2, 12, 3)
        n_init = st.slider("Repeticiones (n_init)", 5, 30, 10)
        init_method = st.selectbox("Método de inicialización", ["k-means++"])

        if st.button("🚀 Ejecutar K-Means", type="primary", key="run_kmeans"):
            try:
                kmeans = KMeans(
                    n_clusters=k, n_init=n_init, init=init_method, random_state=42
                )
                labels = kmeans.fit_predict(df[numerical_cols])
                df["Cluster_KMeans"] = labels

                sil_score = silhouette_score(df[numerical_cols], labels)

                st.success(
                    f"✅ K-Means completado. Silhouette Score: **{sil_score:.4f}**"
                )

                fig = px.scatter(
                    df,
                    x="PC1",
                    y="PC2",
                    color="Cluster_KMeans",
                    title="Clústeres K-Means proyectados en espacio PCA",
                    opacity=0.85,
                    color_discrete_sequence=palette
                    if isinstance(palette, list)
                    else None,
                )
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("📊 Resumen por Clúster")
                summary = df.groupby("Cluster_KMeans")[numerical_cols].mean()
                st.dataframe(summary, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error ejecutando K-Means: {e}")

    # ================================================
    # 🌿 HIERARCHICAL CLUSTERING
    # ================================================
    with st.expander("🌿 Clustering Jerárquico"):
        linkage_method = st.selectbox(
            "Método de enlace (linkage)", ["single", "complete", "average", "ward"]
        )

        num_clusters_h = st.slider("Número de clusters", 2, 12, 3, key="clusters_hier")

        if st.button("🌱 Ejecutar Clustering Jerárquico", key="run_hier"):
            try:
                model = AgglomerativeClustering(
                    n_clusters=num_clusters_h,
                    linkage=linkage_method,
                    metric="euclidean",
                )
                df["Cluster_Hier"] = model.fit_predict(df[numerical_cols])

                st.success("✅ Clustering jerárquico completado.")

                fig2 = px.scatter(
                    df,
                    x="PC1",
                    y="PC2",
                    color="Cluster_Hier",
                    title="Clústeres Jerárquicos proyectados en espacio PCA",
                    opacity=0.85,
                    color_discrete_sequence=palette
                    if isinstance(palette, list)
                    else None,
                )
                st.plotly_chart(fig2, use_container_width=True)

                st.subheader("📊 Resumen por Clúster")
                st.dataframe(
                    df.groupby("Cluster_Hier")[numerical_cols].mean(),
                    use_container_width=True,
                )

                st.subheader("🌳 Dendrograma")
                try:
                    linked = linkage(df[numerical_cols], method=linkage_method)
                    fig_d = ff.create_dendrogram(linked, orientation="left")
                    st.plotly_chart(fig_d, use_container_width=True)
                except Exception as dendro_error:
                    st.error(f"No se pudo generar el dendrograma: {dendro_error}")

            except Exception as e:
                st.error(f"❌ Error ejecutando clustering jerárquico: {e}")

    # ================================================
    # NAVIGATION
    # ================================================
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("➡️ Ir a ANOVA", type="primary"):
            st.session_state.page = "ANOVA"
            st.rerun()


# -------------------------------------------------
# ANOVA PAGE (ENHANCED)
# -------------------------------------------------
def anova_page():
    st.header("🧮 Análisis de Varianza (ANOVA)")

    # Initialize flag
    if "anova_done" not in st.session_state:
        st.session_state.anova_done = False

    with st.expander("ℹ️ ¿Qué es el ANOVA?", expanded=False):
        st.markdown("""
        El **ANOVA** ...
        """)

    COLOR_PALETTES = {
        "QuimioAnalytics (Custom)": [
            "#B0A461",
            "#4A525A",
            "#E0D7B2",
            "#2E3339",
            "#8E9E9A",
        ],
        "Viridis (Default)": "viridis",
        "Plasma": "plasma",
        "Cool Warm": ["#0000FF", "#87CEEB", "#FFFFFF", "#FF6347", "#FF0000"],
        "Greyscale": ["#000000", "#555555", "#AAAAAA", "#CCCCCC", "#FFFFFF"],
    }

    if st.session_state.df is None:
        st.warning("⚠️ Carga un dataset primero.")
        return

    df_anova = st.session_state.df.copy()

    if "plot_color_choice" not in st.session_state:
        st.session_state.plot_color_choice = "QuimioAnalytics (Custom)"

    numerical_cols = df_anova.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_anova.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    if not numerical_cols or not categorical_cols:
        st.error(
            "❌ Se necesita al menos una columna numérica y una categórica para ANOVA."
        )
        return

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        y_var = st.selectbox(
            "🎯 Variable Dependiente (Numérica):", numerical_cols, key="anova_y"
        )

    with col2:
        x_var = st.selectbox(
            "🏷️ Variable Factor/Grupo (Categórica):", categorical_cols, key="anova_x"
        )

    with col3:
        st.session_state.plot_color_choice = st.selectbox(
            "🎨 Paleta:",
            list(COLOR_PALETTES.keys()),
            index=list(COLOR_PALETTES.keys()).index(st.session_state.plot_color_choice),
        )

    st.markdown("---")

    # --------------------------------------------------
    # ▶️ APPLY ANOVA (kept exactly as you had it)
    # --------------------------------------------------

    if st.button("▶️ Aplicar ANOVA", type="primary"):
        df_model = df_anova[[y_var, x_var]].copy()
        df_model = df_model.dropna()

        df_model[x_var] = df_model[x_var].astype("category")

        if df_model.shape[0] < 3:
            st.error("❌ No hay suficientes datos para realizar ANOVA.")
            return

        n_groups = df_model[x_var].nunique()
        if n_groups < 2:
            st.error(
                f"❌ Se necesitan al menos 2 grupos diferentes. Solo se encontró {n_groups} grupo."
            )
            return

        formula = f'Q("{y_var}") ~ C(Q("{x_var}"))'

        try:
            model = ols(formula, data=df_model).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)

            st.success("✅ ANOVA completado")

            # Store values so the button can work later
            st.session_state.anova_done = True
            st.session_state.anova_table = anova_table
            st.session_state.df_model = df_model
            st.session_state.y_var = y_var
            st.session_state.x_var = x_var

        except Exception as e:
            st.error(f"❌ Error al calcular ANOVA: {e}")
            return

    # --------------------------------------------------
    # SHOW THE AI CHAT BUTTON ONLY IF ANOVA WAS RUN
    # --------------------------------------------------

    if st.session_state.anova_done:
        st.markdown("---")
        colA, colB, colC = st.columns([1, 1, 1])
        with colB:
            if st.button(
                "➡️ Ir con Heisenberg", type="primary", use_container_width=True
            ):
                st.session_state.page = "AI Chat"
                st.rerun()

        # --------------------------------------------------
        # Now show all tabs normally (your original code)
        # --------------------------------------------------
        anova_table = st.session_state.anova_table
        df_model = st.session_state.df_model
        y_var = st.session_state.y_var
        x_var = st.session_state.x_var

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "📊 Tabla ANOVA",
                "📈 Visualizaciones",
                "🔍 Test Post-Hoc",
                "📋 Estadísticas Descriptivas",
            ]
        )


# -------------------------
# PAGE: AI CHAT
# -------------------------
def ai_chat_page():
    st.header("💬 Chat de I.A.")

    # Check dataset
    if st.session_state.df is None:
        st.warning("⚠️ Carga un dataset primero.")
        return

    df = st.session_state.df

    # Load image safely
    img_path = pathlib.Path("man.png")
    img_data_uri = None

    try:
        with img_path.open("rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode("utf-8")
            img_data_uri = f"data:image/png;base64,{b64}"
    except Exception as e:
        st.warning(f"No se pudo cargar man.png ({e}).")

    # ----------- CSS (fixed version) ----------
    st.markdown(
        f"""
        <style>
        .floating-image-container {{
            position: fixed;
            top: 500px;
            right: 25px;
            width: 80px;
            z-index: 9999;
            text-align: center;
        }}
        
        .floating-image-container img {{
            width: 100%;
            height: auto;
            border-radius: 0px;
            box-shadow: 0 0px 0px rgba(0,0,0,0);
            display: block;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ----------- Insert floating image (fixed) -----------
    if img_data_uri:
        st.markdown(
            f"""
            <div class="floating-image-container">
                <img src="{img_data_uri}">
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # fallback
        st.image("man.png", width=150)

    # -------------------------
    # Initialize session state
    # -------------------------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "clear_input" not in st.session_state:
        st.session_state.clear_input = False

    # --- The input processing logic needs to stay at the top for the rerun to work correctly. ---

    # Process submitted message from the form that will be defined later
    if st.session_state.get("submitted_input"):
        user_input = st.session_state.submitted_input
        # Remove the flag and value immediately after retrieving them
        del st.session_state.submitted_input

        if user_input.strip():
            st.session_state.chat_history.append(
                {"role": "user", "content": user_input}
            )
            st.session_state.last_user_input = user_input
            # st.session_state.clear_input = True # No longer needed with the new st.chat_input pattern
            st.rerun()
        # else:
        # st.warning("⚠️ Please enter a message before sending.") # Use st.chat_input, which handles empty messages

    # -------------------------
    # Render chat history
    # -------------------------
    # Use st.container() to hold the chat history
    chat_box_container = st.container()

    with chat_box_container:
        # Loop through and display messages
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(msg["content"])

        # Placeholder for streaming text. It must be created *before* the input logic for correct placement.
        streaming_placeholder = st.empty()

    # -------------------------
    # Build dataset description for the AI (kept in place)
    # -------------------------
    dataset_description = f"""
You have access to the user's uploaded dataset.

COLUMN NAMES:
{", ".join(df.columns)}

DATA TYPES:
{df.dtypes.to_string()}

SUMMARY STATISTICS:
{df.describe().to_string()}

FIRST 10 ROWS:
{df.head(10).to_string(index=False)}
"""

    # -------------------------
    # Process LLM response (kept in place)
    # -------------------------
    if "last_user_input" in st.session_state:
        user_msg = st.session_state.pop("last_user_input")

        messages_for_api = [
            {
                "role": "system",
                "content": (
                    "You are an expert data analyst specializing in statistics, chemistry, "
                    "machine learning and dataset interpretation.\n"
                    "Use the dataset description below to answer questions accurately, "
                    "identify patterns, recommend preprocessing steps, "
                    "and generate insights.\n\n"
                    f"{dataset_description}"
                ),
            },
            # Append historical chat messages
            *[
                msg
                for msg in st.session_state.chat_history
                if msg["content"]
                != dataset_description  # Avoid including the system prompt description in the history sent to the model repeatedly
            ],
        ]

        try:
            # client.chat.completions.create is assumed to be available
            stream = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages_for_api,
                stream=True,
            )

            full_response = ""

            # Streaming response inside the placeholder
            with streaming_placeholder.container():
                text_placeholder = st.empty()

                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content
                        text_placeholder.markdown(
                            f"<div style='text-align:left;'>**AI:** {full_response}▌</div>",
                            unsafe_allow_html=True,
                        )

                # Final update
                text_placeholder.markdown(
                    f"<div style='text-align:left;'>**AI:** {full_response}</div>",
                    unsafe_allow_html=True,
                )

                if full_response:
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": full_response}
                    )

            st.rerun()

        except NameError:
            # Handle case where 'client' is not defined in this snippet (simulating error handling)
            st.error("Groq API client not found (assuming external dependency).")
        except Exception as e:
            st.error(f"API Error: {e}")
            # Remove the last user message if the API call failed
            if (
                st.session_state.chat_history
                and st.session_state.chat_history[-1]["role"] == "user"
            ):
                st.session_state.chat_history.pop()

    user_input = st.chat_input(
        "Pregunta algo sobre el dataset cargado", key="chat_input_box"
    )

    # st.chat_input returns the user's message when submitted.
    if user_input:
        # Save the input to a session state variable to process it at the top of the script
        st.session_state.submitted_input = user_input
        st.rerun()


# -------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------
with st.sidebar:
    st.markdown("### 🧪 QuimioAnalytics")
    st.markdown("---")

    if st.button("🏠 Home", use_container_width=True):
        st.session_state.page = "Home"

    if st.button("📂 Cargar Dataset", use_container_width=True):
        st.session_state.page = "Cargar dataset"

    if st.button("🔧 Preprocesamiento", use_container_width=True):
        st.session_state.page = "Preprocesamiento de Datos"

    if st.button("📈 Análisis PCA", use_container_width=True):
        st.session_state.page = "PCA"

    if st.button("🍇 Clustering", use_container_width=True):
        st.session_state.page = "Clustering"

    if st.button("🧮 ANOVA", use_container_width=True):
        st.session_state.page = "ANOVA"

    if st.button("💬 AI Chat", use_container_width=True):
        st.session_state.page = "AI Chat"

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
if st.session_state.page == "Home":
    home_page()

elif st.session_state.page == "Cargar dataset":
    cargar_dataset()

elif st.session_state.page == "Preprocesamiento de Datos":
    preprocessing_page()

elif st.session_state.page == "PCA":
    pca_page()

elif st.session_state.page == "Clustering":
    cluster_page()

elif st.session_state.page == "ANOVA":
    anova_page()

elif st.session_state.page == "AI Chat":
    ai_chat_page()
