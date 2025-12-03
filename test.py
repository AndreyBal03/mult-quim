import streamlit as st
import time
import base64
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.api as sm
from statsmodels.formula.api import ols

# -------------------------------------------------
# MUST BE FIRST: Page config
# -------------------------------------------------
st.set_page_config(
    page_title="QuimioAnalytics",
    page_icon="",
    layout="centered"
)

# -------------------------------------------------
# Initialize session_state variables
# -------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

if "df" not in st.session_state:
    st.session_state.df = None

if "snow_triggered" not in st.session_state:
    st.session_state.snow_triggered = False

# Default color palette value
if "plot_color_choice" not in st.session_state:
    st.session_state.plot_color_choice = "Viridis (Default)"

# -------------------------------------------------
# Title CSS
# -------------------------------------------------
title_css = """
<style>
h1 {
    color: #B0A461 !important; 
    text-align: center;
}
</style>
"""
st.markdown(title_css, unsafe_allow_html=True)

# -------------------------------------------------
# Header (h2) Styling
# -------------------------------------------------
header_css = """
<style>
h2 {
    text-align: center !important;
    color: #99BFF0 !important;
}
</style>
"""
st.markdown(header_css, unsafe_allow_html=True)

# --- CSS Block for Custom Text Size (Place near the top of your script) ---
custom_font_size_css = """
<style>
/* Target the h2 element used by st.subheader */
h2 {
    font-size: 15px; /* Default st.subheader size is usually around 28px. Adjust this value (e.g., 20px, 36px) as needed. */
    color: #4A525A; /* Optional: You can adjust the color too */
}
/* To target subheaders *specifically* inside a column, you can try: */
/* .stColumn > h2 { font-size: 20px; } */
</style>
"""
st.markdown(custom_font_size_css, unsafe_allow_html=True)
# --------------------------------------------------------------------------

# -------------------------------------------------
# Convert image to Base64
# -------------------------------------------------
@st.cache_data
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        st.error(f"Image not found: {image_path}")
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
        background-size: contain;
        background-position: center;
    }}
    </style>
    """
else:
    page_bg_img = """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16ec73aa");
        background-size: contain;
        background-position: center;
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
    st.title(" 👨🏻‍🔬 QuimioAnalytics") 
    st.markdown("¡Hola! 👋 Esta plataforma está diseñada especialmente para estudiantes de química que desean explorar datos de manera intuitiva, sin necesidad de saber programar.")
    st.subheader("Cargar dataset")

    uploaded_file = st.file_uploader("Carga tu archivo CSV ó Excel", type=["csv", "xlsx"])

    if uploaded_file:
        df = None

        # Load CSV
        if uploaded_file.name.endswith(".csv"):
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Error leyendo archivo CSV: {e}")

        # Load Excel
        elif uploaded_file.name.endswith((".xlsx", ".xls")):
            try:
                df = pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"Error leyendo archivo Excel: {e}")

        if df is not None:
            st.session_state.df = df
            st.success("¡Archivo cargado exitosamente! 🎉")

            st.write("### Vista previa de los datos")
            st.dataframe(df.head())

            # ---------------------------------------------------------
            # BASIC STATISTICAL SUMMARY
            # ---------------------------------------------------------
            st.markdown("## 📊 Resumen Estadístico Básico")

            # Select only numeric columns
            numeric_df = df.select_dtypes(include=["number"])

            if numeric_df.empty:
                st.warning("El dataset no contiene columnas numéricas para calcular estadísticas.")
            else:
                col1, col2 = st.columns(2)

                # ----- CENTRAL TENDENCY -----
                with col1:
                    st.subheader("📌 Medidas de Tendencia Central")

                    mean_vals = numeric_df.mean()
                    median_vals = numeric_df.median()
                    mode_vals = numeric_df.mode().iloc[0]

                    st.write("**Media:**")
                    st.dataframe(mean_vals)

                    st.write("**Mediana:**")
                    st.dataframe(median_vals)

                    st.write("**Moda:**")
                    st.dataframe(mode_vals)

                # ----- DISPERSION -----
                with col2:
                    st.subheader("📐 Medidas de Dispersión                            ")

                    range_vals = numeric_df.max() - numeric_df.min()
                    std_vals = numeric_df.std()
                    var_vals = numeric_df.var()
                    iqr_vals = numeric_df.quantile(0.75) - numeric_df.quantile(0.25)

                    st.write("**Rango:**")
                    st.dataframe(range_vals)

                    st.write("**Desviación Estándar:**")
                    st.dataframe(std_vals)

                    st.write("**Varianza:**")
                    st.dataframe(var_vals)

            st.markdown("---")

            # ---------------------------------------------------------
            # Navigation button
            # ---------------------------------------------------------
            if st.button("Ir a Preprocesamiento de Datos 📊"):
                st.session_state.page = "Preprocesamiento de Datos"
                st.rerun()

# -------------------------------------------------
# PREPROCESSING PAGE
# -------------------------------------------------
def preprocessing_page():
    st.header("Preprocesamiento de Datos")

    if st.session_state.df is None:
        st.warning("Aún no has cargado un dataset.")
        if st.button("Regresar a Home"):
            st.session_state.page = "Home"
            st.rerun()
        return

    df_current = st.session_state.df.copy()
    current_rows, current_cols = df_current.shape

    st.info(f"Dataset actual con **{current_rows}** filas y **{current_cols}** columnas.")
    st.dataframe(df_current.head())

    st.markdown("---")
    
    # ----------------------------------------------------
    # 1. Limpieza NaN
    # ----------------------------------------------------
    st.subheader("Limpieza de Valores Nulos (NaN)")
    initial_nan = df_current.isnull().any(axis=1).sum()
    st.metric("Filas con al menos un valor nulo", initial_nan)

    if st.button("Limpiar Dataset (Eliminar filas con NaN)", key="clean_btn"):
        df_clean = df_current.dropna()
        dropped = df_current.shape[0] - df_clean.shape[0]
        st.session_state.df = df_clean 

        st.success(f"Limpieza completada. Se eliminaron **{dropped}** filas.")
        st.rerun()

    st.markdown("---")

    # ----------------------------------------------------
    # 2. Eliminación de Columnas
    # ----------------------------------------------------
    st.subheader("Eliminación de Columnas")
    st.write("Haz clic para eliminar columnas:")

    col_buttons = st.columns(min(current_cols, 8))
    index = 0

    for column in df_current.columns:
        with col_buttons[index]:
            if st.button(f"❌ {column}", key=f"drop_{column}"):
                df_updated = df_current.drop(columns=[column])
                st.session_state.df = df_updated
                st.success(f"Columna '{column}' eliminada.")
                st.rerun()

        index = (index + 1) % len(col_buttons)

    st.markdown("---")

    # ----------------------------------------------------
    # 3. Estandarización
    # ----------------------------------------------------
    st.subheader("Estandarización (Z-Score)")

    numerical_cols = df_current.select_dtypes(include=[np.number]).columns.tolist()

    if not numerical_cols:
        st.warning("No hay columnas numéricas.")
    else:
        st.info(f"Columnas numéricas: **{', '.join(numerical_cols)}**")

        if st.button("Estandarizar Columnas Numéricas", key="standardize_btn"):
            try:
                df_current.replace([np.inf, -np.inf], np.nan, inplace=True)
                df_current.dropna(subset=numerical_cols, inplace=True)

                scaler = StandardScaler()
                df_current[numerical_cols] = scaler.fit_transform(df_current[numerical_cols])

                st.session_state.df = df_current
                st.session_state.standardized = True
                st.success("¡Estandarización completada!")
                st.rerun()

            except Exception as e:
                st.error(f"Error durante estandarización: {e}")

    if st.session_state.get("standardized"):
        st.success("El dataset ya fue estandarizado.")

        if st.button("Ir a PCA ➡️"):
            st.session_state.page = "PCA"
            st.rerun()


# -------------------------------------------------
# PCA PAGE
# -------------------------------------------------
def pca_page():
    st.header("Análisis de Componentes Principales (PCA)")

    # 1. Explanation of PCA
    st.markdown("""
    ### ¿Qué es el PCA?
    
    El **Análisis de Componentes Principales (PCA)** es una técnica fundamental utilizada para la **reducción de dimensionalidad**.
    
    * **Objetivo:** Transforma un conjunto grande de variables interrelacionadas en un conjunto más pequeño de variables nuevas, llamadas **Componentes Principales (PCs)**.
    * **Proceso:** Cada PC es una combinación lineal de las variables originales y está diseñado para capturar la mayor parte posible de la **varianza** total presente en los datos. La primera PC captura la mayor varianza, la segunda la segunda mayor, y así sucesivamente.
    
    **Importante:** PCA siempre debe aplicarse a datos que han sido **escalados o estandarizados** previamente.
    """)
    st.markdown("---")          
                # PALETAS
    COLOR_PALETTES = {
        "QuimioAnalytics (Custom)": ["#B0A461", "#4A525A", "#E0D7B2", "#2E3339", "#8E9E9A"],
        "Viridis (Default)": "viridis",
        "Plasma": "plasma",
        "Cool Warm": ["#0000FF", "#87CEEB", "#FFFFFF", "#FF6347", "#FF0000"],
        "Greyscale": ["#000000", "#555555", "#AAAAAA", "#CCCCCC", "#FFFFFF"],
    }

    # ---------------------------------
    # CHECK DATA
    # ---------------------------------
    if st.session_state.df is None:
        st.warning("Carga un dataset primero.")
        return

    df_pca = st.session_state.df.copy()
    numeric_cols = df_pca.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.error("No hay columnas numéricas en el dataset.")
        return

    if not st.session_state.get("standardized", False):
        st.warning("Debes estandarizar los datos antes de realizar PCA.")
        return

    # ---------------------------------
    # COLOR PALETTE SELECTOR
    # ---------------------------------
    st.session_state.plot_color_choice = st.selectbox(
        "Paleta de color para los gráficos:",
        list(COLOR_PALETTES.keys()),
        index=list(COLOR_PALETTES.keys()).index(st.session_state.plot_color_choice)
    )

    # ---------------------------------
    # VARIABLE SELECTION
    # ---------------------------------
    st.subheader("Selecciona columnas numéricas para aplicar PCA")
    selected_columns = st.multiselect(
        "Columnas disponibles:",
        numeric_cols,
        default=numeric_cols  # all selected by default
    )

    if len(selected_columns) < 2:
        st.warning("Selecciona al menos 2 columnas para aplicar PCA.")
        return

    # ---------------------------------
    # RUN PCA BUTTON (fixed behavior)
    # ---------------------------------
    run_pca = st.button("Aplicar PCA y Mostrar Gráficos", key="run_pca")

    if run_pca:
        st.session_state.pca_ready = True
        st.session_state.pca_columns = selected_columns
        st.rerun()

    # ---------------------------------
    # PCA RESULTS
    # ---------------------------------
    if st.session_state.get("pca_ready", False):

        columns = st.session_state.get("pca_columns", selected_columns)
        X = df_pca[columns].values
        n_components = min(X.shape)

        pca = PCA(n_components=n_components)
        pc_values = pca.fit_transform(X)

        explained = pca.explained_variance_ratio_
        cumulative = np.cumsum(explained)

        st.success("PCA aplicado correctamente.")

        # ---------------------------------
        # SCREE PLOT
        # ---------------------------------
        st.subheader("📊 Varianza Explicada (Scree Plot)")

        df_var = pd.DataFrame({
            "Componente": [f"PC{i+1}" for i in range(n_components)],
            "Varianza": explained,
            "Acumulada": cumulative
        })

        palette = COLOR_PALETTES[st.session_state.plot_color_choice]

        fig_bar = px.bar(
            df_var,
            x="Componente",
            y="Varianza",
            color="Varianza",
            color_continuous_scale=palette if isinstance(palette, str) else None
        )

        fig_bar.add_scatter(
            x=df_var["Componente"],
            y=df_var["Acumulada"],
            mode="lines+markers",
            name="Varianza Acumulada",
            line=dict(color="black")
        )

        st.plotly_chart(fig_bar, use_container_width=True)

        # ---------------------------------
        # 2D PCA SCATTER (PC1 vs PC2)
        # ---------------------------------
        if n_components >= 2:
            st.subheader("📌 PCA (PC1 vs PC2)")

            df_2d = pd.DataFrame(pc_values[:, :2], columns=["PC1", "PC2"])
            df_2d["ID"] = df_pca.index.astype(str)

            fig_scatter = px.scatter(
                df_2d,
                x="PC1",
                y="PC2",
                color="ID",
                title="PCA - Primeras Dos Componentes",
                color_discrete_sequence=palette if isinstance(palette, list) else None
            )

            st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.markdown("### Varianza Explicada")
            st.write(f"PC1: {pca.explained_variance_ratio_[0]*100:.2f}%")
            st.write(f"PC2: {pca.explained_variance_ratio_[1]*100:.2f}%")

        # ---------------------------------
        # NAVIGATION
        # ---------------------------------
        st.markdown("---")
        if st.button("Ir a ANOVA ➡️", key="to_anova"):
            st.session_state.page = "ANOVA"
            st.session_state.pca_ready = False
            st.rerun()

def anova_page():
    st.header("Análisis de Varianza (ANOVA)")
    
    st.markdown("""
    ### ¿Qué es el ANOVA?
        
    El **Análisis de Varianza (ANOVA)** es un método estadístico utilizado para **comparar las medias** de tres o más grupos independientes para determinar si existe una diferencia estadísticamente significativa entre ellos. 

    [Image of ANOVA concept comparing means of three normal distributions]

        
    * **Propósito Principal:** Determinar si la **variación entre los grupos** (inter-grupos) es significativamente mayor que la **variación dentro de los grupos** (intra-grupos).
    * **Hipótesis Nula ($H_0$):** Las medias de todos los grupos son iguales.
    * **$p$-valor:** Si el $p$-valor es **bajo** (típicamente $< 0.05$), rechazamos $H_0$ y concluimos que al menos una media de grupo es diferente.
        
    **Requisitos:** ANOVA requiere una **variable dependiente numérica** y una **variable categórica** que defina los grupos.
    """)
        
    st.markdown("---")
    
    # 2. Palettes (Defined locally for portability, mirroring the structure used in pca_page)
    COLOR_PALETTES = {
        "QuimioAnalytics (Custom)": ["#B0A461", "#4A525A", "#E0D7B2", "#2E3339", "#8E9E9A"],
        "Viridis (Default)": "viridis",
        "Plasma": "plasma",
        "Cool Warm": ["#0000FF", "#87CEEB", "#FFFFFF", "#FF6347", "#FF0000"],
        "Greyscale": ["#000000", "#555555", "#AAAAAA", "#CCCCCC", "#FFFFFF"],
    }

    # ---------------------------------
    # CHECK DATA
    # ---------------------------------
    if st.session_state.df is None:
        st.warning("Carga un dataset primero.")
        return

    df_anova = st.session_state.df.copy()

    # 3. Color Selector (Using the stored choice from PCA/Home page)
    # Ensure a default value is set if 'plot_color_choice' hasn't been initialized
    if 'plot_color_choice' not in st.session_state:
         st.session_state.plot_color_choice = 'QuimioAnalytics (Custom)'
         
    current_choice = st.session_state.plot_color_choice
    
    st.session_state.plot_color_choice = st.selectbox(
        "Paleta de color para los gráficos:",
        list(COLOR_PALETTES.keys()),
        # Fix: Ensure the index is safe if current_choice is valid
        index=list(COLOR_PALETTES.keys()).index(current_choice)
    )
    
    # --- 4. User Input for ANOVA Variables ---
    st.subheader("Selección de Variables")
    
    # Identify numerical and categorical columns
    numerical_cols = df_anova.select_dtypes(include=[np.number]).columns.tolist()
    # Check for both object and category Dtype, which often represent categorical data
    categorical_cols = df_anova.select_dtypes(include=['object', 'category']).columns.tolist()

    if not numerical_cols or not categorical_cols:
        st.error("Se necesita al menos una columna numérica (dependiente) y una categórica (grupos) para ANOVA.")
        return
        
    col_y, col_x = st.columns(2)
    
    with col_y:
        y_var = st.selectbox("Variable Dependiente (Numérica):", numerical_cols, key="anova_y")
    
    with col_x:
        x_var = st.selectbox("Variable Factor/Grupo (Categórica):", categorical_cols, key="anova_x")

    st.markdown("---")

    # --- 5. ANOVA Calculation Button ---
    if st.button("Aplicar ANOVA", type="primary"):
        
        # 5a. Clean data for the selected variables
        df_model = df_anova[[y_var, x_var]].dropna()
        
        # Ensure factor column is treated as category/object
        df_model[x_var] = df_model[x_var].astype('category')
        
        if df_model.shape[0] < 3:
            st.error("No hay suficientes datos limpios para realizar el ANOVA.")
            return
            
        # 5b. Define the OLS model formula
        # C() ensures the variable is treated as a categorical factor
        formula = f'{y_var} ~ C({x_var})' 
        
        # 5c. Fit the model and perform ANOVA
        try:
            model = ols(formula, data=df_model).fit()
            # Use Type 2 ANOVA for simplicity and robustness in single-factor design
            anova_table = sm.stats.anova_lm(model, typ=2) 
        except Exception as e:
            st.error(f"Error al calcular ANOVA. Asegúrese de que el factor '{x_var}' tenga al menos dos niveles (grupos).")
            return

        st.success("ANOVA completado.")

        # --- 6. Display ANOVA Table ---
        st.subheader("Tabla de Resultados ANOVA")
        st.dataframe(anova_table)
        
        # Extract the p-value for the Factor variable
        p_value_key = f'C({x_var})'
        if p_value_key in anova_table.index:
            p_value = anova_table.loc[p_value_key, 'PR(>F)']
        else:
            p_value = 1.0 # Default safe value if indexing fails

        st.markdown(f"#### Conclusión (Nivel $\\alpha=0.05$):")
        if p_value < 0.05:
            st.success(f"El $p$-valor ({p_value:.4f}) es **menor que 0.05**. Existe evidencia significativa para rechazar la hipótesis nula, lo que indica que **hay una diferencia significativa** en la media de '{y_var}' entre los niveles de '{x_var}'.")
        else:
            st.warning(f"El $p$-valor ({p_value:.4f}) es **mayor que 0.05**. No hay suficiente evidencia para rechazar la hipótesis nula, lo que sugiere que **no hay diferencia significativa** en la media de '{y_var}' entre los grupos.")

        # --- 7. Visualization (Box Plot) ---
        st.subheader("Visualización de Grupos")
        
        palette = COLOR_PALETTES[st.session_state.plot_color_choice]
        
        # Use a box plot to visually compare the distribution of Y across groups X
        fig_box = px.box(
            df_model,
            x=x_var,
            y=y_var,
            title=f"Distribución de '{y_var}' por '{x_var}'",
            color=x_var,
            color_discrete_sequence=palette if isinstance(palette, list) else None,
            points="all" # Show all data points
        )
        
        fig_box.update_layout(showlegend=False)
        
        st.plotly_chart(fig_box, use_container_width=True)

# -------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------
with st.sidebar:
    st.title("Toolbar")

    if st.button("Home"):
        st.session_state.page = "Home"

    if st.button("Preprocesamiento de Datos"):
        st.session_state.page = "Preprocesamiento de Datos"

    if st.button("PCA"):
        st.session_state.page = "PCA"

    if st.button("ANOVA"):
        st.session_state.page = "ANOVA"

    if st.button("AI Chat"):
        st.session_state.page = "AI Chat"


# -------------------------------------------------
# PAGE CONTROLLER
# -------------------------------------------------
if st.session_state.page == "Home":
    home_page()

elif st.session_state.page == "Preprocesamiento de Datos":
    preprocessing_page()

elif st.session_state.page == "PCA":
    pca_page()

elif st.session_state.page == "ANOVA":
    anova_page()

elif st.session_state.page == "AI Chat":
    st.header("Chat de I.A.")
