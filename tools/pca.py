import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.tools import Tool
from sklearn.preprocessing import StandardScaler

class PCATool(Tool):
    def __init__(self):
        super().__init__(
            name="PCA",
            description="Análisis de Componentes Principales (manual).",
            options={},
            icon="assets/pca.png"
        )

    def apply(self):
        pass

    def preview(self):
        pass

    def show(self, data: pd.DataFrame):
        st.header("Análisis de Componentes Principales (PCA)")

        # -----------------------------
        # VALIDACIONES
        # -----------------------------
        if data is None or data.empty:
            st.warning("No hay datos para mostrar.")
            return

        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.error("El dataset no contiene columnas numéricas.")
            return

        cols_to_use = st.multiselect(
            "Selecciona variables para el PCA:",
            numeric_cols,
            default=numeric_cols
        )

        if not cols_to_use:
            st.warning("Debes seleccionar al menos una variable.")
            return

        # Limpieza
        df_clean = data[cols_to_use].dropna()
        if len(df_clean) < len(data):
            st.caption(f"Se ignoraron {len(data) - len(df_clean)} filas por valores nulos.")

        # ---------------------------------------
        # PREPROCESAMIENTO: Escalado manual
        # ---------------------------------------
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_clean)

        # ---------------------------------------
        # CÁLCULO MANUAL DE PCA
        # ---------------------------------------

        # 1. Obtener matriz de covarianza
        cov_matrix = np.cov(X_scaled, rowvar=False)

        # 2. Autovalores y autovectores
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 3. Ordenarlos de mayor a menor
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        # 4. Varianza explicada
        explained_var = eigenvalues / np.sum(eigenvalues)
        explained_var_cum = np.cumsum(explained_var)

        # 5. Proyección a componentes principales
        X_pca = np.dot(X_scaled, eigenvectors)

        # ---------------------------------------
        # PESTAÑAS
        # ---------------------------------------
        tab1, tab2, tab3 = st.tabs(["Varianza", "Biplot 2D", "Cargas"])

        # ---------------- TAB 1 -----------------
        with tab1:
            st.subheader("Gráfica Scree (Autovalores)")
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            ax1.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o')
            ax1.set_xlabel("Componente Principal")
            ax1.set_ylabel("Autovalor")
            ax1.set_title("Scree Plot (Autovalores)")
            st.pyplot(fig1)
            plt.close(fig1)

            st.subheader("Varianza Explicada (%)")
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.bar(range(1, len(explained_var) + 1), explained_var * 100)
            ax2.plot(range(1, len(explained_var) + 1), explained_var_cum * 100, marker="o")
            ax2.set_xlabel("Componente Principal")
            ax2.set_ylabel("% Varianza")
            ax2.set_title("Varianza Explicada y Acumulada (%)")
            st.pyplot(fig2)
            plt.close(fig2)

            # Tabla
            var_table = pd.DataFrame({
                "Autovalor": eigenvalues,
                "% Varianza": explained_var * 100,
                "% Acumulada": explained_var_cum * 100
            })
            st.dataframe(var_table.round(4))

        # ---------------- TAB 2 -----------------
        with tab2:
            st.subheader("Biplot (PC1 vs PC2)")

            if len(eigenvalues) < 2:
                st.warning("Se requieren al menos 2 componentes.")
            else:
                fig3, ax3 = plt.subplots(figsize=(7, 6))
                ax3.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)

                # vectores de cargas
                for i, col in enumerate(cols_to_use):
                    ax3.arrow(0, 0, eigenvectors[i, 0]*3, eigenvectors[i, 1]*3,
                              head_width=0.05, color="red")
                    ax3.text(eigenvectors[i, 0]*3.2,
                             eigenvectors[i, 1]*3.2,
                             col, color="red")

                ax3.set_xlabel(f"PC1 ({explained_var[0]*100:.1f}%)")
                ax3.set_ylabel(f"PC2 ({explained_var[1]*100:.1f}%)")
                ax3.set_title("Biplot (PCA Manual)")
                st.pyplot(fig3)
                plt.close(fig3)

        # ---------------- TAB 3 -----------------
        with tab3:
            st.subheader("Matriz de Cargas (Loadings)")

            loadings = pd.DataFrame(
                eigenvectors,
                columns=[f"PC{i+1}" for i in range(len(eigenvalues))],
                index=cols_to_use
            )

            st.dataframe(loadings.style.background_gradient(cmap="coolwarm"), use_container_width=True)

            st.caption("Los loadings indican qué tanto contribuye cada variable a cada componente.")

        st.success("Análisis PCA completado.")