import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.tools import Tool
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

class ClusteringTool(Tool):
    def __init__(self):
        super().__init__(
            name="Clustering",
            description="Analisis de clustering con K-means y cluster jerarquico.",
            options={},
            icon="assets/clustering.png"
        )
    def apply(self):
        pass

    def preview(self):
        pass
    
    def show(self, data: pd.DataFrame):
        st.header("Analisis de Clustering")

      
        if data is None or data.empty:
            st.warning("No data to show.")
            return
        #preprocesamiento
        #mostrar columnas numericas
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.error("El dataset no contiene columnas numericas.")
            return

        cols_to_use = st.multiselect("Selecciona variables para el clustering:", numeric_cols, default=numeric_cols)
        
        if not cols_to_use:
            st.warning("Debes seleccionar al menos una variable.")
            return

        #limpieza
        df_clean = data[cols_to_use].dropna()
        if len(df_clean) < len(data):
            st.caption(f"Se ignoraron {len(data) - len(df_clean)} filas con valores nulos.")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_clean)

        #pestanas para los metodos
        tab1, tab2 = st.tabs(["K-Means", "Jerarquico"])
        
        labels = None
        
        # --- K-MEANS ---
        with tab1:
            st.subheader("K-Means")
            col_k, col_init = st.columns(2)
            with col_k:
                k = st.slider("Numero de clusters (k)", 2, 15, 3, key="kmeans_k")
            with col_init:
                n_init = st.number_input("Repeticiones (n_init)", min_value=1, max_value=50, value=10, help="Veces que se ejecuta el algoritmo con distintas semillas.")

            if st.button("Ejecutar K-Means"):
                kmeans = KMeans(n_clusters=k, init='k-means++', n_init=n_init, random_state=42)
                labels = kmeans.fit_predict(X_scaled)
                
                #metricas
                sil_score = silhouette_score(X_scaled, labels)
                st.metric("Silhouette Score", f"{sil_score:.3f}", help="Indica que tan bien separados estan los clusters (max 1).")

        # --- Jerarquico ---
        with tab2:
            st.subheader("Jerarquico")
            col_link, col_metric = st.columns(2)
            with col_link:
                linkage_method = st.selectbox("Metodo de enlace (Linkage)", ["ward", "complete", "average", "single"])
            with col_metric:
                if linkage_method == "ward":
                    metric = "euclidean"
                    st.text_input("Metrica de distancia", value="euclidean", disabled=True, help="'ward' solo soporta distancia euclidiana.")
                else:
                    metric = st.selectbox("Metrica de distancia", ["euclidean", "cityblock", "cosine"])
            
            st.subheader("Dendrograma")
            try:
                #calculo matriz de enlace
                Z = linkage(X_scaled, method=linkage_method, metric=metric)
                
                fig_dendro, ax = plt.subplots(figsize=(10, 5))
                dendrogram(Z, ax=ax, truncate_mode='lastp', p=30, show_leaf_counts=True)
                plt.title("Dendrograma (truncado a los ultimos 30 nodos)")
                plt.xlabel("Índice de muestra o (tamano del cluster)")
                plt.ylabel("Distancia")
                st.pyplot(fig_dendro)
                plt.close(fig_dendro) #cerrar para liberar memoria
            except Exception as e:
                st.error(f"Error calculando dendrograma: {e}")

            st.divider()
            n_clusters_h = st.slider("Numero de clusters para el corte", 2, 15, 3, key="hierarchical_k")
            
            if st.button("Aplicar corte Jerarquico"):
                hc = AgglomerativeClustering(n_clusters=n_clusters_h, metric=metric, linkage=linkage_method)
                labels = hc.fit_predict(X_scaled)

        # --- RESULTADOS Y PERFILES ---
        if labels is not None:
            st.divider()
            st.header("Resultados del Clustering")
            
            #crear dataframe de resultados
            df_results = df_clean.copy()
            df_results['Cluster'] = labels
            
            #1.conteo de muestras
            st.subheader("Distribución de muestras")
            count_data = df_results['Cluster'].value_counts().sort_index()
            st.bar_chart(count_data)

            #2.centroides (Promedios por cluster)
            st.subheader("Perfil de los Clusters (Promedios)")
            st.info("Esta tabla muestra el valor promedio de cada variable para cada grupo detectado.")
            
            summary = df_results.groupby('Cluster').mean()
            #usamos un mapa de color para destacar valores altos/bajos
            st.dataframe(summary.style.background_gradient(cmap="Blues"), use_container_width=True)
#TODO: proyeccion de los clusters sobre el espacio de las PC (de acuerdo al punto 5 de Estructura de la aplicacion)
#esto se hizo dado que todavia no se hace el PCA lol