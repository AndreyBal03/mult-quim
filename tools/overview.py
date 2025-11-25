from models.tools import Tool
from pandas import DataFrame
import streamlit as st

class Overview(Tool):
    def __init__(self):
        super().__init__(
            name="Overview",
            description="Muestra un resumen del DataFrame.",
            options={},
            icon="assets/overviewicon.png" 
        )
    
    def apply(self):
        pass

    def preview(self):
        pass

    def GuiaUso(self):
        st.subheader("GUÍA DE USO")
        st.markdown(r"""
# Quimiometría - Guía de Uso

## Importancia del análisis multivariante
1. Permitir la carga y limpieza básica de datos  
2. Estandarización de variables  
3. Análisis de varianza (ANOVA)  
4. Análisis de Componentes Principales (PCA)  
5. Análisis de clusters (conglomerados)

---

# ¿Por qué estandarizar?

## Para el ANOVA…
La estandarización de variables generalmente no es un requisito estricto para realizar un ANOVA simple, ya que el procedimiento se centra en comparar las medias de grupos en su escala original.  
Sin embargo, es útil porque:

**Comparabilidad entre variables:**  
Permite comparar efectos cuando las características están en escalas distintas.

**Supuestos del modelo:**  
Puede estabilizar varianzas.

**Interpretación:**  
Los coeficientes estandarizados son más fáciles de comparar.

**Métodos avanzados:**  
PCA y otras técnicas requieren estandarizar.

---

## Para el PCA…
Si no se estandariza, una variable con varianza muy grande dominará el primer componente.  
Al estandarizar:

- Todas las variables tienen media 0 y varianza 1  
- Ninguna “pesa más” por su escala  

Los componentes principales son autovectores de la matriz de correlación y sus autovalores indican cuánta varianza explican.

---

## Para los Clústers…
Los métodos basados en distancia (k-means, jerárquico…) son **muy sensibles a la escala**, por lo que es indispensable estandarizar.

Una variable grande distorsiona la distancia euclidiana:

\[
d = \sqrt{\sum (x_i - y_i)^2}
\]

---

# ANOVA

## ¿Qué es?
Método estadístico que divide la variación observada en:

- Variación entre grupos  
- Variación dentro de los grupos  

Sirve para determinar si las diferencias químicas entre grupos son reales o aleatorias.

---

## Importancia del ANOVA
Permite:

- Identificar variables que diferencian grupos  
- Validar diferencias antes del PCA  
- Detectar señales responsables de variación  
- Interpretar cambios químicos desde un punto univariado  

---

# Análisis de Componentes Principales (PCA)

## ¿Qué es?
Método para reducir dimensionalidad cuando hay correlación entre variables.

\[
Z_1 = a_{11}X_1 + a_{12}X_2 + \dots + a_{1n}X_n
\]

Captura la máxima varianza posible en pocas dimensiones.

---

## Importancia del PCA en Quimiometría
Permite ver:

- Agrupamientos  
- Outliers  
- Variables influyentes (loadings)  
- Patrones químicos globales  

---

# Análisis de clusters (conglomerados)

## ¿Qué es?
Técnica para agrupar objetos según similitud.  
Distancia euclidiana:

\[
d = \sqrt{\sum (x_i - y_i)^2}
\]

Distancia Manhattan:

\[
D = |x_1 - y_1| + |x_2 - y_2|
\]

---

## Importancia del clustering
Permite:

- Distinguir feedstocks  
- Identificar grupos homogéneos  
- Detectar adulteraciones  
- Complementar PCA  
- Detectar valores atípicos  

""")

    def show(self, data: DataFrame | None):
        # Mostrar Guía de uso
        self.GuiaUso()

        st.subheader("Data Overview")
        
        if data is None or data.empty:
            st.warning("No data to show.")
            return

        st.write("Column Types & Non-Null Counts:")
        from io import StringIO
        buffer = StringIO()
        data.info(buf=buffer)
        st.text(buffer.getvalue())

        st.write("Descriptive Statistics:")
        st.dataframe(data.describe(include='all'))
        
        st.write("Null Value Counts:")
        st.dataframe(data.isnull().sum())
