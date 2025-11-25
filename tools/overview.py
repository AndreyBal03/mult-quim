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
        st.markdown(r"""""")

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
