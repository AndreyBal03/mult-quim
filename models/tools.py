from typing import Dict, Any
from abc import ABC, abstractmethod
from pandas import DataFrame
import streamlit as st

class Tool(ABC):
    def __init__(self, name: str, description: str, options: Dict[str, Any]):
        self.name = name
        self.description = description
        self.options = options 
        self.icon = ""

    @abstractmethod
    def apply(self):
        pass

    @abstractmethod
    def preview(self):
        pass

    @abstractmethod
    def show(self):
        pass
    
    def render_tool(self, data: DataFrame) -> DataFrame:
        st.write(f"Ejecutando la herramienta: **{self.name}**")
        st.info(f"Icono: {self.icon}")
        return data