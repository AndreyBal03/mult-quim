from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from pandas import DataFrame
import streamlit as st

class Tool(ABC):
    def __init__(self, name: str, description: str, options: Dict[str, Any], icon: Optional[str] = None):
        self.name = name
        self.description = description
        self.options = options
        self.icon = icon 

    @abstractmethod
    def apply(self):
        pass

    @abstractmethod
    def preview(self):
        pass

    @abstractmethod
    def show(self, data: DataFrame): 
        pass
    
    def render_tool(self, data: DataFrame) -> DataFrame:
        """
        Este método se llama desde el laboratorio.
        Renderiza la UI de la herramienta en el panel principal.
        """
        self.show(data)
        return data