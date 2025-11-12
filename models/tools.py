from typing import Dict, Any
from abc import ABC, abstractmethod

class Tool(ABC):
    def __init__(self, name: str, description: str, options: Dict[str, Any]):
        self.name = name
        self.description = description
        self.options = options 

    @abstractmethod
    def apply(self):
        pass

    @abstractmethod
    def preview(self):
        pass

    @abstractmethod
    def show(self):
        pass
    
   