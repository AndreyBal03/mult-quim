import streamlit as st
from models.tools import Tool


class Overview(Tool):
    def __init__(self):
        super().__init__(
            name="Overview",
            description="easy too for review data",
            options= {})
        ...