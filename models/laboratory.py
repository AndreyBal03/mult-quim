import streamlit as st
from .tools import Tool 
from typing import Optional, Dict
from pandas import DataFrame, read_csv, read_excel

class Laboratory:
    def __init__(self, data: DataFrame) -> None:
        self.tools: Dict[str, Tool] = {} 
        self.current_tool: Optional[Tool] = None

        if "lab_data" not in st.session_state:
            st.session_state["lab_data"] = data
        
        self.current_data = st.session_state["lab_data"]
    
    def add_tool(self, tool: Tool):
        self.tools[tool.name] = tool

    def render(self):
        st.title("Laboratory")

        st.download_button(
            label="Download CSV",
            data=self.download(),
            file_name="data.csv",
            mime="text/csv",
            icon=":material/download:",
        )

        self.handle_uploads()
        
        st.write(self.current_data)

    def change_tool(self, tool_name: str):
        self.current_tool = self.tools.get(tool_name)

    def download(self) -> bytes:
        data_to_download = st.session_state.get("lab_data")
        if data_to_download is None or data_to_download.empty:
            return b"" 
        return data_to_download.to_csv(index=False).encode('utf-8')
    
    def handle_uploads(self) -> None:
        uploaded_file = st.file_uploader("Current Data", 
                                         type = ["csv", "xlsx", "xls"])

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    new_data = read_csv(uploaded_file)
                else:
                    new_data = read_excel(uploaded_file)
                
                st.session_state["lab_data"] = new_data
                self.current_data = new_data 
            
            except Exception as e:
                st.error(f"Error reading file: {e}")
        