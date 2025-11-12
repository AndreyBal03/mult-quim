import streamlit as st
from .tools import Tool 
from typing import Optional, Dict
from pandas import DataFrame, read_csv, read_excel

class Laboratory:
    def __init__(self, data: DataFrame) -> None:
        self.tools: Dict[str, Tool] = {} 

        if "lab_data" not in st.session_state:
            st.session_state["lab_data"] = data

        if "current_tool_name" not in st.session_state:
            st.session_state["current_tool_name"] = None
        
    
    @property
    def current_tool(self) -> Optional[Tool]:
        tool_name = st.session_state.get("current_tool_name")
        if tool_name:
            return self.tools.get(tool_name)
        return None

    def add_tool(self, tool: Tool):
        self.tools[tool.name] = tool

    def render(self):
        with st.sidebar:
            st.subheader("Tools")
            
            if not self.tools:
                st.caption("No Tools implemented.")
            
            for tool_name in self.tools:
                
                button_label = tool_name
                button_icon = None 

                """ No funciona el icono xd
                if tool.icon:
                    if tool.icon.endswith(".png") or tool.icon.endswith(".jpg"):
                        try:
                            st.image(tool.icon, width=40) 
                        except Exception as e:
                            st.caption(f" No se encontró {tool.icon}, error: {e}")
                            button_icon = "️x"
                    else:
                        # Es un emoji, úsalo *dentro* del botón
                        button_icon = tool.icon
                """
                
                is_active = (st.session_state.get("current_tool_name") == tool_name)
                
                if st.button(
                    button_label, 
                    icon=button_icon, 
                    key=f"tool_btn_{tool_name}", 
                    use_container_width=True,
                    type="primary" if is_active else "secondary" 
                ):
                    self.change_tool(tool_name)
                    st.rerun()

        st.title("Laboratory")
        
        col1, col2 = st.columns(2)
        with col1:
            self.handle_uploads() 
        with col2:
            _spacer, btn_col = st.columns([0.6, 0.4])
            with btn_col:
                self.handle_download()
        
        st.header(f"Tool: {self.current_tool.name}")
        st.divider() 

        col3, col4 = st.columns(2)
        with col3:
            st.header("Current Data (Sample)")
            st.dataframe(st.session_state.get("lab_data", DataFrame({"Info": ["No data"]})).head(), use_container_width=True)

        with col4:

            if self.current_tool:
                self.current_tool.render_tool(st.session_state["lab_data"])
        
        


    def change_tool(self, tool_name: str):
        st.session_state["current_tool_name"] = tool_name
    
    def download(self) -> bytes:
        data_to_download = st.session_state.get("lab_data")
        if data_to_download is None or data_to_download.empty:
            return b"" 
        return data_to_download.to_csv(index=False).encode('utf-8')
    
    def handle_uploads(self) -> None:
        uploaded_file = st.file_uploader("Upload a new DataFrame", 
                                         type=["csv", "xlsx", "xls"],
                                         label_visibility="collapsed")

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    new_data = read_csv(uploaded_file)
                else:
                    new_data = read_excel(uploaded_file)
                st.session_state["lab_data"] = new_data
                st.success("Data has been updated")
            except Exception as e:
                st.error(f"Error while reading the file: {e}")

    def handle_download(self) -> None:
        st.download_button(
            label="Download CSV",
            data=self.download(),
            file_name="data.csv",
            mime="text/csv",
            icon=":material/download:",
        )