import streamlit as st
from models.laboratory import Laboratory
import pandas as pd
from tools.overview import Overview
from tools.clustering import ClusteringTool

st.set_page_config(layout="wide")
data = st.session_state.get("data", None) 
lab = Laboratory(data=pd.DataFrame(data))  

overview_tool = Overview()
clustering_tool = ClusteringTool()

lab.add_tool(overview_tool)
lab.add_tool(clustering_tool)

lab.render()