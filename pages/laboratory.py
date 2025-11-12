import streamlit as st
from models.laboratory import Laboratory
import pandas as pd



data = st.session_state.get("data", None) 
lab = Laboratory(data=pd.DataFrame(data))  

lab.render()