import streamlit as st

##Pages
home = st.Page("pages/home.py", title="Home")

pg = st.navigation([home])
pg.run()