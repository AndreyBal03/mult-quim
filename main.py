import streamlit as st

##Pages
home = st.Page("pages/home.py", title="Home")
laboratory = st.Page("pages/laboratory.py", title="Laboratory")

pg = st.navigation([home, laboratory])
pg.run()