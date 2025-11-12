import streamlit as st


" # Welcome :) "

if not "data" in st.session_state:
    st.session_state["data"] = st.file_uploader("Add dataFrame", 
                        type = ["csv", "xlsx", "xls"])
    if st.button("Ir al Laboratorio"):
        st.switch_page("pages/laboratory.py")

else:
    if st.button("Continuar Laboratorio"):
        st.switch_page("pages/laboratory.py")

