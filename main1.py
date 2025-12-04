import streamlit as st
import pandas as pd
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# -------------------------
# Initialize Groq client
# -------------------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(page_title="AI Model Advisor", page_icon="📊", layout="wide")


# -------------------------
# Placeholder model function
# Replace this with your actual statistical model logic
# -------------------------
def run_statistical_model(df):
    # Example: simple summary (replace with real model)
    summary = df.describe(include="all").to_string()
    return f"Model summary:\n{summary}"


# -------------------------
# Initialize session state
# -------------------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

# -------------------------
# Sidebar Navigation
# -------------------------
with st.sidebar:
    st.title("Toolbar")

    if st.button("Home"):
        st.session_state.page = "Home"

    if st.button("Analysis"):
        st.session_state.page = "Analysis"

    if st.button("AI Chat"):
        st.session_state.page = "AI Chat"


# -------------------------
# PAGE: HOME
# -------------------------
def home_page():
    st.title("📊 AI Statistical Model Advisor")
    st.write("Upload a dataset, run your model, and chat with the AI about the results.")

    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file:
        # Load file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.session_state["df"] = df

        st.success("File uploaded successfully!")
        st.write("### Preview of your data")
        st.dataframe(df.head())

        # Run model
        if st.button("Run Statistical Model"):
            with st.spinner("Running model..."):
                model_output = run_statistical_model(df)

            st.session_state["model_output"] = model_output
            st.success("Model run completed!")
            st.write("### Model Output")
            st.text(model_output)


# -------------------------
# PAGE: ANALYSIS
# -------------------------
def analysis_page():
    st.header("📈 Model Analysis")

    # 🔥 CHECK THAT DATASET EXISTS
    if "df" not in st.session_state:
        st.warning("Please upload a dataset on the Home page first.")
        return

    # 🔥 RETRIEVE THE DATASET
    df = st.session_state["df"]

    # --- Slider to control rows ---
    max_rows = len(df)
    rows_to_show = st.slider(
        "Select number of rows to display",
        1, max_rows, min(5, max_rows)
    )

    st.write("### Data Preview")
    st.dataframe(df.head(rows_to_show))

    if "model_output" not in st.session_state:
        st.info("Run the model on the Home page to enable model analysis.")
        return

    model_output = st.session_state["model_output"]

# -------------------------
# PAGE: AI CHAT
# -------------------------
def get_ai_recommendations():
    st.header("💬 Ask the AI About Your Model")

    if "model_output" not in st.session_state:
        st.info("Please upload a file and run the model first.")
        return

    user_input = st.chat_input("Ask something about the model...")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_ai_recommendations(
                    st.session_state["model_output"],
                    user_input
                )
            st.write(response)


# -------------------------
# RENDER THE CURRENT PAGE
# -------------------------
if st.session_state.page == "Home":
    home_page()
elif st.session_state.page == "Analysis":
    analysis_page()
elif st.session_state.page == "AI Chat":
    get_ai_recommendations()
