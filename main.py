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

# -------------------------
# Placeholder model function
# Replace this with your actual statistical model logic
# -------------------------
def run_statistical_model(df):
    # Example: simple summary (replace with real model)
    summary = df.describe(include="all").to_string()
    return f"Model summary:\n{summary}"

# -------------------------
# LLM function (Groq)
# -------------------------
def get_ai_recommendations(model_output, user_message):
    chat_prompt = f"""
    You are an AI assistant analyzing a user's statistical model output.
    Model results:
    {model_output}

    User question:
    {user_message}

    Provide:
    - Clear interpretation of the model output
    - Actionable recommendations
    - Alternative strategies or improvements
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",   
        messages=[
            {"role": "system", "content": "You are an expert data analyst."},
            {"role": "user", "content": chat_prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="AI Model Advisor", page_icon="📊")

st.title("📊 AI Statistical Model Advisor")
st.write("Upload a dataset, run your model, and chat with the AI about the results.")

# -------------------------
# File Upload
# -------------------------
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    # Load file into pandas
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("File uploaded successfully!")
    st.write("### Preview of your data")
    st.dataframe(df.head())

    # Run model button
    if st.button("Run Statistical Model"):
        with st.spinner("Running model..."):
            model_output = run_statistical_model(df)
        st.success("Model run completed!")
        
        # Store output in session state
        st.session_state["model_output"] = model_output
        st.write("### Model Output")
        st.text(model_output)

# -------------------------
# Chat Section
# -------------------------
st.write("---")
st.header("💬 Ask the AI About Your Model")

# Ensure model output is available
if "model_output" not in st.session_state:
    st.info("Please upload a file and run the model first.")
else:
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
