import streamlit as st
import requests
from utils import load_models, create_new_page

st.set_page_config(page_title="smol. 🦎", page_icon="🦎", layout="wide", initial_sidebar_state="collapsed")

# Initialize session state variables if they don't exist
if "delete_model" not in st.session_state:
    st.session_state["delete_model"] = False
if "model_name_to_delete" not in st.session_state:
    st.session_state["model_name_to_delete"] = ""

# Title and description
st.title("smol. 🦎")
st.sidebar.write("")

# Main container
with st.container():
    st.header("API Configuration")
    
    GROQ_API_KEY= st.text_input("GROQ API Key")  
    FIREWORKS_API_KEY = st.text_input("FIREWORKS API Key")
    MISTRAL_API_KEY = st.text_input("MISTRAL API Key")
    oracles, students = load_models()
    button_submit_key = st.button("Submit API Keys 🔑")
    
    st.header("Configuration")
    name_input = st.text_input("Name")
    theme_input = st.text_input("Theme")
    oracle_input = st.selectbox("Oracle", oracles)
    student_model_input = st.selectbox("Student", students)
    button_train = st.button("Train Model 🚀")

if button_submit_key:
    UPDATE_ENV_ENDPOINT = "http://127.0.0.1:105/update_env"
    payload = {}
    if GROQ_API_KEY:
        payload["GROQ_API_KEY"] = GROQ_API_KEY
    if FIREWORKS_API_KEY:
        payload["FIREWORKS_API_KEY"] = FIREWORKS_API_KEY
    if MISTRAL_API_KEY:
        payload["MISTRAL_API_KEY"] = MISTRAL_API_KEY

    if payload:
        response = requests.post(UPDATE_ENV_ENDPOINT, json=payload)
        st.success(f"API Keys submitted successfully!")

if button_train:
    if theme_input and oracle_input and student_model_input:
        CREATE_MODEL_ENV_ENDPOINT = "http://127.0.0.1:105/create_model"
        payload = {
            "theme": theme_input,
            "oracle": oracle_input,
            "student_model": student_model_input
        }
        response = requests.post(CREATE_MODEL_ENV_ENDPOINT, json=payload)
        create_new_page(name_input, theme_input, oracle_input, student_model_input)
        st.success(f"Page {name_input} created successfully!")
    else:
        st.error("Please fill in all fields before starting the training.")
