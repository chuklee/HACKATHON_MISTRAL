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
    if GROQ_API_KEY and FIREWORKS_API_KEY and MISTRAL_API_KEY:
        api_endpoint = "http://"
        payload = {
            "GROQ_API_KEY": GROQ_API_KEY,
            "FIREWORKS_API_KEY": FIREWORKS_API_KEY,
            "MISTRAL_API_KEY": MISTRAL_API_KEY
        }
        response = requests.post(api_endpoint, json=payload)
        st.success(f"API Keys submitted successfully!")

if button_train:
    if theme_input and oracle_input and student_model_input:
        api_endpoint = "http://127.0.0.1:105/create_model"
        payload = {
            "theme": theme_input,
            "oracle": oracle_input,
            "student_model": student_model_input
        }
        response = requests.post(api_endpoint, json=payload)
        create_new_page(name_input, theme_input, oracle_input, student_model_input)
        st.success(f"Page {name_input} created successfully!")
    else:
        st.error("Please fill in all fields before starting the training.")
