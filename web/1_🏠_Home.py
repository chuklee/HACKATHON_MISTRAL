import streamlit as st
import requests
from utils import load_models, create_new_page

st.set_page_config(page_title="smol. 🦎", page_icon="🦎", layout="wide" , initial_sidebar_state="collapsed")

# Title and description
st.title("smol. 🦎")
st.sidebar.write("")

# Main container
with st.container():
    oracles, students = load_models()
    st.header("Configuration")
    name_input = st.text_input("Name")
    theme_input = st.text_input("Theme")
    oracle_input = st.selectbox("Oracle", oracles)
    student_model_input = st.selectbox("Student", students)
    button_train = st.button("Train Model 🚀")


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
        
    else:
        st.error("Please fill in all fields before starting the training.")

