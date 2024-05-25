import json

MODEL_PATH = "../models.json"

def read_json(path: str) -> dict:
    with open(path, 'r') as f:
        data = json.load(f)
        return data
    
def load_models():
    models = read_json(MODEL_PATH)
    oracles = models["oracle"]
    students = models["student"]
    return oracles, students



def create_new_page(name_input, theme_input, oracle_input, student_model_input):
    content = f"""import streamlit as st
import os
import time
        
st.set_page_config(page_title="Model Details", page_icon="🤖")

st.title("Model Details")
st.write(f"**Name**: {name_input}")
st.write(f"**Theme**: {theme_input}")
st.write(f"**Oracle**: {oracle_input}")
st.write(f"**Student**: {student_model_input}")
st.text_area("")
submit = st.button('Generate')  

if submit:
    with st.spinner(text="This may take a moment..."):
        time.sleep(2)
        text2 = "response_message"
        st.write(text2)


if st.button(f"Delete Model {name_input}"):
    os.remove(f"pages/{name_input}.py")
    st.session_state.runpage= "Home"
    st.rerun()
"""
    filename = f"pages/{name_input}.py"
    with open(filename, "w") as file:
        file.write(content)