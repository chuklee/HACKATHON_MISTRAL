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

messages = st.container(height=300)
if prompt := st.chat_input("Say something"):
    messages.chat_message("user").write(prompt)
    messages.chat_message("assistant").write("On attend Camil pour répondre à votre question.")

if st.button(f"Delete Model {name_input}"):
    st.session_state["delete_model"] = True
    st.session_state["model_name_to_delete"] = "{name_input}"
    st.experimental_rerun()

if "delete_model" in st.session_state and st.session_state["delete_model"]:
    if st.session_state["model_name_to_delete"] == "{name_input}":
        model_file_path = f"pages/{name_input}.py"
        if os.path.exists(model_file_path):
            os.remove(model_file_path)
            st.success(f"Model {name_input} deleted successfully!")
        else:
            st.error(f"Model {name_input} does not exist!")
        st.session_state["delete_model"] = False
        st.session_state["model_name_to_delete"] = ""
        st.experimental_set_query_params(page="home")
        st.experimental_rerun()
"""
    filename = f"pages/{name_input}.py"
    with open(filename, "w") as file:
        file.write(content)
