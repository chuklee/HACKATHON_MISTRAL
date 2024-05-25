import streamlit as st
import pandas as pd
import requests
from utils import Row, load_models
import time  # For simulating progress

st.set_page_config(page_title="smol. 🦎", page_icon="🦎", layout="wide")

st.markdown(
    """
    <style>
    .css-1outpf7, .css-1a32fsj, .css-1cpxqw2, .css-1fv8s86, .css-2trqyj, .css-4gd8zi {
        display: flex;
        justify-content: center;
    }
    .dataframe thead th, .dataframe tbody td {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and description
st.title("smol. 🦎")

# Sidebar inputs
with st.sidebar:
    oracles, students = load_models()
    st.header("Configuration")
    theme_input = st.text_input("Theme")
    oracle_input = st.selectbox("Oracle", oracles)
    student_model_input = st.selectbox("Student", students)
    button_train = st.button("Train Model 🚀")

# Initialize session state to store rows
if "row_list" not in st.session_state:
    st.session_state.row_list = []

# Function to simulate progress
def simulate_training(progress_bar):
    for i in range(101):
        time.sleep(0.005)  # Simulate time delay for training
        progress_bar.progress(i)

# Add new row to the session state
if button_train:
    if theme_input and oracle_input and student_model_input:
        row = Row(theme_input, oracle_input, student_model_input)
        st.session_state.row_list.append(row.to_dict())
        
        # Create a progress bar
        progress_bar = st.progress(0)
        
        # Simulate training progress
        simulate_training(progress_bar)
        
        api_endpoint = "http://127.0.0.1:105/create_model"
        payload = {
            "theme": theme_input,
            "oracle": oracle_input,
            "student_model": student_model_input
        }
        response = requests.post(api_endpoint, json=payload)
        
    else:
        st.error("Please fill in all fields before starting the training.")

# Display the dataframe using st.data_editor
if st.session_state.row_list:
    df = pd.DataFrame(
        st.session_state.row_list, columns=["Theme", "Oracle", "Student", "Link"]
    )
    st.data_editor(
        df,
        column_config={
            "Link": st.column_config.LinkColumn(
                "Hugging Face",
                validate="^https://example.com/.*$",
                display_text="Model",
            )
        },
        hide_index=True,
        use_container_width=True,
        disabled=True,
    )
else:
    st.write("No models trained yet.")
