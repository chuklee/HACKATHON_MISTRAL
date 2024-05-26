import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langserve import RemoteRunnable

st.set_page_config(page_title="Model Details", page_icon="🤖")
name_input = "Mistral Small"
theme_input = "Python Coding Interview Exercises on Data Structures and Algorithms"
oracle_input = "groq_llama3-70b-8192"
student_model_input = "Mistral-7B-v0.1"
st.title("Model Details")
st.write(f"**Name**: {name_input}")
st.write(f"**Theme**: {theme_input}")
st.write(f"**Oracle**: {oracle_input}")
st.write(f"**Student**: {student_model_input}")


model = RemoteRunnable("https://87be-195-242-24-207.ngrok-free.app/smol/")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages_smol = []

# Display chat messages from history on app rerun
for message in st.session_state.messages_smol:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages_smol.append({"role": "user", "content": prompt})
    
if prompt:
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", prompt)
        ]
    ).format_messages()
    response = model.invoke(chat_prompt).content
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages_smol.append({"role": "assistant", "content": response})