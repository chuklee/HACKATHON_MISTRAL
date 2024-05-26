import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langserve import RemoteRunnable

st.set_page_config(page_title="Model Details", page_icon="🤖")
name_input = "Mistral Small"
theme_input = ""
oracle_input = ""
student_model_input = ""
st.title("Model Details")
st.write(f"**Name**: {name_input}")
model = RemoteRunnable("http://localhost:8000/mistral_small/")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
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
    st.session_state.messages.append({"role": "assistant", "content": response})