"""
DEMO 1: Building a Multi-SDK LLM Interface (OpenAI Focus)
Module 1: LLM Foundations & Prompt Engineering
---------------------------------------------------------
Goal: Create a simple, functional ChatBot using the OpenAI SDK and Streamlit.
Learners will see how to:
1. Initialize the OpenAI Client.
2. Manage session-based chat history.
3. Handle streaming responses for a better UI experience.
"""

import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from a local .env file (if present)
load_dotenv()
# Read OPENAI_API_KEY from environment as default for the sidebar
env_openai_api_key = os.getenv("OPENAI_API_KEY", "")

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Agentic AI: OpenAI Demo", layout="centered")
st.title("🤖 Simple OpenAI Chatbot")
st.caption("A Module 1 Demo for 'Agentic AI Architecture Foundations'")

# --- SIDEBAR: API KEY MANAGEMENT ---
# In a real app, you might use a .env file or environment variables.
# For this demo, we provide a sidebar input for the API Key.
with st.sidebar:
    st.header("Configuration")
    openai_api_key = st.text_input("Enter OpenAI API Key", value=env_openai_api_key, type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

# --- INITIALIZE CHAT HISTORY ---
# Streamlit's session_state allows us to keep the conversation across reruns.
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! I am your OpenAI-powered assistant. How can I help you today?"}
    ]

# Display existing chat messages from history on app rerun
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- CHAT LOGIC ---
# Handle user input
if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    # 1. Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 2. Initialize the OpenAI client with the provided key
    client = OpenAI(api_key=openai_api_key)

    # 3. Generate response from OpenAI
    with st.chat_message("assistant"):
        # We use a placeholder to update the text as it streams in
        response_placeholder = st.empty()
        full_response = ""
        
        # Calling the Chat Completion API
        # Model 'gpt-4o' is used for high-quality reasoning (Module 1 Concepts)
        try:
            stream = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                stream=True,
                temperature=0.7,      # Adjust randomness
                max_tokens=500,       # Limit length
                presence_penalty=0.6  # Encourage variety
            )
            
            # Iterate through the stream of chunks
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    response_placeholder.markdown(full_response + "▌")
            
            # Finalize the response display
            response_placeholder.markdown(full_response)
            
            # 4. Save the assistant's response to history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            print(st.session_state.messages)  # For debugging/logging purposes
            
        except Exception as e:
            st.error(f"An error occurred: {e}")


#command to run: streamlit run p6_openai_chatbot_demo.py

