"""
DEMO 1: Comedy Chatbot with OpenAI SDK + Streamlit
API key is loaded server-side only — never rendered in the UI or sent to the browser.
"""

import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# --- API KEY: loaded silently from .env / Streamlit secrets, never shown in UI ---
api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Comedy Bot Demo", layout="centered")
st.title("😄 Comedy Chatbot")
st.caption("Ask me anything — clean comedy guaranteed.")

# --- Playful sidebar, no key input at all ---
with st.sidebar:
    st.header("About")
    st.write("This bot runs on a securely configured API key. No peeking! 🕵️")
    if st.button("🔍 Try to find the API key"):
        st.warning("Nice try. The key's not in this app, it's not in your browser, "
                   "it's not even in this galaxy. It lives quietly in a `.env` file, "
                   "sipping tea, minding its own business. 🍵🔐")

if not api_key:
    st.error("No API key configured on the server. Contact the app owner.")
    st.stop()

SYSTEM_PROMPT = """
You are a witty, family-friendly comedian chatbot.
Rules:
- Answer every message with humor — puns, wordplay, playful exaggeration, light sarcasm.
- Keep it clean: no vulgarity, no profanity, no offensive stereotypes.
- Still be genuinely helpful for real questions, just deliver it with comedic flair.
- Keep responses concise unless asked for a longer bit.
"""

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hey there! Ask me anything — I'll answer it with a smile."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    client = OpenAI(api_key=api_key)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            api_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
                {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
            ]

            stream = client.chat.completions.create(
                model="gpt-4o",
                messages=api_messages,
                stream=True,
                temperature=0.9,
                max_tokens=400,
                presence_penalty=0.6
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"An error occurred: {e}")