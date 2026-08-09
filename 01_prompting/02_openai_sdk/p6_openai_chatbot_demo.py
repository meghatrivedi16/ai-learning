"""
DEMO 1: Comedy Chatbot with OpenAI SDK + Streamlit
API key is loaded server-side only — never rendered in the UI or sent to the browser.
Built by Megha — a developer learning to build with AI.
"""

import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")

# --- USAGE LIMIT CONFIG ---
MAX_MESSAGES = 3  # total user messages allowed per session (tune this to taste)

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Comedy Bot Demo", layout="centered")
st.title("😄 Comedy Chatbot")
st.caption("Ask me anything — clean comedy guaranteed (while my wallet lasts).")

with st.sidebar:
    st.header("About this bot")
    st.write("This bot runs on a securely configured API key. No peeking! 🕵️")
    if st.button("🔍 Try to find the API key"):
        st.warning("Nice try. The key's not in this app, it's not in your browser, "
                   "it's not even in this galaxy. It lives quietly in a `.env` file, "
                   "sipping tea, minding its own business. 🍵🔐")

    st.divider()

    st.header("About the developer")
    st.markdown(
        """
        **Hi, I'm Megha 👋**

        I'm a developer learning to build with AI — figuring out
        LLMs, prompt engineering, and agentic systems one project
        at a time. This comedy chatbot is part of a hands-on
        practice series exploring the same concept across
        different frameworks (OpenAI SDK, LangChain, AutoGen).

        Currently exploring: prompting → chunking → RAG → agents.
        """
    )
    st.caption("Built with Streamlit + OpenAI SDK")

    st.divider()
    st.caption(f"💬 Messages used this session: {st.session_state.get('msg_count', 0)}/{MAX_MESSAGES}")

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

if "msg_count" not in st.session_state:
    st.session_state["msg_count"] = 0

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- LIMIT REACHED: show a funny paywall-ish message instead of the chat input ---
if st.session_state.msg_count >= MAX_MESSAGES:
    st.chat_message("assistant").write(
        "Whoa there, comedy fan! 🎤 I've cracked my quota of jokes for this session — "
        "turns out being funny costs real money (who knew?). I'm a broke developer's "
        "side project, not a Silicon Valley unicorn. 🦄💸\n\n"
        "If you enjoyed this, share it with a friend, star the repo, or just imagine "
        "10 more jokes — they were probably pretty good. 😄"
    )
    st.info("Session limit reached. Refresh the page to start a new session (resets the counter).")
    st.stop()

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.msg_count += 1
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
                model="gpt-4o-mini",   # cheaper model — see note below
                messages=api_messages,
                stream=True,
                temperature=0.9,
                max_tokens=250,        # capped lower to control cost per response
                presence_penalty=0.6
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

            # Warn on the second-to-last message, building the joke before the cutoff
            remaining = MAX_MESSAGES - st.session_state.msg_count
            if remaining == 1:
                st.info("😅 Psst — one message left before I go full 'starving artist' on you.")

        except Exception as e:
            st.error(f"An error occurred: {e}")

# command to run: streamlit run p6_comedy_chatbot_demo.py