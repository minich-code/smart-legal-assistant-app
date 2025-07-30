import streamlit as st
import requests
import os
from dotenv import load_dotenv

# --- Load environment variables from .env.locals ---
# --- Load environment variables from .env.locals ---
load_dotenv(dotenv_path="frontend/.env.locals")
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000/query")

# --- Page Configuration ---
st.set_page_config(
    page_title="LegalAI RAG Demo",
    page_icon="⚖️",
    layout="wide"
)

# --- Backend API URL ---
# Use the environment variable from .env.locals, with a fallback for local testing
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000/query")

# --- UI Components ---
st.title("⚖️ LegalAI Demo")
st.caption("A RAG-powered assistant for navigating Kenyan Corporate Law.")

# Use a sidebar for inputs to keep the main chat interface clean
with st.sidebar:
    st.header("Query Controls")
    namespace = st.text_input(
        "Knowledge Base Namespace:",
        value="companies-act-2015-v1",
        help="Specify the legal document set to search within."
    )
    st.info("This demo is configured to answer questions based on the **Kenya Companies Act, 2015**.")

# --- Chat History Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input and API Call ---
if prompt := st.chat_input("Ask a legal question..."):
    # Add user's message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display AI response in a "thinking" state
    with st.chat_message("assistant"):
        with st.spinner("Analyzing legal documents..."):
            try:
                # Prepare the request payload
                payload = {"query": prompt, "namespace": namespace}

                # Send the request to the backend
                response = requests.post(BACKEND_API_URL, json=payload)
                response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

                # Parse the JSON response
                data = response.json()
                answer = data.get("answer", "Sorry, I could not generate a response.")
                citations = data.get("citations", [])

                # --- Format and Display the Response ---
                st.markdown(answer)

                # Display the citations in an expander for a clean look
                if citations:
                    with st.expander("Show Sources"):
                        for i, citation in enumerate(citations, 1):
                            st.markdown(
                                f"""
                                **{i}. {citation.get('title', 'N/A')}** (Section: {citation.get('section', 'N/A')})
                                - *Source:* [{citation.get('source_url')}]({citation.get('source_url')})
                                """
                            )

                # Add the full response to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer  # Storing just the answer for simplicity in history
                })

            except requests.exceptions.RequestException as e:
                error_message = f"Could not connect to the backend API. Please make sure it's running. Error: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
            except Exception as e:
                error_message = f"An unexpected error occurred: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})