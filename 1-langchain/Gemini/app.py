import getpass
from dotenv import load_dotenv

import streamlit as st

load_dotenv()

import os
print("Google Application Credentials:", os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

st.title("Langchain Demo With Gemini-1.5 pro Model")

input_text = st.text_input("What question do you have in mind?")

if input_text:
    try:

        messages = [
            {"role": "system", "content": "You are a helpful assistant that gives the steps to be followed in a list format (high level not including planning and all) to build a website."},
            {"role": "user", "content": input_text}
        ]


        result = llm.invoke(messages)

        st.write(result)

    except ConnectionError:
        st.error("Failed to connect to the service. Please ensure the service is running.")
    except Exception as e:
        st.error(f"An error occurred: {e}")