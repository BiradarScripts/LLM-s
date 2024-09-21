import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI
from langserve import add_routes

# Load environment variables
load_dotenv()

# Initialize the Google Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Initialize message history
chat_history = ChatMessageHistory()

# The template for generating the response
generic_template = '''You are an expert software developer. Provide initial setup for frontend application Initial Setup. Don't include file structure and remember at last Ensure the following:
give Initial Setup
1. **Terminal commands** should start with "terminal:" followed by the command.
2. only setup, don't include code and all, only setup

6)example output:

terminal: mkdir my-frontend-app
terminal: cd my-frontend-app
terminal: npm init -y

terminal: npm install react react-dom
terminal: mkdir src
terminal: touch src/index.js
terminal: touch src/App.js
'''

# Create a structured prompt using ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", generic_template),
        ("user", "{text}")
    ]
)

parser = StrOutputParser()

# Initialize the Langchain pipeline (chain)
chain = prompt | llm | parser

# Create FastAPI app
app = FastAPI(title="Langchain Server",
              version="1.0",
              description="A simple API server using Langchain runnable interfaces")

def main():
    # Console-based input loop
    while True:
        # Input box for user question
        input_text = input("What question do you have in mind? (Type 'exit' to quit): ")

        if input_text.lower() == 'exit':
            break

        if input_text:
            try:
                # Add the user message to the history
                chat_history.add_user_message(input_text)
                
                # Prepare the messages for the chain using LangChain message classes
                messages = []
                
                # Convert history into LangChain's message types
                for message in chat_history.messages:
                    if message.type == "human":
                        messages.append(HumanMessage(content=message.content))
                    else:
                        messages.append(AIMessage(content=message.content))

                # Execute the LLM with the generated messages
                result = chain.invoke({"text": input_text})

                # Add the assistant's response to the chat history
                chat_history.add_ai_message(result)  # Access the content directly

                # Display the result
                print(f"AI: {result}\n")

            except ConnectionError:
                print("Failed to connect to the service. Please ensure the service is running.")
            except Exception as e:
                print(f"An error occurred: {e}")

# Add routes to the FastAPI app
add_routes(
    app,
    chain,  # Ensure `chain` is passed here correctly
    path="/chain"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
