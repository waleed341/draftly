
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_ollama import ChatOllama


# Load environment variables
load_dotenv()

model_name = os.getenv("MODEL_NAME")

if not model_name:
    raise ValueError("MODEL_NAME is missing in your .env file")


# Initialize Ollama model
llm = ChatOllama(
    model=model_name,
    base_url=os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434"
    ),
    temperature=0.7,
    reasoning=False
)


# Create Draftly AI email agent
agent = create_agent(
    model=llm,
    tools=[],
    system_prompt=(
        "You are Draftly, a professional AI email writing assistant. "
        "Your job is to WRITE complete emails based on the user's idea. "
        "The user provides an email idea, NOT an existing email draft. "
        "Never ask the user to provide an email draft. "
        "Never explain what you are going to write. "
        "Immediately generate the complete email. "
        "Always include Subject, Greeting, Body, and Closing. "
        "If information is missing, use reasonable placeholders. "
        "Return ONLY the finished email."
    )
)


# Get email idea from user
query = input("Enter email draft idea: ").strip()

if not query:
    raise ValueError("Email draft idea cannot be empty")


# Prepare messages
messages = [
    {
        "role": "user",
        "content": (
            f"Write a complete professional email based on this idea:\n"
            f"{query}\n\n"
            "Generate the email now. Do not ask questions."
        )
    }
]


# Invoke agent
result = agent.invoke({"messages": messages})


# Display generated email
print("\n--- Generated Email Draft ---\n")
print(result["messages"][-1].text)
