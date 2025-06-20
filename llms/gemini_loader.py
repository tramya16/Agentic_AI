from crewai import LLM
from dotenv import load_dotenv

import os

load_dotenv()

def load_gemini_llm():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")  # Consider using .env file

    return LLM(
        model="gemini/gemini-1.5-pro",
        api_key=os.environ["GEMINI_API_KEY"]
    )