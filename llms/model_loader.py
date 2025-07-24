# llms/model_loader.py

import os
from crewai import LLM
from dotenv import load_dotenv

load_dotenv()


def load_llm(model_name: str = None, seed: int = None, temperature: float = 0.6) -> LLM:
    """Load an LLM with proper token limits and rate limiting"""

    model_name = model_name or os.getenv("DEFAULT_LLM", "gemini")

    common_kwargs = {
        "temperature": temperature,
        "max_tokens": 2000,  # Reasonable limit
        "timeout": 60,  # 1 minute timeout
    }

    if seed is not None:
        common_kwargs["seed"] = seed

    if model_name == "gemini":
        print("Loading Gemini with token limits...")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment.")
        return LLM(
            model="gemini/gemini-1.5-pro",
            api_key=api_key,
            **common_kwargs
        )

    elif model_name == "deepseek":
        print("Loading Deepseek with token limits...")
        api_key = os.getenv("HF_API_KEY")
        if not api_key:
            raise ValueError("HF_API_KEY is not set in environment.")
        os.environ["HF_TOKEN"] = api_key
        return LLM(
            model="huggingface/deepseek-ai/DeepSeek-R1",
            api_key=api_key,
            **common_kwargs
        )

    elif model_name == "ollama":
        print("Loading Ollama with token limits...")
        return LLM(
            model="ollama/llama3.2",
            base_url="http://localhost:11434",
            **common_kwargs
        )

    else:
        raise ValueError(f"Unsupported model name: {model_name}")