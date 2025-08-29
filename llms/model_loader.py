# llms/model_loader.py

import os
from crewai import LLM
from dotenv import load_dotenv

from config import ExperimentConfig


load_dotenv()


def load_llm(model_name: str = None, seed: int = None) -> LLM:
    """Load an LLM with proper token limits and rate limiting"""

    model_name = model_name or os.getenv("DEFAULT_LLM", "gemini")

    common_kwargs = {
        "temperature": ExperimentConfig.TEMPERATURE,
        "max_tokens": ExperimentConfig.MAX_TOKENS,
        "timeout": ExperimentConfig.TIMEOUT,
    }

    if seed is not None:
        common_kwargs["seed"] = seed

    if model_name.lower().__contains__("gemini"):
        print("Loading Gemini with token limits...")
        print("Model:",model_name)
        print("Details:",common_kwargs.get("temperature"))

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment.")
        return LLM(
            model="gemini/"+model_name,
            api_key=api_key,
            **common_kwargs
        )

    elif model_name.lower().__contains__("deepseek"):
        print("Loading Deepseek with token limits...")
        api_key = os.getenv("HF_API_KEY")
        if not api_key:
            raise ValueError("HF_API_KEY is not set in environment.")
        os.environ["HF_TOKEN"] = api_key
        return LLM(
            model="huggingface/together/deepseek-ai/"+model_name,
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