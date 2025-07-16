# llms/model_loader.py

import os
from crewai import LLM
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

load_dotenv()
SEED=16
def load_llm(model_name: str = None, seed: int | None = None,
             temperature: float = 0.9) -> LLM:
    """
    Load an LLM based on environment variables or explicit model name.
    Supports 'gemini', 'deepseek' etc.
    """

    model_name = model_name or os.getenv("DEFAULT_LLM", "gemini")

    common_kwargs = {"temperature": temperature}
    if seed is not None:
        common_kwargs["seed"] = seed          # litellm supports this

    if model_name == "gemini":
        print("Loading Gemini")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment.")
        return LLM(
            model="gemini/gemini-1.5-pro",
            api_key=api_key,
            **common_kwargs
        )

    elif model_name == "ollama":
        print("Loading ollama/llama3.2")
        return LLM(
            model="ollama/llama3",  # or whatever model you're running
            base_url="http://localhost:11434",
            api_key=None,
            provider="ollama"
        )
    elif model_name == "deepseek":
        print("Loading Deepseek")
        api_key = os.getenv("HF_API_KEY")
        if not api_key:
            raise ValueError("HF_API_KEY is not set in environment.")
        # Set the HF_TOKEN environment variable for litellm compatibility
        os.environ["HF_TOKEN"] = api_key
        return LLM(
            model="huggingface/together/deepseek-ai/DeepSeek-R1",
            api_key=api_key,
            provider="huggingface"
        )

    else:
        raise ValueError(f"Unsupported model name: {model_name}")


def load_deepseek_llm() -> LLM:
    """
    Convenience function to load DeepSeek-R1 specifically.
    """
    return load_llm("deepseek")

def load_gemini_llm() -> LLM:
    """
    Convenience function to load DeepSeek-R1 specifically.
    """
    return load_llm("gemini")