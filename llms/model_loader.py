import os
from crewai import LLM
from dotenv import load_dotenv

load_dotenv()


def load_llm(model_config: dict = None, seed: int = None) -> LLM:
    """Load an LLM with model configuration"""
    
    if model_config is None:
        # Fallback to default
        model_config = {
            "name": "gemini",
            "model_id": "gemini/gemini-2.0-flash",
            "temperature": 0.9
        }
    
    common_kwargs = {
        "temperature": model_config.get("temperature", 0.9),
        "max_tokens": 1300,
        "timeout": 60,
    }
    
    if seed is not None:
        common_kwargs["seed"] = seed
    
    model_name = model_config["name"]
    model_id = model_config["model_id"]
    
    if model_name == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment.")
        return LLM(
            model=model_id,
            api_key=api_key,
            **common_kwargs
        )
    
    elif model_name == "deepseek":
        api_key = os.getenv("HF_API_KEY")
        if not api_key:
            raise ValueError("HF_API_KEY is not set in environment.")
        os.environ["HF_TOKEN"] = api_key
        return LLM(
            model=model_id,
            api_key=api_key,
            **common_kwargs
        )
    
    elif model_name == "ollama":
        return LLM(
            model=model_id,
            base_url="http://localhost:11434",
            **common_kwargs
        )
    
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
