"""
Configuration file for LLM molecular generation experiments
"""

import os
from pathlib import Path

from scripts.improved_queries import get_query_list


# Experiment Configuration
class ExperimentConfig:
    # Model settings
    TEMPERATURE = 0.9
    MAX_TOKENS = 2000
    TIMEOUT = 60

    # Experiment settings
    RUNS_PER_QUERY = 1
    MAX_ITERATIONS = 2  # For iterative pipeline
    TOP_N = 5

    # Model configurations
    MODELS = {
        "gemini_1_5_pro": {
            "name": "gemini_1_5_pro",
            "model_id":"gemini-1.5-pro",
            "display_name": "Gemini 1.5 Pro",
            "folder_name": f"Gemini_1.5_Pro_Temp_{TEMPERATURE}_Results"
        },
        "gemini_2_5_pro": {
            "name": "gemini_2_5_pro",
            "model_id": "gemini-2.5-pro",
            "display_name": "Gemini 2.5 Pro",
            "folder_name": f"Gemini_2.5_Pro_Temp_{TEMPERATURE}_Results"
        },
        "gemini_2_0_flash": {
            "name": "gemini_2_0_flash",
            "model_id": "gemini-2.0-flash",
            "display_name": "Gemini 2.0 Flash",
            "folder_name": f"Gemini_2.0_Flash_Temp_{TEMPERATURE}_Results"
        },
        # "deepseek_v3": {
        #     "name": "deepseek_v3",
        #     "model_id":"DeepSeek-V3",
        #     "display_name": "DeepSeek V3",
        #     "folder_name": f"DeepSeek_V3_Temp_{TEMPERATURE}_Results"
        # }
    }

    # Paths
    RESULTS_DIR = Path("results")

    # Query settings
    # QUERY_LIST = get_query_list()
    QUERY_LIST = ["albuterol_similarity"]

    @classmethod
    def get_model_result_dir(cls, model_key):
        """Get the results directory for a specific model"""
        if model_key not in cls.MODELS:
            raise ValueError(f"Unknown model: {model_key}")
        return cls.RESULTS_DIR / cls.MODELS[model_key]["folder_name"]

    @classmethod
    def get_all_model_result_dirs(cls):
        """Get all model result directories"""
        return {model_key: cls.get_model_result_dir(model_key)
                for model_key in cls.MODELS.keys()}

    @classmethod
    def ensure_directories(cls):
        """Ensure all necessary directories exist"""
        cls.RESULTS_DIR.mkdir(exist_ok=True)

        for model_key in cls.MODELS.keys():
            cls.get_model_result_dir(model_key).mkdir(parents=True, exist_ok=True)


# Validation
def validate_config():
    """Validate configuration settings based on active models"""
    required_env_vars = []
    
    # Dynamically determine required environment variables based on active models
    for model_key, model_config in ExperimentConfig.MODELS.items():
        model_id = model_config.get("model_id", "")
        
        if model_id.startswith("gemini"):
            if "GEMINI_API_KEY" not in required_env_vars:
                required_env_vars.append("GEMINI_API_KEY")
        elif model_id.startswith("DeepSeek") or "deepseek" in model_id.lower():
            if "HF_API_KEY" not in required_env_vars:
                required_env_vars.append("HF_API_KEY")
        # Add other model types as needed
    
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print(f"⚠️ Missing environment variables: {missing_vars}")
        print("Please set these in your .env file")
        return False

    return True


if __name__ == "__main__":
    # Test configuration
    print("Testing configuration...")

    if validate_config():
        print("✅ Configuration valid")
        ExperimentConfig.ensure_directories()
        print("✅ Directories created")

        print("\nModel configurations:")
        for model_key, config in ExperimentConfig.MODELS.items():
            print(f"  {config['display_name']}: {ExperimentConfig.get_model_result_dir(model_key)}")
    else:
        print("❌ Configuration invalid")