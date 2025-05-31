import os
from openai import OpenAI


class GeneratorAgent:
    """
    Uses OpenRouter (via the OpenAI Python 1.x client) to propose a new SMILES string,
    given an input SMILES and an optimization objective.
    """

    def __init__(
            self,
            model: str = "mistralai/mixtral-8x7b-instruct",
            temperature: float = 0.7,
    ):
        # Get API key from environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model = model
        self.temperature = temperature

    def generate(self, base_smiles: str, objective: str) -> str:
        prompt = (
            f"You are a molecule optimization assistant. Your task is to modify the given SMILES string to improve the specified properties.\n\n"
            f"Base SMILES: {base_smiles}\n"
            f"Optimization objective: {objective}\n\n"
            f"Return ONLY the modified SMILES string with NO additional text or explanations.\n"
            f"Ensure the SMILES string is valid and chemically correct.\n"
            f"Modified SMILES: "
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content

        # Extract just the SMILES part
        return self.extract_smiles(text)

    def extract_smiles(self, text: str) -> str:
        """Extract the SMILES string from the response text"""
        # Look for text within backticks (common for SMILES)
        if '`' in text:
            parts = text.split('`')
            if len(parts) >= 2:
                return parts[1].strip()

        # Look for the last word that looks like a SMILES
        words = text.split()
        for word in reversed(words):
            if any(c in word for c in ['=', '@', '(', ')', '[', ']', '/', '\\']):
                return word.strip("`\"' \n")

        # Fallback: return the entire text stripped
        return text.strip("`\"' \n")