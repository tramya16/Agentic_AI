from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from utils.chemistry_utils import check_toxicity
import json

class ToxicityInput(BaseModel):
    smiles: str = Field(..., description="SMILES string to assess toxicity")

class ToxicityCheckTool(BaseTool):
    name: str = "toxicity_check"
    description: str = "Flag potential toxicophores or known toxic fragments"
    args_schema: Type[BaseModel] = ToxicityInput

    def _run(self, smiles: str) -> str:
        result = check_toxicity(smiles)
        return json.dumps(result)
