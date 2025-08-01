from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from utils.chemistry_utils import check_duplicate
import json

class DuplicateCheckInput(BaseModel):
    smiles1: str = Field(..., description="First SMILES")
    smiles2: str = Field(..., description="Second SMILES")

class DuplicateCheckTool(BaseTool):
    name: str = "duplicate_checker"
    description: str = "Check if two SMILES are equivalent molecules"
    args_schema: Type[BaseModel] = DuplicateCheckInput

    def _run(self, smiles1: str, smiles2: str) -> str:
        result = check_duplicate(smiles1, smiles2)
        return json.dumps(result)
