import json
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from utils.chemistry_utils import is_patented

class PatentCheckInput(BaseModel):
    smiles: str = Field(..., description="SMILES string to check for patents")

class PatentCheckTool(BaseTool):
    name: str = "patent_check"
    description: str = "Check if the molecule is patented"
    args_schema: Type[BaseModel] = PatentCheckInput

    def _run(self, smiles: str) -> str:
        result = is_patented(smiles)
        return json.dumps(result)
