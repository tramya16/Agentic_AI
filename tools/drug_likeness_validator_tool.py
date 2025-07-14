from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from utils.chemistry_utils import check_drug_likeness
import json

class DrugLikenessInput(BaseModel):
    smiles: str = Field(..., description="SMILES string to evaluate")

class DrugLikenessValidatorTool(BaseTool):
    name: str = "drug_likeness_validator"
    description: str = "Evaluate molecule using Lipinski, Veber, and PAINS filters"
    args_schema: Type[BaseModel] = DrugLikenessInput

    def _run(self, smiles: str) -> str:
        result = check_drug_likeness(smiles)
        return json.dumps(result)
