from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from utils.chemistry_utils import validate_smiles
import json

class ValidatorInput(BaseModel):
    smiles: str = Field(..., description="SMILES string to validate")

class SmilesValidatorTool(BaseTool):
    name: str = "smiles_validator"
    description: str = "Validate SMILES and return canonical form"
    args_schema: Type[BaseModel] = ValidatorInput

    def _run(self, smiles: str) -> str:
        result = validate_smiles(smiles)
        return json.dumps(result)