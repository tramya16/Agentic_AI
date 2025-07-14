from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from utils.chemistry_utils import get_rdkit_properties
import json

class LookupInput(BaseModel):
    smiles: str = Field(..., description="Valid SMILES string")

class PropertiesLookupTool(BaseTool):
    name: str = "chemical_lookup"
    description: str = "Get comprehensive chemical properties"
    args_schema: Type[BaseModel] = LookupInput

    def _run(self, smiles: str) -> str:
        result = get_rdkit_properties(smiles)
        return json.dumps(result)