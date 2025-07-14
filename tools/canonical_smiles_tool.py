from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from utils.chemistry_utils import resolve_to_smiles
import json

class CanonicalSmilesInput(BaseModel):
    input_value: str = Field(..., description="Chemical identifier (name, SMILES, CAS, InChI, etc.)")
    input_type: str = Field("auto", description="Input type or 'auto' for detection")

class CanonicalSmilesTool(BaseTool):
    name:str = "canonical_smiles_generator"
    description:str = "Generate canonical SMILES from any valid chemical input"
    args_schema: Type[BaseModel] = CanonicalSmilesInput

    def _run(self, input_value: str, input_type: str = "auto") -> str:
        result = resolve_to_smiles(input_value, input_type)
        return json.dumps(result)
