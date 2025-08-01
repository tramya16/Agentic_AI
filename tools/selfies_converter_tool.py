from typing import Type, Literal
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from utils.chemistry_utils import smiles_to_selfies, selfies_to_smiles
import json

class SelfiesConverterInput(BaseModel):
    input_value: str = Field(..., description="Input string: SMILES or SELFIES")
    input_type: Literal["smiles", "selfies"] = Field(..., description="Input type")

class SelfiesConverterTool(BaseTool):
    name:str = "selfies_converter"
    description:str = "Convert between SELFIES and SMILES"
    args_schema: Type[BaseModel] = SelfiesConverterInput

    def _run(self, input_value: str, input_type: str) -> str:
        try:
            if input_type == "smiles":
                res = smiles_to_selfies(input_value)
            elif input_type == "selfies":
                res = selfies_to_smiles(input_value)
            else:
                return json.dumps({"status": "error", "error": "Invalid input_type"})
            return json.dumps(res)
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})
