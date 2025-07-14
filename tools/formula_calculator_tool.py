from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from utils.chemistry_utils import calculate_molecular_formula
import json

class FormulaCalculatorInput(BaseModel):
    smiles: str = Field(..., description="SMILES string for molecular formula calculation")

class FormulaCalculatorTool(BaseTool):
    name:str = "formula_calculator"
    description:str = "Calculate molecular formula from SMILES"
    args_schema: Type[BaseModel] = FormulaCalculatorInput

    def _run(self, smiles: str) -> str:
        result = calculate_molecular_formula(smiles)
        return json.dumps(result)
