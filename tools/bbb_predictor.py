# tools/bbb_predictor.py

from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from adme_py import ADME
import json

class BBBPermeantInput(BaseModel):
    smiles: str = Field(..., description="Valid SMILES string of the compound")

class BBBPermeantPredictionTool(BaseTool):
    name: str = "bbb_penetration_predictor"
    description: str = "Predicts whether the compound can cross the blood-brain barrier using the BOILED-Egg model"
    args_schema: Type[BaseModel] = BBBPermeantInput

    def _run(self, smiles: str) -> str:
        try:
            result = ADME(smiles).calculate()
            permeant = result.get("pharmacokinetics", {}).get("blood_brain_barrier_permeant", None)
            if permeant is None:
                return json.dumps({"error": "Unable to determine BBB permeability"})
            return json.dumps({"blood_brain_barrier_permeant": bool(permeant)})
        except Exception as e:
            return json.dumps({"error": str(e)})
