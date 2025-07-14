# tools/complexity_calculator_tool.py
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import json

class ComplexityCalculatorInput(BaseModel):
    smiles: str = Field(..., description="Valid SMILES string")

class ComplexityCalculatorTool(BaseTool):
    name: str = "complexity_calculator"
    description: str = "Calculate Bertz complexity, rotatable bonds, H-bond donors and acceptors"
    args_schema: Type[BaseModel] = ComplexityCalculatorInput

    def _run(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return json.dumps({"error": "Invalid SMILES input"})

        bertz_complexity = Descriptors.BertzCT(mol)
        rotatable_bonds = Lipinski.NumRotatableBonds(mol)
        hbond_donors = Lipinski.NumHDonors(mol)
        hbond_acceptors = Lipinski.NumHAcceptors(mol)

        result = {
            "bertz_complexity": bertz_complexity,
            "rotatable_bonds": rotatable_bonds,
            "hbond_donors": hbond_donors,
            "hbond_acceptors": hbond_acceptors,
        }
        return json.dumps(result)
