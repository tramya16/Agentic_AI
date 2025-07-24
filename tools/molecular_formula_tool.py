# tools/molecular_formula_tool.py

from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import json

class MolecularFormulaInput(BaseModel):
    smiles: str = Field(..., description="Valid SMILES string of the compound")
    target_formula: str = Field(..., description="Target molecular formula (e.g., 'C19H17N3O2')")

class MolecularFormulaValidatorTool(BaseTool):
    name: str = "molecular_formula_validator"
    description: str = "Validates if a SMILES string matches an exact molecular formula"
    args_schema: Type[BaseModel] = MolecularFormulaInput

    def _run(self, smiles: str, target_formula: str) -> str:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return json.dumps({"error": f"Invalid SMILES: {smiles}"})
            
            actual_formula = rdMolDescriptors.CalcMolFormula(mol)
            matches = actual_formula == target_formula
            
            return json.dumps({
                "smiles": smiles,
                "target_formula": target_formula,
                "actual_formula": actual_formula,
                "matches": matches
            })
        except Exception as e:
            return json.dumps({"error": str(e)})
