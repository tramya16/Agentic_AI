# tools/smarts_pattern_tool.py

from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from rdkit import Chem
import json

class SmartsPatternInput(BaseModel):
    smiles: str = Field(..., description="Valid SMILES string of the compound")
    smarts_pattern: str = Field(..., description="SMARTS pattern to search for")

class SmartsPatternTool(BaseTool):
    name: str = "smarts_pattern_matcher"
    description: str = "Checks if a molecule contains a specific SMARTS pattern and counts matches"
    args_schema: Type[BaseModel] = SmartsPatternInput

    def _run(self, smiles: str, smarts_pattern: str) -> str:
        try:
            mol = Chem.MolFromSmiles(smiles)
            pattern = Chem.MolFromSmarts(smarts_pattern)
            
            if not mol:
                return json.dumps({"error": f"Invalid SMILES: {smiles}"})
            if not pattern:
                return json.dumps({"error": f"Invalid SMARTS pattern: {smarts_pattern}"})
                
            matches = mol.HasSubstructMatch(pattern)
            match_count = len(mol.GetSubstructMatches(pattern))
            
            return json.dumps({
                "smiles": smiles,
                "smarts_pattern": smarts_pattern,
                "contains_pattern": matches,
                "match_count": match_count
            })
        except Exception as e:
            return json.dumps({"error": str(e)})
