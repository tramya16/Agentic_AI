# tools/scaffold_extraction_tool.py
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import json

class ScaffoldExtractionInput(BaseModel):
    smiles: str = Field(..., description="Valid SMILES string")
    scaffold_type: str = Field("murcko", description="Type of scaffold to extract: 'murcko' (default)")

class ScaffoldExtractionTool(BaseTool):
    name: str = "scaffold_extraction"
    description: str = "Extract Murcko scaffolds or Bemis-Murcko frameworks"
    args_schema: Type[BaseModel] = ScaffoldExtractionInput

    def _run(self, smiles: str, scaffold_type: str = "murcko") -> str:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return json.dumps({"error": "Invalid SMILES input"})

        if scaffold_type == "murcko":
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        else:
            # extend if you want other scaffold types
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)

        scaffold_smiles = Chem.MolToSmiles(scaffold)
        return json.dumps({"scaffold_smiles": scaffold_smiles})
