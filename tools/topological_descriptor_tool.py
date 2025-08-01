# tools/topological_descriptor_tool.py

from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.Chem import Descriptors
import json

class TopologicalDescriptorInput(BaseModel):
    smiles: str = Field(..., description="Valid SMILES string")

class TopologicalDescriptorTool(BaseTool):
    name: str = "topological_descriptor"
    description: str = "Compute topological descriptors like Chi indices and Kappa indices."
    args_schema: Type[BaseModel] = TopologicalDescriptorInput

    def _run(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return json.dumps({"error": "Invalid SMILES input"})

        try:
            result = {
                "chi0": Descriptors.Chi0(mol),
                "chi1": Descriptors.Chi1(mol),
                "chi2n": Descriptors.Chi2n(mol),
                "chi3n": Descriptors.Chi3n(mol),
                "kappa1": Descriptors.Kappa1(mol),
                "kappa2": Descriptors.Kappa2(mol),
                "kappa3": Descriptors.Kappa3(mol),
            }
        except Exception as e:
            return json.dumps({"error": str(e)})

        return json.dumps(result)
