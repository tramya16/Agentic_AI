# tools/shape_descriptor_tool.py

from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
import json

class ShapeDescriptorInput(BaseModel):
    smiles: str = Field(..., description="Valid SMILES string")

class ShapeDescriptorTool(BaseTool):
    name: str = "shape_descriptor"
    description: str = "Calculate 3D shape descriptors: molecular volume, asphericity, eccentricity."
    args_schema: Type[BaseModel] = ShapeDescriptorInput

    def _run(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return json.dumps({"error": "Invalid SMILES input"})

        try:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.UFFOptimizeMolecule(mol)

            result = {
                "mol_volume": rdMolDescriptors.CalcExactMolWt(mol),
                "asphericity": rdMolDescriptors.CalcAsphericity(mol),
                "eccentricity": rdMolDescriptors.CalcEccentricity(mol),
                "inertial_shape_factor": rdMolDescriptors.CalcInertialShapeFactor(mol)
            }
        except Exception as e:
            return json.dumps({"error": str(e)})

        return json.dumps(result)
