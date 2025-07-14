# tools/electronic_descriptor_tool.py

from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.Chem import rdPartialCharges, rdMolDescriptors
import json

class ElectronicDescriptorInput(BaseModel):
    smiles: str = Field(..., description="Valid SMILES string")

class ElectronicDescriptorTool(BaseTool):
    name: str = "electronic_descriptor"
    description: str = "Calculate electronic descriptors: Gasteiger charges, TPSA, partial charge range."
    args_schema: Type[BaseModel] = ElectronicDescriptorInput

    def _run(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return json.dumps({"error": "Invalid SMILES input"})

        try:
            # Correct usage
            rdPartialCharges.ComputeGasteigerCharges(mol)
            charges = [
                float(atom.GetProp('_GasteigerCharge'))
                for atom in mol.GetAtoms()
                if atom.HasProp('_GasteigerCharge')
            ]
            charge_range = (min(charges), max(charges)) if charges else (None, None)

            result = {
                "tpsa": rdMolDescriptors.CalcTPSA(mol),
                "min_gasteiger_charge": charge_range[0],
                "max_gasteiger_charge": charge_range[1],
            }
        except Exception as e:
            return json.dumps({"error": str(e)})

        return json.dumps(result)
