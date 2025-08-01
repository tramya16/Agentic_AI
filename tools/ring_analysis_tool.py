# tools/ring_analysis_tool.py
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

import json

class RingAnalysisInput(BaseModel):
    smiles: str = Field(..., description="Valid SMILES string")

class RingAnalysisTool(BaseTool):
    name: str = "ring_analysis"
    description: str = "Analyze ring count, sizes, spiro atoms, macrocycles"
    args_schema: Type[BaseModel] = RingAnalysisInput

    def _run(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return json.dumps({"error": "Invalid SMILES input"})

        ring_info = mol.GetRingInfo()
        ring_sizes = ring_info.AtomRings()  # tuple of atom idxs per ring
        ring_sizes_list = [len(r) for r in ring_sizes]

        # Count spiro atoms
        spiro_atoms = rdMolDescriptors.CalcNumSpiroAtoms(mol)

        # Macrocycles (rings with size >= 8)
        macrocycles_count = sum(1 for size in ring_sizes_list if size >= 8)

        result = {
            "ring_count": ring_info.NumRings(),
            "ring_sizes": ring_sizes_list,
            "spiro_atoms_count": spiro_atoms,
            "macrocycles_count": macrocycles_count
        }
        return json.dumps(result)
