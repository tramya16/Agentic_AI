# tools/fragmentation_tool.py
from typing import Type, Dict
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr
from rdkit import Chem
from utils.chemistry_utils import load_rdkit_functional_groups
import json


class FragmentationInput(BaseModel):
    smiles: str = Field(..., description="Valid SMILES string")


class FragmentationTool(BaseTool):
    name: str = "fragmentation_analysis"
    description: str = "Count functional groups, key fragments, and rings"
    args_schema: Type[BaseModel] = FragmentationInput
    _functional_groups: Dict[str, str] = PrivateAttr(default_factory=load_rdkit_functional_groups)

    def _run(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return json.dumps({"error": "Invalid SMILES input"})

        fg_counts = {}
        for name, smarts in self._functional_groups.items():
            patt = Chem.MolFromSmarts(smarts)
            if patt is None:
                fg_counts[name] = "Invalid SMARTS"
                continue
            matches = mol.GetSubstructMatches(patt)
            fg_counts[name] = len(matches)

        ring_count = mol.GetRingInfo().NumRings()
        fragments = Chem.GetMolFrags(mol, asMols=True)
        fragment_smiles = [Chem.MolToSmiles(frag) for frag in fragments]

        result = {
            "functional_groups_counts": fg_counts,
            "ring_count": ring_count,
            "fragment_count": len(fragments),
            "fragments_smiles": fragment_smiles,
        }
        return json.dumps(result)
