from rdkit import Chem
from rdkit.Chem import Descriptors

def score_logp(smiles: str) -> float:
    """Compute octanol-water partition coefficient (LogP)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES for LogP scoring")
    return Descriptors.MolLogP(mol)

def score_mol_wt(smiles: str) -> float:
    """Compute molecular weight."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES for molecular weight scoring")
    return Descriptors.MolWt(mol)

def score_tpsa(smiles: str) -> float:
    """Compute topological polar surface area (TPSA)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES for TPSA scoring")
    return Descriptors.TPSA(mol)

#This gives  three correlates of ADMET:
# lipophilicity (Log P),
# size (MolWt),
# and polarity (TPSA).

def get_all_scores(smiles: str) -> dict:
    """Return a dict of all baseline descriptor scores."""
    return {
        "LogP": round(score_logp(smiles), 2),
        "MolWt": round(score_mol_wt(smiles), 1),
        "TPSA": round(score_tpsa(smiles), 1),
    }



