# simplified_queries.py

from rdkit import Chem

IMPROVED_PMO_QUERIES = {
    "albuterol_similarity": {
        "prompt": """Design molecules similar to albuterol while preserving key functional groups.

Target: Albuterol (CC(C)(C)NCC(O)c1ccc(O)c(CO)c1)
- Beta-2 agonist for asthma
- Key features: tert-butyl group, beta-hydroxyl, catechol pattern, benzyl alcohol

Requirements:
- Keep the core pharmacophore
- Maintain drug-like properties
- Don't copy albuterol exactly

Example changes:
- Modify alkyl groups on nitrogen
- Change hydroxyl positions
- Replace with bioisosteres""",
        "target_smiles": "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1"
    },

    "amlodipine_mpo": {
        "prompt": """Generate amlodipine-like molecules with good drug properties.

Target: Amlodipine (CCOC(=O)C1=C(COCCN)NC(=C(C1c1ccccc1Cl)C(=O)OC)C)
- Calcium channel blocker
- Keep dihydropyridine core and 3 rings
- Optimize MW (300-450), LogP (2-4), TPSA (60-90)

Example changes:
- Vary ester groups
- Replace chlorophenyl
- Modify side chains""",
        "target_smiles": "CCOC(=O)C1=C(COCCN)NC(=C(C1c1ccccc1Cl)C(=O)OC)C"
    },

    "celecoxib_rediscovery": {
        "prompt": """Design celecoxib-like COX-2 inhibitors.

Target: Celecoxib (Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1)
- COX-2 selective inhibitor
- Keep pyrazole core, sulfonamide, trifluoromethyl
- Maintain anti-inflammatory activity

Example changes:
- Vary aromatic substituents
- Replace trifluoromethyl group
- Modify sulfonamide""",
        "target_smiles": "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1"
    },

    "isomers_c7h8n2o2": {
        "prompt": """Generate molecules with exact formula C7H8N2O2.

HARD CONSTRAINT: Must have exactly 7C, 8H, 2N, 2O atoms.

Common patterns:
- Aromatic amines with functional groups  
- Nitroaromatic compounds
- Aminobenzoic acid derivatives

Strategy: Start with benzene ring + add remaining atoms as functional groups.""",
        "molecular_formula": "C7H8N2O2"
    },

    "drd2_binding": {
        "prompt": """Design molecules that bind strongly to dopamine D2 receptor.

Target: High DRD2 binding affinity
- Need basic nitrogen, aromatic rings, proper linker
- CNS drug-like properties (MW<450, LogP 2-5)
- Reference: haloperidol, risperidone, aripiprazole

Key features:
- Basic nitrogen (protonatable)
- Aromatic system
- 3-4 atom linker
- Avoid toxicity""",
        "target_protein": "DRD2"
    }
}

def get_query_list():
    """Get list of available queries"""
    return list(IMPROVED_PMO_QUERIES.keys())

def get_query_prompt(query_name):
    """Get prompt for a query"""
    return IMPROVED_PMO_QUERIES.get(query_name, {}).get("prompt", "")

def get_query_data(query_name):
    """Get all data for a query"""
    return IMPROVED_PMO_QUERIES.get(query_name, {})

def validate_smiles():
    """Check if SMILES are valid"""
    for name, data in IMPROVED_PMO_QUERIES.items():
        if "target_smiles" in data:
            mol = Chem.MolFromSmiles(data["target_smiles"])
            status = "✅" if mol else "❌"
            print(f"{name}: {status}")

if __name__ == "__main__":
    print("Available queries:", get_query_list())
    print("\nSMILES validation:")
    validate_smiles()
