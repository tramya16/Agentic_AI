# improved_queries.py

IMPROVED_PMO_QUERIES = {
    "albuterol_similarity": {
        "prompt": """Design molecules similar to albuterol (salbutamol) while preserving key functional groups.

Target molecule: Albuterol
SMILES: CC(C)(C)NCC(c1ccc(c(c1)CO)O)O
Key features: Beta-2 agonist, bronchodilator, phenolic hydroxyl groups, secondary amine

Requirements:
- Maintain similarity to albuterol structure
- Preserve beta-agonist pharmacophore
- Keep drug-like properties
- Tanimoto similarity > 0.6 to target

Example modifications:
- Modify alkyl substituents on nitrogen
- Vary hydroxyl group positions
- Introduce bioisosteric replacements""",
        "target_smiles": "CC(C)(C)NCC(c1ccc(c(c1)CO)O)O",
        "similarity_threshold": 0.6
    },

    "amlodipine_mpo": {
        "prompt": """Generate molecules similar to amlodipine with optimized drug-like properties.

Target molecule: Amlodipine
SMILES: CCOC(=O)C1=C(COCCN)NC(=C(C1c1ccccc1Cl)C(=O)OC)C
Key features: Calcium channel blocker, dihydropyridine core, 3-ring topology

Multi-parameter optimization targets:
- Molecular weight: 300-450 Da
- LogP: 2-4
- TPSA: 60-90 Ų
- Maintain 3-ring system topology
- Good oral bioavailability

Example modifications:
- Vary ester groups
- Modify dihydropyridine substituents  
- Replace chlorophenyl with other aromatics""",
        "target_smiles": "CCOC(=O)C1=C(COCCN)NC(=C(C1c1ccccc1Cl)C(=O)OC)C",
        "properties": {"MW": "300-450", "logP": "2-4", "TPSA": "60-90"}
    },

    "celecoxib_rediscovery": {
        "prompt": """Recreate or design molecules similar to the anti-inflammatory drug celecoxib.

Target molecule: Celecoxib  
SMILES: Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1
Key features: COX-2 selective inhibitor, pyrazole core, sulfonamide group, trifluoromethyl

Requirements:
- Anti-inflammatory activity
- COX-2 selectivity preferred
- Maintain core pyrazole-sulfonamide structure
- Good drug-like properties

Example modifications:
- Vary aromatic substituents
- Modify trifluoromethyl group
- Replace sulfonamide with bioisosteres
- Adjust pyrazole substitution pattern""",
        "target_smiles": "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
        "activity": "COX-2_inhibition"
    },

    "isomers_c7h8n2o2": {
        "prompt": """Generate molecules that are exact isomers of the molecular formula C7H8N2O2.

Molecular formula: C7H8N2O2
Molecular weight: 152.15 Da
Atom counts: C=7, H=8, N=2, O=2

Requirements:
- EXACT molecular formula match: C7H8N2O2
- Valid chemical structure
- Reasonable stability
- No additional or missing atoms

Example structures with C7H8N2O2:
- 4-nitroaniline derivatives
- Aminobenzoic acid derivatives  
- Pyrimidine carboxylic acids
- Imidazole derivatives

Validation: Generated molecules must have exactly 7 carbons, 8 hydrogens, 2 nitrogens, and 2 oxygens.""",
        "molecular_formula": "C7H8N2O2",
        "exact_mass": 152.15
    },

    "drd2_binding": {
        "prompt": """Design molecules predicted to bind strongly to the dopamine D2 receptor.

Target: Dopamine D2 receptor
Binding mode: Antagonist or partial agonist preferred
Reference compounds: Haloperidol, risperidone, aripiprazole

Key pharmacophore features:
- Basic nitrogen (protonatable at physiological pH)
- Aromatic ring system
- Appropriate linker length (3-4 atoms)
- Lipophilic aromatic region

Requirements:
- High predicted D2 binding affinity
- CNS drug-like properties
- Blood-brain barrier penetration
- Avoid excessive lipophilicity (LogP < 5)

Example scaffolds:
- Benzisoxazole derivatives
- Butyrophenone analogs  
- Quinolinone derivatives
- Phenothiazine-like structures""",
        "target_protein": "DRD2",
        "activity_type": "binding_affinity",
        "properties": {"CNS_MPO": ">4", "logP": "<5", "MW": "<450"}
    }
}


def get_improved_query_list():
    """Return list of improved query names"""
    return list(IMPROVED_PMO_QUERIES.keys())


def get_improved_query_data(query_name):
    """Get complete query data including prompt, SMILES, and metadata"""
    return IMPROVED_PMO_QUERIES.get(query_name, {})


def get_improved_query_prompt(query_name):
    """Get improved prompt for a specific query"""
    query_data = IMPROVED_PMO_QUERIES.get(query_name, {})
    return query_data.get("prompt", "")


# Add this function to improved_queries.py
def get_query_list():
    """Backward compatibility function"""
    return get_improved_query_list()