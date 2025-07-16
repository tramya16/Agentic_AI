# queries.py

PMO_1K_QUERIES = {
    "albuterol_similarity": "Design a molecule similar to albuterol while preserving key functional groups.",
    "amlodipine_mpo": "Generate molecules similar to amlodipine with good drug-like properties (e.g., 3-ring topology).",
    "celecoxib_rediscovery": "Recreate the anti-inflammatory drug celecoxib.",
    "deco_hop": "Modify the decorations of a molecule while preserving a fixed scaffold. Avoid forbidden substructures and stay below similarity cap.",
    "drd2": "Generate molecules predicted to strongly bind to the dopamine D2 receptor using a predictive model.",
    "fexofenadine_mpo": "Create molecules structurally similar to fexofenadine with TPSA ≈ 90 and logP ≈ 4.",
    "gsk3b": "Design molecules predicted to have high binding affinity for the GSK3β protein.",
    "isomers_c7h8n2o2": "Generate any molecule that is an exact isomer of C7H8N2O2. Must match the molecular formula exactly.",
    "isomers_c9h10n2o2pf2cl": "Generate an exact isomer of C9H10N2O2PF2Cl.",
    "jnk3": "Design molecules with high predicted inhibitory activity against the JNK3 protein."
}

def get_query_list():
    """Return list of query names for iteration"""
    return list(PMO_1K_QUERIES.keys())

def get_query_prompt(query_name):
    """Get prompt for a specific query"""
    return PMO_1K_QUERIES.get(query_name, "")