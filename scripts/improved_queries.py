from rdkit import Chem
from tools.smarts_pattern_tool import SmartsPatternTool
from tools.molecular_formula_tool import MolecularFormulaValidatorTool
import json

IMPROVED_PMO_QUERIES = {
    "albuterol_similarity": {
        "prompt": """Design molecules with high structural similarity to albuterol, preserving key pharmacophoric features for beta-2 agonist activity.

Target: Albuterol (SMILES: CC(C)(C)NCC(O)c1ccc(O)c(CO)c1)
- Task Type: similarity_search
- Objective: Beta-2 agonist for asthma treatment
- Key Features: tert-butyl group, beta-hydroxyl, catechol pattern, benzyl alcohol
- Similarity Threshold: 0.7 (Tanimoto, ECFP4 fingerprint)

Requirements:
- Maintain core pharmacophore (tert-butylamine, catechol, beta-hydroxyl)
- Ensure drug-like properties (MW 200-400, LogP 1-3, TPSA 60-100)
- Avoid exact replication of albuterol
- Validate SMILES with SmilesValidatorTool

Design Strategies:
- Modify alkyl groups on nitrogen (e.g., isopropyl instead of tert-butyl)
- Adjust hydroxyl positions on aromatic ring
- Replace catechol with bioisosteres (e.g., resorcinol, pyridone)
- Extend or shorten benzyl alcohol chain

Constraints:
- Molecular Weight: 200-400 Da
- LogP: 1-3
- TPSA: 60-100 Å²
- Essential Groups: tertiary amine, hydroxyl, aromatic ring""",
        "target_smiles": "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",
        "task_type": "similarity_search",
        "similarity_threshold": 0.7,
        "property_targets": {"MW": "200-400", "logP": "1-3", "TPSA": "60-100", "HBD": "2-4", "HBA": "2-4"}
    },

    "amlodipine_mpo": {
        "prompt": """Generate molecules structurally similar to amlodipine with optimized physicochemical properties.

Target: Amlodipine (SMILES: CCOC(=O)C1=C(COCCN)NC(=C(C1c1ccccc1Cl)C(=O)OC)C)
- Task Type: optimization
- Objective: Calcium channel blocker for hypertension
- Key Features: dihydropyridine core, chlorophenyl, ester groups, amine side chain
- Similarity Threshold: 0.65 (Tanimoto, ECFP4)

Requirements:
- Preserve dihydropyridine core (SMARTS: c1c(c(c(c(c1)C)C(=O)OCC)C(=O)OC)N)
- Include 3 rings (1 dihydropyridine, 1 chlorophenyl, 1 flexible)
- Optimize properties: MW 300-450, LogP 2-4, TPSA 60-90
- Avoid exact replication of amlodipine
- Validate core SMARTS with SmartsPatternTool

Design Strategies:
- Vary ester groups (e.g., methyl to ethyl ester)
- Replace chlorophenyl with other substituted aromatics
- Modify amine side chain length or substituents
- Introduce polar groups to adjust TPSA

Constraints:
- Core SMARTS: c1c(c(c(c(c1)C)C(=O)OCC)C(=O)OC)N
- Molecular Weight: 300-450 Da
- LogP: 2-4
- TPSA: 60-90 Å²""",
        "target_smiles": "CCOC(=O)C1=C(COCCN)NC(=C(C1c1ccccc1Cl)C(=O)OC)C",
        "task_type": "optimization",
        "core_smarts": "c1c(c(c(c(c1)C)C(=O)OCC)C(=O)OC)N",
        "similarity_threshold": 0.65,
        "property_targets": {"MW": "300-450", "logP": "2-4", "TPSA": "60-90", "HBD": "1-3", "HBA": "4-6"}
    },

    "celecoxib_rediscovery": {
        "prompt": """Design COX-2 selective inhibitors similar to celecoxib.

Target: Celecoxib (SMILES: Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1)
- Task Type: similarity_search
- Objective: COX-2 inhibition for anti-inflammatory activity
- Key Features: pyrazole core, sulfonamide, trifluoromethyl, two aromatic rings
- Similarity Threshold: 0.7 (Tanimoto, ECFP4)

Requirements:
- Preserve pyrazole core (SMARTS: c1cn(nn1)c2ccccc2)
- Maintain sulfonamide and trifluoromethyl groups
- Ensure drug-like properties (MW 300-500, LogP 2-4, TPSA 60-100)
- Avoid exact replication of celecoxib
- Validate core SMARTS with SmartsPatternTool

Design Strategies:
- Vary substituents on aromatic rings (e.g., methyl to ethyl)
- Replace trifluoromethyl with other fluorinated groups
- Modify sulfonamide to sulfone or other bioisosteres
- Adjust pyrazole substitution pattern

Constraints:
- Core SMARTS: c1cn(nn1)c2ccccc2
- Molecular Weight: 300-500 Da
- LogP: 2-4
- TPSA: 60-100 Å²""",
        "target_smiles": "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
        "task_type": "similarity_search",
        "core_smarts": "c1cn(nn1)c2ccccc2",
        "similarity_threshold": 0.7,
        "property_targets": {"MW": "300-500", "logP": "2-4", "TPSA": "60-100", "HBD": "1-2", "HBA": "3-5"}
    },

    "deco_hop": {
        "prompt": """Design drug-like molecules via scaffold hopping, preserving the propoxy-benzothiazole decoration while replacing the quinazoline core.

Target: CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C
- Task Type: scaffold_hopping
- Objective: Kinase inhibitor with novel scaffold
- Core SMARTS to Replace: [#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12 (quinazoline)
- Decoration SMARTS to Preserve: [#6]-[#6]-[#6]-[#8]-[#6]~[#6]~[#6]~[#6]~[#6]-[#7]-c1ccc2ncsc2c1 (propoxy-benzothiazole)
- Forbidden SMARTS: [CS([#6])(=O)=O], [#7]-c1ccc2ncsc2c1
- Similarity Cap: 0.85 (Tanimoto, ECFP4)

Requirements:
- Replace quinazoline core with alternative heterocycles (e.g., pyrimidine, pyrazine, quinoline)
- Preserve propoxy-benzothiazole decoration
- Avoid forbidden motifs (validate with SmartsPatternTool)
- Ensure drug-like properties (MW 300-500, LogP 2-4, TPSA 60-100)
- Validate new scaffold with SmartsPatternTool

Design Strategies:
- Use pyrimidine, pyrazine, or quinoline as core replacements
- Maintain ether linkage and sulfonamide group
- Vary thiazole substituents
- Adjust propoxy chain length or substituents

Constraints:
- Core SMARTS to Replace: [#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12
- Decoration SMARTS to Preserve: [#6]-[#6]-[#6]-[#8]-[#6]~[#6]~[#6]~[#6]~[#6]-[#7]-c1ccc2ncsc2c1
- Forbidden SMARTS: [CS([#6])(=O)=O], [#7]-c1ccc2ncsc2c1
- Molecular Weight: 300-500 Da
- LogP: 2-4
- TPSA: 60-100 Å²""",
        "target_smiles": "CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C",
        "task_type": "scaffold_hopping",
        "core_smarts": "[#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12",
        "forbidden_smarts": ["CS([#6])(=O)=O", "[#7]-c1ccc2ncsc2c1"],
        "similarity_cap": 0.85,
        "property_targets": {"MW": "300-500", "logP": "2-4", "TPSA": "60-100", "HBD": "1-3", "HBA": "3-5"}
    },

    "drd2_binding": {
        "prompt": """Design molecules with high predicted binding affinity to dopamine D2 receptor (DRD2).

Target: High DRD2 binding affinity
- Task Type: optimization
- Objective: Antipsychotic or CNS-active molecules
- Reference Molecules: haloperidol, risperidone, aripiprazole
- Key Features: basic nitrogen (protonatable), aromatic rings, 3-4 atom linker

Requirements:
- Ensure CNS penetration (use BBBPermeantPredictionTool)
- Optimize properties: MW <450, LogP 2-5, TPSA 40-80
- Include basic nitrogen and aromatic systems
- Avoid toxicity (validate with ToxicityCheckTool)
- Similarity Threshold: 0.6 (Tanimoto, ECFP4) to references

Design Strategies:
- Incorporate piperazine or piperidine for basic nitrogen
- Use arylpiperazine or benzisoxazole scaffolds
- Vary linker length (3-4 atoms, e.g., propyl, butyl)
- Add polar groups to balance TPSA and LogP

Constraints:
- Molecular Weight: <450 Da
- LogP: 2-5
- TPSA: 40-80 Å²
- Essential Features: basic nitrogen, 2-3 aromatic rings, 3-4 atom linker""",
        "target_protein": "DRD2",
        "task_type": "optimization",
        "similarity_threshold": 0.6,
        "property_targets": {"MW": "<450", "logP": "2-5", "TPSA": "40-80", "HBD": "1-2", "HBA": "2-4"}
    },

    "fexofenadine_mpo": {
        "prompt": """Design molecules structurally similar to fexofenadine with optimized physicochemical properties.

Target: Fexofenadine (SMILES: CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4)
- Task Type: optimization
- Objective: Non-sedating antihistamine (H1 receptor antagonist)
- Key Features: carboxylic acid, hydroxyl groups, piperidine ring, diphenylmethyl core
- Similarity Threshold: 0.7 (Tanimoto, ECFP4)

Requirements:
- Maintain antihistamine pharmacophore (piperidine, diphenylmethyl)
- Target TPSA ~90 Å², LogP ~4
- Ensure drug-like properties (MW 400-600, HBD 2-3, HBA 4-6)
- Avoid exact replication of fexofenadine
- Validate SMILES with SmilesValidatorTool

Design Strategies:
- Replace carboxylic acid with bioisosteres (e.g., tetrazole)
- Modify piperidine substituents or ring size
- Adjust alkyl chain length between aromatic rings
- Replace hydroxyl with other polar groups (e.g., amine)

Constraints:
- Molecular Weight: 400-600 Da
- LogP: ~4
- TPSA: ~90 Å²
- Essential Groups: carboxylic acid or bioisostere, piperidine, diphenylmethyl""",
        "target_smiles": "CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4",
        "task_type": "optimization",
        "similarity_threshold": 0.7,
        "property_targets": {"MW": "400-600", "logP": "~4", "TPSA": "~90", "HBD": "2-3", "HBA": "4-6"}
    },

    "gsk3b_activity": {
        "prompt": """Design molecules with high predicted binding affinity to GSK3B (Glycogen Synthase Kinase 3 Beta).

Target: GSK3B kinase inhibitors
- Task Type: optimization
- Objective: Inhibit GSK3B for diabetes, Alzheimer's, or cancer
- Key Features: heterocyclic core, 1-2 H-bond donors, 2-4 H-bond acceptors, aromatic rings
- Pharmacophore: Heterocyclic core with H-bond donors/acceptors for ATP site

Requirements:
- Maximize GSK3B binding probability (0-1 score)
- Ensure drug-like properties (Lipinski's Rule: MW 200-500, LogP 1-4, TPSA 40-100)
- Validate heterocyclic core with FragmentationTool and RingAnalysisTool
- Avoid known inhibitors (novel structures)

Design Strategies:
- Use pyrimidine, purine, or indole scaffolds
- Incorporate 1-2 H-bond donors and 2-4 acceptors
- Add aromatic rings for π-π stacking
- Include polar substituents for selectivity

Constraints:
- Molecular Weight: 200-500 Da
- LogP: 1-4
- TPSA: 40-100 Å²
- Essential Features: heterocyclic core, H-bond donors/acceptors""",
        "target_smiles": None,
        "task_type": "optimization",
        "property_targets": {"MW": "200-500", "logP": "1-4", "TPSA": "40-100", "HBD": "1-2", "HBA": "2-4"}
    },

    "isomers_c7h8n2o2": {
        "prompt": """Generate molecules with exact molecular formula C7H8N2O2.

Target Formula: C7H8N2O2
- Task Type: isomers
- Objective: Chemically valid isomers with exact atom counts
- Hard Constraint: Exactly 7C, 8H, 2N, 2O (validate with MolecularFormulaValidatorTool)

Requirements:
- Generate diverse isomers (aromatic, aliphatic, heterocyclic)
- Ensure exact atom counts (C7H8N2O2)
- Maintain drug-like properties (MW ~150-300, LogP 1-3, TPSA 40-80)
- Validate SMILES with SmilesValidatorTool

Design Strategies:
- Start with benzene ring and add functional groups (e.g., nitro, amide)
- Explore heterocyclic cores (e.g., pyrimidine, pyrazole)
- Include polar groups (e.g., hydroxyl, amine) for TPSA
- Vary connectivity (e.g., ortho vs. meta substitution)

Constraints:
- Molecular Formula: C7H8N2O2
- Molecular Weight: ~150-300 Da
- LogP: 1-3
- TPSA: 40-80 Å²""",
        "molecular_formula": "C7H8N2O2",
        "task_type": "isomers",
        "property_targets": {"MW": "150-300", "logP": "1-3", "TPSA": "40-80", "HBD": "1-3", "HBA": "2-4"}
    },

    "isomers_c9h10n2o2pf2cl": {
        "prompt": """Design isomers with exact molecular formula C9H10N2O2PF2Cl.

Target Formula: C9H10N2O2PF2Cl
- Task Type: isomers
- Objective: Chemically valid isomers with exact atom counts
- Hard Constraint: Exactly 9C, 10H, 2N, 2O, 1P, 2F, 1Cl (validate with MolecularFormulaValidatorTool)

Requirements:
- Generate diverse isomers (aromatic, phosphonate, heterocyclic)
- Ensure exact atom counts
- Maintain drug-like properties (MW 250-400, LogP 1-4, TPSA 40-100)
- Validate SMILES with SmilesValidatorTool

Design Strategies:
- Incorporate phosphonate or phosphinate groups
- Use aromatic rings with halogen substituents
- Explore nitrogen-containing heterocycles (e.g., pyridine, triazole)
- Vary halogen positions (F, Cl) on aromatic or aliphatic chains

Constraints:
- Molecular Formula: C9H10N2O2PF2Cl
- Molecular Weight: 250-400 Da
- LogP: 1-4
- TPSA: 40-100 Å²""",
        "molecular_formula": "C9H10N2O2PF2Cl",
        "task_type": "isomers",
        "property_targets": {"MW": "250-400", "logP": "1-4", "TPSA": "40-100", "HBD": "1-2", "HBA": "3-5"}
    },

    "jnk3_inhibition": {
        "prompt": """Design drug-like molecules with high predicted JNK3 (c-Jun N-terminal kinase 3) inhibitory activity.

Target: JNK3 kinase inhibitors
- Task Type: optimization
- Objective: Inhibit JNK3 for neurodegenerative diseases, diabetes, or cancer
- Key Features: heterocyclic core, 1-2 H-bond donors, 2-4 H-bond acceptors, aromatic rings
- Reference: 4-phenyl-1,3-thiazol-2-amine (SMILES: C1=CC=C(C=C1)C2=NC(=CS2)N)
- Similarity Threshold: 0.6 (Tanimoto, ECFP4)

Requirements:
- Maximize JNK3 inhibition probability (0-1 score)
- Ensure drug-like properties (MW 200-500, LogP 1-4, TPSA 40-100)
- Validate heterocyclic core with FragmentationTool and RingAnalysisTool
- Avoid exact replication of reference

Design Strategies:
- Use aminothiazole, pyrazole, or indole scaffolds
- Incorporate 1-2 H-bond donors and 2-4 acceptors
- Add aromatic rings for hydrophobic interactions
- Include polar substituents for selectivity

Constraints:
- Molecular Weight: 200-500 Da
- LogP: 1-4
- TPSA: 40-100 Å²
- Essential Features: heterocyclic core, H-bond donors/acceptors""",
        "target_smiles": "C1=CC=C(C=C1)C2=NC(=CS2)N",
        "task_type": "optimization",
        "similarity_threshold": 0.6,
        "property_targets": {"MW": "200-500", "logP": "1-4", "TPSA": "40-100", "HBD": "1-2", "HBA": "2-4"}
    },

    "median1_similarity": {
        "prompt": """Design molecules simultaneously similar to camphor and menthol based on ECFP4 fingerprint similarity.

Target Molecules:
- Camphor: CC1(C)C2CCC1(C)C(=O)C2 (bicyclic ketone)
- Menthol: CC(C)C1CCC(C)CC1O (cyclohexane with hydroxyl)
- Task Type: similarity_search
- Objective: Balance monoterpene features for drug-like molecules
- Similarity Threshold: 0.7 (Tanimoto, ECFP4, balanced for both)

Requirements:
- Achieve high similarity to both molecules
- Maintain drug-like properties (MW 150-300, LogP 2-4, TPSA 20-60)
- Avoid exact replication of either molecule
- Validate SMILES with SmilesValidatorTool

Design Strategies:
- Combine bicyclic and cyclohexane elements
- Incorporate ketone and hydroxyl groups
- Maintain C10 monoterpene-like framework
- Adjust methyl or isopropyl substituents

Constraints:
- Molecular Weight: 150-300 Da
- LogP: 2-4
- TPSA: 20-60 Å²
- Essential Features: saturated carbon framework, ketone or hydroxyl""",
        "target_smiles": ["CC1(C)C2CCC1(C)C(=O)C2", "CC(C)C1CCC(C)CC1O"],
        "task_type": "similarity_search",
        "similarity_threshold": 0.7,
        "property_targets": {"MW": "150-300", "logP": "2-4", "TPSA": "20-60", "HBD": "0-2", "HBA": "1-3"}
    },

    "median2_similarity": {
        "prompt": """Design molecules simultaneously similar to tadalafil and sildenafil based on ECFP4 fingerprint similarity.

Target Molecules:
- Tadalafil: O=C1N(CC(N2C1CC3=C(C2C4=CC5=C(OCO5)C=C4)NC6=C3C=CC=C6)=O)C
- Sildenafil: CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C
- Task Type: similarity_search
- Objective: PDE5 inhibitors for erectile dysfunction
- Similarity Threshold: 0.65 (Tanimoto, ECFP4, balanced for both)

Requirements:
- Balance features from tadalafil (tricyclic, indole-like) and sildenafil (pyrazolopyrimidinone)
- Ensure drug-like properties (MW 300-500, LogP 2-4, TPSA 60-100)
- Avoid exact replication of either molecule
- Validate SMILES with SmilesValidatorTool

Design Strategies:
- Combine indole and pyrazole-pyrimidine cores
- Incorporate lactam and sulfonamide features
- Add methylenedioxyphenyl or piperazine groups
- Balance rigidity and flexibility

Constraints:
- Molecular Weight: 300-500 Da
- LogP: 2-4
- TPSA: 60-100 Å²
- Essential Features: heterocyclic core, amide/lactam, aromatic rings""",
        "target_smiles": ["O=C1N(CC(N2C1CC3=C(C2C4=CC5=C(OCO5)C=C4)NC6=C3C=CC=C6)=O)C",
                          "CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C"],
        "task_type": "similarity_search",
        "similarity_threshold": 0.65,
        "property_targets": {"MW": "300-500", "logP": "2-4", "TPSA": "60-100", "HBD": "1-3", "HBA": "4-6"}
    },

    "mestranol_similarity": {
        "prompt": """Design molecules structurally similar to mestranol, preserving the steroid scaffold and key pharmacophores.

Target: Mestranol (SMILES: COc1ccc2[C@H]3CC[C@@]4(C)[C@@H](CC[C@@]4(O)C#C)[C@@H]3CCc2c1)
- Task Type: similarity_search
- Objective: Synthetic estrogen for oral contraceptives
- Key Features: steroid backbone, ethinyl group at C17, methoxy at C3, phenolic A-ring
- Similarity Threshold: 0.7 (Tanimoto, ECFP4)

Requirements:
- Preserve steroid scaffold (SMARTS: C1CC2C(C(C1)C3CCC4(C)C(C3)CCC4)
- Maintain ethinyl and phenolic A-ring
- Ensure drug-like properties (MW 300-400, LogP 3-5, TPSA 20-60)
- Avoid exact replication of mestranol
- Validate core SMARTS with SmartsPatternTool

Design Strategies:
- Replace methoxy with ethoxy or propoxy
- Modify ethinyl to other alkynyl groups
- Add halogens to A-ring
- Adjust stereochemistry or alkyl substituents

Constraints:
- Core SMARTS: C1CC2C(C(C1)C3CCC4(C)C(C3)CCC4
- Molecular Weight: 300-400 Da
- LogP: 3-5
- TPSA: 20-60 Å²""",
        "target_smiles": "COc1ccc2[C@H]3CC[C@@]4(C)[C@@H](CC[C@@]4(O)C#C)[C@@H]3CCc2c1",
        "task_type": "similarity_search",
        "core_smarts": "C1CC2C(C(C1)C3CCC4(C)C(C3)CCC4",
        "similarity_threshold": 0.7,
        "property_targets": {"MW": "300-400", "logP": "3-5", "TPSA": "20-60", "HBD": "1-2", "HBA": "2-3"}
    },

    "osimertinib_mpo": {
        "prompt": """Design molecules similar to osimertinib with optimized physicochemical properties.

Target: Osimertinib (SMILES: COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34)
- Task Type: optimization
- Objective: EGFR tyrosine kinase inhibitor for NSCLC
- Key Features: pyrimidine core, indole system, acrylamide warhead, tertiary amines
- Similarity Threshold: 0.8 (ECFP6), 0.65 (FCFP4)

Requirements:
- Preserve pyrimidine-indole-acrylamide framework (SMARTS: c1cc2c(c(c1)Nc3nccc(n3)c4cn(c)c5ccccc45)C(=O)NC=C)
- Target TPSA ~100 Å², LogP ~1
- Ensure drug-like properties (MW 400-600, HBD 1-2, HBA 4-6)
- Avoid exact replication of osimertinib
- Validate core SMARTS with SmartsPatternTool

Design Strategies:
- Replace methoxy with hydroxyl or amino
- Add polar substituents to indole or pyrimidine
- Modify tertiary amine alkyl groups
- Introduce nitrogen-containing heterocycles

Constraints:
- Core SMARTS: c1cc2c(c(c1)Nc3nccc(n3)c4cn(c)c5ccccc45)C(=O)NC=C
- Molecular Weight: 400-600 Da
- LogP: ~1
- TPSA: ~100 Å²""",
        "target_smiles": "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34",
        "task_type": "optimization",
        "core_smarts": "c1cc2c(c(c1)Nc3nccc(n3)c4cn(c)c5ccccc45)C(=O)NC=C",
        "similarity_threshold": {"ECFP6": 0.8, "FCFP4": 0.65},
        "property_targets": {"MW": "400-600", "logP": "~1", "TPSA": "~100", "HBD": "1-2", "HBA": "4-6"}
    },

    "perindopril_mpo": {
        "prompt": """Design molecules similar to perindopril with approximately 2 aromatic rings.

Target: Perindopril (SMILES: O=C(OCC)C(NC(C(=O)N1C(C(=O)O)CC2CCCCC12)C)CCC)
- Task Type: optimization
- Objective: ACE inhibitor for hypertension
- Key Features: bicyclic lactam, ethyl ester, carboxylic acid, secondary amide
- Similarity Threshold: 0.65 (Tanimoto, ECFP4)

Requirements:
- Incorporate ~2 aromatic rings (e.g., replace cyclohexane with benzene)
- Preserve bicyclic lactam and carboxylic acid
- Ensure drug-like properties (MW 300-500, LogP 1-3, TPSA 60-100)
- Avoid exact replication of perindopril
- Validate SMILES with SmilesValidatorTool

Design Strategies:
- Replace cyclohexane with benzene
- Add phenyl group to propyl chain
- Introduce benzyl substituents on amide
- Use indoline or tetrahydroquinoline scaffolds

Constraints:
- Molecular Weight: 300-500 Da
- LogP: 1-3
- TPSA: 60-100 Å²
- Essential Features: bicyclic lactam, carboxylic acid, 2 aromatic rings""",
        "target_smiles": "O=C(OCC)C(NC(C(=O)N1C(C(=O)O)CC2CCCCC12)C)CCC",
        "task_type": "optimization",
        "similarity_threshold": 0.65,
        "property_targets": {"MW": "300-500", "logP": "1-3", "TPSA": "60-100", "HBD": "1-2", "HBA": "4-6"}
    },

    "qed_optimization": {
        "prompt": """Design molecules with maximum QED (Quantitative Estimation of Drug-likeness) score.

Target: High QED score (~1.0)
- Task Type: optimization
- Objective: Optimize drug-likeness across multiple descriptors
- Key Features: balanced MW, LogP, TPSA, rotatable bonds, aromatic rings

Requirements:
- Maximize QED score (0-1, target ~1.0)
- Ensure drug-like properties (Lipinski's Rule)
- Avoid reactive or toxic groups (validate with ToxicityCheckTool)
- Create synthetically feasible molecules

Design Strategies:
- Use benzene, pyridine, or thiophene scaffolds
- Include 1-2 aromatic rings with polar substituents
- Add hydroxyl, amine, or amide groups for TPSA
- Limit rotatable bonds (<10)

Constraints:
- Molecular Weight: 150-500 Da (optimal ~300)
- LogP: 1-3 (optimal ~2.5)
- TPSA: 20-130 Å² (optimal ~60)
- Rotatable Bonds: 0-10
- Aromatic Rings: 1-4""",
        "target_smiles": None,
        "task_type": "optimization",
        "property_targets": {"MW": "150-500", "logP": "1-3", "TPSA": "20-130", "rotatable_bonds": "0-10",
                             "aromatic_rings": "1-4"}
    },

    "ranolazine_mpo": {
        "prompt": """Design molecules similar to ranolazine with optimized physicochemical properties and 1 fluorine atom.

Target: Ranolazine (SMILES: COc1ccccc1OCC(O)CN2CCN(CC(=O)Nc3c(C)cccc3C)CC2)
- Task Type: optimization
- Objective: Anti-anginal agent for chronic stable angina
- Key Features: methoxyphenyl ether, piperazine, secondary alcohol, amide, dimethylaniline
- Similarity Threshold: 0.7 (Tanimoto, ECFP4)

Requirements:
- Include exactly 1 fluorine atom
- Target TPSA ~95 Å², LogP ~7
- Ensure drug-like properties (MW 400-600, HBD 1-3, HBA 4-6)
- Avoid exact replication of ranolazine
- Validate SMILES with SmilesValidatorTool

Design Strategies:
- Add fluorine to dimethylaniline or methoxyphenyl
- Replace methoxy with polar groups to adjust TPSA
- Modify piperazine substituents
- Adjust linker chain to balance LogP

Constraints:
- Molecular Weight: 400-600 Da
- LogP: ~7
- TPSA: ~95 Å²
- Fluorine Count: 1
- Essential Features: piperazine, amide, aromatic rings""",
        "target_smiles": "COc1ccccc1OCC(O)CN2CCN(CC(=O)Nc3c(C)cccc3C)CC2",
        "task_type": "optimization",
        "similarity_threshold": 0.7,
        "property_targets": {"MW": "400-600", "logP": "~7", "TPSA": "~95", "HBD": "1-3", "HBA": "4-6",
                             "fluorine_count": 1}
    },

    "scaffold_hop": {
        "prompt": """Design molecules via scaffold hopping, replacing the quinazoline core while preserving key decorations.

Target: CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C
- Task Type: scaffold_hopping
- Objective: Kinase inhibitor with novel scaffold
- Core SMARTS to Replace: [#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12 (quinazoline)
- Decoration SMARTS to Preserve: [#6]-[#6]-[#6]-[#8]-[#6]~[#6]~[#6]~[#6]~[#6]-[#7]-c1ccc2ncsc2c1 (propoxy-benzothiazole)
- Forbidden SMARTS: [CS([#6])(=O)=O], [#7]-c1ccc2ncsc2c1
- Similarity Cap: 0.75 (Tanimoto, ECFP4)

Requirements:
- Replace quinazoline with alternative heterocycles (e.g., pyrimidine, pyrazine)
- Preserve propoxy-benzothiazole decoration
- Avoid forbidden motifs (validate with SmartsPatternTool)
- Ensure drug-like properties (MW 300-500, LogP 2-4, TPSA 60-100)
- Validate new scaffold with SmartsPatternTool

Design Strategies:
- Use pyrimidine, pyrazine, or benzimidazole cores
- Maintain ether and sulfonamide groups
- Vary benzothiazole substituents
- Adjust propoxy chain length

Constraints:
- Core SMARTS to Replace: [#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12
- Decoration SMARTS to Preserve: [#6]-[#6]-[#6]-[#8]-[#6]~[#6]~[#6]~[#6]~[#6]-[#7]-c1ccc2ncsc2c1
- Forbidden SMARTS: [CS([#6])(=O)=O], [#7]-c1ccc2ncsc2c1
- Molecular Weight: 300-500 Da
- LogP: 2-4
- TPSA: 60-100 Å²""",
        "target_smiles": "CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C",
        "task_type": "scaffold_hopping",
        "core_smarts": "[#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12",
        "forbidden_smarts": ["CS([#6])(=O)=O", "[#7]-c1ccc2ncsc2c1"],
        "similarity_cap": 0.75,
        "property_targets": {"MW": "300-500", "logP": "2-4", "TPSA": "60-100", "HBD": "1-3", "HBA": "3-5"}
    },

    "sitagliptin_mpo": {
        "prompt": """
    Design molecules similar to sitagliptin with the exact molecular formula C16H15F6N5O, maintaining key pharmacophoric features (trifluoromethyl, triazole, primary amine, fluorinated aromatic, amide groups). Optimize for properties evaluated by the Sitagliptin_MPO oracle, including similarity to sitagliptin (Tanimoto, ECFP4), molecular weight (MW), LogP, TPSA, HBD, and HBA. Ensure synthetic feasibility and drug-likeness.
    Requirements:
    - Exact molecular formula: C16H15F6N5O
    - Similarity threshold: 0.7 (Tanimoto, ECFP4)
    - Avoid exact replication of sitagliptin
    - Validate SMILES and formula using MolecularFormulaValidatorTool
    Design Strategies:
    - Rearrange fluorine atoms on the aromatic ring
    - Modify triazole substitution or position
    - Adjust amine or amide connectivity
    - Reposition trifluoromethyl group
    Constraints:
    - MW: 407.31 Da (exact)
    - LogP: ~2.1
    - TPSA: ~95.5 Å²
    - HBD: 2
    - HBA: 8
    """,
        "target_smiles": "Fc1cc(c(F)cc1F)CC(N)CC(=O)N3Cc2nnc(n2CC3)C(F)(F)F",
        "task_type": "optimization",
        "similarity_threshold": 0.7,
        "molecular_formula": "C16H15F6N5O",
        "property_targets": {
            "MW": 407.31,
            "LogP": 2.1,
            "TPSA": 95.5,
            "HBD": 2,
            "HBA": 8
        }
    },

    "thiothixene_similarity": {
        "prompt": """Design molecules structurally similar to thiothixene, preserving the thioxanthene scaffold.

Target: Thiothixene (SMILES: CN(C)S(=O)(=O)c1ccc2Sc3ccccc3C(=CCCN4CCN(C)CC4)c2c1)
- Task Type: similarity_search
- Objective: Antipsychotic for dopamine D2 receptor antagonism
- Key Features: thioxanthene core, piperazine, sulfonamide, alkene linker
- Similarity Threshold: 0.7 (Tanimoto, ECFP4)

Requirements:
- Preserve thioxanthene core (SMARTS: c1cc2c(c(c1)S(=O)(=O)N(C)C)Sc3ccccc3C=C)
- Maintain piperazine and sulfonamide
- Ensure CNS penetration (TPSA 40-80, use BBBPermeantPredictionTool)
- Avoid exact replication of thiothixene
- Validate core SMARTS with SmartsPatternTool

Design Strategies:
- Vary piperazine substituents (e.g., ethyl instead of methyl)
- Modify sulfonamide alkyl groups
- Replace alkene with alkyl linker
- Add halogens to aromatic rings

Constraints:
- Core SMARTS: c1cc2c(c(c1)S(=O)(=O)N(C)C)Sc3ccccc3C=C
- Molecular Weight: 400-600 Da
- LogP: 2-5
- TPSA: 40-80 Å²""",
        "target_smiles": "CN(C)S(=O)(=O)c1ccc2Sc3ccccc3C(=CCCN4CCN(C)CC4)c2c1",
        "task_type": "similarity_search",
        "core_smarts": "c1cc2c(c(c1)S(=O)(=O)N(C)C)Sc3ccccc3C=C",
        "similarity_threshold": 0.7,
        "property_targets": {"MW": "400-600", "logP": "2-5", "TPSA": "40-80", "HBD": "0-2", "HBA": "3-5"}
    },

    "troglitazone_similarity": {
        "prompt": """Design molecules structurally similar to troglitazone, preserving the thiazolidinedione pharmacophore.

Target: Troglitazone (SMILES: Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O)
- Task Type: similarity_search
- Objective: PPAR-γ agonist for type 2 diabetes
- Key Features: thiazolidinedione ring, chroman core, phenolic hydroxyl
- Similarity Threshold: 0.7 (Tanimoto, ECFP4)

Requirements:
- Preserve thiazolidinedione ring (SMARTS: c1cc(OC)cc(c1)CC2SC(=O)NC2=O)
- Maintain chroman or similar bicyclic scaffold
- Ensure drug-like properties (MW 400-600, LogP 2-5, TPSA 60-100)
- Avoid hepatotoxic features (validate with ToxicityCheckTool)
- Validate core SMARTS with SmartsPatternTool

Design Strategies:
- Vary methyl groups on chroman
- Replace chroman with benzofuran or indane
- Modify ether linker length
- Add polar substituents for safety

Constraints:
- Core SMARTS: c1cc(OC)cc(c1)CC2SC(=O)NC2=O
- Molecular Weight: 400-600 Da
- LogP: 2-5
- TPSA: 60-100 Å²""",
        "target_smiles": "Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O",
        "task_type": "similarity_search",
        "core_smarts": "c1cc(OC)cc(c1)CC2SC(=O)NC2=O",
        "similarity_threshold": 0.7,
        "property_targets": {"MW": "400-600", "logP": "2-5", "TPSA": "60-100", "HBD": "1-3", "HBA": "4-6"}
    },

    "valsartan_smarts": {
        "prompt": """
    Design molecules containing the SMARTS pattern CN(C=O)Cc1ccc(c2ccccc2)cc1, targeting ARB-like antihypertensive molecules. Optimize for properties evaluated by the Valsartan_SMARTS oracle, including presence of the SMARTS pattern, MW, LogP, TPSA, HBD, HBA, and Bertz complexity. Ensure drug-like properties and synthetic feasibility.
    Requirements:
    - Must contain SMARTS: CN(C=O)Cc1ccc(c2ccccc2)cc1
    - Validate SMILES with SmilesValidatorTool
    Design Strategies:
    - Extend biphenyl with polar groups (e.g., carboxylic acid, tetrazole)
    - Add hydroxyl or amine substituents
    - Incorporate heterocycles for complexity
    - Balance LogP and TPSA
    Constraints:
    - MW: 300-600 Da
    - LogP: ~2.0
    - TPSA: ~95 Å²
    - HBD: 1-3
    - HBA: 3-5
    - Bertz Complexity: ~800
    """,
        "core_smarts": "CN(C=O)Cc1ccc(c2ccccc2)cc1",
        "task_type": "smarts_matching",
        "property_targets": {
            "MW": [300, 600],
            "LogP": 2.0,
            "TPSA": 95,
            "HBD": [1, 3],
            "HBA": [3, 5],
            "Bertz Complexity": 800
        }
    },

    "zaleplon_similarity": {
        "prompt": """Design molecules similar to zaleplon with exact molecular formula C19H17N3O2.

Target: Zaleplon (SMILES: O=C(C)N(CC)C1=CC=CC(C2=CC=NC3=C(C=NN23)C#N)=C1)
- Task Type: similarity_search
- Objective: Non-benzodiazepine hypnotic for insomnia
- Key Features: pyrazolopyrimidine core, acetamide, nitrile
- Molecular Formula: C19H17N3O2 (validate with MolecularFormulaValidatorTool)
- Similarity Threshold: 0.75 (Tanimoto, atom-pair fingerprint)

Requirements:
- Preserve pyrazolopyrimidine core (SMARTS: c1cc2c(c(c1)N(C)C(=O)C)n(c(c(n2)C#N)N=C)C)
- Match exact formula: C19H17N3O2
- Ensure drug-like properties (MW ~350, LogP 1-3, TPSA 60-100)
- Avoid exact replication of zaleplon
- Validate core SMARTS and formula

Design Strategies:
- Vary alkyl groups on amide (e.g., methyl instead of ethyl)
- Modify phenyl ring substituents
- Replace acetyl with propionyl
- Adjust nitrile position

Constraints:
- Core SMARTS: c1cc2c(c(c1)N(C)C(=O)C)n(c(c(n2)C#N)N=C)C
- Molecular Formula: C19H17N3O2
- Molecular Weight: ~350 Da
- LogP: 1-3
- TPSA: 60-100 Å²""",
        "target_smiles": "O=C(C)N(CC)C1=CC=CC(C2=CC=NC3=C(C=NN23)C#N)=C1",
        "task_type": "similarity_search",
        "core_smarts": "c1cc2c(c(c1)N(C)C(=O)C)n(c(c(n2)C#N)N=C)C",
        "molecular_formula": "C19H17N3O2",
        "similarity_threshold": 0.75,
        "property_targets": {"MW": "~350", "logP": "1-3", "TPSA": "60-100", "HBD": "0-2", "HBA": "3-5"}
    }
}


def get_query_list():
    """Get list of available queries."""
    return list(IMPROVED_PMO_QUERIES.keys())


def get_query_prompt(query_name):
    """Get prompt for a query."""
    return IMPROVED_PMO_QUERIES.get(query_name, {}).get("prompt", "")


def get_query_data(query_name):
    """Get all data for a query."""
    return IMPROVED_PMO_QUERIES.get(query_name, {})