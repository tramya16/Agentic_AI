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

"deco_hop": {
    "prompt": """Design drug-like molecules that preserve the quinazoline core scaffold while modifying peripheral decorations to explore chemical diversity.

Target Pharmacophore: CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C
- Quinazoline-based kinase inhibitor scaffold
- Key features: quinazoline core, ether linkage, sulfonamide group, thiazole decoration

Requirements:
- Preserve the quinazoline scaffold matching SMARTS: [#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12
- Avoid forbidden motifs: CS([#6])(=O)=O and [#7]-c1ccc2ncsc2c1
- Maintain moderate similarity to reference (capped at 0.85)
- Explore peripheral decorations creatively
- Don't copy the reference exactly

Example modifications:
- Replace tert-butyl sulfonamide with other groups
- Modify the propoxy chain length
- Change thiazole decoration
- Add/remove substituents on quinazoline""",
    "target_smiles": "CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C",
    "core_smarts": "[#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12",
    "forbidden_smarts": ["CS([#6])(=O)=O", "[#7]-c1ccc2ncsc2c1"],
    "similarity_cap": 0.85
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
    },
    "fexofenadine_mpo": {
    "prompt": """Create molecules structurally similar to fexofenadine with TPSA ≈90 and logP ≈4.

Target: Fexofenadine (CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4)
- Non-sedating antihistamine (H1 receptor antagonist)
- Key features: carboxylic acid, hydroxyl groups, piperidine ring, diphenylmethyl core, aromatic systems

Requirements:
- Maintain structural similarity to fexofenadine
- Target TPSA around 90
- Aim for LogP close to 4 (moderate lipophilicity)
- Preserve antihistamine pharmacophore
- Don't copy fexofenadine exactly

Example modifications:
- Alter carboxylic acid to bioisosteres (tetrazole, hydroxamic acid)
- Modify piperidine ring substituents
- Change aromatic ring substitution patterns
- Adjust alkyl chain length between aromatic rings
- Replace hydroxyl groups with other polar groups""",
    "target_smiles": "CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4"
},

"gsk3b_activity": {
    "prompt": """Design molecules with high predicted binding affinity to GSK3B (Glycogen Synthase Kinase 3 Beta).

Target: GSK3B kinase inhibitors
- Key enzyme involved in glycogen metabolism and Wnt signaling
- Therapeutic target for diabetes, Alzheimer's disease, and cancer
- ATP-competitive and allosteric binding sites available

Requirements:
- Maximize predicted GSK3B binding probability (0-1 score)
- Design novel structures, don't copy known inhibitors exactly
- Consider drug-like properties (Lipinski's Rule of Five)

Common GSK3B inhibitor features:
- Heterocyclic cores (pyrimidines, purines, indoles)
- Hydrogen bond donors/acceptors for ATP site
- Aromatic rings for π-π stacking
- Polar substituents for selectivity
- Molecular weight 200-500 Da

Example scaffolds to consider:
- Maleimide derivatives
- Pyrazine-based compounds
- Indirubins and analogues
- Thiadiazolidinone cores
- Benzothiazole frameworks""",
    "target_smiles": None
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

    "isomers_c9h10n2o2pf2cl": {
    "prompt": """Design isomers with the exact molecular formula C9H10N2O2PF2Cl.

Target Formula: C9H10N2O2PF2Cl
- Exactly 9 Carbon atoms
- Exactly 10 Hydrogen atoms  
- Exactly 2 Nitrogen atoms
- Exactly 2 Oxygen atoms
- Exactly 1 Phosphorus atom
- Exactly 2 Fluorine atoms
- Exactly 1 Chlorine atom

Requirements:
- Generate chemically valid isomers with this exact composition
- No missing or extra atoms allowed
- Create structurally diverse arrangements
- Consider different connectivity patterns

Structural considerations:
- Phosphorus can form 3-5 bonds (common oxidation states)
- Halogens (F, Cl) typically form single bonds
- Nitrogen can be sp3, sp2, or sp hybridized
- Consider aromatic vs aliphatic carbon frameworks
- Possible functional groups: phosphonates, amines, amides, esters

Example structural motifs:
- Aromatic rings with heteroatom substituents
- Phosphonate or phosphinate groups
- Halogenated aromatic compounds
- Nitrogen-containing heterocycles""",
    "target_smiles": None
},

    "jnk3_inhibition": {
    "prompt": """Design drug-like molecules with high predicted JNK3 (c-Jun N-terminal kinase 3) inhibitory activity.

Target: JNK3 kinase inhibitors
- Key enzyme in MAPK signaling pathway
- Therapeutic target for neurodegenerative diseases, diabetes, and cancer
- ATP-competitive binding site in kinase domain

Requirements:
- Maximize predicted JNK3 inhibition probability (0-1 score)
- Design chemically valid, drug-like molecules
- Follow Lipinski's Rule of Five guidelines
- Don't copy known inhibitors exactly

Common JNK3 inhibitor features:
- Heterocyclic cores (pyrimidines, thiazoles, pyrazoles, indoles)
- Hydrogen bond donors/acceptors for ATP site interaction
- Aromatic rings for hydrophobic interactions
- Molecular weight 200-500 Da
- Moderate lipophilicity (LogP 1-4)

Example scaffolds to consider:
- Aminothiazole derivatives
- Pyrazolopyrimidines  
- Indazole-based compounds
- Quinoline/quinazoline cores
- Benzimidazole frameworks
- Substituted anilines with heterocycles

Reference structure: 4-phenyl-1,3-thiazol-2-amine scaffold""",
    "target_smiles": "C1=CC=C(C=C1)C2=NC(=CS2)N"
},

"median1_similarity": {
    "prompt": """Design molecules that are simultaneously similar to both camphor and menthol based on ECFP4 fingerprint similarity.

Target molecules: 
- Camphor: CC1(C)C2CCC1(C)C(=O)C2 (bicyclic ketone, monoterpene)
- Menthol: CC(C)C1CCC(C)CC1O (cyclohexane with hydroxyl, monoterpene)

Requirements:
- Achieve high structural similarity to both reference molecules
- Balance features from both camphor and menthol
- Create novel structures, don't copy either molecule exactly
- Maintain drug-like properties

Key structural features to consider:
- Camphor: bicyclic structure, ketone group, quaternary carbons, methyl substituents
- Menthol: cyclohexane ring, secondary alcohol, isopropyl group, axial methyl
- Common: saturated carbon frameworks, multiple methyl groups, monoterpene-like scaffolds

Design strategies:
- Combine bicyclic elements with hydroxyl functionality
- Incorporate both ketone and alcohol groups
- Use similar carbon frameworks (C10 monoterpene-like)
- Maintain similar substitution patterns
- Consider stereochemistry for optimal similarity

Example hybrid features:
- Bicyclic alcohols combining camphor's rigidity with menthol's OH
- Cyclohexane derivatives with camphor-like substitution
- Bridged systems with both ketone and alcohol groups""",
    "target_smiles": ["CC1(C)C2CCC1(C)C(=O)C2", "CC(C)C1CCC(C)CC1O"]
},
"median2_similarity": {
    "prompt": """Design molecules that are simultaneously similar to both tadalafil and sildenafil based on structural similarity.

Target molecules:
- Tadalafil: O=C1N(CC(N2C1CC3=C(C2C4=CC5=C(OCO5)C=C4)NC6=C3C=CC=C6)=O)C (PDE5 inhibitor for erectile dysfunction)
- Sildenafil: CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C (PDE5 inhibitor, Viagra)

Requirements:
- Achieve balanced structural similarity to both reference molecules
- Capture key pharmacophoric features from both compounds
- Create novel structures, don't copy either molecule exactly
- Maintain drug-like properties and PDE5 inhibitor potential

Shared structural features to consider:
- Both are PDE5 inhibitors with similar mechanism of action
- Heterocyclic core systems (indole-like in tadalafil, pyrazolopyrimidinone in sildenafil)
- Aromatic ring systems with substitution patterns
- Nitrogen-containing heterocycles
- Amide/lactam functionality
- Bulky substituents for selectivity

Key differences to balance:
- Tadalafil: fused tricyclic system, methylenedioxyphenyl group, more rigid structure
- Sildenafil: pyrazole-pyrimidine core, sulfonamide group, piperazine moiety, more flexible

Design strategies:
- Combine heterocyclic cores from both molecules
- Include both rigid and flexible elements
- Incorporate nitrogen heterocycles with appropriate substitution
- Balance aromatic content and molecular complexity
- Consider bioisosteric replacements for key functional groups

Example hybrid approaches:
- Indole-pyrazole fused systems
- Pyrimidine cores with methylenedioxyphenyl substituents
- Lactam structures with sulfonamide modifications""",
    "target_smiles": ["O=C1N(CC(N2C1CC3=C(C2C4=CC5=C(OCO5)C=C4)NC6=C3C=CC=C6)=O)C", "CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C"]
},

"mestranol_similarity": {
    "prompt": """Design molecules structurally similar to mestranol while preserving key functional groups and core scaffold.

Target: Mestranol (COc1ccc2[C@H]3CC[C@@]4(C)[C@@H](CC[C@@]4(O)C#C)[C@@H]3CCc2c1)
- Synthetic estrogen used in oral contraceptives
- Key features: steroid backbone, ethinyl group at C17, methoxy group at C3, phenolic A-ring

Requirements:
- Maintain structural similarity to mestranol
- Preserve core steroid scaffold (four-ring system)
- Keep key functional groups for hormonal activity
- Don't copy mestranol exactly
- Maintain drug-like properties

Critical structural elements:
- Steroid backbone (gonane core structure)
- Ethinyl group (-C≡C-H) at position 17α for oral activity
- Aromatic A-ring (phenol derivative)
- Methoxy substitution at C3 position
- Tertiary alcohol at C17
- Proper stereochemistry at ring junctions

Possible modifications to consider:
- Replace methoxy with other alkoxy groups (ethoxy, propoxy)
- Modify ethinyl to other alkynyl groups
- Change aromatic ring substitution patterns
- Alter alkyl substituents while preserving core
- Consider bioisosteric replacements for key groups
- Maintain 17α-ethinyl for oral bioavailability

Example strategies:
- Ethoxy analogue: replace -OCH3 with -OCH2CH3
- Hydroxyl derivative: replace methoxy with -OH (ethynylestradiol-like)
- Halogenated variants: add F, Cl to aromatic ring
- Extended alkynyl: replace ethinyl with propynyl""",
    "target_smiles": "COc1ccc2[C@H]3CC[C@@]4(C)[C@@H](CC[C@@]4(O)C#C)[C@@H]3CCc2c1"
},
"osimertinib_mpo": {
    "prompt": """Design molecules similar to osimertinib while optimizing physicochemical properties.

Target: Osimertinib (COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34)
- Third-generation EGFR tyrosine kinase inhibitor
- Used for EGFR T790M mutation-positive non-small cell lung cancer
- Key features: pyrimidine core, indole system, acrylamide warhead, methoxy groups, tertiary amines

Requirements:
- Maintain structural similarity to osimertinib
- Target TPSA around 100 Ų (topological polar surface area)
- Aim for LogP close to 1 (low-to-moderate lipophilicity)
- Preserve key pharmacophoric elements
- Don't copy osimertinib exactly

Critical structural elements:
- Pyrimidine core for EGFR binding
- Indole ring system for selectivity
- Acrylamide group (Michael acceptor for covalent binding)
- Methoxy substituents for potency
- Tertiary amine groups for solubility
- Aromatic aniline linker

Design strategies for property optimization:
- Increase polarity: add hydroxyl groups, replace methoxy with polar groups
- Reduce lipophilicity: introduce hydrophilic substituents, polar heterocycles
- Maintain activity: preserve pyrimidine-aniline-indole framework
- Balance properties: optimize substitution patterns

Example modifications:
- Replace methoxy with hydroxyl or amino groups
- Add polar substituents to aromatic rings
- Introduce additional nitrogen atoms in rings
- Modify alkyl chains to more polar groups
- Consider bioisosteric replacements for key groups

Fingerprint considerations:
- FCFP4 similarity should be moderate (≤0.8) - functional class preservation
- ECFP6 similarity should be high (≈0.85) - extended connectivity maintenance""",
    "target_smiles": "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34"
},
"perindopril_mpo": {
    "prompt": """Design molecules similar to perindopril while incorporating approximately 2 aromatic rings.

Target: Perindopril (O=C(OCC)C(NC(C(=O)N1C(C(=O)O)CC2CCCCC12)C)CCC)
- ACE inhibitor for hypertension and heart failure
- Key features: bicyclic lactam core, ethyl ester, carboxylic acid, secondary amide, cyclohexane ring
- Currently has 0 aromatic rings - needs modification to include ~2 aromatic rings

Requirements:
- Maintain structural similarity to perindopril
- Incorporate approximately 2 aromatic rings into the structure
- Preserve key pharmacophoric elements for ACE inhibition
- Don't copy perindopril exactly
- Maintain drug-like properties

Critical structural elements to preserve:
- Bicyclic lactam system (proline-like structure)
- Carboxylic acid group for zinc coordination
- Secondary amide linkage
- Ester functionality
- Overall molecular framework

Design strategies for aromatic incorporation:
- Replace cyclohexane ring with benzene ring
- Add aromatic substituents to existing scaffold
- Introduce phenyl groups as side chains
- Replace aliphatic chains with aromatic rings
- Create aromatic-containing analogs of key fragments

Example modifications:
- Convert cyclohexane to benzene ring (1 aromatic ring)
- Replace propyl chain with phenyl group (additional aromatic ring)
- Add phenyl substituents to amide nitrogen
- Introduce benzyl groups instead of ethyl groups
- Create indoline or tetrahydroquinoline analogs

Pharmacological considerations:
- Maintain ACE binding elements
- Preserve zinc coordination capability
- Keep appropriate molecular size and flexibility
- Balance lipophilicity with aromatic content""",
    "target_smiles": "O=C(OCC)C(NC(C(=O)N1C(C(=O)O)CC2CCCCC12)C)CCC"
},
"qed_optimization": {
    "prompt": """Design molecules with maximum QED (Quantitative Estimation of Drug-likeness) score.

Target: High QED score (close to 1.0)
- QED combines multiple drug-likeness descriptors into a single score
- Considers molecular weight, lipophilicity, polar surface area, rotatable bonds, aromatic rings, alerts

Requirements:
- Maximize QED score (target close to 1.0)
- Create chemically realistic and synthetically feasible molecules
- Balance all QED components optimally
- Don't copy existing high-QED molecules exactly

QED scoring factors to optimize:
- Molecular Weight: 150-500 Da (optimal ~300 Da)
- LogP: 1-3 (optimal ~2.5)
- Polar Surface Area: 20-130 Ų (optimal ~60 Ų)
- Rotatable Bonds: 0-15 (optimal ~6)
- Aromatic Rings: 1-4 (optimal ~2)
- Structural Alerts: minimize problematic groups

Design strategies for high QED:
- Use moderate-sized aromatic scaffolds (benzene, pyridine, thiophene)
- Include 1-2 aromatic rings with appropriate substitution
- Add polar groups for balanced hydrophilicity (OH, NH2, amide)
- Keep molecular weight in optimal range (250-400 Da)
- Limit rotatable bonds and flexibility
- Avoid reactive or toxic functional groups

Example high-QED scaffolds:
- Substituted benzenes with polar groups
- Pyridine derivatives with amide/ester linkages
- Thiophene compounds with hydroxyl/amino groups
- Simple heterocycles with balanced properties
- Phenol derivatives with moderate substitution

Avoid:
- Very large molecules (MW > 500)
- Highly lipophilic compounds (LogP > 5)
- Molecules with many rotatable bonds
- Reactive groups (aldehydes, epoxides, etc.)
- Multiple aromatic rings without polarity""",
    "target_smiles": None
},
"ranolazine_mpo": {
    "prompt": """Design molecules similar to ranolazine while optimizing specific physicochemical properties.

Target: Ranolazine (COc1ccccc1OCC(O)CN2CCN(CC(=O)Nc3c(C)cccc3C)CC2)
- Anti-anginal agent for chronic stable angina
- Key features: methoxyphenyl ether, piperazine ring, secondary alcohol, amide linkage, dimethylaniline

Requirements:
- Maintain structural similarity to ranolazine
- Target TPSA around 95 Ų (topological polar surface area)
- Aim for LogP around 7 (high lipophilicity)
- Include approximately 1 fluorine atom
- Don't copy ranolazine exactly

Critical structural elements:
- Methoxyphenyl ether system
- Piperazine ring (6-membered N-heterocycle)
- Secondary alcohol group
- Amide bond connecting to aromatic ring
- Dimethylaniline moiety
- Flexible alkyl chain linkers

Design strategies for property optimization:
- Maintain core scaffold for similarity
- Introduce fluorine for metabolic stability and lipophilicity
- Balance polar groups (TPSA ~95) with high LogP (~7)
- Consider fluorinated aromatic rings
- Modify substituents while preserving key pharmacophore

Example modifications:
- Replace one methoxy with fluorine atom
- Add fluorine to dimethylaniline ring
- Introduce fluorinated alkyl chains
- Replace hydroxyl with fluorinated groups
- Modify piperazine with fluorinated substituents

Property considerations:
- High LogP (7) requires significant lipophilic character
- TPSA ~95 needs balanced polar surface area
- Single fluorine atom for enhanced properties
- Maintain ranolazine's therapeutic profile

Fluorine incorporation strategies:
- Fluorinated aromatic rings (4-fluorophenyl, 3-fluorophenyl)
- Trifluoromethyl groups (but count as 3 F atoms)
- Single fluorine substitution on aromatic rings
- Fluorinated alkyl chains or ethers""",
    "target_smiles": "COc1ccccc1OCC(O)CN2CCN(CC(=O)Nc3c(C)cccc3C)CC2"
},
"scaffold_hop": {
    "prompt": """Design molecules by removing a specific scaffold while preserving key decorations (scaffold hopping).

Target: Scaffold hopping from pharmacophore CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C

Requirements:
- REMOVE scaffold matching SMARTS: [#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12 (quinazoline core)
- PRESERVE decoration matching SMARTS: [#6]-[#6]-[#6]-[#8]-[#6]~[#6]~[#6]~[#6]~[#6]-[#7]-c1ccc2ncsc2c1 (propoxy-benzothiazole chain)
- Maintain pharmacophore similarity (capped at 0.75)
- Create drug-like molecules with novel scaffolds

Scaffold to remove:
- Quinazoline core system (bicyclic N-heterocycle)
- This is the central scaffold that must be replaced

Decoration to preserve:
- Propoxy chain (CCCOc-)
- Benzothiazole moiety (-Nc1ccc2ncsc2c1)
- The connecting aromatic system
- Overall spatial arrangement of key pharmacophoric elements

Design strategies:
- Replace quinazoline with alternative heterocycles (pyrimidine, pyrazine, triazine)
- Use different bicyclic systems (benzimidazole, benzoxazole, quinoline)
- Try monocyclic alternatives (pyrimidine, triazine, pyridine)
- Maintain similar electronic properties and hydrogen bonding patterns
- Preserve spatial orientation of decorations

Example scaffold replacements:
- Pyrimidine core instead of quinazoline
- Benzimidazole system
- Pyrazine-based scaffolds
- Triazine derivatives
- Quinoline or isoquinoline variants

Key considerations:
- Maintain drug-like properties
- Preserve binding interactions through bioisosteric replacement
- Keep similar molecular size and shape
- Ensure synthetic feasibility of new scaffold
- Balance similarity (not too high, not too low)""",
    "target_smiles": "CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C"
},
"sitagliptin_mpo": {
    "prompt": """Design molecules similar to sitagliptin while matching exact molecular formula and physicochemical properties.

Target: Sitagliptin (Fc1cc(c(F)cc1F)CC(N)CC(=O)N3Cc2nnc(n2CC3)C(F)(F)F)
- DPP-4 inhibitor for type 2 diabetes
- Key features: trifluoromethyl group, triazole ring, primary amine, fluorinated aromatic ring, amide linkage

Requirements:
- Match exact molecular formula: C16H15F6N5O (no missing or extra atoms)
- Maintain structural similarity to sitagliptin (Tanimoto ~0.7)
- Target LogP similar to sitagliptin (~{round(sitagliptin_logP, 4)})
- Target TPSA similar to sitagliptin (~{round(sitagliptin_TPSA, 4)} Ų)
- Don't copy sitagliptin exactly

Critical structural elements:
- Trifluoromethyl group (CF3) - key for DPP-4 binding
- Triazole ring system (1,2,4-triazole)
- Primary amine group for activity
- Fluorinated aromatic ring (trifluorophenyl)
- Amide bond connecting fragments
- Exact atom counts: C16, H15, F6, N5, O1

Design constraints:
- Must contain exactly 6 fluorine atoms
- Must contain exactly 5 nitrogen atoms
- Must contain exactly 1 oxygen atom
- Maintain similar molecular complexity
- Preserve key pharmacophoric elements

Modification strategies:
- Rearrange fluorine atoms on aromatic rings
- Modify triazole ring position or substitution
- Change amide linkage position
- Alter alkyl chain connectivity
- Reposition functional groups while maintaining formula

Example approaches:
- Move fluorines to different positions on phenyl ring
- Change triazole substitution pattern
- Modify amine position on alkyl chain
- Rearrange trifluoromethyl group location
- Alter connectivity between ring systems

Property considerations:
- LogP ~{round(sitagliptin_logP, 4)}: balance hydrophobic fluorines with polar groups
- TPSA ~{round(sitagliptin_TPSA, 4)}: maintain similar polar surface area
- Similarity ~0.7: moderate structural similarity without exact copying""",
    "target_smiles": "Fc1cc(c(F)cc1F)CC(N)CC(=O)N3Cc2nnc(n2CC3)C(F)(F)F"
},
"thiothixene_similarity": {
    "prompt": """Design molecules structurally similar to thiothixene while preserving core scaffold and pharmacophores.

Target: Thiothixene (CN(C)S(=O)(=O)c1ccc2Sc3ccccc3C(=CCCN4CCN(C)CC4)c2c1)
- Typical antipsychotic medication
- Key features: thioxanthene core, piperazine ring, sulfonamide group, alkene linker

Requirements:
- Maintain structural similarity to thiothixene
- Preserve core thioxanthene scaffold (dibenzothiopyran system)
- Keep key pharmacophoric elements for antipsychotic activity
- Don't copy thiothixene exactly
- Maintain drug-like properties

Critical structural elements:
- Thioxanthene core (tricyclic system with sulfur)
- Piperazine ring (N-methylpiperazine moiety)
- Sulfonamide group (dimethylsulfonamide)
- Alkene linker connecting fragments
- Aromatic substitution patterns

Key pharmacophoric features:
- Tricyclic aromatic system for dopamine receptor binding
- Basic nitrogen (piperazine) for receptor interaction
- Appropriate spatial arrangement of aromatic and basic centers
- Lipophilic character for CNS penetration

Possible modifications to consider:
- Alter piperazine substituents (ethyl instead of methyl)
- Modify sulfonamide group (different alkyl groups)
- Change alkene to alkyl chain
- Substitute aromatic rings with different groups
- Vary linker chain length
- Replace sulfur with oxygen (xanthene analogs)

Example strategies:
- N-ethylpiperazine instead of N-methylpiperazine
- Diethylsulfonamide instead of dimethylsulfonamide
- Saturated alkyl linker instead of alkene
- Halogenated aromatic rings
- Modified chain length between rings

Maintain antipsychotic profile:
- Dopamine D2 receptor antagonism
- Appropriate CNS penetration
- Similar molecular size and flexibility
- Balanced lipophilicity and solubility""",
    "target_smiles": "CN(C)S(=O)(=O)c1ccc2Sc3ccccc3C(=CCCN4CCN(C)CC4)c2c1"
},
"troglitazone_similarity": {
    "prompt": """Design molecules structurally similar to troglitazone while preserving core scaffold and pharmacophores.

Target: Troglitazone (Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O)
- PPAR-γ agonist for type 2 diabetes (withdrawn due to hepatotoxicity)
- Key features: thiazolidinedione ring, chroman core, phenolic hydroxyl, multiple methyl groups

Requirements:
- Maintain structural similarity to troglitazone
- Preserve core thiazolidinedione ring (essential for PPAR-γ activity)
- Keep chroman scaffold or similar bicyclic system
- Don't copy troglitazone exactly
- Maintain drug-like properties

Critical structural elements:
- Thiazolidinedione ring (2,4-thiazolidinedione) - essential pharmacophore
- Chroman core (benzopyran system) - provides rigidity and orientation
- Phenolic hydroxyl group - contributes to binding
- Aromatic ether linkage connecting fragments
- Multiple methyl substituents on chroman ring

Key pharmacophoric features:
- Thiazolidinedione for PPAR-γ receptor binding
- Aromatic systems for hydrophobic interactions
- Hydroxyl group for hydrogen bonding
- Appropriate spatial arrangement of polar and nonpolar regions
- Conformational flexibility in linker region

Possible modifications to consider:
- Alter methyl substitution patterns on chroman ring
- Replace some methyl groups with ethyl or other alkyl groups
- Modify the aromatic ether linker length
- Substitute phenolic OH with other polar groups
- Change chroman to related bicyclic systems (benzofuran, indane)
- Vary thiazolidinedione substitution (but keep the ring intact)

Example strategies:
- Ethyl groups instead of some methyl groups
- Different substitution pattern on chroman ring  
- Modified linker chain length
- Hydroxyl group in different position
- Benzofuran instead of chroman core
- Additional polar substituents for improved safety profile

Safety considerations:
- Avoid structural features associated with hepatotoxicity
- Maintain PPAR-γ selectivity
- Consider metabolic stability improvements
- Balance efficacy with safety profile""",
    "target_smiles": "Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O"
},
"valsartan_smarts": {
    "prompt": """Design molecules containing a specific SMARTS pattern with optimized physicochemical properties.

Target SMARTS pattern: CN(C=O)Cc1ccc(c2ccccc2)cc1
- This represents a formamide group attached to a biphenylmethyl moiety
- Similar to structural elements found in valsartan (ARB antihypertensive)

Requirements:
- MUST contain the exact SMARTS pattern: CN(C=O)Cc1ccc(c2ccccc2)cc1
- Target LogP around 2.0 (moderate lipophilicity)
- Target TPSA around 95 Ų (appropriate polar surface area)
- Target Bertz complexity around 800 (moderate molecular complexity)
- Create drug-like molecules with this core motif

SMARTS pattern breakdown:
- CN(C=O): N-methylformamide group
- Cc1ccc(c2ccccc2)cc1: 4-phenylbenzyl group (biphenyl system with methyl linker)
- Combined: N-methyl-N-(4-phenylbenzyl)formamide core

Design strategies:
- Build around the required SMARTS motif as core structure
- Add substituents to achieve target LogP (~2.0)
- Include polar groups to reach TPSA (~95)
- Balance complexity to approach Bertz score (~800)
- Consider additional aromatic rings, heteroatoms, or functional groups

Property optimization approaches:
- LogP ~2.0: balance hydrophobic aromatic systems with polar groups
- TPSA ~95: include hydrogen bond donors/acceptors (OH, NH, C=O, etc.)
- Bertz ~800: moderate molecular complexity with multiple ring systems
- Maintain drug-like characteristics (MW 200-600, rotatable bonds <10)

Example extensions:
- Add carboxylic acid groups for polarity
- Include hydroxyl or amino substituents
- Incorporate additional heterocycles
- Add ester or amide linkages
- Consider tetrazole or other bioactive groups

Structural considerations:
- Preserve the biphenyl-formamide core exactly
- Extend with compatible functional groups
- Maintain synthetic feasibility
- Consider metabolic stability
- Balance all three property targets simultaneously""",
    "target_smiles": None
},
"zaleplon_similarity": {
    "prompt": """Design molecules similar to zaleplon while maintaining high structural similarity.

Target: Zaleplon (O=C(C)N(CC)C1=CC=CC(C2=CC=NC3=C(C=NN23)C#N)=C1)
- Non-benzodiazepine hypnotic for insomnia
- Key features: pyrazolopyrimidine core, acetamide group, ethyl substitution, nitrile group

Requirements:
- Maintain the pyrazolopyrimidine bicyclic system
- Preserve the amide linkage to phenyl ring
- Keep molecular formula C19H17N3O2 exactly
- Achieve high Tanimoto similarity using atom-pair fingerprints
- Don't copy zaleplon exactly

Example modifications:
- Vary alkyl groups on amide nitrogen
- Modify substituents on phenyl ring
- Change acetyl to other acyl groups
- Replace ethyl with other small alkyl groups""",
    "target_smiles": "O=C(C)N(CC)C1=CC=CC(C2=CC=NC3=C(C=NN23)C#N)=C1"
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
