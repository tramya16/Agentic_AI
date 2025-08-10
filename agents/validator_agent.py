import json
from crewai import Agent, Task
from llms.model_loader import load_llm
from tools.tool_registry import VALIDATOR_TOOLS


def create_validator_agent(model_config=None, llm_seed: int = None):
    llm = load_llm(model_config=model_config, seed=llm_seed)
    return Agent(
        role="Molecular Validation Specialist",
        goal="Apply comprehensive validation filters to ensure molecular quality, compliance, and drug-likeness",
        backstory="""You are a computational chemistry expert specializing in molecular validation, 
        ADMET prediction, and drug-likeness assessment. You apply systematic filters to ensure 
        generated molecules meet quality standards for drug discovery while maintaining reasonable 
        acceptance criteria to avoid over-filtering promising candidates.""",
        tools=VALIDATOR_TOOLS,
        verbose=False,
        llm=llm,
        allow_delegation=False
    )


def create_validation_task(candidates: list, parsed_spec: str, agent: Agent):
    return Task(
        description=f"""
Perform comprehensive validation of {len(candidates)} molecular candidates against design specifications.

MOLECULAR CANDIDATES: {json.dumps(candidates)}
DESIGN SPECIFICATION: {parsed_spec}

VALIDATION PROTOCOL:
1. SMILES Validity Check:
   - Verify each SMILES can be parsed by RDKit
   - Check for proper chemical valence and bonding
   - Identify and flag malformed structures

2. Duplicate Detection:
   - Remove exact duplicates within the candidate set
   - Check against reference molecules to avoid target replication
   - Use canonical SMILES for accurate comparison

3. Drug-likeness Assessment (Relaxed Criteria):
   - Apply Lipinski's Rule of Five with reasonable flexibility
   - Check molecular weight (150-600 Da, flexible boundaries)
   - Assess LogP, HBD, HBA with context-appropriate ranges
   - Consider TPSA and rotatable bonds

4. Basic Safety Screening:
   - Flag only severe toxicity concerns (reactive groups, known toxicophores)
   - Apply PAINS filters selectively to avoid over-filtering
   - Check for problematic functional groups in drug discovery context

5. Specification Compliance:
   - Verify adherence to structural requirements
   - Check property constraints from design specification
   - Validate against similarity requirements if specified

VALIDATION CRITERIA:
- Use RELAXED acceptance criteria to preserve promising candidates
- Only reject molecules with clear validity or safety issues
- Provide specific reasons for rejection to enable learning
- Maintain scientific rigor while avoiding over-conservative filtering

OUTPUT FORMAT - Return valid JSON:
{{
  "valid": ["list_of_valid_smiles_strings"],
  "invalid": ["list_of_invalid_smiles_strings"],
  "validation_details": {{
    "total_processed": {len(candidates)},
    "duplicates_found": ["duplicate_smiles"],
    "target_matches": ["molecules_matching_reference"],
    "invalid_smiles": ["chemically_invalid_structures"],
    "drug_likeness_failures": ["molecules_failing_drug_likeness"],
    "toxicity_flags": ["molecules_with_safety_concerns"],
    "specification_violations": ["molecules_not_meeting_requirements"],
    "validation_summary": "brief_summary_of_validation_results"
  }}
}}

IMPORTANT GUIDELINES:
- Apply validation filters judiciously - err on the side of inclusion for borderline cases
- Provide clear, specific reasons for molecule rejection
- Ensure all valid molecules meet basic chemical validity standards
- Balance quality control with preservation of structural diversity

RETURN: Valid JSON with comprehensive validation results and detailed reasoning.
""",
        agent=agent,
        expected_output="Valid JSON with validated molecules and comprehensive validation analysis"
    )
