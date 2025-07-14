# tool_registry.py

# --- Imports ---

# Validation & Conversion Tools
from tools.smiles_validator import SmilesValidatorTool
from tools.universal_converter_tool import UniversalConverterTool
from tools.canonical_smiles_tool import CanonicalSmilesTool
from tools.duplicate_check_tool import DuplicateCheckTool
from tools.patent_check_tool import PatentCheckTool
from tools.molecular_fingerprint_tool import MolecularFingerprintTool

# Similarity & Lookup Tools
from tools.similar_molecule_finder import SimilarMoleculeFinderTool
from tools.similarity_search_tool import SimilaritySearchTool
from tools.properties_lookup import PropertiesLookupTool

# Descriptor Tools
from tools.topological_descriptor_tool import TopologicalDescriptorTool
from tools.electronic_descriptor_tool import ElectronicDescriptorTool
from tools.shape_descriptor_tool import ShapeDescriptorTool
from tools.complexity_calculator_tool import ComplexityCalculatorTool
from tools.fragmentation_tool import FragmentationTool
from tools.ring_analysis_tool import RingAnalysisTool

# Drug-likeness & Safety Tools
from tools.drug_likeness_validator_tool import DrugLikenessValidatorTool
from tools.toxicity_check_tool import ToxicityCheckTool
from tools.bbb_predictor import BBBPermeantPredictionTool
from tools.formula_calculator_tool import FormulaCalculatorTool

# Scaffold & SELFIES Tools
from tools.scaffold_extraction_tool import ScaffoldExtractionTool
from tools.selfies_converter_tool import SelfiesConverterTool

# Wikipedia
from tools.wikipedia_search_tool import WikipediaSearchTool

# SA Score
from tools.calculate_synthetic_accessibility import CalculateSA


# --- Registry ---

ALL_TOOLS = {
    "validation": [
        SmilesValidatorTool(),
        DuplicateCheckTool(),
        PatentCheckTool(),
        CanonicalSmilesTool(),
    ],
    "conversion": [
        UniversalConverterTool(),
        SelfiesConverterTool(),
        MolecularFingerprintTool()
    ],
    "similarity": [
        SimilarMoleculeFinderTool(),
        SimilaritySearchTool(),
    ],
    "lookup": [
        PropertiesLookupTool(),
        WikipediaSearchTool(),
    ],
    "descriptors": [
        TopologicalDescriptorTool(),
        ElectronicDescriptorTool(),
        ShapeDescriptorTool(),
        ComplexityCalculatorTool(),
        FragmentationTool(),
        RingAnalysisTool(),
        FormulaCalculatorTool(),
    ],
    "drug_likeness_and_safety": [
        DrugLikenessValidatorTool(),
        ToxicityCheckTool(),
        BBBPermeantPredictionTool(),
        CalculateSA(),
    ],
    "scaffold": [
        ScaffoldExtractionTool(),
    ]
}


# Optional: flattened list of all tools
ALL_TOOLS_FLAT = [tool for group in ALL_TOOLS.values() for tool in group]


AGENT_TOOLS = {
    # ---------------- Parser Agent -----------------------------------------
    "parser": [
        UniversalConverterTool(),
        CanonicalSmilesTool(),
        SmilesValidatorTool(),
        DuplicateCheckTool(),
        PatentCheckTool(),
        PropertiesLookupTool(),
    ],

    # ---------------- Property Agent ---------------------------------------
    "property": [
        TopologicalDescriptorTool(),
        ElectronicDescriptorTool(),
        ShapeDescriptorTool(),
        ComplexityCalculatorTool(),
        FormulaCalculatorTool(),
        CalculateSA(),
        DrugLikenessValidatorTool(),
        ToxicityCheckTool(),
        BBBPermeantPredictionTool(),
    ],

    # ---------------- Fragment Agent ---------------------------------------
    "fragment": [
        FragmentationTool(),
        RingAnalysisTool(),
        #BioisostereGenerator
    ],

    # ---------------- Scaffold Agent ---------------------------------------
    "scaffold": [
        ScaffoldExtractionTool(),
        MolecularFingerprintTool()
    ],

    # ---------------- Generator Agent --------------------------------------
    "generator": [
        SimilarMoleculeFinderTool(),
        SimilaritySearchTool(),
        SelfiesConverterTool(),
        UniversalConverterTool(),
        DuplicateCheckTool()
    ],

    # ---------------- Validator Agent --------------------------------------
    "validator": [
        SmilesValidatorTool(),
        DrugLikenessValidatorTool(),
        ToxicityCheckTool(),
        BBBPermeantPredictionTool(),
        DuplicateCheckTool(),
        PatentCheckTool(),
    ],

}

PARSER_TOOLS = AGENT_TOOLS["parser"]
PROPERTY_TOOLS = AGENT_TOOLS["property"]
FRAGMENT_TOOLS = AGENT_TOOLS["fragment"]
SCAFFOLD_TOOLS = AGENT_TOOLS["scaffold"]
GENERATOR_TOOLS = AGENT_TOOLS["generator"]
VALIDATOR_TOOLS = AGENT_TOOLS["validator"]
