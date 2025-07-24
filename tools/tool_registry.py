# tools/tool_registry.py

# --- Imports ---
from tools.smiles_validator import SmilesValidatorTool
from tools.universal_converter_tool import UniversalConverterTool
from tools.canonical_smiles_tool import CanonicalSmilesTool
from tools.duplicate_check_tool import DuplicateCheckTool
from tools.patent_check_tool import PatentCheckTool
from tools.molecular_fingerprint_tool import MolecularFingerprintTool
from tools.similar_molecule_finder import SimilarMoleculeFinderTool
from tools.similarity_search_tool import SimilaritySearchTool
from tools.properties_lookup import PropertiesLookupTool
from tools.topological_descriptor_tool import TopologicalDescriptorTool
from tools.electronic_descriptor_tool import ElectronicDescriptorTool
from tools.shape_descriptor_tool import ShapeDescriptorTool
from tools.complexity_calculator_tool import ComplexityCalculatorTool
from tools.fragmentation_tool import FragmentationTool
from tools.ring_analysis_tool import RingAnalysisTool
from tools.drug_likeness_validator_tool import DrugLikenessValidatorTool
from tools.toxicity_check_tool import ToxicityCheckTool
from tools.bbb_predictor import BBBPermeantPredictionTool
from tools.formula_calculator_tool import FormulaCalculatorTool
from tools.scaffold_extraction_tool import ScaffoldExtractionTool
from tools.selfies_converter_tool import SelfiesConverterTool
from tools.wikipedia_search_tool import WikipediaSearchTool
from tools.calculate_synthetic_accessibility import CalculateSA
from tools.molecular_formula_tool import MolecularFormulaValidatorTool
from tools.smarts_pattern_tool import SmartsPatternTool

# --- Agent-Specific Tool Sets ---

PARSER_TOOLS = [
    UniversalConverterTool(),
    CanonicalSmilesTool(),
    SmilesValidatorTool(),
    PropertiesLookupTool(),
    WikipediaSearchTool(),
]

GENERATOR_TOOLS = [
    SimilarMoleculeFinderTool(),
    SimilaritySearchTool(),
    DuplicateCheckTool(),
    UniversalConverterTool(),
    SelfiesConverterTool(),
    CanonicalSmilesTool(),
    SmilesValidatorTool(),
    ScaffoldExtractionTool(),
    MolecularFormulaValidatorTool(),
    SmartsPatternTool(),
]

VALIDATOR_TOOLS = [
    SmilesValidatorTool(),
    CanonicalSmilesTool(),
    DuplicateCheckTool(),
    DrugLikenessValidatorTool(),
    ToxicityCheckTool(),
    PatentCheckTool(),
    ComplexityCalculatorTool(),
    MolecularFormulaValidatorTool(),
    SmartsPatternTool(),
]

CRITIC_TOOLS = [
    TopologicalDescriptorTool(),
    ElectronicDescriptorTool(),
    ShapeDescriptorTool(),
    ComplexityCalculatorTool(),
    FormulaCalculatorTool(),
    CalculateSA(),
    DrugLikenessValidatorTool(),
    ToxicityCheckTool(),
    BBBPermeantPredictionTool(),
    SimilaritySearchTool(),
    FragmentationTool(),
    RingAnalysisTool(),
]

# Legacy support
ALL_TOOLS_FLAT = PARSER_TOOLS + GENERATOR_TOOLS + VALIDATOR_TOOLS + CRITIC_TOOLS