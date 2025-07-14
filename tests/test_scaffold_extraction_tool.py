import json
from tools.scaffold_extraction_tool import ScaffoldExtractionTool

def test_scaffold_extraction_valid_smiles():
    tool = ScaffoldExtractionTool()
    smiles = "CCOc1ccc2nc(SCc3ccccc3)sc2c1"  # Example molecule

    result_json = tool._run(smiles)
    result = json.loads(result_json)

    assert "scaffold_smiles" in result
    assert isinstance(result["scaffold_smiles"], str)
    assert len(result["scaffold_smiles"]) > 0

def test_scaffold_extraction_invalid_smiles():
    tool = ScaffoldExtractionTool()
    invalid_smiles = "XYZ123"

    result_json = tool._run(invalid_smiles)
    result = json.loads(result_json)

    assert "error" in result
