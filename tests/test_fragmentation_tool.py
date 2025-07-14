import json
from tools.fragmentation_tool import FragmentationTool

def test_fragmentation_valid_smiles():
    tool = FragmentationTool()
    smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin

    result_json = tool._run(smiles)
    result = json.loads(result_json)

    assert "functional_groups_counts" in result
    assert "ring_count" in result
    assert "fragment_count" in result
    assert "fragments_smiles" in result
    # Check some reasonable value types
    assert isinstance(result["functional_groups_counts"], dict)
    assert isinstance(result["ring_count"], int)
    assert isinstance(result["fragment_count"], int)
    assert isinstance(result["fragments_smiles"], list)

def test_fragmentation_invalid_smiles():
    tool = FragmentationTool()
    invalid_smiles = "ABC"

    result_json = tool._run(invalid_smiles)
    result = json.loads(result_json)

    assert "error" in result
