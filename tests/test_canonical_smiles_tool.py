import json
from tools.canonical_smiles_tool import CanonicalSmilesTool

tool = CanonicalSmilesTool()

def test_valid_smiles_input():
    smiles = "CCO"
    output_json = tool._run(smiles, "smiles")
    output = json.loads(output_json)
    assert output["status"] == "success"
    assert output["smiles"] == "CCO" or output["smiles"] == "OCC"  # canonical form could reorder

def test_valid_name_input():
    name = "acetone"
    output_json = tool._run(name, "name")
    output = json.loads(output_json)
    assert output["status"] == "success"
    assert "smiles" in output

def test_invalid_input():
    invalid = "not_a_molecule"
    output_json = tool._run(invalid, "auto")
    output = json.loads(output_json)
    assert output["status"] == "error"

def test_empty_input():
    empty = ""
    output_json = tool._run(empty, "auto")
    print(output_json)
    output = json.loads(output_json)
    assert output["status"] == "error"
