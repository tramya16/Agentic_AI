import json
from tools.toxicity_check_tool import ToxicityCheckTool

def test_non_toxic_molecule():
    tool = ToxicityCheckTool()
    result = json.loads(tool._run("CCO"))
    assert not result.get("pains_alert", False)

def test_toxic_pains_molecule():
    tool = ToxicityCheckTool()
    # A PAINS substructure — should trigger alert
    result = json.loads(tool._run("c1ccccc1N=Nc2ccccc2"))
    assert result.get("pains_alert", False)
