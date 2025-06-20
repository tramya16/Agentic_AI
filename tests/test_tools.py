from tools.chem_tools import PubChemLookupTool, Smilesvalidatortool

def test_pubChemTool():
    """Test individual tools to ensure they work correctly."""
    print("Testing individual tools...")
    # Test PubChem lookup
    print("\n Testing PubChem lookup for aspirin...")
    pubchem_tool = PubChemLookupTool()
    pubchem_result = pubchem_tool._run("aspirin", "name")
    print(pubchem_result)