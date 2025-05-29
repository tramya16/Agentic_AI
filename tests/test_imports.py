import pytest

def test_import_rdkit():
    import rdkit
    # Basic sanity: RDKit has Chem module
    from rdkit import Chem
    assert hasattr(Chem, 'MolFromSmiles')

def test_import_streamlit():
    import streamlit as st
    # Confirm core functions exist
    assert hasattr(st, 'write')
    assert hasattr(st, 'title')

def test_import_crewai():
    import crewai
    # At minimum the SDK module should load
    assert hasattr(crewai, '__version__'), "crewai should have __version__ attribute"
