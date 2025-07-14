# tests/test_similar_molecule_finder.py
import pytest
import json
from tools.similar_molecule_finder import SimilarMoleculeFinderTool


@pytest.fixture
def finder():
    return SimilarMoleculeFinderTool()


@pytest.mark.parametrize(
    "smiles, similarity_threshold, max_results, expect_success",
    [
        ("CCO", 0.7, 3, True),  # ethanol, expect success
        ("C1CCCCC1", 0.8, 5, True),  # cyclohexane, expect success
        ("invalidsmiles", 0.7, 5, False),  # invalid SMILES
        ("", 0.7, 5, False),  # empty input
        ("CCO", 1.1, 5, False),  # invalid similarity threshold >1 (should handle gracefully)
        ("CCO", 0.7, 0, False),  # invalid max_results = 0
    ]
)
def test_find_similar(finder, smiles, similarity_threshold, max_results, expect_success):
    response_json = finder._run(smiles, similarity_threshold, max_results)
    response = json.loads(response_json)

    if expect_success:
        assert response["status"] == "success"
        assert "results" in response
        assert isinstance(response["results"], list)
        # Each result must have cid, smiles, name, similarity keys
        for res in response["results"]:
            assert "cid" in res
            assert "smiles" in res
            assert "name" in res
            assert "similarity" in res
            assert 0 <= res["similarity"] <= 1
    else:
        assert response["status"] == "error"

