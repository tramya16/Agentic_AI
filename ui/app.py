import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
from chemtools.scorer import get_all_scores
from io import BytesIO

# --- Helpers ---
@st.cache_data
def load_amoxicillin_smiles(path="data/amoxicillin.smi"):
    try:
        with open(path) as f:
            return f.read().strip()
    except FileNotFoundError:
        st.error(f"SMILES file not found at {path}")
        return ""

def mol_to_image_bytes(mol, size=(300, 300)):
    """Return PNG bytes for an RDKit molecule."""
    img = Draw.MolToImage(mol, size=size)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# --- Page Config ---
st.set_page_config(page_title="Molecular Optimizer", layout="centered")

# --- Load Default SMILES ---
default_smiles = load_amoxicillin_smiles()
st.title("Agentic AI Molecular Optimizer — Week 1 Day 3")
st.markdown(
    "Paste any SMILES string below to view its 2D structure. "
    "By default, we load **Amoxicillin**."
)

# --- User Input ---
smiles_input = st.text_input("Enter SMILES", value=default_smiles)

# --- Molecule Rendering ---
if smiles_input:
    mol = Chem.MolFromSmiles(smiles_input)
    if mol:
        img_bytes = mol_to_image_bytes(mol)
        st.image(img_bytes, caption=f"Structure for SMILES: `{smiles_input}`")
          # --- Scoring Section ---
        st.subheader("Baseline Descriptor Scores")

        try:
            scores = get_all_scores(smiles_input)
            st.table(scores.items())
        except ValueError as e:
            st.error(str(e))

    else:
        st.error("❌ Invalid SMILES. Please check your input.")
