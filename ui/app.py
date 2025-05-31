import streamlit as st
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw
from io import BytesIO

from agents.generator_agent import GeneratorAgent
from agents.validator_agent import ValidatorAgent
from chemtools.scorer import get_all_scores
from dotenv import load_dotenv

load_dotenv()


# --- Helpers ---
def mol_to_image_bytes(mol, size=(350, 300)):
    img = Draw.MolToImage(mol, size=size)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@st.cache_data
def load_base_smiles(path="data/amoxicillin.smi"):
    return Path(path).read_text().strip()


# --- Page Config ---
st.set_page_config(
    page_title="Molecular Optimizer",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    :root {
        --primary: #2c3e50;
        --secondary: #3498db;
        --light: #ecf0f1;
        --dark: #2c3e50;
    }
    .header-style {
        font-size: 32px;
        font-weight: 700;
        color: var(--primary);
        padding-bottom: 10px;
        border-bottom: 3px solid var(--secondary);
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 15px;
        border-left: 4px solid var(--secondary);
    }
    .stButton>button {
        background-color: var(--secondary) !important;
        color: white !important;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: var(--primary) !important;
        transform: scale(1.03);
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<p class="header-style">Molecular Optimizer</p>', unsafe_allow_html=True)
st.caption("Agentic AI Pharmaceutical Chemistry Suite")

# Info section
with st.expander("ℹ️ About this tool", expanded=True):
    st.markdown("""
    - **Visualize** molecular structures from SMILES strings
    - **Generate** optimized molecules using AI agents
    - **Analyze** key pharmaceutical properties
    - **Compare** against baseline compounds
    """)
    st.markdown("Default compound: **Amoxicillin** (β-lactam antibiotic)")

# Load baseline
base_smiles = load_base_smiles()
st.subheader("Baseline Molecule: Amoxicillin")

# Display baseline molecule and properties
col1, col2 = st.columns(2)

with col1:
    mol0 = Chem.MolFromSmiles(base_smiles)
    st.image(mol_to_image_bytes(mol0), use_container_width=True)
    with st.expander("View SMILES Notation"):
        st.code(base_smiles)

with col2:
    st.subheader("Molecular Properties")
    try:
        scores = get_all_scores(base_smiles)
        # Display properties in cards
        cols = st.columns(3)
        for i, (k, v) in enumerate(scores.items()):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:14px; color:#7f8c8d">{k}</div>
                    <div style="font-size:24px; font-weight:bold; color:#2c3e50">{v:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error calculating properties: {str(e)}")

st.markdown("---")

# --- Generation Section ---
st.subheader("Molecule Optimization")

# Inputs in columns
col_left, col_right = st.columns([3, 1])
with col_left:
    objective = st.text_input(
        "Optimization Objective",
        value="reduce toxicity and increase solubility",
        help="Describe your desired property improvements.",
        placeholder="e.g., increase bioavailability, reduce side effects"
    )
with col_right:
    st.write("")  # Vertical spacer
    generate_btn = st.button("Generate Optimized Molecule", use_container_width=True)

if generate_btn:
    gen_agent = GeneratorAgent()
    val_agent = ValidatorAgent()

    with st.spinner("Generating molecule with AI agents..."):
        candidate = gen_agent.generate(base_smiles, objective)

    st.markdown(f"**Generated SMILES:** `{candidate}`")

    if not val_agent.validate(candidate):
        st.error("Generated SMILES is chemically invalid. Please try again.")
    else:
        # Display generated molecule
        can_smiles = val_agent.sanitize(candidate)
        mol1 = Chem.MolFromSmiles(can_smiles)

        gen_col1, gen_col2 = st.columns(2)

        with gen_col1:
            st.subheader("Optimized Structure")
            st.image(mol_to_image_bytes(mol1), use_container_width=True)
            with st.expander("View SMILES Notation"):
                st.code(can_smiles)

        with gen_col2:
            st.subheader("Molecular Properties")
            try:
                scores = get_all_scores(can_smiles)
                # Display properties in cards
                cols = st.columns(3)
                for i, (k, v) in enumerate(scores.items()):
                    with cols[i % 3]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="font-size:14px; color:#7f8c8d">{k}</div>
                            <div style="font-size:24px; font-weight:bold; color:#2c3e50">{v:.3f}</div>
                        </div>
                        """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error calculating properties: {str(e)}")

        # Action buttons
        st.download_button("Download Structure", can_smiles, file_name="optimized_molecule.smi")
        if st.button("Generate Another Variation"):
            st.experimental_rerun()

# Footer
st.markdown("---")
st.caption("© 2023 Agentic AI Research | Pharmaceutical Chemistry Suite v1.0")