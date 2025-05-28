# Agentic AI System for SMILES-Based Molecular Optimization

## Overview
A prototype multi-agent pipeline that generates, validates, and optimizes small molecules (SMILES) using a modular set of AI agents. Built with CrewAI, RDKit, and HuggingFace, the system focuses on Amoxicillin as a case study and provides an interactive Streamlit UI for real-time feedback.

## Features
- **Modular Agents**  
  - **Generator**: LLM-driven SMILES editor  
  - **Validator**: RDKit-based chemical validity checks  
  - **Scorer**: ADMET property estimation via TDC/HuggingFace  
  - **Critic**: LLM-driven evaluation & next-step guidance  
- **Interactive Streamlit UI**  
  - Input arbitrary SMILES  
  - Visualize 2D structures  
  - Display key ADMET metrics  
- **Logging & Exports**  
  - Session logging of all generations and scores  
  - CSV & image export support

## Week 1 Deliverables
- Project folder structure:
  ```text
  agents/
  chemtools/
  ui/
  crew/
  data/
  notebooks/

* Python virtual environment setup
* `requirements.txt` with core dependencies
* Basic Streamlit app stub in `ui/app.py`
* Initial CI smoke-test configuration (import checks)
* Amoxicillin SMILES stored in `data/amoxicillin.smi`

## Setup Instructions

1. **Clone the repo**

   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```

2. **Create & activate a virtual environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit UI**

   ```bash
   streamlit run ui/app.py
   ```

## Next Steps

* **Week 2**: Develop the Generator Agent to produce SMILES edits via an LLM
* **Week 3**: Build the Validator Agent and integrate the basic pipeline
* **Week 4**: Add the Critic Agent and refine multi-agent workflow




