# Data Sources

- **Amoxicillin SMILES**  
  - **Source**: PubChem REST API  
  - **CID**: 33613  
  - **Endpoint**:  
    `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/33613/property/IsomericSMILES/TXT`  
  - **Retrieved on**: 2025-05-30

To update the file, run:
```bash
python scripts/fetch_amoxicillin_smiles.py
