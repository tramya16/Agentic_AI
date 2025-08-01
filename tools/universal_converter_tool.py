import json
from typing import Type, Literal

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from utils.chemistry_utils import resolve_to_smiles, get_iupac_name_from_smiles, \
    get_common_name_from_smiles


class UniversalConverterInput(BaseModel):
    input_value: str = Field(..., description="Chemical input (name, SMILES, CAS, InChI, SELFIES, etc.)")
    input_type: Literal["auto", "name", "smiles", "cas", "inchi", "selfies", "cid"] = Field(
        default="auto", description="Type of input")
    output_type: Literal["smiles", "selfies", "inchi", "iupac_name", "common_name"] = Field(
        default="smiles", description="Desired output type")

class UniversalConverterTool(BaseTool):
    name:str = "universal_converter"
    description:str = "Convert chemical identifiers interchangeably between SMILES, SELFIES, InChI, names, CAS, CID, etc."
    args_schema: Type[BaseModel] = UniversalConverterInput

    def _run(self, input_value: str, input_type: str = "auto", output_type: str = "smiles") -> str:
        # Handle SELFIES output separately
        try:
            # First resolve input to SMILES if output is not selfies or smiles
            if output_type in ["smiles", "iupac_name", "name", "common_name", "inchi"]:
                if input_type != "smiles":
                    resolved = resolve_to_smiles(input_value, input_type)
                    if resolved["status"] != "success":
                        return json.dumps(resolved)
                    smiles = resolved["smiles"]
                else:
                    smiles = input_value

                if output_type == "smiles":
                    return json.dumps({"status": "success", "output": smiles})

                elif output_type == "inchi":
                    from rdkit import Chem
                    mol = Chem.MolFromSmiles(smiles)
                    if not mol:
                        return json.dumps({"status": "error", "error": "Failed to convert SMILES to molecule"})
                    inchi = Chem.MolToInchi(mol)
                    return json.dumps({"status": "success", "output": inchi})

                elif output_type in ["iupac_name", "common_name","name"]:
                    if input_type != "smiles":
                        resolved = resolve_to_smiles(input_value, input_type)
                        if resolved["status"] != "success":
                            return json.dumps(resolved)
                        smiles = resolved["smiles"]
                    else:
                        smiles = input_value

                    if output_type == "iupac_name":
                        res = get_iupac_name_from_smiles(smiles)
                    else:
                        res = get_common_name_from_smiles(smiles)
                    return json.dumps(res)


            elif output_type == "selfies":
                if input_type != "smiles":
                    resolved = resolve_to_smiles(input_value, input_type)
                    if resolved["status"] != "success":
                        return json.dumps(resolved)
                    smiles = resolved["smiles"]
                else:
                    smiles = input_value

                from utils.chemistry_utils import smiles_to_selfies
                res = smiles_to_selfies(smiles)
                return json.dumps(res)

            else:
                return json.dumps({"status": "error", "error": f"Unsupported output_type: {output_type}"})

        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})
