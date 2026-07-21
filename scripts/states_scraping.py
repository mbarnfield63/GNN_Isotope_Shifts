"""
Ingest a single ExoMol .states file (one isotopologue, MARVEL source embedded
via a per-row Source-code flag) into this project's standard states CSV.
Column layout is read per-molecule from its ExoMol JSON .def file (see
src/def_reader.py) rather than assumed -- it varies with Hund's coupling case
and which optional fields (gfactor, lifetime) the linelist includes.

Usage:
    uv run python -m scripts.states_scraping \
        --molecule SiO --isotope 28Si16O \
        --source "C:/path/to/28Si-16O__SiOUVenIR.states" \
        --def-json "C:/path/to/28Si-16O__SiOUVenIR.def.json"
"""

import argparse
import os

import pandas as pd

from src.def_reader import read_states_schema
from src.masses import isotopologue_masses
from src.registry import get_molecule

# Band-key columns (ElecState/Omega/parity) are retained alongside the core
# physics columns when present, so the PS baseline can group by vibronic
# band. Omega is absent for Hund's case (b) molecules (e.g. O2, C2) -- those
# fall back to grouping by (ElecState, v, parity) alone, same degraded path
# already used for legacy-ingested molecules (see dataset.py).
OUTPUT_COLUMNS = [
    "mass_A",
    "mass_B",
    "reduced_mass",
    "v",
    "J",
    "ECalc",
    "EMarv",
    "unc",
    "Ediff",
    "ElecState",
    "Omega",
    "parity",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecule", required=True, help="Registry molecule name, e.g. SiO")
    parser.add_argument("--isotope", required=True, help="Isotopologue formula, e.g. 28Si16O")
    parser.add_argument("--source", required=True, help="Path to the raw ExoMol .states file")
    parser.add_argument("--def-json", required=True, help="Path to the raw ExoMol JSON .def file")
    parser.add_argument("--output-dir", default="data/states")
    parser.add_argument("--registry", default="configs/molecules.yaml")
    return parser.parse_args()


def process_states(
    molecule: str, isotope: str, source_path: str, def_json_path: str, output_dir: str, registry_path: str
):
    mol_entry = get_molecule(molecule, registry_path)
    if mol_entry["electronic_state"] is None or mol_entry["marvel_source_code"] is None:
        raise ValueError(
            f"{molecule}: electronic_state/marvel_source_code not set in {registry_path} -- "
            "fill these in from the molecule's ExoMol .def file before ingesting."
        )

    os.makedirs(output_dir, exist_ok=True)

    column_names = read_states_schema(def_json_path)
    df = pd.read_csv(source_path, sep=r"\s+", names=column_names, header=None)

    df = df[df["ElecState"] == mol_entry["electronic_state"]].copy()
    df.loc[df["Source"] != mol_entry["marvel_source_code"], "EMarv"] = pd.NA

    masses = isotopologue_masses(isotope)
    df = df.assign(**masses)
    df["Ediff"] = df["EMarv"] - df["ECalc"]

    out_df = df[[c for c in OUTPUT_COLUMNS if c in df.columns]]
    save_path = os.path.join(output_dir, f"{isotope}.csv")
    out_df.to_csv(save_path, index=False)
    print(f"Saved {len(out_df)} states to {save_path}")


if __name__ == "__main__":
    args = parse_args()
    process_states(args.molecule, args.isotope, args.source, args.def_json, args.output_dir, args.registry)
