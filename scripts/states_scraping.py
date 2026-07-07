"""
Ingest a single ExoMol .states file (one isotopologue, MARVEL source embedded
via a per-row Source-code flag) into this project's standard states CSV.

Usage:
    uv run python -m scripts.states_scraping \
        --molecule SiO --isotope 28Si16O \
        --source "C:/path/to/28Si-16O__SiOUVenIR.states"
"""

import argparse
import os

import pandas as pd

from src.masses import isotopologue_masses
from src.registry import get_molecule

# ExoMol .states files are whitespace-separated with no header; columns beyond
# these core ones vary by molecule (see the associated .def file).
FULL_COLUMN_NAMES = [
    "ID",
    "E",
    "gtot",
    "J",
    "unc",
    "tau",
    "gfactor",
    "parity",
    "rotlessparity",
    "hunda:ElectronicState",
    "hunda:v",
    "hunda:Lambda",
    "hunda:Sigma",
    "hunda:Omega",
    "Source",
    "Ecalc",
]

RENAMING_MAP = {
    "E": "EMarv",
    "Ecalc": "ECalc",
    "hunda:v": "v",
    "hunda:ElectronicState": "ElecState",
    "hunda:Omega": "Omega",
}

# Band-key columns (ElecState/Omega/parity) are always retained alongside the
# core physics columns, so the PS baseline can group by vibronic band for any
# molecule uniformly.
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
    parser.add_argument("--output-dir", default="data/states")
    parser.add_argument("--registry", default="configs/molecules.yaml")
    return parser.parse_args()


def process_states(molecule: str, isotope: str, source_path: str, output_dir: str, registry_path: str):
    mol_entry = get_molecule(molecule, registry_path)
    if mol_entry["electronic_state"] is None or mol_entry["marvel_source_code"] is None:
        raise ValueError(
            f"{molecule}: electronic_state/marvel_source_code not set in {registry_path} -- "
            "fill these in from the molecule's ExoMol .def file before ingesting."
        )

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(source_path, sep=r"\s+", names=FULL_COLUMN_NAMES, header=None)
    df = df.rename(columns=RENAMING_MAP)

    df = df[df["ElecState"] == mol_entry["electronic_state"]].copy()
    df.loc[df["Source"] != mol_entry["marvel_source_code"], "EMarv"] = pd.NA

    masses = isotopologue_masses(isotope)
    df = df.assign(**masses)
    df["Ediff"] = df["EMarv"] - df["ECalc"]

    out_df = df[OUTPUT_COLUMNS]
    save_path = os.path.join(output_dir, f"{isotope}.csv")
    out_df.to_csv(save_path, index=False)
    print(f"Saved {len(out_df)} states to {save_path}")


if __name__ == "__main__":
    args = parse_args()
    process_states(args.molecule, args.isotope, args.source, args.output_dir, args.registry)
