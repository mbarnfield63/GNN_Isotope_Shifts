"""
Ingest a molecule's isotopologues from separate DUO calculated-energy output
files and MARVEL experimental-energy files (the path used historically for
CO, where ExoMol didn't ship a single combined .states file with an embedded
MARVEL-source flag).

Usage:
    uv run python -m scripts.combining_duo_marv \
        --molecule CO --duo-dir "C:/path/to/duo_outputs" --marvel-dir "C:/path/to/marvel"

Expects filenames of the form "{molecule}{iso_id}_output_duo.out" and
"MARVEL_Energies_{molecule}{iso_id}.txt", where iso_id is ExoMol's standard
isotopologue ID (see src.masses.exomol_isotope_id).
"""

import argparse
import os

import pandas as pd

from src.masses import isotopologue_masses
from src.registry import get_molecule

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
    parser.add_argument("--molecule", required=True, help="Registry molecule name, e.g. CO")
    parser.add_argument("--duo-dir", required=True, help="Directory of *_output_duo.out files")
    parser.add_argument("--marvel-dir", required=True, help="Directory of MARVEL_Energies_*.txt files")
    parser.add_argument("--output-dir", default="data/states")
    parser.add_argument("--registry", default="configs/molecules.yaml")
    return parser.parse_args()


def parse_duo_output(filename: str) -> pd.DataFrame:
    with open(filename, "r") as f:
        lines = f.readlines()

    rows = []
    for line in lines:
        if not line.strip() or line.startswith("#") or line.startswith("*"):
            continue
        parts = line.split()
        if len(parts) < 11:
            continue
        try:
            rows.append(
                {
                    "J": int(float(parts[0])),
                    "ECalc": float(parts[2]),
                    "ElecState": int(parts[3]),
                    "v": int(parts[4]),
                    "Omega": float(parts[8]),
                    "parity": parts[9],
                }
            )
        except (ValueError, IndexError):
            continue

    return pd.DataFrame(rows)


def parse_marvel(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename, sep=r"\s+", header=None)
    df.columns = ["v", "J", "EMarv", "unc", "N"]
    return df[["v", "J", "EMarv", "unc"]]


def combine_isotopologue(duo_path: str, marvel_path: str, isotope: str) -> pd.DataFrame:
    duo_df = parse_duo_output(duo_path)
    marvel_df = parse_marvel(marvel_path)

    combined = pd.merge(duo_df, marvel_df, on=["v", "J"], how="outer")
    combined = combined.assign(**isotopologue_masses(isotope))
    combined["Ediff"] = combined["EMarv"] - combined["ECalc"]

    return combined[OUTPUT_COLUMNS]


def process_molecule(molecule: str, duo_dir: str, marvel_dir: str, output_dir: str, registry_path: str):
    from src.masses import exomol_isotope_id

    mol_entry = get_molecule(molecule, registry_path)
    os.makedirs(output_dir, exist_ok=True)

    for isotope in mol_entry["isotopes"]:
        iso_id = exomol_isotope_id(isotope)
        duo_path = os.path.join(duo_dir, f"{molecule}{iso_id}_output_duo.out")
        marvel_path = os.path.join(marvel_dir, f"MARVEL_Energies_{molecule}{iso_id}.txt")

        if not os.path.exists(duo_path):
            print(f"Skipping {isotope}: {duo_path} not found.")
            continue
        if not os.path.exists(marvel_path):
            print(f"Skipping {isotope}: {marvel_path} not found.")
            continue

        combined = combine_isotopologue(duo_path, marvel_path, isotope)
        save_path = os.path.join(output_dir, f"{isotope}.csv")
        combined.to_csv(save_path, index=False)
        print(f"Saved {len(combined)} states to {save_path}")


if __name__ == "__main__":
    args = parse_args()
    process_molecule(args.molecule, args.duo_dir, args.marvel_dir, args.output_dir, args.registry)
