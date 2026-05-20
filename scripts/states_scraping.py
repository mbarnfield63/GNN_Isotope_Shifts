import pandas as pd
import os

# ==========================================================
# CONFIGURATION SECTION
# ==========================================================

# Molecule name
MOLECULE_NAME = "31P14N"
MASS_A = 30.9737619985  # Atomic mass of P
MASS_B = 14.0030740048  # Atomic mass of N
X_ELEC = "X1Sigma+"  # Target electronic state to extract

# 1. Path to the source ExoMol .states file
SOURCE_STATES_FILE = r"C:\Code\Work\raw_data_store\Diatomics\PN\31P-14N__PaiN.states"

# 2. List of ALL entries/columns as they appear in the source file
# Note: ExoMol formats vary by molecule; check the associated .def file.
FULL_COLUMN_NAMES = [
    "ID",  # State ID
    "E",  # Energy (cm-1)
    "gtot",  # Total degeneracy
    "J",  # Total angular momentum
    "unc",  # Uncertainty
    "tau",  # Lifetime (optional)
    "gfactor",  # Lande g-factor (optional)
    "parity",  # Parity (e.g., '+' or '-')
    "rotlessparity",  # Rotationless parity (optional)
    "hunda:ElectronicState",  # Electronic state label
    "hunda:v",  # Vibrational quantum number
    "hunda:Lambda",  # Projection of electronic angular momentum
    "hunda:Sigma",  # Projection of electronic spin
    "hunda:Omega",  # Projection of total angular momentum
    "Source",  # Source of the data (e.g., MARVEL, calculated)
    "Ecalc",  # Calculated energy (optional)
]

# 3. Rename columns as required
RENAMING_MAP = {
    "E": "EMarv",
    "Ecalc": "ECalc",
    "hunda:v": "v",
    "hunda:ElectronicState": "ElecState",
}

# 4. The specific columns for new CSV
COLUMNS = [
    "mass_A",
    "mass_B",
    "reduced_mass",
    "v",
    "J",
    "ECalc",
    "EMarv",
    "unc",
    "Ediff",
]

# 5. Output Naming Format
OUTPUT_DIRECTORY = "data/states"
OUTPUT_FILENAME = f"{MOLECULE_NAME}.csv"

# ==========================================================
# SCRAPER LOGIC
# ==========================================================


def process_states():
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    print(f"Opening file: {SOURCE_STATES_FILE}")

    try:
        # ExoMol .states files are space-separated (variable whitespace)
        # and do not contain headers.
        df = pd.read_csv(
            SOURCE_STATES_FILE,
            sep=r"\s+",
            names=FULL_COLUMN_NAMES,
            header=None,
        )

        # Rename columns if needed
        if RENAMING_MAP:
            df.rename(columns=RENAMING_MAP, inplace=True)

        # Filter to only Electronic State X
        df = df[df["ElecState"] == X_ELEC].copy()

        # If source is not MARVEL, set EMarv to NaN since it's not from MARVEL
        df.loc[df["Source"] != "Ma", "EMarv"] = pd.NA

        # Masses and reduced mass calcs
        mu = (MASS_A * MASS_B) / (MASS_A + MASS_B)
        df = df.assign(
            mass_A=MASS_A,
            mass_B=MASS_B,
            reduced_mass=mu,
        )
        # Calculate energy difference where both EMarv and ECalc are available
        df["Ediff"] = df["EMarv"] - df["ECalc"]

        # Select only the columns needed for the final output
        extracted_df = df[COLUMNS]

        # Save to CSV
        save_path = os.path.join(OUTPUT_DIRECTORY, OUTPUT_FILENAME)
        extracted_df.to_csv(save_path, index=False)

        print(f"Success! Extracted {len(COLUMNS)} columns.")
        print(f"Saved to: {save_path}")

    except FileNotFoundError:
        print(f"Error: {SOURCE_STATES_FILE} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    process_states()
