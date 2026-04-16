import pandas as pd
import os

# ==========================================================
# CONFIGURATION SECTION
# ==========================================================

# 1. Path to the source ExoMol .states file
SOURCE_STATES_FILE = (
    r"C:\Code\Work\raw_data_store\Diatomics\SO\Full\32S-16O__SOLIS.states"
)

# 2. List of ALL entries/columns as they appear in the source file
# Note: ExoMol formats vary by molecule; check the associated .def file.
FULL_COLUMN_NAMES = [
    "ID",  # State ID
    "E",  # Energy (cm-1)
    "gtot",  # Total degeneracy
    "J",  # Total angular momentum
    "unc",  # Uncertainty
    "tau",  # Lifetime (optional)
    "hunda:+/-",  # Parity
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
    "hunda:v": "v",
    "hunda:ElectronicState": "ElecState",
}

# 4. The specific columns you want to "scrape" into your new CSV
COLUMNS_NEEDED = [
    "v",
    "J",
    "Ecalc",
    "EMarv",
    "unc",
]

# 5. Output Naming Format
MOLECULE_NAME = "32S-16O"
OUTPUT_DIRECTORY = "data/preprocessed"
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
        df = df[df["ElecState"] == "X(3SIGMA-)"]

        # If source is not MARVEL, set EMarv to NaN since it's not from MARVEL
        df.loc[df["Source"] != "Ma", "EMarv"] = pd.NA

        # Scrape only the requested columns
        extracted_df = df[COLUMNS_NEEDED]

        # Save to CSV
        save_path = os.path.join(OUTPUT_DIRECTORY, OUTPUT_FILENAME)
        extracted_df.to_csv(save_path, index=False)

        print(f"Success! Extracted {len(COLUMNS_NEEDED)} columns.")
        print(f"Saved to: {save_path}")

    except FileNotFoundError:
        print(f"Error: {SOURCE_STATES_FILE} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    process_states()
