import os
import pandas as pd
from src.config import ISOTOPOLOGUE_CONFIGS

# CONFIG
ISO_NUMBERS = [26, 27, 28, 36, 37, 38]
DUO_DIR = r"C:\Code\Work\raw_data_store\Diatomics\CO"
MARVEL_DIR = r"C:\Code\Work\raw_data_store\Diatomics\CO"


def parse_duo_output(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    print(f"Parsing DUO output file: {filename}")
    duo_data = []

    # Skip header lines and look for level data
    for line in lines:
        # Check if line contains level information with all required columns
        if line.strip() and not line.startswith("#") and not line.startswith("*"):
            try:
                # Split the line and convert to appropriate types
                parts = line.split()
                if len(parts) >= 11:  # Ensure line has all required columns
                    level = {
                        "J": int(
                            float(parts[0])
                        ),  # Rotational quantum number (column 1)
                        "i": int(parts[1]),  # State index (column 2)
                        "ECalc": float(parts[2]),  # Energy level (column 3)
                        "State": int(parts[3]),  # Electronic state (column 4)
                        "v": int(parts[4]),  # Vibrational quantum number (column 5)
                        "lambda": int(parts[5]),  # Lambda (column 6)
                        "spin": float(parts[6]),  # Spin (column 7)
                        "sigma": float(parts[7]),  # Sigma (column 8)
                        "omega": float(parts[8]),  # Omega (column 9)
                        "parity": parts[9],  # Parity (column 10)
                        "label": parts[10],  # State label (column 11)
                    }
                    duo_data.append(level)
            except (ValueError, IndexError):
                continue

    return pd.DataFrame(duo_data)


def load_duo_data(duo_directory):
    # Get all DUO output files
    duo_files = sorted(
        [f for f in os.listdir(duo_directory) if f.endswith("_output_duo.out")]
    )

    # Create a dictionary to store all dataframes
    duo_data_dict = {}

    # Process each file
    for file in duo_files:
        # Extract number from filename (assuming format COxx_output_duo.out)
        number = file.split("_")[0][2:]  # This extracts "xx" from "COxx"

        # Parse the file and process the data
        duo_data = parse_duo_output(f"{duo_directory}/{file}")
        duo_data = duo_data[
            ["v", "J", "ECalc"]
            + [col for col in duo_data.columns if col not in ["v", "J", "ECalc"]]
        ]
        duo_data.sort_values(by=["ECalc"], inplace=True)

        # Store in dictionary with name duo_data_xx
        duo_data_dict[f"duo_data_{number}"] = duo_data

    print("DUO data loaded for isotopologues:", list(duo_data_dict.keys()))
    return duo_data_dict


def load_marvel_data(marvel_directory):
    # Get all MARVEL files
    marvel_files = sorted(
        [f for f in os.listdir(marvel_directory) if f.startswith("MARVEL_Energies_CO")]
    )

    # Create a dictionary to store all MARVEL dataframes
    marvel_data_dict = {}

    # Process each file
    for file in marvel_files:
        # Extract number from filename (assuming format MARVEL_Energies_COxx.txt)
        number = file.split("_")[-1].replace("CO", "").replace(".txt", "")
        print(f"Processing MARVEL file for isotopologue: {number}")

        # Read and process the MARVEL file
        marvel_data = pd.read_csv(f"{marvel_directory}/{file}", sep=r"\s+", header=None)
        marvel_data.columns = ["v", "J", "EMarv", "unc", "N"]
        marvel_data.sort_values(by=["EMarv"], inplace=True)

        # Store in dictionary with name marvel_data_xx
        marvel_data_dict[f"marvel_data_{number}"] = marvel_data

    print("MARVEL data loaded for isotopologues:", list(marvel_data_dict.keys()))
    return marvel_data_dict


def combine_datasets(duo_data_dict, marvel_data_dict):
    # Isotopologue numbers between DUO and MARVEL data
    iso_numbers = [26, 27, 28, 36, 37, 38]

    # Process each isotopologue
    for number in iso_numbers:
        # Get the corresponding DUO and MARVEL data
        duo_data = duo_data_dict[f"duo_data_{number}"]
        marvel_data = marvel_data_dict[f"marvel_data_{number}"]

        # Get iso name from config based on number
        iso_name = None
        for name, config in ISOTOPOLOGUE_CONFIGS.items():
            if config["iso"] == int(number):
                iso_name = name
                break

        # Merge the data
        combined = pd.merge(duo_data, marvel_data, on=["v", "J"], how="outer")
        combined = combined[["v", "J", "ECalc", "EMarv", "unc"]]

        # Save the combined data for this isotopologue
        combined.to_csv(f"data/preprocessed/{iso_name}.csv", index=False)


if __name__ == "__main__":
    duo_data_dict = load_duo_data(DUO_DIR)
    marvel_data_dict = load_marvel_data(MARVEL_DIR)

    combined_data = combine_datasets(duo_data_dict, marvel_data_dict)
    print("Datasets combined and saved successfully.")
