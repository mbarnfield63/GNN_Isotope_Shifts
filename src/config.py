import os

PREPROCESSED_DIR = "data/preprocessed"
PROCESSED_DIR = "data/processed"

# Atomic weights
ISOTOPIC_MASSES = {
    "12C": 12.0000000,
    "13C": 13.0033548,
    "14C": 14.0032419,
    "16O": 15.9949146,
    "17O": 16.9991317,
    "18O": 17.9991596,
    "32S": 31.9720711,
    "33S": 32.9714587,
    "34S": 33.9678670,
}


def get_reduced_mass(m1, m2):
    return (m1 * m2) / (m1 + m2)


# Isotopologue configs
ISOTOPOLOGUE_CONFIGS = {
    # Carbon monoxide isotopologues
    "12C16O": {
        "molecule": "CO",
        "iso": 26,
        "mass_A": ISOTOPIC_MASSES["12C"],
        "mass_B": ISOTOPIC_MASSES["16O"],
        "mu": get_reduced_mass(ISOTOPIC_MASSES["12C"], ISOTOPIC_MASSES["16O"]),
    },
    "12C17O": {
        "molecule": "CO",
        "iso": 27,
        "mass_A": ISOTOPIC_MASSES["12C"],
        "mass_B": ISOTOPIC_MASSES["17O"],
        "mu": get_reduced_mass(ISOTOPIC_MASSES["12C"], ISOTOPIC_MASSES["17O"]),
    },
    "12C18O": {
        "molecule": "CO",
        "iso": 28,
        "mass_A": ISOTOPIC_MASSES["12C"],
        "mass_B": ISOTOPIC_MASSES["18O"],
        "mu": get_reduced_mass(ISOTOPIC_MASSES["12C"], ISOTOPIC_MASSES["18O"]),
    },
    "13C16O": {
        "molecule": "CO",
        "iso": 36,
        "mass_A": ISOTOPIC_MASSES["13C"],
        "mass_B": ISOTOPIC_MASSES["16O"],
        "mu": get_reduced_mass(ISOTOPIC_MASSES["13C"], ISOTOPIC_MASSES["16O"]),
    },
    "13C17O": {
        "molecule": "CO",
        "iso": 37,
        "mass_A": ISOTOPIC_MASSES["13C"],
        "mass_B": ISOTOPIC_MASSES["17O"],
        "mu": get_reduced_mass(ISOTOPIC_MASSES["13C"], ISOTOPIC_MASSES["17O"]),
    },
    "13C18O": {
        "molecule": "CO",
        "iso": 38,
        "mass_A": ISOTOPIC_MASSES["13C"],
        "mass_B": ISOTOPIC_MASSES["18O"],
        "mu": get_reduced_mass(ISOTOPIC_MASSES["13C"], ISOTOPIC_MASSES["18O"]),
    },
    # # Carbon monosulfide isotopologues
    # "12C32S": {
    #     "molecule": "CS",
    #     "iso": 22,
    #     "mass_A": ISOTOPIC_MASSES["12C"],
    #     "mass_B": ISOTOPIC_MASSES["32S"],
    #     "mu": get_reduced_mass(ISOTOPIC_MASSES["12C"], ISOTOPIC_MASSES["32S"]),
    # },
    # "12C34S": {
    #     "molecule": "CS",
    #     "iso": 24,
    #     "mass_A": ISOTOPIC_MASSES["12C"],
    #     "mass_B": ISOTOPIC_MASSES["34S"],
    #     "mu": get_reduced_mass(ISOTOPIC_MASSES["12C"], ISOTOPIC_MASSES["34S"]),
    # },
    # "13C32S": {
    #     "molecule": "CS",
    #     "iso": 32,
    #     "mass_A": ISOTOPIC_MASSES["13C"],
    #     "mass_B": ISOTOPIC_MASSES["32S"],
    #     "mu": get_reduced_mass(ISOTOPIC_MASSES["13C"], ISOTOPIC_MASSES["32S"]),
    # },
    # # Sulfur monoxide isotopologues
    # "32S16O": {
    #     "molecule": "SO",
    #     "iso": 26,
    #     "mass_A": ISOTOPIC_MASSES["32S"],
    #     "mass_B": ISOTOPIC_MASSES["16O"],
    #     "mu": get_reduced_mass(ISOTOPIC_MASSES["32S"], ISOTOPIC_MASSES["16O"]),
    # },
    # "32S18O": {
    #     "molecule": "SO",
    #     "iso": 28,
    #     "mass_A": ISOTOPIC_MASSES["32S"],
    #     "mass_B": ISOTOPIC_MASSES["18O"],
    #     "mu": get_reduced_mass(ISOTOPIC_MASSES["32S"], ISOTOPIC_MASSES["18O"]),
    # },
    # "34S16O": {
    #     "molecule": "SO",
    #     "iso": 34,
    #     "mass_A": ISOTOPIC_MASSES["34S"],
    #     "mass_B": ISOTOPIC_MASSES["16O"],
    #     "mu": get_reduced_mass(ISOTOPIC_MASSES["34S"], ISOTOPIC_MASSES["16O"]),
    # },
    # "34S18O": {
    #     "molecule": "SO",
    #     "iso": 36,
    #     "mass_A": ISOTOPIC_MASSES["34S"],
    #     "mass_B": ISOTOPIC_MASSES["18O"],
    #     "mu": get_reduced_mass(ISOTOPIC_MASSES["34S"], ISOTOPIC_MASSES["18O"]),
    # },
}
