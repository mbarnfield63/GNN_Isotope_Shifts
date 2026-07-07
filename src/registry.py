from functools import lru_cache

import yaml

from src.masses import exomol_isotope_id

DEFAULT_REGISTRY_PATH = "configs/molecules.yaml"


@lru_cache(maxsize=None)
def _load_raw(registry_path: str) -> dict:
    with open(registry_path, "r") as file:
        return yaml.safe_load(file)


def load_registry(registry_path: str = DEFAULT_REGISTRY_PATH) -> dict:
    """Load every molecule entry from the registry, with isotope_ids derived
    automatically (ExoMol convention) rather than stored."""
    raw = _load_raw(registry_path)
    return {name: get_molecule(name, registry_path) for name in raw}


def get_molecule(name: str, registry_path: str = DEFAULT_REGISTRY_PATH) -> dict:
    """Resolve a single molecule's registry entry, deriving isotope_ids from
    each isotope's formula string."""
    raw = _load_raw(registry_path)
    if name not in raw:
        raise KeyError(f"Molecule {name!r} not found in registry {registry_path!r}")

    entry = dict(raw[name])
    isotopes = entry["isotopes"]
    parent_isotope = entry["parent_isotope"]

    if parent_isotope not in isotopes:
        raise ValueError(
            f"{name}: parent_isotope {parent_isotope!r} is not listed in isotopes {isotopes!r}"
        )

    entry["isotope_ids"] = [exomol_isotope_id(iso) for iso in isotopes]
    return entry


if __name__ == "__main__":
    co = get_molecule("CO")
    assert co["parent_isotope"] == "12C16O"
    assert co["isotope_ids"] == [26, 36, 28, 38, 27, 37]

    cn = get_molecule("CN")
    assert cn["isotope_ids"] == [24]

    print("registry.py self-check passed")
