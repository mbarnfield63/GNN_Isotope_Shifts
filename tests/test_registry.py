import pytest

from src.registry import get_molecule, load_registry

REGISTRY_PATH = "configs/molecules.yaml"


def test_co_resolves_isotope_ids_from_registry():
    co = get_molecule("CO", REGISTRY_PATH)
    assert co["parent_isotope"] == "12C16O"
    assert co["isotope_ids"] == [26, 36, 28, 38, 27, 37]


def test_single_isotope_molecule_resolves_one_id():
    cn = get_molecule("CN", REGISTRY_PATH)
    assert cn["isotope_ids"] == [24]


def test_unknown_molecule_raises():
    with pytest.raises(KeyError):
        get_molecule("NOT_A_MOLECULE", REGISTRY_PATH)


def test_load_registry_returns_every_entry():
    registry = load_registry(REGISTRY_PATH)
    assert {"CO", "CN", "PN", "SiN", "SiO"} <= set(registry)


def test_parent_isotope_must_be_in_isotope_list(tmp_path):
    bad_registry = tmp_path / "bad_molecules.yaml"
    bad_registry.write_text(
        "XX:\n"
        "  parent_isotope: '99Xx99Yy'\n"
        "  isotopes: ['12C16O']\n"
        "  electronic_state: null\n"
        "  marvel_source_code: null\n"
    )
    with pytest.raises(ValueError):
        get_molecule("XX", str(bad_registry))
