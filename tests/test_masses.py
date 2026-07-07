import pytest

from src.masses import exomol_isotope_id, isotopologue_masses, parse_isotopologue


def test_parse_isotopologue():
    assert parse_isotopologue("12C16O") == ((12, "C"), (16, "O"))
    assert parse_isotopologue("28Si14N") == ((28, "Si"), (14, "N"))
    assert parse_isotopologue("31P14N") == ((31, "P"), (14, "N"))


def test_parse_isotopologue_rejects_garbage():
    with pytest.raises(ValueError):
        parse_isotopologue("not-a-formula")


def test_isotopologue_masses_uses_exact_isotope_mass_not_atomic_weight():
    co = isotopologue_masses("12C16O")
    assert co["mass_A"] == pytest.approx(12.0, abs=1e-9)
    assert co["mass_B"] == pytest.approx(15.9949146, abs=1e-6)
    assert co["reduced_mass"] == pytest.approx(6.8562086, abs=1e-5)

    # Exact isotope mass, not the natural-abundance-averaged atomic weight
    # (28.0855) that the old hand-typed SiO config used.
    sio = isotopologue_masses("28Si16O")
    assert sio["mass_A"] == pytest.approx(27.97692653, abs=1e-6)
    assert sio["reduced_mass"] == pytest.approx(10.1767072, abs=1e-5)


def test_exomol_isotope_id_matches_known_convention():
    assert exomol_isotope_id("12C16O") == 26
    assert exomol_isotope_id("13C16O") == 36
    assert exomol_isotope_id("12C18O") == 28
    assert exomol_isotope_id("13C18O") == 38
    assert exomol_isotope_id("12C17O") == 27
    assert exomol_isotope_id("13C17O") == 37
    assert exomol_isotope_id("12C14N") == 24
