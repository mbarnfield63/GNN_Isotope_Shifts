import re
from functools import lru_cache

from molmass import ELEMENTS

# Isotopologue formula strings (also used as data filenames) are already an
# unambiguous encoding of isotope composition, e.g. "13C16O" = C-13 + O-16.
# ponytail: skips InChI/.def-file parsing -- the formula string alone is
# enough to resolve exact isotope masses via molmass, no extra lookup needed.
_ISOTOPOLOGUE_RE = re.compile(r"^(\d+)([A-Z][a-z]?)(\d+)([A-Z][a-z]?)$")


@lru_cache(maxsize=None)
def atomic_mass(symbol: str, mass_number: int) -> float:
    """Exact mass (u) of a single isotope, e.g. atomic_mass('C', 13)."""
    return ELEMENTS[symbol].isotopes[mass_number].mass


def parse_isotopologue(formula: str) -> tuple[tuple[int, str], tuple[int, str]]:
    """Split a diatomic isotopologue formula (e.g. "28Si14N") into its two
    (mass_number, element_symbol) atoms, in formula order."""
    match = _ISOTOPOLOGUE_RE.match(formula)
    if not match:
        raise ValueError(f"Could not parse diatomic isotopologue formula: {formula!r}")
    mass_a, sym_a, mass_b, sym_b = match.groups()
    return (int(mass_a), sym_a), (int(mass_b), sym_b)


def isotopologue_masses(formula: str) -> dict:
    """Resolve mass_A, mass_B, and reduced_mass (u) for a diatomic isotopologue
    formula string, using exact isotope masses (not averaged atomic weights)."""
    (mass_num_a, sym_a), (mass_num_b, sym_b) = parse_isotopologue(formula)
    mass_a = atomic_mass(sym_a, mass_num_a)
    mass_b = atomic_mass(sym_b, mass_num_b)
    reduced_mass = (mass_a * mass_b) / (mass_a + mass_b)
    return {"mass_A": mass_a, "mass_B": mass_b, "reduced_mass": reduced_mass}


def exomol_isotope_id(formula: str) -> int:
    """ExoMol's standard isotopologue ID: the last digit of each atom's mass
    number, concatenated in formula order (e.g. "13C16O" -> 36)."""
    (mass_num_a, _), (mass_num_b, _) = parse_isotopologue(formula)
    return (mass_num_a % 10) * 10 + (mass_num_b % 10)


if __name__ == "__main__":
    assert parse_isotopologue("12C16O") == ((12, "C"), (16, "O"))
    assert parse_isotopologue("28Si14N") == ((28, "Si"), (14, "N"))

    co = isotopologue_masses("12C16O")
    assert abs(co["mass_A"] - 12.0) < 1e-9
    assert abs(co["mass_B"] - 15.9949146) < 1e-6
    assert abs(co["reduced_mass"] - 6.8562) < 1e-3

    sio = isotopologue_masses("28Si16O")
    assert abs(sio["reduced_mass"] - 10.1767) < 1e-3

    assert exomol_isotope_id("12C16O") == 26
    assert exomol_isotope_id("13C18O") == 38
    assert exomol_isotope_id("12C14N") == 24

    print("masses.py self-check passed")
