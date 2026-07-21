"""
Reads an ExoMol JSON .def file (dataset.states.states_file_fields) to recover
the exact, molecule-specific column layout of that molecule's .states file.

Why this exists: .states files are whitespace-delimited with no header, so
ingestion has to know column order up front. That order is NOT uniform across
molecules -- it depends on Hund's coupling case and which optional fields
(gfactor, lifetime) the dataset includes. Observed so far: CN/PN/SiN/NO/SiO
use Hund's case (a) with gfactor (16 fields); SO drops gfactor (15 fields);
O2 is Hund's case (b) with no gfactor and no Omega at all, using Fi/N instead
(14 fields); C2 has both Hund's case (b) *and* gfactor (18 fields). A single
hardcoded schema silently misaligns columns for any molecule that doesn't
match it exactly.
"""

import json

# Hund's-case quantum-number fields are prefixed "hunda:"/"hundb:" in the
# ExoMol def; stripped here since none of Lambda/Sigma/Fi/N are used
# downstream -- only ElecState/v/Omega/parity (the PS band key) and the core
# physics columns are.
_PREFIXES = ("hunda:", "hundb:", "Auxiliary:")

# PS baseline bands are defined by rotationless parity (e/f), not total
# parity (+/-) -- total parity flips sign with J within a band, so grouping
# by it would split one continuous J-progression in half. See
# src/ps_baseline.py and notes/predicted_shift_methodology.md.
_BASE_NAME_MAP = {
    "ID": "ID",
    "E": "EMarv",
    "gtot": "gtot",
    "J": "J",
    "unc": "unc",
    "tau": "tau",
    "gfactor": "gfactor",
    "+/-": "total_parity",
    "e/f": "parity",
    "ElecState": "ElecState",
    "v": "v",
    "Lambda": "Lambda",
    "Sigma": "Sigma",
    "Omega": "Omega",
    "Fi": "Fi",
    "N": "N",
    "SourceType": "Source",
    "Ecal": "ECalc",
}


def _canonical_name(raw_name: str) -> str:
    base = raw_name
    for prefix in _PREFIXES:
        if base.startswith(prefix):
            base = base[len(prefix):]
            break
    if base not in _BASE_NAME_MAP:
        raise ValueError(
            f"Unrecognized states-file field {raw_name!r} -- add it to "
            "_BASE_NAME_MAP in src/def_reader.py"
        )
    return _BASE_NAME_MAP[base]


def read_states_schema(def_json_path: str) -> list[str]:
    """Returns canonical column names for a molecule's .states file, in file
    order -- pass straight as pandas' `names=` when reading it."""
    with open(def_json_path) as f:
        spec = json.load(f)
    fields = spec["dataset"]["states"]["states_file_fields"]
    return [_canonical_name(f["name"]) for f in fields]


if __name__ == "__main__":
    import tempfile

    hunda_spec = {
        "dataset": {
            "states": {
                "states_file_fields": [
                    {"name": n}
                    for n in [
                        "ID", "E", "gtot", "J", "unc", "tau", "gfactor",
                        "+/-", "e/f", "ElecState", "v",
                        "hunda:Lambda", "hunda:Sigma", "hunda:Omega",
                        "Auxiliary:SourceType", "Auxiliary:Ecal",
                    ]
                ]
            }
        }
    }
    hundb_no_gfactor_spec = {
        "dataset": {
            "states": {
                "states_file_fields": [
                    {"name": n}
                    for n in [
                        "ID", "E", "gtot", "J", "unc", "tau",
                        "+/-", "e/f", "ElecState", "v",
                        "hundb:Fi", "hundb:N",
                        "Auxiliary:SourceType", "Auxiliary:Ecal",
                    ]
                ]
            }
        }
    }

    with tempfile.NamedTemporaryFile("w", suffix=".def.json", delete=False) as f:
        json.dump(hunda_spec, f)
        hunda_path = f.name
    with tempfile.NamedTemporaryFile("w", suffix=".def.json", delete=False) as f:
        json.dump(hundb_no_gfactor_spec, f)
        hundb_path = f.name

    hunda_cols = read_states_schema(hunda_path)
    assert hunda_cols == [
        "ID", "EMarv", "gtot", "J", "unc", "tau", "gfactor",
        "total_parity", "parity", "ElecState", "v",
        "Lambda", "Sigma", "Omega", "Source", "ECalc",
    ], hunda_cols
    assert len(hunda_cols) == 16

    hundb_cols = read_states_schema(hundb_path)
    assert "Omega" not in hundb_cols
    assert "gfactor" not in hundb_cols
    assert hundb_cols[-4:] == ["Fi", "N", "Source", "ECalc"]
    assert len(hundb_cols) == 14

    try:
        _canonical_name("hunda:Nonsense")
        raise AssertionError("expected ValueError for unrecognized field")
    except ValueError:
        pass

    print("def_reader.py self-check passed")
