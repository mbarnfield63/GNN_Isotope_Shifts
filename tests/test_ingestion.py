import pandas as pd
import pytest

from scripts.combining_duo_marv import parse_duo_output, parse_marvel
from scripts.states_scraping import process_states


def test_parse_duo_output_extracts_band_key_columns(tmp_path):
    duo_file = tmp_path / "CO26_output_duo.out"
    duo_file.write_text(
        "# header line, ignored\n"
        "0.0   1   0.000000   1   0   0   0.0   0.0   0.0   +   label\n"
        "1.0   1   3.845000   1   0   0   0.0   0.0   1.0   -   label\n"
    )

    df = parse_duo_output(str(duo_file))

    assert list(df["J"]) == [0, 1]
    assert list(df["v"]) == [0, 0]
    assert list(df["ElecState"]) == [1, 1]
    assert list(df["Omega"]) == [0.0, 1.0]
    assert list(df["parity"]) == ["+", "-"]


def test_parse_duo_output_skips_short_lines(tmp_path):
    duo_file = tmp_path / "CO26_output_duo.out"
    duo_file.write_text("too short\n0.0 1 0.0 1 0 0 0.0 0.0 0.0 + label\n")

    df = parse_duo_output(str(duo_file))

    assert len(df) == 1


def test_parse_marvel(tmp_path):
    marvel_file = tmp_path / "MARVEL_Energies_CO26.txt"
    marvel_file.write_text("0 0 0.0 0.001 1\n0 1 3.8 0.001 2\n")

    df = parse_marvel(str(marvel_file))

    assert list(df.columns) == ["v", "J", "EMarv", "unc"]
    assert list(df["EMarv"]) == [0.0, 3.8]


def test_process_states_masses_and_marvel_masking(tmp_path):
    import json

    # SiO is a real registry entry (X(1SIGMA+), MARVEL code "Ma")
    def_json = tmp_path / "28Si-16O.def.json"
    def_json.write_text(
        json.dumps(
            {
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
        )
    )

    source = tmp_path / "28Si-16O.states"
    # ID E gtot J unc tau gfactor +/- e/f ElecState v Lambda Sigma Omega Source Ecalc
    # Total parity (+/-) and rotationless parity (e/f) deliberately differ here.
    source.write_text(
        "1 0.0 1 0 0.000001 0 0 + - X(1SIGMA+) 0 0 0 0 Ma 0.0\n"
        "2 100.0 1 1 0.000001 0 0 - + X(1SIGMA+) 0 0 0 0 Duo 99.5\n"
    )

    output_dir = tmp_path / "out"
    process_states("SiO", "28Si16O", str(source), str(def_json), str(output_dir), "configs/molecules.yaml")

    out_df = pd.read_csv(output_dir / "28Si16O.csv")
    assert len(out_df) == 2
    assert out_df["mass_A"].iloc[0] == pytest.approx(27.97692653, abs=1e-6)
    # Row from "Duo" source (not the MARVEL code) must have EMarv masked to NaN
    assert pd.isna(out_df["EMarv"].iloc[1])
    assert out_df["EMarv"].iloc[0] == 0.0
    assert set(["ElecState", "Omega", "parity"]).issubset(out_df.columns)
    # PS bands use rotationless parity (e/f), not total parity (+/-)
    assert list(out_df["parity"]) == ["-", "+"]
