import numpy as np
import pandas as pd

from src.ps_baseline import fit_predict_ps_baseline


def _synthetic_band(j_max=20, noise_std=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    J = np.arange(0, j_max)
    true_correction = 0.001 * J * (J + 1)
    noise = rng.normal(0, noise_std, size=len(J))
    return pd.DataFrame(
        {
            "molecule": "TEST",
            "iso_id": 1,
            "ElecState": "X",
            "v": 0,
            "Omega": 0.0,
            "parity": "+",
            "J": J,
            "Ediff": true_correction + noise,
            "is_known_marvel": True,
        }
    )


def test_extrapolates_beyond_fit_range():
    df = _synthetic_band()
    fit_mask = df["J"] <= 14

    result = fit_predict_ps_baseline(df, fit_mask, degree=2, min_points=4)

    held_out = result[~fit_mask]
    err = (held_out["Ediff"] - held_out["ps_predicted_correction"]).abs()
    assert err.max() < 1e-3


def test_sparse_band_falls_back_to_constant_not_crash():
    df = _synthetic_band().iloc[:2].copy()
    result = fit_predict_ps_baseline(df, pd.Series(True, index=df.index))
    assert not result["ps_predicted_correction"].isna().any()


def test_nan_omega_does_not_silently_drop_rows():
    # Regression test: pandas groupby drops NaN group keys by default, which
    # previously made every row in a legacy (no Omega recorded) molecule
    # vanish from the PS fit/predict output entirely.
    df = _synthetic_band()
    df["Omega"] = np.nan
    fit_mask = df["J"] <= 14

    result = fit_predict_ps_baseline(df, fit_mask, degree=2, min_points=4)

    assert len(result) == len(df)
    held_out = result[~fit_mask]
    assert not held_out["ps_predicted_correction"].isna().any()


def test_unfitted_band_predicts_zero_not_nan():
    df = _synthetic_band()
    no_fit = pd.Series(False, index=df.index)
    result = fit_predict_ps_baseline(df, no_fit)
    assert (result["ps_predicted_correction"] == 0.0).all()
