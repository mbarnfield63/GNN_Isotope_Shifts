"""
ExoMol's "Predicted Shift" (PS) baseline (RAS Techniques and Instruments,
Vol. 3, Issue 1, Jan 2024, doi:10.1093/rasti/rzae037, Sec 2.2.3): for a given
vibronic band (electronic state, v, Omega, rotationless parity), fit the
trend in obs.-calc. energy differences as a function of J, then extrapolate
that fit to predict synthetic obs.-calc. corrections for J outside the
experimentally observed range.

This is the numeric baseline the GNN is compared against for single-
isotopologue molecules, where no isotopologue-extrapolation axis exists and
the GNN's job reduces to the same J-extrapolation task PS was built for.
"""

import numpy as np
import pandas as pd

BAND_COLS = ["molecule", "iso_id", "ElecState", "v", "Omega", "parity"]


def fit_predict_ps_baseline(
    nodes_df: pd.DataFrame,
    fit_mask,
    band_cols=BAND_COLS,
    degree: int = 2,
    min_points: int = 4,
) -> pd.DataFrame:
    """
    Fits a per-band polynomial in J_ext(J_ext+1) to the obs.-calc. residuals
    (Ediff) of rows marked True in `fit_mask`, then predicts a PS correction
    for every row in nodes_df (including rows outside fit_mask, extrapolating
    the band's fit to their J).

    ponytail: the RASTI paper gives the exact functional form for the PS
    *uncertainty* growth (a*J_ext(J_ext+1)+sigma) but not for the correction
    fit itself -- it only says "fit the obs.-calc. trend". A degree-2
    polynomial in J(J+1) is used here (the standard form for rotational-
    energy trends, and the same functional shape as the published
    uncertainty formula); raise `degree` if a band's trend needs more terms.

    Returns nodes_df with three added columns: ps_predicted_correction,
    ps_band_sigma (std of in-band fit residuals), ps_j_max_fit (max J used
    to fit the band, i.e. how far each prediction extrapolates).
    """
    fit_mask = pd.Series(fit_mask, index=nodes_df.index) if not isinstance(fit_mask, pd.Series) else fit_mask

    pred = np.zeros(len(nodes_df))
    sigma = np.zeros(len(nodes_df))
    j_max_fit = np.full(len(nodes_df), np.nan)
    j_all = nodes_df["J"].values.astype(float)
    x_all = j_all * (j_all + 1)

    for _, positions in nodes_df.groupby(band_cols, dropna=False).indices.items():
        fit_positions = positions[fit_mask.values[positions]]

        if len(fit_positions) == 0:
            continue
        elif len(fit_positions) < min_points:
            fit_ediff = nodes_df["Ediff"].values[fit_positions]
            pred[positions] = fit_ediff.mean()
            sigma[positions] = fit_ediff.std() if len(fit_positions) > 1 else 0.0
            j_max_fit[positions] = j_all[fit_positions].max()
        else:
            x_fit = x_all[fit_positions]
            ediff_fit = nodes_df["Ediff"].values[fit_positions]
            fit_degree = min(degree, len(fit_positions) - 1)
            coeffs = np.polyfit(x_fit, ediff_fit, fit_degree)
            pred[positions] = np.polyval(coeffs, x_all[positions])
            sigma[positions] = (ediff_fit - np.polyval(coeffs, x_fit)).std()
            j_max_fit[positions] = j_all[fit_positions].max()

    out = nodes_df.copy()
    out["ps_predicted_correction"] = pred
    out["ps_band_sigma"] = sigma
    out["ps_j_max_fit"] = j_max_fit
    return out


def ps_uncertainty(j: np.ndarray, j_max_fit: np.ndarray, sigma: np.ndarray, a: float) -> np.ndarray:
    """ExoMol's PS uncertainty growth formula (RASTI 2024, Sec 2.2.3):
    deltaE_PS = a * J_ext(J_ext+1) + sigma, with J_ext = J - J_max_marvel."""
    j_ext = np.maximum(np.asarray(j, dtype=float) - np.asarray(j_max_fit, dtype=float), 0.0)
    return a * j_ext * (j_ext + 1) + np.asarray(sigma, dtype=float)


if __name__ == "__main__":
    # Synthetic band: true trend is 0.001*J(J+1), some J held out of the fit.
    rng = np.random.default_rng(0)
    J = np.arange(0, 20)
    true_correction = 0.001 * J * (J + 1)
    noise = rng.normal(0, 1e-4, size=len(J))

    df = pd.DataFrame(
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

    fit_mask = df["J"] <= 14  # hold out J=15..19 as unseen extrapolation targets
    result = fit_predict_ps_baseline(df, fit_mask, degree=2, min_points=4)

    held_out = result[~fit_mask]
    err = (held_out["Ediff"] - held_out["ps_predicted_correction"]).abs()
    assert err.max() < 1e-3, f"PS baseline extrapolation error too high: {err.max()}"

    # A band with too few fit points falls back to a constant, not a crash.
    sparse = df.iloc[:2].copy()
    sparse_result = fit_predict_ps_baseline(sparse, pd.Series(True, index=sparse.index))
    assert not sparse_result["ps_predicted_correction"].isna().any()

    print("ps_baseline.py self-check passed")
