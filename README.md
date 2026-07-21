# GNN Isotope Shifts

A physics-informed Graph Attention Network (`HybridIsotopologueGATv2`) that
corrects calculated diatomic energy levels toward experimental (MARVEL)
accuracy. It generalizes the MLP method from the published
[Isotopologue Extrapolation paper](https://doi.org/10.1016/j.jms.2026.112084)
and doubles as a from-scratch reimplementation of ExoMol's
[Predicted Shift (PS) method](https://doi.org/10.1093/rasti/rzae037) (Sec. 2.2.3).

## The core idea

The model predicts `Ediff = E_MARVEL - E_calc`, the same obs.-calc. residual
both methods correct for, along two different extrapolation axes:

- **Isotopologue Extrapolation (IE)** -- learn the residual on a molecule's
  well-studied parent isotopologue and transfer it to minor isotopologues
  that lack experimental data. Requires >1 MARVEL-covered isotopologue.
- **Predicted Shift (PS)** -- fit the residual's trend in J within a
  vibronic band (electronic state, v, Omega, parity) and extrapolate to J
  values outside the experimentally observed range. Applies to any molecule,
  and is the *only* available correction for molecules with just one
  MARVEL-covered isotopologue.

**There is no mode switch.** Both axes are always present in the graph and
the model:

- `generate_graph_edges` (`src/dataset.py`) builds physical (rovibrational,
  same isotopologue) and isotopic (same v/J, different isotopologue) edges.
  A single-isotopologue molecule produces zero isotopic edges automatically
  -- the model then trains on physical edges alone, which *is* PS-style
  J-extrapolation, no special-casing required.
- The model's linear physics bypass (`src/model.py`) always includes both a
  mass-ratio term (the IE axis, zero-centered so the parent isotopologue is
  baseline 0) and a `J_ext(J_ext+1)` term (the PS axis, mirroring ExoMol's
  own uncertainty-growth formula, left uncentered since it's naturally zero
  at J=0).
- `main.py` runs LOIO cross-validation automatically whenever a molecule has
  >1 MARVEL isotopologue, and always runs the J-extrapolation split (holding
  out each band's highest-J known states) regardless of isotope count. A
  real PS baseline (`src/ps_baseline.py`, a per-band polynomial fit in
  `J(J+1)`) is fit alongside the GNN in the J-extrapolation split, so its
  performance is reported next to the GNN's rather than just claimed.

## Adding a new molecule

1. Add an entry to `configs/molecules.yaml`: parent isotopologue formula,
   isotope list, electronic state label, MARVEL source-flag code. Masses
   and ExoMol isotope IDs are derived automatically (`src/masses.py`, via
   `molmass`) from the formula strings -- never hand-type a mass.
2. Ingest the raw data into `data/states/<isotopologue>.csv`:
   - One combined `.states` file per isotopologue with an embedded
     MARVEL-source flag: `uv run python -m scripts.states_scraping --molecule X --isotope <formula> --source <path>`
   - Separate DUO output + MARVEL files (CO's historical path):
     `uv run python -m scripts.combining_duo_marv --molecule X --duo-dir <dir> --marvel-dir <dir>`
3. Add an experiment config under `configs/` (copy an existing one, change
   `molecules: ["X"]`).
4. `uv run python main.py --config configs/x_standard.yaml`

## Running experiments

```bash
uv run python main.py --config configs/co_standard.yaml    # multi-isotopologue: LOIO + J-extrapolation
uv run python main.py --config configs/cn_standard.yaml    # single-isotopologue: J-extrapolation only
uv run python main.py --config configs/pn_standard.yaml
uv run python main.py --config configs/sin_standard.yaml
uv run python main.py --config configs/sio_standard.yaml
```

Set `execution.ensemble_run: true` in a config to repeat across
`execution.num_seeds` seeds and aggregate mean +/- std instead of a single run.

## Layout

| Path | Role |
|---|---|
| `configs/molecules.yaml` | Molecule registry: identity only (isotopes, parent, electronic state, MARVEL code) -- no physical constants |
| `configs/*_standard.yaml` | Per-experiment config: which molecule(s), training hyperparameters |
| `src/masses.py` | Isotopologue formula -> exact isotope masses (via `molmass`) and ExoMol isotope IDs |
| `src/registry.py` | Loads `configs/molecules.yaml`, resolves isotope IDs |
| `src/dataset.py` | Loads per-isotopologue CSVs, computes mass-ratio + `J(J+1)` features, builds the PyG graph |
| `src/model.py` | `HybridIsotopologueGATv2` -- GATv2 trunk + linear physics bypass (mass ratios + J(J+1)) |
| `src/ps_baseline.py` | Per-band polynomial-in-`J(J+1)` fit -- the real ExoMol PS baseline |
| `src/train.py` | `run_loio_cross_validation`, `run_j_extrapolation_split`, training loop |
| `src/plotting.py` | Result plots |
| `main.py` | Orchestrator: builds the graph, auto-resolves which eval(s) to run, saves results |
| `scripts/states_scraping.py` | Ingest a single `.states` file (embedded MARVEL flag) into `data/states/` |
| `scripts/combining_duo_marv.py` | Ingest separate DUO + MARVEL files into `data/states/` |
| `tests/` | pytest coverage for ingestion parsers, mass/ID lookup, edge construction, PS band-fit |

## Known limitations

- Every `data/states/*.csv` currently checked in (CN, PN, SiN, SiO, NO, CO)
  predates `scripts/states_scraping.py` and lacks `ElecState`/`Omega`/
  `parity` columns -- the pipeline falls back to grouping by `v` alone for
  the PS baseline (logged as a warning at load time), even for molecules
  whose `configs/molecules.yaml` entry is already filled in. None of these
  have been re-ingested through the current script yet. `states_scraping.py`
  now derives each molecule's `.states` column layout from its ExoMol JSON
  `.def` file (`src/def_reader.py`) rather than assuming one fixed schema --
  it varies by Hund's coupling case and optional fields (e.g. O2/C2 have no
  `Omega`; SO/O2 have no `gfactor`). Re-ingest via `scripts/states_scraping.py
  --def-json <path>` once each molecule's electronic state and MARVEL source
  code are filled into its registry entry to get proper per-band PS fits.
- Cross-molecule multi-task learning (shared trunk across e.g. CO+SiO) is
  intentionally out of scope here -- see the archived `CO_SiO_testing`
  notebook if that direction is picked back up.

## Archived exploratory notebooks

The notebook lineage that led to this architecture (single-isotopologue PS-
style notebooks, then CO's isotopologue-extrapolation and cross-molecule
notebooks) is archived outside this repo at
`C:\Code\Isotopologue Extrapolation\GNN_Testing_Notebooks\GNN_Isotope_Shifts_notebooks\`,
along with the RASTI PS methodology excerpt (`ps.txt.txt`). See
`notes/predicted_shift_methodology.md` for how each notebook maps to PS/IE.
