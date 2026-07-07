# Notebook work vs. ExoMol's Predicted Shift methodology

Source for the "Predicted shift" (PS) definition: RAS Techniques and Instruments,
Vol. 3, Issue 1, Jan 2024, pp. 565-583 (doi:10.1093/rasti/rzae037), Section 2.2.3
(full text archived at `C:\Code\Isotopologue Extrapolation\GNN_Testing_Notebooks\GNN_Isotope_Shifts_notebooks\ps.txt.txt`,
alongside the exploratory notebooks referenced below — all moved out of this repo
once their ideas were folded into `src/`).

## PS method, in brief

For a given vibronic state (electronic state, v, Omega, rotationless parity), fit
the trend in obs.-calc. (MARVEL - Duo) energy differences as a function of J, then
extrapolate that per-band fit to predict synthetic obs.-calc. corrections for
levels outside the experimentally observed J range. Uncertainty grows with J via
`deltaE_PS = a * J_ext(J_ext+1) + sigma`. Used in the AlO and VO ExoMol line lists.

## Two lineages in `notebooks/`

**Single-isotopologue notebooks** (`CN_testing`, `PN_testing`, `SiN_testing`,
`SiO_testing`) — molecules where ExoMol only has one MARVEL-covered isotopologue.
Each builds a graph over `(v, J)` states with edges for allowed rovibrational
transitions (deltaJ=+-1, deltav=0,-1), trains a GATv2 to predict
`target_residual = EMarv - ECalc` on states with MARVEL data, and evaluates on
held-out states within the *same* isotopologue.

This is structurally a GNN reimplementation of the predicted-shift correction:
instead of a hand-fit quadratic-in-J per vibronic band, the network learns the
obs.-calc. trend directly from the graph of neighbouring (v,J) states and predicts
it for states without experimental coverage. `finalize_line_list()` (in
`CO_parent.ipynb` and `CO_testing.ipynb`) reconstructs a PS-style final line list:
`Final_Energy = MARVEL` where known, `E_calc + predicted_residual` otherwise —
the same construction pattern as a PS-corrected line list.

**Multi-isotopologue notebooks** (`CO_parent`, `CO_testing`, `CO_SiO_testing`) —
extend this into isotopologue extrapolation (IE):

- `CO_parent.ipynb` does the PS-style correction on parent 12C16O only, then in
  "Test New IE on Minor" transfers the learned correction (`deltaML`) to five
  minor CO isotopologues by matching on `(v,J)`, compared against the classical
  IE baseline (plain obs.-calc. transfer, `delta`). This cell is the bridge from
  intra-isotopologue PS-style correction to cross-isotopologue extrapolation.
- `CO_testing.ipynb` generalizes this into one graph spanning all 6 CO
  isotopologues, typed edges (physical rovibrational vs. cross-isotope) with an
  edge embedding, and Leave-One-Isotopologue-Out CV. This became the production
  `HybridIsotopologueGATv2` in `src/model.py`.
- `CO_SiO_testing.ipynb` pushes further into cross-*molecule* multi-task
  learning (shared GCN trunk + molecule-specific heads, PCGrad gradient surgery,
  stratified K-fold, LOO), comparing a joint CO+SiO model against isolated
  single-molecule baselines.

## The connection

PS is a per-band, hand-fit polynomial extrapolation of obs.-calc. in J, filling
in unobserved levels *within one isotopologue*. The single-isotopologue notebooks
ask whether a GNN over the (v,J) transition graph can learn that trend more
generally than a per-band quadratic fit. The CO notebooks then reuse that same
learned obs.-calc. signal as the correction term for classical isotopologue
extrapolation — i.e. PS correction and IE correction are the same underlying
quantity (`EMarv - ECalc`), applied along two different axes: J-extrapolation
within an isotopologue vs. mass-extrapolation across isotopologues.

**Caveat**: none of the notebooks cite the RASTI paper or PS method by name, and
none reimplement the `a*J_ext(J_ext+1)+sigma` uncertainty formula or the
per-vibronic-band split (by electronic state, Omega, parity) that the real PS
method uses — the graph-based GNN is a much less structured, more general
stand-in for it.

## Relevance to the diatomic-landscape goal

If this project generalizes to applying GNNs across ExoMol's diatomic molecules,
the PS method is the existing baseline this work should eventually be benchmarked
against for molecules that only have one MARVEL isotopologue (no IE is possible
there, so PS-style J-extrapolation is the only available comparison). For
molecules with multiple isotopologues (like CO), both PS (J-axis) and IE
(mass-axis) baselines are relevant, and the GNN's edge structure could in
principle be extended to do both simultaneously in one graph.
