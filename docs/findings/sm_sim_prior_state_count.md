# Findings: Prior-predictive occupied-state count

_Script: `script/sm_sim_prior_state_count.py`. Tier: SUPPLEMENT (Methods support)._

_Characterizes how many distinct states the finite (weak-limit) sticky-HDP-HMM
prior favors at the production hyperparameters, before any data are observed.
Fit-free Monte-Carlo; no fMRI data are read._

## Method (as run)

- **Prior construction** (mirrors `utils/hdphmm.py`):
  - Base measure `beta ~ GEM(gamma)` via truncated stick-breaking, renormalized.
  - Start distribution `= beta`.
  - Transition row `i ~ Dirichlet(alpha * beta + kappa * rho * e_i)` — the HDP base
    measure plus the sticky self-transition pseudo-count on the diagonal.
- **Hyperparameters:** production setting `gamma=1, kappa=10, alpha=1, rho=1`,
  truncation `K_max=50`. The sticky bias `kappa=10` gives a prior-mean
  self-transition of ~0.91 (= `kappa*rho / (alpha + kappa*rho)`), i.e. a prior-mean
  dwell of ~11 TR (~16 s at TR = 1.49 s) — a hemodynamic timescale.
- **"Occupied" definition:** state fractional occupancy `> 1%` (usage-based),
  matching the 1% training-usage threshold used for the data-driven active-state count.
- **Subject-agnostic:** the prior is shared across participants, so this reads no
  subject data; the finite-T lengths/run-counts (91,547–137,913 TRs; 194–292 runs)
  are representative of the Friends corpus, and data-driven occupied counts vary per
  participant.
- **Estimators:**
  1. _Asymptotic_ — count states whose stationary probability exceeds the
     threshold (long-sequence limit). 3000 prior draws, seed 0.
  2. _Finite-T_ — simulate Markov chains at representative sequence lengths
     (91,547 and 137,913 TRs; 194 and 292 runs), counting usage `> 1%`.

## Results

At the production setting (`gamma = 1`) the prior alone favors only a few states,
far below the truncation capacity:

| Quantity | Value |
|---|---|
| Occupied states (mean) | 2.6 |
| Occupied states (median) | 2 |
| 95% interval | [1, 6] |
| Range over 3000 draws | [1, 9] |
| Finite-T validation (both lengths) | ~2-3 |

The two estimators agree, and results are stable across seeds.

**Against the data.** Applying the same 1% global occupancy threshold to each
participant's full Viterbi-decoded sequence yields 37–42 occupied states — an order
of magnitude above the prior. (That is the usage-based count; the recurrence-based
repertoire of 42–47 used in downstream analyses applies a different per-run 2%
threshold and is not the comparison here.)

**Why this is exact in structure.** With `alpha=1` and `kappa*rho=10`, every prior
transition row has total concentration `alpha + kappa*rho = 11`, so the prior-mean
matrix is `row_i = (1/11)*beta + (10/11)*e_i`, whose stationary distribution is exactly
`beta`; individual sampled matrices vary around it (producing the spread in the table
above). Long-run occupancy therefore tracks `beta`, and at
realistic sequence lengths the 1% usage threshold reduces to roughly `#{beta_j > 0.01}`.

### Sensitivity to gamma

The favored repertoire grows with the global concentration `gamma`:

| gamma | occupied states (median) | 95% interval |
|---|---|---|
| 1 (production) | 2 | [1, 6] |
| 5 | 6 | [1, 12] |

## Reproduce

```bash
python script/sm_sim_prior_state_count.py                 # production setting
python script/sm_sim_prior_state_count.py --gamma 5       # sensitivity
python script/tests/test_sim_prior_state_count.py         # unit tests
```

Output JSON: `$SCRATCH_DIR/prior_state_count/prior_state_count.json`.
