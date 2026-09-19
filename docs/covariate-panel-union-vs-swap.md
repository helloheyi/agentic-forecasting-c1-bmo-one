# Why `SERIES_ID_2Y10Y_SPREAD` is the "swap" candidate

Documents the rationale behind `DROPPED_SERIES_IDS = [SERIES_ID_2Y10Y_SPREAD]`
in `01_BAA10Y_multivariate_backtest.ipynb`'s Section 1
(`COVARIATE_PANEL_VARIANTS["swap"]`). Read this before changing
`DROPPED_SERIES_IDS` to a different series, or before treating the current
choice as more rigorously justified than it actually is — see the honest
limitation in §3.

## 1. The question this was answering

Adding new covariates on top of the existing panel (`"union"`) isn't
guaranteed to beat *replacing* a weak existing covariate with a new one
(`"swap"`) — more feature columns aren't free for a tree model fit fresh at
every walk-forward origin (see the tuning guide §1: more features wants more
capacity *and* more regularization). If an existing covariate is already
contributing mostly noise, it may be better dropped than kept alongside the
new signal. `DROPPED_SERIES_IDS` names which existing covariate(s) to test
dropping.

## 2. How the candidate was actually chosen

A quick, deliberately cheap screen (not a backtest, not a model — plain
Pearson correlation against the `baa10y_change_5b` target, computed directly
from already-registered series, no training involved) was run to sanity-check
whether the *new* H.4.1-derived series showed any signal at all, using a
**few existing covariates as calibration reference points** — not to rank the
new series in isolation, since a bare correlation number is meaningless
without something already-known-useful to compare it against.

Results (Pearson correlation vs. `baa10y_change_5b`, full-history / 2025-2026
window):

| Series | Full history | 2025-2026 |
|---|---|---|
| `vix_level_l1b` | 0.2452 | 0.3228 |
| `hyoas_observed_change_1b_bps_l1b` | 0.3204 | 0.3592 |
| `hyoas_hyg_dgs3_proxy_change_1b_bps_l1b` | 0.2147 | 0.2814 |
| **`ust2y10y_spread_l1b`** | **-0.0370** | **-0.1513** |
| `qt_intensity_l1b` (new) | 0.1398 | 0.1872 |
| `facility_stress_l1b` (new) | -0.0630 | 0.1380 |
| `liquidity_narrative_median_l1b` (new) | -0.0355 | 0.1335 |

`ust2y10y_spread_l1b` was the weakest of the existing covariates checked —
noticeably below VIX and both HYOAS series, and in the same low range as (or
weaker than) several of the brand-new candidate series. That's the entire
basis for picking it as the swap candidate: among what was checked, it looked
the most like "occupying feature budget without earning it."

## 3. The honest limitation — read this before trusting the choice too far

**Only 4 of the 11 `DEFAULT_COVARIATE_SERIES_IDS` members were actually
checked**: `vix_level_l1b`, `ust2y10y_spread_l1b`, and the two HYOAS series.
The other seven — `vix_log_ret_1b_l1b`, `ust10y_level_l1b`,
`fed_funds_level_l1b`, `cpi_mom_logdiff_l1b`, `unemployment_rate_l1b`,
`oil_log_ret_1b_l1b`, `gold_log_ret_1b_l1b`, `dollar_index_log_ret_1b_l1b`,
`nasdaq_log_ret_1b_l1b` — were never run through this screen. It is entirely
possible one of them is *weaker* than `ust2y10y_spread_l1b` and would have been
a better swap candidate. **"2Y10Y spread is the swap candidate" reflects "the
weakest of the four spot-checked," not "the weakest of the eleven."** Treat
this as a reasonable, cheap starting hypothesis, not a completed ranking.

Two further caveats already noted when this check was run, worth repeating
here since they apply directly to this decision:

- **Correlation-only, so nonlinear-blind.** This screen can't see whatever
  nonlinear or interaction value LightGBM might extract from a covariate that
  looks weak under plain linear correlation. A tree model could still find
  `ust2y10y_spread_l1b` useful in combination with other features even though
  its marginal correlation is low.
- **Small-sample noise.** The 2025-2026 window is a few dozen origins;
  differences in this range (e.g. 0.02 vs. 0.20 elsewhere in the same table)
  aren't statistically robust at that sample size. The full-history number
  (-0.037) and the recent-window number (-0.151) agreeing in sign and both
  being clearly negative is somewhat reassuring, but this is still a
  screening heuristic, not a significance test.

## 4. What would make this more rigorous, if it turns out to matter

Run the same correlation screen against the remaining seven covariates before
committing further, and/or run the actual `"union"` vs `"swap"` CRPS
comparison (which this whole toggle exists to enable) and let the backtest
settle it empirically rather than relying on the correlation screen at all —
the screen was only ever meant to produce a cheap, defensible *starting
point* for `DROPPED_SERIES_IDS`, not a final answer.
