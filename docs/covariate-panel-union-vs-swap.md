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

## 5. Update (2026-09-22): the full screen, run against all 21 covariates

§3's limitation — "only 4 of 11 `DEFAULT_COVARIATE_SERIES_IDS` checked" — is
now resolved. The same correlation screen was re-run against all 11
`DEFAULT_COVARIATE_SERIES_IDS`, all 9 `H41_RAG_OPTIONAL_COVARIATE_SERIES_IDS`,
and both `HYOAS_OPTIONAL_COVARIATE_SERIES_IDS` (21 series total; `gold_log_ret_1b_l1b`
failed to fetch — see §5c), against `baa10y_change_5b`, using the notebook's
current `DATA_HISTORY_START="2007-01-01"` (the original screen's "full
history" window predates that change and may have used a shorter span).
Full results: `data/predictions/covariate_correlation_screen_2007start.csv`.
~5,100-5,140 observations per series (2007-01-01 to 2026-09-22) for the
"full history" column, ~430-450 for "recent" (2025-2026).

### 5a. The original rationale holds, and is now better supported

| Series | Panel | Full-history corr | Recent (2025-26) corr |
|---|---|---:|---:|
| `ust2y10y_spread_l1b` | DEFAULT | **-0.0124** | -0.0989 |
| `ust10y_level_l1b` | DEFAULT | 0.0238 | **-0.2004** |
| `liquidity_narrative_median_l1b` | H4.1, *selected* | -0.0409 | 0.1335 |
| `cpi_mom_logdiff_l1b` | DEFAULT | 0.0429 | -0.0563 |
| `fed_funds_level_l1b` | DEFAULT | 0.0474 | 0.1149 |
| `credit_stress_narrative_max_l1b` | H4.1 | -0.0664 | -0.1139 |
| `oil_log_ret_1b_l1b` | DEFAULT | -0.0715 | -0.1288 |
| `dollar_index_log_ret_1b_l1b` | DEFAULT | 0.0744 | 0.0402 |
| `facility_stress_l1b` | H4.1 | -0.0752 | 0.1380 |
| `vix_log_ret_1b_l1b` | DEFAULT | 0.0773 | 0.2192 |
| `reserve_balances_level_l1b` | H4.1 | -0.0791 | 0.0796 |
| `unemployment_rate_l1b` | DEFAULT | -0.0970 | -0.1153 |
| `credit_stress_narrative_median_l1b` | H4.1 | -0.1022 | -0.0596 |
| `intervention_narrative_max_l1b` | H4.1 | -0.1131 | 0.1999 |
| `nasdaq_log_ret_1b_l1b` | DEFAULT | -0.1194 | -0.2229 |
| `liquidity_narrative_max_l1b` | H4.1 | -0.1522 | 0.0139 |
| `qt_intensity_l1b` | H4.1, *selected* | 0.1540 | 0.1872 |
| `intervention_narrative_median_l1b` | H4.1 | 0.1598 | 0.0127 |
| `hyoas_hyg_dgs3_proxy_change_1b_bps_l1b` | HYOAS | 0.2145 | 0.2754 |
| `vix_level_l1b` | DEFAULT | 0.2670 | 0.3315 |
| `hyoas_observed_change_1b_bps_l1b` | HYOAS | 0.3162 (n=779, short history) | 0.3451 |
| `gold_log_ret_1b_l1b` | DEFAULT | fetch failed (n=0) | — |

`ust2y10y_spread_l1b` is the weakest of all 21 successfully-measured series,
not just the weakest of the 4 originally spot-checked. §2's original basis —
"among what was checked, it looked the most like occupying feature budget
without earning it" — upgrades from a 4-series spot-check to a full-panel
result. §3's two caveats (correlation-only/nonlinear-blind, small-sample
noise) still apply unchanged; nothing about them is resolved by checking more
series, only the *coverage* gap is.

### 5b. New: `ust10y_level_l1b` is a second, comparably strong drop candidate

Second-weakest full-history (0.0238), and the single most negative of all 21
in the recent window (-0.2004) — worse there than `ust2y10y_spread_l1b`
itself. It's also the only raw *level* feature in a panel that's otherwise
changes/log-returns; a non-stationary level over a 19-year window spanning
near-zero rates (2008-2021) through the post-2022 hiking cycle risks a tree
model using it more as an "which era is this" proxy than a genuine
forward-looking signal — a structural reason to be suspicious of it beyond
just its correlation number. Recommendation: add a third `COVARIATE_PANEL`
variant, `"swap2"`, dropping both `SERIES_ID_2Y10Y_SPREAD` and
`SERIES_ID_10Y_YIELD`, and compare it against the existing `"union"`/`"swap"`
CRPS results rather than assuming a two-series drop is strictly better —
this is still a correlation-screen hypothesis, not a backtest result (§3's
caveats apply here too).

### 5c. Separate finding: `gold_log_ret_1b_l1b` silently fails to fetch

Not a predictive-value question — a data-availability bug.
`build_baa10y_multivariate_service` warns
(`Skipping unavailable covariate 'gold_log_ret_1b_l1b': Could not fetch any
configured gold FRED series (GOLDAMGBD228NLBM, GOLDPMGBD228NLBM)`) and
silently drops it rather than raising, so the `"union"` panel has been
running with 10 of its 11 nominal `DEFAULT_COVARIATE_SERIES_IDS` in this
environment, matching the "12 / 13 registered" count Section 1's own printed
diagnostics already show — easy to miss since nothing fails loudly. Those
FRED series ids may be deprecated/renamed; worth fixing independently of the
swap-candidate question.

### 5d. Caveat on the current `SELECTED_H41_RAG_SERIES_IDS` choice

`liquidity_narrative_median_l1b` — one of the two H.4.1 series actually
selected for the panel — has a full-history correlation (-0.0409) about as
weak as `ust2y10y_spread_l1b` itself. `qt_intensity_l1b`, the other selected
series, looks comfortably stronger (0.1540, ahead of most of the DEFAULT
panel). This is not a recommendation to drop `liquidity_narrative_median` —
narrative "crisis flag" scores are plausible candidates for real nonlinear or
threshold-triggered value a linear correlation can't see, arguably more so
than a macro level/return series — but it's worth knowing going in rather
than assuming both selected series are equally well-supported by this
screen.

### 5e. A mechanism-level point the original write-up didn't cover

2Y10Y curve inversion is a well-documented *long-lead* recession/regime
indicator (historically ~12-24 month lead times before recessions/credit
stress), not a mechanism for moving credit spreads day-to-day. This screen
tests a 1-business-day-lagged value against a 5-business-day-forward target —
essentially the wrong horizon for the channel through which the yield curve
is economically believed to matter. So this result is good evidence 2Y10Y
isn't pulling weight *at this specific short horizon*, but shouldn't be
read as "the yield curve doesn't matter for credit spreads" more broadly —
a longer-horizon task (e.g. `baa10y_change_21b`, or a purpose-built
longer-lag feature) might tell a different story. Out of scope for
`DROPPED_SERIES_IDS` as currently used (all three tasks share one panel
choice — see the tuning guide's one-shared-search discussion), but worth
keeping in mind before generalizing this screen's conclusion beyond the 5b
horizon it was actually run at.
