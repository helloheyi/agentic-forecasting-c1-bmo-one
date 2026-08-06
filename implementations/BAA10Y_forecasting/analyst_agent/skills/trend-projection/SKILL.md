---
name: trend-projection
description: >-
  Copy-pasteable pandas and numpy patterns for producing a conservative BAA10Y
  spread-change projection from target history, with optional statistical and
  credit-driver skill inputs. Load references/projection-patterns.md before
  writing trend-projection code.
---

# BAA10Y trend-projection skill

This skill synthesizes recent one-business-day BAA10Y behavior with optional
outputs from the `statistical-analysis` and `credit-driver-analysis` skills. It
creates a conservative projection rationale; it does not replace a supplied
model forecast or mechanically set forecast quantiles.

Load `references/projection-patterns.md` via:

```python
load_skill_resource(
    "trend-projection",
    "references/projection-patterns.md",
)
```

## Optional upstream inputs

Both upstream skills are independently optional. Declare the choices before
running the patterns:

```python
use_statistical_analysis = True
use_credit_driver_analysis = True
```

When `use_statistical_analysis` is `True`, run that skill first and retain:

- `analysis_window`
- `recent_center`
- `recent_vol`
- `regime`
- `z_score`

When `use_credit_driver_analysis` is `True`, run that skill first and retain:

- `overall_signal`
- `combined_score`
- `confidence`
- `widening_evidence`
- `tightening_evidence`

If either skill is disabled or unavailable, the projection patterns use a
target-history fallback and explicitly report that the corresponding input was
not used.

## Quick-reference steps

1. Parse the daily diagnostic history from the JSON payload.
2. Use the statistical skill's selected window when enabled; otherwise use a
   conservative 30-observation fallback.
3. Estimate a robust recent target center and dispersion.
4. Use the credit-driver conclusion only as a modest directional tilt when
   enabled and sufficiently supported.
5. Reduce confidence and preserve broad uncertainty under an anomalous or
   elevated-volatility regime.
6. Report the anchor, optional inputs used, projection direction, and
   limitations.

Positive BAA10Y values mean spread widening; negative values mean tightening.
`daily_change_history_csv` always contains one-business-day changes for diagnostics.
The requested 1b, 5b, or 21b target remains separate and is summarized in
`target_summary`. Do not use the daily diagnostic center as a mechanical point
forecast for a 5b or 21b target.

**No scripts in this skill. Do not call `run_skill_script`.**
