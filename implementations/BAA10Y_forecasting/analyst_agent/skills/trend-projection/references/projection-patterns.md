# BAA10Y trend-projection patterns

These read-only patterns produce a conservative BAA10Y requested-target
projection from the task payload. They can independently use results from
`statistical-analysis` and `credit-driver-analysis`, but remain usable when
either is disabled or unavailable.

The result is a transparent analytical anchor and directional rationale. It
does not replace the supplied model point forecast or calculate final forecast
quantiles.

---

## Section 0: Choose optional inputs and parse target history

All data enters through the JSON task payload. Do not open files, connect to
external data, or install packages. `target_history_csv` contains history on
the requested 1b, 5b, or 21b target scale. `daily_change_history_csv` contains
the shared one-business-day diagnostic history used by
`statistical-analysis`.

```python
import io
import json

import numpy as np
import pandas as pd

# These switches are independent. Set either to False when its upstream skill
# was not run or its result is unavailable.
use_statistical_analysis = True
use_credit_driver_analysis = True

target_history = pd.read_csv(
    io.StringIO(payload["target_history_csv"]),
    parse_dates=["date"],
)
required_columns = {"date", "spread_change_bps"}
missing_columns = required_columns.difference(target_history.columns)
if missing_columns:
    raise ValueError(
        "target_history_csv must contain date and spread_change_bps; "
        f"missing: {sorted(missing_columns)}."
    )

target_history = (
    target_history[["date", "spread_change_bps"]]
    .assign(
        spread_change_bps=lambda frame: pd.to_numeric(
            frame["spread_change_bps"],
            errors="raise",
        )
    )
    .dropna()
    .drop_duplicates(subset="date", keep="last")
    .sort_values("date")
    .reset_index(drop=True)
)

if len(target_history) < 15:
    raise ValueError(
        "Need at least 15 target observations for a conservative projection; "
        f"received {len(target_history)}."
    )

statistical_inputs_used = None
if use_statistical_analysis:
    statistical_inputs_used = {
        "analysis_window": int(analysis_window),
        "regime": regime,
        "z_score": float(z_score),
    }

# Daily diagnostics choose a regime and confidence adjustment; they do not set
# a target-scale estimation window. Use a six-month target history when
# available, independent of the 15/30/45 daily diagnostic window.
target_window = min(126, len(target_history))
recent_target = target_history.tail(target_window)["spread_change_bps"].astype(float)

print(
    json.dumps(
        {
            "target_series_id": payload["target_series_id"],
            "target_window": target_window,
            "use_statistical_analysis": use_statistical_analysis,
            "use_credit_driver_analysis": use_credit_driver_analysis,
            "statistical_inputs": statistical_inputs_used,
        }
    )
)
```

**Interpretation:** Disabling a skill does not prevent projection. It removes
only that skill's contribution and records a limitation in the final result.

---

## Pattern 1: Establish a robust target-history anchor

Use the requested target's median as the projection anchor because BAA10Y
changes can contain large shocks. The target window is independent of the daily
diagnostic window. For 5b/21b cumulative targets, estimate dispersion from
approximately non-overlapping observations so adjacent rolling windows are not
mistaken for independent outcomes.

```python
target_window_business_days = int(payload["target_window_business_days"])
if target_window_business_days == 1:
    dispersion_sample = recent_target
    dispersion_method = "all target observations"
else:
    dispersion_sample = recent_target.iloc[::target_window_business_days]
    dispersion_method = (
        f"every {target_window_business_days}th target observation"
    )

if len(dispersion_sample) < 2:
    raise ValueError(
        "Need at least two non-overlapping target observations to estimate "
        f"dispersion for a {target_window_business_days}b target."
    )

anchor_center = float(recent_target.median())
anchor_vol = float(dispersion_sample.std(ddof=1))
anchor_source = "requested-target history"

if not np.isfinite(anchor_vol) or anchor_vol <= 0:
    raise ValueError("Cannot project: recent target dispersion must be positive.")

print(
    f"ANCHOR: {anchor_center:+.2f} bps | vol={anchor_vol:.2f} bps | "
    f"source={anchor_source} | target_window={target_window} | "
    f"dispersion_sample={len(dispersion_sample)} ({dispersion_method})"
)
```

**Example output:**

```text
ANCHOR: +1.75 bps | vol=7.40 bps | source=requested-target history | target_window=126 | dispersion_sample=26 (every 5th target observation)
```

**Interpretation:** The anchor reflects recent realized BAA10Y changes on the
requested target's scale. Its dispersion uses non-overlapping observations for
rolling 5b/21b targets and is not a mechanically extrapolated trend line.

---

## Pattern 2: Apply an optional, bounded driver tilt

Credit-driver evidence can confirm or challenge the anchor but should not
override it. Apply a small tilt only when the driver skill reports a clear
direction with at least medium confidence. The tilt is capped at one-quarter
of recent dispersion.

```python
driver_inputs_used = None
driver_tilt_bps = 0.0
projection_center = anchor_center

if use_credit_driver_analysis:
    driver_inputs_used = {
        "overall_signal": overall_signal,
        "combined_score": float(combined_score),
        "confidence": confidence,
        "widening_evidence": list(widening_evidence),
        "tightening_evidence": list(tightening_evidence),
    }

    confidence_scale = {"high": 1.0, "medium": 0.5}.get(confidence, 0.0)
    direction = {
        "widening": 1.0,
        "tightening": -1.0,
    }.get(overall_signal, 0.0)
    evidence_scale = min(abs(float(combined_score)) / 2.0, 1.0)

    max_tilt_bps = 0.25 * anchor_vol
    driver_tilt_bps = (
        direction * confidence_scale * evidence_scale * max_tilt_bps
    )

projection_center = anchor_center + driver_tilt_bps
print(
    f"DRIVER_TILT: {driver_tilt_bps:+.2f} bps | "
    f"projection_center={projection_center:+.2f} bps"
)
```

**Example output:**

```text
DRIVER_TILT: +0.39 bps | projection_center=+0.74 bps
```

**Interpretation:** A driver result can make the directional rationale more or
less persuasive, but the cap prevents correlated market signals from causing a
large mechanical shift. The tilt is measured on the requested target's scale.

---

## Pattern 3: Produce a conservative projection summary

Higher daily volatility and a daily anomaly reduce confidence; they do not
determine whether BAA10Y will widen or tighten. The projection result
explicitly records which optional inputs were used.

```python
limitations = []
if not use_statistical_analysis:
    limitations.append(
        "Statistical-analysis was not used; daily regime and anomaly diagnostics are unavailable."
    )
if not use_credit_driver_analysis:
    limitations.append(
        "Credit-driver-analysis was not used; the projection has no covariate direction tilt."
    )

if use_statistical_analysis and (
    regime in {"elevated", "extreme", "stressed"} or abs(z_score) > 2.5
):
    projection_confidence = "low"
    limitations.append(
        "Elevated daily volatility or an anomalous daily observation warrants broad uncertainty."
    )
elif use_credit_driver_analysis and confidence == "low":
    projection_confidence = "low"
elif use_credit_driver_analysis and confidence in {"medium", "high"}:
    projection_confidence = "moderate"
else:
    projection_confidence = "moderate"

if projection_center > 0:
    projection_direction = "modest widening bias"
elif projection_center < 0:
    projection_direction = "modest tightening bias"
else:
    projection_direction = "near-zero directional bias"

projection_summary = {
    "target_series_id": payload["target_series_id"],
    "horizon_business_days": int(payload["horizons"][0]),
    "anchor_center_bps": round(anchor_center, 2),
    "driver_tilt_bps": round(driver_tilt_bps, 2),
    "projection_center_bps": round(projection_center, 2),
    "recent_dispersion_bps": round(anchor_vol, 2),
    "direction": projection_direction,
    "confidence": projection_confidence,
    "used_statistical_analysis": use_statistical_analysis,
    "used_credit_driver_analysis": use_credit_driver_analysis,
    "statistical_inputs": statistical_inputs_used,
    "driver_inputs": driver_inputs_used,
    "limitations": limitations,
}
print(json.dumps(projection_summary))
```

**Example output:**

```text
{"target_series_id": "baa10y_change_5b", "horizon_business_days": 5, "anchor_center_bps": 1.75, "driver_tilt_bps": 0.46, "projection_center_bps": 2.21, "recent_dispersion_bps": 7.4, "direction": "modest widening bias", "confidence": "moderate", "used_statistical_analysis": true, "used_credit_driver_analysis": true, "statistical_inputs": {"analysis_window": 30, "regime": "normal", "z_score": 0.4}, "driver_inputs": {"overall_signal": "widening", "combined_score": 1.0, "confidence": "medium", "widening_evidence": ["hyoas_observed_change_1b_bps_l1b"], "tightening_evidence": []}, "limitations": []}
```

**Interpretation:** Use this summary to explain how daily credit conditions and
covariate evidence relate to a requested-target forecast. Keep final forecast
ownership and required quantiles under the forecasting agent's control.
