"""Forecast GTA VI's launch-day Twitch viewers, hour by hour (make.fun
gta6-twitch-ccu-forecast-01).

Method: learn the shape of a big game launch's Twitch release day from the
comparable launches shipped inside the case file, then anchor that shape to
the one number the case actually gives us — hour 1's viewer count.

For every comparable launch we take its release-day hourly curve and express
each hour as a log-ratio to that launch's own hour-1 count. The forecast for
an hour is the weighted empirical quantiles of those log-ratios across all
comparables, exponentiated and multiplied by this case's hour-1 count. The
weights favour comparables whose launch resembles the target's: unlock
timing, launch shape (a spike at hour 1 vs a ramp through the day as
regions unlock), and launch size. The case's notes (and each comparable's
unlock text) say which kind each launch is.

The spread is honest, not tuned to look confident: the 5% and 95% quantiles
come from the actual spread of past launches, widened slightly with horizon
through the empirical distribution itself (later hours genuinely disagree
more). Where too few comparables have data for an hour we fall back to the
house baseline's flat shape with its widening spread.

Deterministic: fixed sort orders, no randomness, no wall-clock, no network.
Standard library only. Reads every <id>.in in the input directory and writes
<id>.out per case: 23 rows, hours 2..24, five non-decreasing quantiles each.
"""

import json
import math
import os
import re
import sys
from pathlib import Path

# The grader's -P leaves the entry directory OFF sys.path; nothing beside
# main.py is imported, so no path fix is needed.
INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
# House baseline spread, log units per sqrt hour; used only as the fallback
# shape where comparables are too thin.
BASE_SIGMA = 0.15
# How far a comparable's log-ratio spread is stretched around its median.
# 1.0 is the raw empirical spread; tuned on leave-one-out over the
# comparables (see validate.py in the build notes, not shipped).
SPREAD_MULT = 1.0
# Gaussian bandwidth (log units) for weighting comparables by launch size
# (hour-1 magnitude); None disables the size weight. Tuned on
# leave-one-out; see validate.py.
MAG_BANDWIDTH = 1.0
# Master switch for all similarity weighting (unlock, shape, size). When
# False every comparable counts equally.
WEIGHTS_ON = True
# Use the unlock-timing weight (True) or ignore unlock text (False).
USE_UNLOCK = True
# Use the launch-shape (spike vs ramp) weight (True) or ignore it (False).
USE_CLASS = True
# Shrinkage of the empirical median log-ratio toward the flat baseline
# (0.0): 1.0 keeps the learned shape whole, lower values pull the median
# toward "every hour stays at hour 1". Tuned on leave-one-out.
MEDIAN_SHRINK = 1.0
# Minimum comparables with data at an hour before trusting the empirical
# quantiles there; below it we blend toward the baseline shape.
MIN_COUNT = 5


def round2(x):
    """Round to 2 decimals, half up (as JavaScript Math.round(x*100)/100)."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value == int(value) else value


def us_eastern_unlock_hour(text):
    """The hour (UTC) the US Eastern midnight unlock lands, from free text.

    Looks for a clock time tied to Eastern/US in the string, e.g.
    "US Eastern = 04:00Z" or "US: Eastern midnight ... = 05:00Z". Returns
    an int hour 0-23, or None when the text names no usable time.
    """
    if not text:
        return None
    times = [(m.start(), int(m.group(1))) for m in re.finditer(r"(?<![\d:])(\d{1,2}):00(?!:)(?:Z| UTC(?![+-]))", text)]
    if not times:
        return None
    east = [m.end() for m in re.finditer(r"Eastern|US East|East Coast", text, re.IGNORECASE)]
    if not east:
        return times[0][1]
    anchor = east[0]
    # The time mentioned closest to the word "Eastern".
    _, hour = min(times, key=lambda t: abs(t[0] - anchor))
    return hour


def launch_class_from_text(text):
    """'spike' or 'ramp' for a launch, from its unlock description.

    A single global/simultaneous unlock spikes at hour 1; a rolling
    midnight-local unlock ramps through the day as regions come online.
    Returns None when the text says neither.
    """
    if not text:
        return None
    low = text.lower()
    if "rolling midnight" in low or "midnight local" in low:
        return "ramp"
    if "single global" in low or "simultaneous" in low:
        return "spike"
    return None


def launch_class_from_curve(curve):
    """'spike' or 'ramp' from a launch's own release-day curve.

    Spike: hour 1 already holds at least half the day's peak. Ramp: the
    peak comes later, after regions unlock through the day.
    """
    if not curve or 1 not in curve:
        return None
    peak = max(curve.values())
    return "spike" if curve[1] >= 0.5 * peak else "ramp"


def release_day_curve(comp):
    """Hour -> viewers for the comparable's release day (UTC hours 1..24).

    The comparable's hourly points cover release-14d to release+3d; keep only
    the 24 points of day_start's own date, keyed by UTC hour (1-based).
    """
    day = comp.get("day_start", "")[:10]
    curve = {}
    for p in comp.get("hourly", []):
        start = p.get("from", "")
        if not start.startswith(day):
            continue
        try:
            hour = int(start[11:13]) + 1
            viewers = p.get("viewers")
        except (ValueError, TypeError):
            continue
        if viewers is None or viewers <= 0:
            continue
        if 1 <= hour <= 24:
            curve[hour] = float(viewers)
    return curve


def log_ratios(curve):
    """Hour -> ln(viewers_h / viewers_1) for hours 2..24 with data."""
    if 1 not in curve:
        return {}
    v1 = curve[1]
    return {h: math.log(v / v1) for h, v in sorted(curve.items()) if h >= 2}


def unlock_weight(comp_hour, target_hour):
    """Similarity weight from unlock timing (1.0 = same unlock hour)."""
    if comp_hour is None or target_hour is None:
        return 0.5
    diff = abs(comp_hour - target_hour)
    if diff <= 1:
        return 1.0
    if diff <= 3:
        return 0.6
    return 0.3


def weighted_quantile(pairs, level):
    """Weighted empirical quantile of (value, weight) pairs at `level`.

    Pairs are sorted by value; the quantile is the smallest value whose
    cumulative weight reaches level * total weight (linear interpolation
    between the bracketing values when the level falls between steps).
    """
    ordered = sorted(pairs, key=lambda p: p[0])
    total = sum(w for _, w in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0] if ordered else 0.0
    target = level * total
    acc = 0.0
    prev_value = ordered[0][0]
    for value, w in ordered:
        acc += w
        if acc >= target:
            if acc - w <= 0:
                return value
            # Interpolate between the previous step and this one.
            frac = (target - (acc - w)) / w if w > 0 else 0.0
            return prev_value + frac * (value - prev_value)
        prev_value = value
    return ordered[-1][0]


def forecast_case(case):
    """Forecast hours 2..24 for one case dict, as the grader's .out format."""
    if case.get("levels") != LEVELS:
        raise ValueError("case asks for levels %r, not %r" % (case.get("levels"), LEVELS))
    known = {k["hour"]: k for k in case.get("known", [])}
    if 1 not in known:
        raise ValueError("case has no hour-1 print")
    p1 = max(1.0, float(known[1]["viewers"]))

    target_hour = us_eastern_unlock_hour(case.get("notes") or "")
    target_class = launch_class_from_text(case.get("notes") or "")

    # Learn one log-ratio curve per comparable, weighted by how similar the
    # launch is to the target: unlock timing, launch shape (spike at hour 1
    # vs ramp through the day), and launch size (hour-1 magnitude).
    curves = []
    for comp in case.get("comparables") or []:
        curve = release_day_curve(comp)
        ratios = log_ratios(curve)
        if not ratios:
            continue
        w = unlock_weight(us_eastern_unlock_hour(comp.get("unlock") or ""), target_hour) if USE_UNLOCK else 1.0
        comp_class = launch_class_from_curve(curve)
        if USE_CLASS:
            if target_class is not None and comp_class is not None:
                w *= 1.0 if comp_class == target_class else 0.4
            elif target_class is not None or comp_class is not None:
                w *= 0.7
        if MAG_BANDWIDTH and 1 in curve:
            diff = math.log(max(1.0, curve[1])) - math.log(p1)
            w *= math.exp(-0.5 * (diff / MAG_BANDWIDTH) ** 2)
        if not WEIGHTS_ON:
            w = 1.0
        curves.append((w, ratios))

    rows = []
    for row in case.get("forecast", []):
        h = row["hour"]
        pairs = [(ratios[h], w) for w, ratios in curves if h in ratios]
        if len(pairs) >= MIN_COUNT:
            median = weighted_quantile(pairs, 0.5)
            qs = []
            for level in LEVELS:
                q = weighted_quantile(pairs, level)
                # Shrink the whole distribution toward flat (median 0) while
                # keeping its spread: q = shrink*median + spread*(q - median).
                q = MEDIAN_SHRINK * median + SPREAD_MULT * (q - median)
                qs.append(q)
        else:
            # Too thin: the house baseline's flat median, widening spread.
            qs = [Z[i] * BASE_SIGMA * math.sqrt(h - 1) for i in range(len(LEVELS))]
        # Quantiles must be non-decreasing, finite, above 0, at most 1e9.
        values = []
        prev = None
        for q in qs:
            v = p1 * math.exp(max(-30.0, min(30.0, q)))
            v = min(max(v, 1.0), 1e9)
            v = round2(v)
            if prev is not None and v < prev:
                v = prev
            values.append(v)
            prev = v
        rows.append({"hour": h, "q": values})
    rows.sort(key=lambda r: r["hour"])
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / ("%s.out" % path.stem)
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
