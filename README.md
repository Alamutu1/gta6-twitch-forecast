# gta6-twitch-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Alamutu1/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Alamutu1/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Hourly Twitch viewer forecasts for GTA VI launch day: five log-scale quantiles for hours 2–24, learned from 38 past big-game launch curves, anchored to the hour-1 count.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The grader runs `python3 -P -s /bot/main.py`; the program reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes one `<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (else `/output`): 23 rows, hours 2–24, five non-decreasing quantiles each. Standard library only, no network, deterministic, ~0.1s per case.

## How it works

For each comparable launch, take its release-day hourly curve and form the log-ratio curve `ln(viewers_h / viewers_1)` for hours 2–24. The forecast for an hour is the weighted empirical quantiles of those log-ratios across the comparables, exponentiated and multiplied by the target's hour-1 count. Comparables are weighted by how similar the launch is to the target: unlock timing (the US Eastern midnight hour parsed from the unlock text), launch shape (a spike at hour 1 for a single global unlock vs a ramp through the day for rolling midnight-local unlocks), and launch size (a Gaussian kernel on log hour-1 viewers, bandwidth 1.0). Hours with fewer than five comparables fall back to the house baseline's shape (flat at hour 1, spread `0.15·sqrt(h−1)` in log space).

Leave-one-out over the 38 comparables (each one as pseudo-target, the other 37 as training): mean pinball loss **0.404** vs **0.930** for the house baseline on the same targets; on big launches only (hour-1 ≥ 50k viewers, the GTA VI-like ones) **0.198** vs **0.251**. The house baseline scores ~0.387 on the four practice cases.

---

<sub>Entered on [make.fun](https://make.fun/?ref=7qgp7z5x&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
