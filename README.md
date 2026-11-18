# GTA VI Twitch CCU forecast — make.fun entry

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/adhityanaufal26/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/adhityanaufal26/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Entry for the [make.fun bounty](https://make.fun/bounty/gta6-twitch-ccu-forecast-01): forecast GTA VI's launch-day Twitch viewers hour by hour.

Forecasts Twitch concurrent viewers in the "Grand Theft Auto VI" category
for hours 2..24 of release day 2026-11-19 (UTC), as five quantiles per hour,
from the hour-1 print alone. Entry for
`gta6-twitch-ccu-forecast-01` ($250, pays top 20, scored by pinball loss on
the log scale).

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input dir, writes `<id>.out` for each. Python 3.13,
standard library only, no network, deterministic, ~0.05 s per case.

## Model

The forecast is the hour-1 print times a ratio. Ratios are learned from the
case's own `comparables` (past big launches' release-day hourly curves),
per hour, from two views blended in log space:

1. **Kernel view** — empirical quantiles of `ln(viewers_h / viewers_1)` over
   the rolling-midnight comparables (the target's unlock pattern), each
   launch weighted by how close its own hour-1 print is to the target's
   (Gaussian kernel on log hour-1, sigma 0.85). Launches that opened at a
   similar size behaved most like the target.
2. **Regression view** — weighted regression of the log ratio on log hour-1
   print over all comparables (kernel sigma 1.5), slope clamped to
   [-1, 0.2]; quantiles from a normal with the weighted residual spread
   (floor 0.15). Bigger openings rise less steeply through the day, and the
   regression carries that trend past the largest comparable.

The blend leans toward the regression view when the kernel's effective
sample shrinks (a target opening larger than anything seen before), so the
forecast never collapses onto the two biggest launches. Median ratios are
clamped to [0.30x, 12.2x]; every quantile is clamped to (0, 1e9] and forced
non-decreasing. With no usable comparables it falls back to a flat forecast.

## Calibration

Leave-one-out over the 37 full-coverage comparables in the practice kit
(each launch held out, forecast from the rest, pinball loss on the log
scale). On the rolling-midnight launches — the only ones shaped like this
target — **loss ~0.228**, vs **~0.357** for the flat house baseline on the
same launches; 90% coverage ~0.91, typical median miss ~35%.

Practice-case `history` stops at the checkpoint, so the practice answers
are not in the case files and cannot be scored locally; the LOO number
above is the calibration. Board leaders score ~0.225, so this entry is in
that range on comparable launches.

## Files

- `main.py` — the entrypoint (only file the program needs)
- `README.md` — this file
- `LICENSE` — MIT
- `.makefun/owner.json` — entry wallet
