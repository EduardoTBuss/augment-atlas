# Queue board

Generated UTC: 2026-10-29T12:00:00Z

Queue commit: <code>cccccccccccccccccccccccccccccccccccccccc</code>

Total jobs: 17

| pending | claimed | running | succeeded | failed | interrupted | cancelled |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | 1 | 4 | 3 | 1 | 1 | 1 |

## Campaign progress

Completed = succeeded / total. Claimed remains a separate queue state.

| Dataset | Version | Campaign | Phase | Seed | Total | Succeeded | Running | Pending | Failed | Interrupted | Cancelled | Claimed | Completed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic-alpha | synthetic-alpha-v1 | campaign:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa | hpo | 7 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 100.0% |
| synthetic-alpha | synthetic-alpha-v1 | campaign:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa | final | 11 | 12 | 1 | 4 | 3 | 1 | 1 | 1 | 1 | 8.3% |
| synthetic-alpha | synthetic-alpha-v1 | campaign:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa | final | 23 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0.0% |
| synthetic-beta | synthetic-beta-v1 | campaign:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb | final | 11 | 2 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 50.0% |
| synthetic-tiny | synthetic-tiny-v1 | recipe:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff | smoke | 11 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0.0% |

## Running jobs

Stale = observation older than 30 min; unknown/future observations are flagged. No automatic recovery.

| Job | Dataset | Campaign | Arm | Fold | Seed | Server | GPU | Epoch | Started UTC | Last observation UTC | Observation age (hh:mm:ss) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0909d128028613be6e39f9643c69d42746d7f9740eacfebca8ae35d16758eed1 | synthetic-alpha | campaign:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa | running-boundary | 0 | 11 | synthetic-node | 1 | 3/20 | 2026-10-29T11:00:00Z | 2026-10-29T11:30:00Z | 00:30:00 |
| 2ce9de08d400e060cf1d2a37cf19937cf5e955173eeb676c9d581f0be9e7e7b7 | synthetic-alpha | campaign:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa | running-stale | 0 | 11 | synthetic-node | 2 | 3/20 | 2026-10-29T11:00:00Z | 2026-10-29T11:29:59Z | 00:30:01 stale |
| 97f2c5b928b4a54a76937a93f8da582d11d81e4702684025858ae5228c964b23 | synthetic-alpha | campaign:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa | running-fresh | 0 | 11 | synthetic-node | 0 | 3/20 | 2026-10-29T11:00:00Z | 2026-10-29T11:55:00Z | 00:05:00 |
| c47cac4ffc949013f53ab33689665433cb9cd30af4d662afa16a039171fbb400 | synthetic-alpha | campaign:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa | running-same-gpu | 0 | 11 | synthetic-node | GPU-synthetic-0 | 3/20 | 2026-10-29T11:00:00Z | 2026-10-29T11:55:00Z | 00:05:00 |

## Recent failures / interruptions

Latest 20 failure/interruption events, including prior attempts; observation time is preferred.

| Job | Dataset | Campaign | State | Observed UTC | Reason |
| --- | --- | --- | --- | --- | --- |
| 8229fc5d0f3ca5b0ec9bdd85bd4a3308ceb3ab2cd0205a4dc8f16722fa52b330 | synthetic-alpha | campaign:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa | failed | 2026-10-29T11:55:00Z | gpu_error &#124; synthetic failure |
| 32c2c988a973e821caadeabf9f5c0d0197104121636f224640206946a55c8474 | synthetic-alpha | campaign:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa | interrupted | 2026-10-29T11:55:00Z | thermal_stop |

## Campaign ETA (estimate)

Estimate = median successful current-attempt duration × pending / active GPUs in this campaign.
Active GPUs are unique server + GPU UUID (GPU index fallback) on running jobs observed within 30 min.
Stale/unknown/future observations and claimed jobs add no capacity; running remainder and retries are excluded.

| Dataset | Version | Campaign | Succeeded timed | Median (hh:mm:ss) | Pending | Active GPUs | ETA (hh:mm:ss) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic-alpha | synthetic-alpha-v1 | campaign:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa | 2 | 00:20:00 | 4 | 2 | 00:40:00 estimate |
| synthetic-beta | synthetic-beta-v1 | campaign:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb | 1 | 00:01:00 | 1 | 0 | unavailable estimate |
| synthetic-tiny | synthetic-tiny-v1 | recipe:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff | 0 | unknown | 1 | 0 | unavailable estimate |

## Seed requests

| Request | Dataset | Seeds | Node | Status | Succeeded / expected | Jobs per seed | Detail |
| --- | --- | --- | --- | --- | --- | --- | --- |

No seed requests supplied.
