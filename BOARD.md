# Queue board

Generated UTC: 2026-10-30T18:20:20.350117Z

Queue commit: <code>a85d663ab04ee60fbca70abe73936227e1205da1</code>

Total jobs: 231

| pending | claimed | running | succeeded | failed | interrupted | cancelled |
| --- | --- | --- | --- | --- | --- | --- |
| 71 | 0 | 0 | 155 | 2 | 2 | 1 |

## Campaign progress

Completed = succeeded / total. Claimed remains a separate queue state.

| Dataset | Version | Campaign | Phase | Seed | Total | Succeeded | Running | Pending | Failed | Interrupted | Cancelled | Claimed | Completed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| inbreast | legacy-proposal-v1-converted | campaign:b56e51504619d210d6dc936b7ecbbf4d11471b916de9adee4d7747226c661064 | final | 11 | 50 | 50 | 0 | 0 | 0 | 0 | 0 | 0 | 100.0% |
| inbreast | legacy-proposal-v1-converted | campaign:e91bfa11279d002676ac720299574d2769d39ad3a88434b5e0ce0ef10c053019 | hpo | 101 | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 100.0% |
| inbreast | legacy-proposal-v1-converted | campaign:e91bfa11279d002676ac720299574d2769d39ad3a88434b5e0ce0ef10c053019 | final | 11 | 50 | 50 | 0 | 0 | 0 | 0 | 0 | 0 | 100.0% |
| inbreast-pilot-endurance | inbreast-existing-classification-integrity-20261016 | recipe:3a40d11636818200565d6b3fdea4d5a9f05bde216033d4c00f2698a12c4e93a3 | smoke | 3 | 2 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 50.0% |
| inbreast-private-smoke | local-pilot-subset-technical-v1 | recipe:283423038b9d7630a864d249cdc0137d643729a7a37d7d35905059a1ed88399f | smoke | 3 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 100.0% |
| isic2020 | isic2020-derived-512-v1 | campaign:d5f515f7c13fa36aeb025a58b476c9b9e506088f5ca82f8687e92c90db786333 | hpo | 101 | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 100.0% |
| isic2020 | isic2020-derived-512-v1 | campaign:d5f515f7c13fa36aeb025a58b476c9b9e506088f5ca82f8687e92c90db786333 | final | 11 | 50 | 27 | 0 | 21 | 2 | 0 | 0 | 0 | 54.0% |
| rsna2018 | rsna2018-derived-512-v1 | campaign:3a463fa6953277f0bc1c1b19f1e836c0b2a2a67c1ab628229084a0cb799c105d | hpo | 101 | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 100.0% |
| rsna2018 | rsna2018-derived-512-v1 | campaign:3a463fa6953277f0bc1c1b19f1e836c0b2a2a67c1ab628229084a0cb799c105d | final | 11 | 50 | 0 | 0 | 50 | 0 | 0 | 0 | 0 | 0.0% |
| synthetic-recovery-1024 | v1 | recipe:9facdb72604e40535840a4dcc4da8c59c8cbe0a821085026aaeaed321ab6ad78 | smoke | 3 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0.0% |
| synthetic-tiny | v1 | recipe:283423038b9d7630a864d249cdc0137d643729a7a37d7d35905059a1ed88399f | smoke | 3 | 3 | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 66.7% |
| synthetic-tiny | v1 | recipe:80fa7714d248e25529412da9eb71d4be01943b2576671b289798714897c88ed1 | smoke | 5 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 100.0% |
| synthetic-tiny | v1 | recipe:e217128750003c64c1aab1ae50d18930a377e42fda0335e474f49c68992b0a8e | smoke | 3 | 10 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 100.0% |

## Running jobs

Stale = observation older than 30 min; unknown/future observations are flagged. No automatic recovery.

| Job | Dataset | Campaign | Arm | Fold | Seed | Server | GPU | Epoch | Started UTC | Last observation UTC | Observation age (hh:mm:ss) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

No running jobs.

## Recent failures / interruptions

Latest 20 failure/interruption events, including prior attempts; observation time is preferred.

| Job | Dataset | Campaign | State | Observed UTC | Reason |
| --- | --- | --- | --- | --- | --- |
| 8891225318fb4665ade80fff927c0736e9a112b1fa8cca5635ccd55435bd1c17 | isic2020 | campaign:d5f515f7c13fa36aeb025a58b476c9b9e506088f5ca82f8687e92c90db786333 | failed | 2026-10-30T16:23:06.543526Z | unspecified |
| 90cc18804b9c933168f56037c36613b5d48092540e217cd3ee53e2d10c1d0b81 | isic2020 | campaign:d5f515f7c13fa36aeb025a58b476c9b9e506088f5ca82f8687e92c90db786333 | failed | 2026-10-30T16:21:06.269451Z | unspecified |
| 139aebd2557521a024ba25f47c7234bf64c932edfc43cbbabb506916bb7a087b | inbreast | campaign:b56e51504619d210d6dc936b7ecbbf4d11471b916de9adee4d7747226c661064 | interrupted | 2026-10-28T21:12:46.406889Z | unspecified |
| 97c890c28ff79ea130b9acae53178fc7ecf777a1392680280884ed9ce3a479d0 | synthetic-recovery-1024 | recipe:9facdb72604e40535840a4dcc4da8c59c8cbe0a821085026aaeaed321ab6ad78 | interrupted | 2026-10-16T22:06:09.804768Z | controller exception |
| 97c890c28ff79ea130b9acae53178fc7ecf777a1392680280884ed9ce3a479d0 | synthetic-recovery-1024 | recipe:9facdb72604e40535840a4dcc4da8c59c8cbe0a821085026aaeaed321ab6ad78 | interrupted | 2026-10-16T15:58:24.799606Z | controller exception |
| 97c890c28ff79ea130b9acae53178fc7ecf777a1392680280884ed9ce3a479d0 | synthetic-recovery-1024 | recipe:9facdb72604e40535840a4dcc4da8c59c8cbe0a821085026aaeaed321ab6ad78 | interrupted | 2026-10-16T15:52:27.171754Z | unspecified |
| 5fdd1d7c2cf541a861953d350fc7e63205741810eb384a2bc9a68a7c6c016f03 | inbreast-pilot-endurance | recipe:3a40d11636818200565d6b3fdea4d5a9f05bde216033d4c00f2698a12c4e93a3 | interrupted | 2026-10-16T15:44:34.059438Z | thermal_guard: historical conservative stop at75C; checkpoint preserved. Future user ceiling90C, preventive89C. |
| 12c3dd6b577cc9365a870d1c151d6bfb8a58827f245b53ef8856cced6255be3c | synthetic-tiny | recipe:80fa7714d248e25529412da9eb71d4be01943b2576671b289798714897c88ed1 | interrupted | 2026-10-15T22:19:53.091264Z | unspecified |
| 12c3dd6b577cc9365a870d1c151d6bfb8a58827f245b53ef8856cced6255be3c | synthetic-tiny | recipe:80fa7714d248e25529412da9eb71d4be01943b2576671b289798714897c88ed1 | interrupted | 2026-10-15T22:19:10.154188Z | Operator confirmed stopped child after test-controller preflight signal race |

## Campaign ETA (estimate)

Estimate = median successful current-attempt duration × pending / active GPUs in this campaign.
Active GPUs are unique server + GPU UUID (GPU index fallback) on running jobs observed within 30 min.
Stale/unknown/future observations and claimed jobs add no capacity; running remainder and retries are excluded.

| Dataset | Version | Campaign | Succeeded timed | Median (hh:mm:ss) | Pending | Active GPUs | ETA (hh:mm:ss) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| inbreast | legacy-proposal-v1-converted | campaign:b56e51504619d210d6dc936b7ecbbf4d11471b916de9adee4d7747226c661064 | 50 | 00:02:53 | 0 | 0 | 00:00:00 estimate |
| inbreast | legacy-proposal-v1-converted | campaign:e91bfa11279d002676ac720299574d2769d39ad3a88434b5e0ce0ef10c053019 | 54 | 00:02:52 | 0 | 0 | 00:00:00 estimate |
| inbreast-pilot-endurance | inbreast-existing-classification-integrity-20261016 | recipe:3a40d11636818200565d6b3fdea4d5a9f05bde216033d4c00f2698a12c4e93a3 | 1 | 00:01:40 | 0 | 0 | 00:00:00 estimate |
| inbreast-private-smoke | local-pilot-subset-technical-v1 | recipe:283423038b9d7630a864d249cdc0137d643729a7a37d7d35905059a1ed88399f | 1 | 00:00:38 | 0 | 0 | 00:00:00 estimate |
| isic2020 | isic2020-derived-512-v1 | campaign:d5f515f7c13fa36aeb025a58b476c9b9e506088f5ca82f8687e92c90db786333 | 31 | 01:58:47 | 21 | 0 | unavailable estimate |
| rsna2018 | rsna2018-derived-512-v1 | campaign:3a463fa6953277f0bc1c1b19f1e836c0b2a2a67c1ab628229084a0cb799c105d | 4 | 00:19:27 | 50 | 0 | unavailable estimate |
| synthetic-recovery-1024 | v1 | recipe:9facdb72604e40535840a4dcc4da8c59c8cbe0a821085026aaeaed321ab6ad78 | 0 | unknown | 0 | 0 | 00:00:00 estimate |
| synthetic-tiny | v1 | recipe:283423038b9d7630a864d249cdc0137d643729a7a37d7d35905059a1ed88399f | 2 | 00:00:38 | 0 | 0 | 00:00:00 estimate |
| synthetic-tiny | v1 | recipe:80fa7714d248e25529412da9eb71d4be01943b2576671b289798714897c88ed1 | 2 | 00:03:04 | 0 | 0 | 00:00:00 estimate |
| synthetic-tiny | v1 | recipe:e217128750003c64c1aab1ae50d18930a377e42fda0335e474f49c68992b0a8e | 10 | 00:00:37 | 0 | 0 | 00:00:00 estimate |
