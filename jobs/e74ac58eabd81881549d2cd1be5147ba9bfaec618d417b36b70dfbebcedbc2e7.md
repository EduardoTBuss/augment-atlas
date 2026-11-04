# Job e74ac58eabd81881549d2cd1be5147ba9bfaec618d417b36b70dfbebcedbc2e7

State: **running** | phase: final | fold: 1 | train seed: 11 | arm: basic_gamma_heavy

Attempt: 01ef10d10fec415fa113f90a4c100bd2 | owner: {'gpu': '0', 'gpu_uuid': 'GPU-add635a3-5ecf-0c1c-2f3e-113c84c5f4ea', 'server': 'x299'} | last observation UTC: 2026-11-04T15:11:11.274162+00:00 | sequence: 2

Epoch: 0/? | progress observed UTC: 2026-11-04T15:11:11.274162+00:00

<!-- augment-atlas-record-v1 -->
```json
{
  "attempt_id": "01ef10d10fec415fa113f90a4c100bd2",
  "history": [
    {
      "at": "2026-11-04T15:10:29.078906+00:00",
      "attempt_id": "01ef10d10fec415fa113f90a4c100bd2",
      "kind": "claim",
      "owner": {
        "gpu": "0",
        "gpu_uuid": "GPU-add635a3-5ecf-0c1c-2f3e-113c84c5f4ea",
        "server": "x299"
      },
      "seq": 1,
      "state": "claimed",
      "token": "d4319dd937ef461bb12e6e0fd7ce3e8f"
    },
    {
      "at": "2026-11-04T15:11:21.653945+00:00",
      "attempt_id": "01ef10d10fec415fa113f90a4c100bd2",
      "epoch": 0,
      "event_id": "1ad116864df843de91d870b0a909cdbf",
      "kind": "update",
      "observed_at_utc": "2026-11-04T15:11:11.274162+00:00",
      "payload_hash": "e3d5f19c838f2e2d2453ce03a1ee675188893cb159d7181c0a8aab581c683fab",
      "seq": 2,
      "state": "running"
    }
  ],
  "job_id": "e74ac58eabd81881549d2cd1be5147ba9bfaec618d417b36b70dfbebcedbc2e7",
  "owner": {
    "gpu": "0",
    "gpu_uuid": "GPU-add635a3-5ecf-0c1c-2f3e-113c84c5f4ea",
    "server": "x299"
  },
  "schema_version": 1,
  "seq": 2,
  "spec": {
    "arm": "basic_gamma_heavy",
    "code_commit": "d89fbe87239e32b6412c1103aa21123072aac03a",
    "compatibility": {
      "min_memory_mb": 1000
    },
    "config": "configs/global/rsna2018/final/basic_gamma_heavy-f1-s11.json",
    "config_sha256": "d41a84d628bdad35da4453a710d337d0276f3c259a42c1d799e6bd17d0ec3831",
    "dataset": "rsna2018",
    "dataset_version": "rsna2018-derived-512-v1",
    "dependencies": [],
    "fold": 1,
    "global": {
      "campaign_sha256": "3a463fa6953277f0bc1c1b19f1e836c0b2a2a67c1ab628229084a0cb799c105d",
      "candidate_id": null,
      "freeze_kind": "hpo_d",
      "freeze_sha256": "bb130dc00686c6c8c344f2a50169717ac059ef6eac6bcf443f6eeb0d416667e5",
      "plan_sha256": null,
      "pools_sha256": "72d332b60d1b39276d203a5f3db9782593f412b2c5a0a598f6f4a58ae5e22151",
      "search_space_sha256": null
    },
    "job_id": "e74ac58eabd81881549d2cd1be5147ba9bfaec618d417b36b70dfbebcedbc2e7",
    "phase": "final",
    "protocol_version": "global-final-v1",
    "recipe_hash": "bb130dc00686c6c8c344f2a50169717ac059ef6eac6bcf443f6eeb0d416667e5",
    "split_hash": "400ba8d70eda9d6c8f295d34431dbc75e236e568ec880010190ce46a03a7acfe",
    "train_seed": 11
  },
  "state": "running",
  "token": "d4319dd937ef461bb12e6e0fd7ce3e8f",
  "version": 2
}
```
