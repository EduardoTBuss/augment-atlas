# Job 4cdd9aa0d7ff74f17ef9d4cd12ab4c694b1d8dda575a7dbdacbc609f5ad5a40a

State: **claimed** | phase: final | fold: 3 | train seed: 11 | arm: basic

Attempt: 4c65ebf7c2d249fe8fc1724bbfe76c26 | owner: {'server': 'x299', 'gpu': '1', 'gpu_uuid': 'GPU-e8692fec-c636-9edc-2415-81b5a103531b'} | last observation UTC: 2026-11-03T13:03:24.300771+00:00 | sequence: 1

<!-- augment-atlas-record-v1 -->
```json
{
  "attempt_id": "4c65ebf7c2d249fe8fc1724bbfe76c26",
  "history": [
    {
      "at": "2026-11-03T13:03:24.300771+00:00",
      "attempt_id": "4c65ebf7c2d249fe8fc1724bbfe76c26",
      "kind": "claim",
      "owner": {
        "gpu": "1",
        "gpu_uuid": "GPU-e8692fec-c636-9edc-2415-81b5a103531b",
        "server": "x299"
      },
      "seq": 1,
      "state": "claimed",
      "token": "bf010b6e73914540b3ebad2db7e01a7f"
    }
  ],
  "job_id": "4cdd9aa0d7ff74f17ef9d4cd12ab4c694b1d8dda575a7dbdacbc609f5ad5a40a",
  "owner": {
    "gpu": "1",
    "gpu_uuid": "GPU-e8692fec-c636-9edc-2415-81b5a103531b",
    "server": "x299"
  },
  "schema_version": 1,
  "seq": 1,
  "spec": {
    "arm": "basic",
    "code_commit": "d89fbe87239e32b6412c1103aa21123072aac03a",
    "compatibility": {
      "min_memory_mb": 1000
    },
    "config": "configs/global/rsna2018/final/basic-f3-s11.json",
    "config_sha256": "fb71f045e1da7f836927ab8de5c75e8a845e23936814943db20221e74ceec5e5",
    "dataset": "rsna2018",
    "dataset_version": "rsna2018-derived-512-v1",
    "dependencies": [],
    "fold": 3,
    "global": {
      "campaign_sha256": "3a463fa6953277f0bc1c1b19f1e836c0b2a2a67c1ab628229084a0cb799c105d",
      "candidate_id": null,
      "freeze_kind": "hpo_d",
      "freeze_sha256": "bb130dc00686c6c8c344f2a50169717ac059ef6eac6bcf443f6eeb0d416667e5",
      "plan_sha256": null,
      "pools_sha256": "72d332b60d1b39276d203a5f3db9782593f412b2c5a0a598f6f4a58ae5e22151",
      "search_space_sha256": null
    },
    "job_id": "4cdd9aa0d7ff74f17ef9d4cd12ab4c694b1d8dda575a7dbdacbc609f5ad5a40a",
    "phase": "final",
    "protocol_version": "global-final-v1",
    "recipe_hash": "bb130dc00686c6c8c344f2a50169717ac059ef6eac6bcf443f6eeb0d416667e5",
    "split_hash": "da8070b6592b117dd292a22649bbef3873b9e650face4c51ffdc9360168349c2",
    "train_seed": 11
  },
  "state": "claimed",
  "token": "bf010b6e73914540b3ebad2db7e01a7f",
  "version": 1
}
```
