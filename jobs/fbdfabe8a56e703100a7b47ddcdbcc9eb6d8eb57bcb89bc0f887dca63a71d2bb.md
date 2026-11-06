# Job fbdfabe8a56e703100a7b47ddcdbcc9eb6d8eb57bcb89bc0f887dca63a71d2bb

State: **claimed** | phase: final | fold: 4 | train seed: 19 | arm: none

Attempt: ad62a69af7a24954a3d83b29e2cdb504 | owner: {'server': 'qflex', 'gpu': '0', 'gpu_uuid': 'GPU-d98acf69-eea9-96f1-4f5e-d5af8e7190d1'} | last observation UTC: 2026-11-06T23:26:21.162788+00:00 | sequence: 1

<!-- augment-atlas-record-v1 -->
```json
{
  "attempt_id": "ad62a69af7a24954a3d83b29e2cdb504",
  "history": [
    {
      "at": "2026-11-06T23:26:21.162788+00:00",
      "attempt_id": "ad62a69af7a24954a3d83b29e2cdb504",
      "kind": "claim",
      "owner": {
        "gpu": "0",
        "gpu_uuid": "GPU-d98acf69-eea9-96f1-4f5e-d5af8e7190d1",
        "server": "qflex"
      },
      "seq": 1,
      "state": "claimed",
      "token": "180f06eeb53d45579c85c3cae3014964"
    }
  ],
  "job_id": "fbdfabe8a56e703100a7b47ddcdbcc9eb6d8eb57bcb89bc0f887dca63a71d2bb",
  "owner": {
    "gpu": "0",
    "gpu_uuid": "GPU-d98acf69-eea9-96f1-4f5e-d5af8e7190d1",
    "server": "qflex"
  },
  "schema_version": 1,
  "seq": 1,
  "spec": {
    "arm": "none",
    "code_commit": "f58b8d4b29b332d915e19d6bca2bec30cd651aa1",
    "compatibility": {
      "min_memory_mb": 1000
    },
    "config": "configs/global/rsna2018/final/stage-2/none-f4-s19.json",
    "config_sha256": "5c50b20ddc70e4238b77c696130225661224199430db3ae939c390c4dc2237a8",
    "dataset": "rsna2018",
    "dataset_version": "rsna2018-derived-512-v1",
    "dependencies": [],
    "fold": 4,
    "global": {
      "campaign_sha256": "3a463fa6953277f0bc1c1b19f1e836c0b2a2a67c1ab628229084a0cb799c105d",
      "candidate_id": null,
      "freeze_kind": "hpo_d",
      "freeze_sha256": "bb130dc00686c6c8c344f2a50169717ac059ef6eac6bcf443f6eeb0d416667e5",
      "plan_sha256": null,
      "pools_sha256": "72d332b60d1b39276d203a5f3db9782593f412b2c5a0a598f6f4a58ae5e22151",
      "search_space_sha256": null
    },
    "job_id": "fbdfabe8a56e703100a7b47ddcdbcc9eb6d8eb57bcb89bc0f887dca63a71d2bb",
    "phase": "final",
    "protocol_version": "global-final-v1",
    "recipe_hash": "bb130dc00686c6c8c344f2a50169717ac059ef6eac6bcf443f6eeb0d416667e5",
    "split_hash": "5d13188584622317cc96786a0f330078d51d22e058ec2553a64e0f5dc3da316b",
    "train_seed": 19
  },
  "state": "claimed",
  "token": "180f06eeb53d45579c85c3cae3014964",
  "version": 1
}
```
