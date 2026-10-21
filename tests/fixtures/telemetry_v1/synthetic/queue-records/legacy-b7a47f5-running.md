# Job 941f53fe4092ca0a5477b13d22c2e6af399e40d6965e3456b4ecb55410746f04

State: **running** | phase: smoke | fold: 0 | train seed: 7 | arm: none

Attempt: 11111111111111111111111111111111 | owner: {'gpu': '0', 'gpu_uuid': 'GPU-00000000-0000-0000-0000-000000000000', 'server': 'sim-a'} | last observation UTC: 2026-10-19T12:00:05Z | sequence: 2

Epoch: 0/? | progress observed UTC: 2026-10-19T12:00:05Z

<!-- augment-atlas-record-v1 -->
```json
{
  "attempt_id": "11111111111111111111111111111111",
  "history": [
    {
      "at": "2026-10-19T12:00:00.000000+00:00",
      "attempt_id": "11111111111111111111111111111111",
      "kind": "claim",
      "owner": {
        "gpu": "0",
        "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
        "server": "sim-a"
      },
      "seq": 1,
      "state": "claimed",
      "token": "22222222222222222222222222222222"
    },
    {
      "at": "2026-10-19T12:00:06.000000+00:00",
      "attempt_id": "11111111111111111111111111111111",
      "epoch": 0,
      "event_id": "start-0",
      "kind": "update",
      "observed_at_utc": "2026-10-19T12:00:05Z",
      "payload_hash": "7f8e0806ead131a8b4885a7f6962fe55b8f78bfea84e905510c0f371cb86d786",
      "seq": 2,
      "state": "running"
    }
  ],
  "job_id": "941f53fe4092ca0a5477b13d22c2e6af399e40d6965e3456b4ecb55410746f04",
  "owner": {
    "gpu": "0",
    "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
    "server": "sim-a"
  },
  "schema_version": 1,
  "seq": 2,
  "spec": {
    "arm": "none",
    "code_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "compatibility": {
      "dataset": "synthetic-tiny",
      "min_memory_mb": 1000
    },
    "config": {
      "epochs": 2
    },
    "config_sha256": "e2826b0417c39ae7520afcdf3563ca9f61a24848990f7cdb5423ccaa85ea5ecc",
    "dataset": "synthetic-tiny",
    "dataset_version": "synthetic-v1",
    "dependencies": [],
    "fold": 0,
    "job_id": "941f53fe4092ca0a5477b13d22c2e6af399e40d6965e3456b4ecb55410746f04",
    "phase": "smoke",
    "protocol_version": "smoke-v1",
    "recipe_hash": "fixture-recipe-sha256",
    "split_hash": "fixture-split-sha256",
    "train_seed": 7
  },
  "state": "running",
  "token": "22222222222222222222222222222222",
  "version": 2
}
```
