# Job 853e7f2dafe63e83b8493e916d16de58d9bebcf77934f73fdc3914897e09f6dd

State: **claimed** | phase: smoke | fold: 0 | train seed: 8 | arm: none

Attempt: 44444444444444444444444444444444 | owner: {'server': 'sim-a', 'node_id': 'sim-a', 'worker': 'gpu0', 'gpu': '0', 'gpu_uuid': 'GPU-00000000-0000-0000-0000-000000000000'} | last observation UTC: 2026-10-19T12:00:00.000000+00:00 | sequence: 1

<!-- augment-atlas-record-v1 -->
```json
{
  "attempt_id": "44444444444444444444444444444444",
  "history": [
    {
      "at": "2026-10-19T12:00:00.000000+00:00",
      "attempt_id": "44444444444444444444444444444444",
      "fleet": {
        "boot_id": "aaaaaaaa-0000-4000-8000-00000000000a",
        "coordinator_commit": "7777777777777777777777777777777777777777",
        "gate": {
          "cpu_pkg_temp_c": 38,
          "free_disk_bytes": 43892981760,
          "gpu_idle": true,
          "gpu_temp_c": 41,
          "observed_at_utc": "2026-10-19T12:00:00Z"
        },
        "heartbeat_ref": "refs/heads/fleet-hb/sim-a/gpu0",
        "image_id": "sha256:3333333333333333333333333333333333333333333333333333333333333333",
        "launch_plan_sha256": "8888888888888888888888888888888888888888888888888888888888888888",
        "lease_seconds": 3600,
        "node_profile_sha256": "6666666666666666666666666666666666666666666666666666666666666666",
        "protocol": "atlas.fleet.v1",
        "requirements_sha256": "4b292f393d88915f2a09b5c4b3b0994e8bad45833827340873b4a035b630370f"
      },
      "kind": "claim",
      "owner": {
        "gpu": "0",
        "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
        "node_id": "sim-a",
        "server": "sim-a",
        "worker": "gpu0"
      },
      "seq": 1,
      "state": "claimed",
      "token": "55555555555555555555555555555555"
    }
  ],
  "job_id": "853e7f2dafe63e83b8493e916d16de58d9bebcf77934f73fdc3914897e09f6dd",
  "owner": {
    "gpu": "0",
    "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
    "node_id": "sim-a",
    "server": "sim-a",
    "worker": "gpu0"
  },
  "schema_version": 1,
  "seq": 1,
  "spec": {
    "arm": "none",
    "code_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "compatibility": {
      "dataset": "synthetic-tiny",
      "image_id": "sha256:3333333333333333333333333333333333333333333333333333333333333333",
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
    "job_id": "853e7f2dafe63e83b8493e916d16de58d9bebcf77934f73fdc3914897e09f6dd",
    "phase": "smoke",
    "protocol_version": "smoke-v1",
    "recipe_hash": "fixture-recipe-sha256",
    "requirements": {
      "allowed_nodes": [
        "sim-a"
      ],
      "container": {
        "cpus": 2,
        "memory_bytes": 4294967296
      },
      "dataset": {
        "id": "synthetic-tiny",
        "manifest_sha256": null,
        "version": "synthetic-v1"
      },
      "expected_epoch_seconds_max": 1800,
      "gpu_required": true,
      "hardware_class": null,
      "image_id": "sha256:3333333333333333333333333333333333333333333333333333333333333333",
      "image_ref": null,
      "min_compute_capability": [
        5,
        2
      ],
      "min_free_disk_bytes": 5368709120,
      "min_vram_mib": 1000,
      "schema": "atlas.requirements.v1",
      "weights_sha256": null
    },
    "scheduling": {
      "order": 120,
      "priority": 0
    },
    "split_hash": "fixture-split-sha256",
    "train_seed": 8
  },
  "state": "claimed",
  "token": "55555555555555555555555555555555",
  "version": 1
}
```
