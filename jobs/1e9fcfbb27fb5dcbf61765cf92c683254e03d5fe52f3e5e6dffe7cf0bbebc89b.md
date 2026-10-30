# Job 1e9fcfbb27fb5dcbf61765cf92c683254e03d5fe52f3e5e6dffe7cf0bbebc89b

State: **claimed** | phase: final | fold: 0 | train seed: 13 | arm: basic

Attempt: 2f89d52fc5fa45a2ae86714f88829e61 | owner: {'server': 'qflex', 'gpu': '0', 'gpu_uuid': 'GPU-d98acf69-eea9-96f1-4f5e-d5af8e7190d1'} | last observation UTC: 2026-10-30T21:32:42.321839+00:00 | sequence: 1

<!-- augment-atlas-record-v1 -->
```json
{
  "attempt_id": "2f89d52fc5fa45a2ae86714f88829e61",
  "history": [
    {
      "at": "2026-10-30T21:32:42.321839+00:00",
      "attempt_id": "2f89d52fc5fa45a2ae86714f88829e61",
      "kind": "claim",
      "owner": {
        "gpu": "0",
        "gpu_uuid": "GPU-d98acf69-eea9-96f1-4f5e-d5af8e7190d1",
        "server": "qflex"
      },
      "seq": 1,
      "state": "claimed",
      "token": "5a3e00fd938247c292a18244fd3c84f0"
    }
  ],
  "job_id": "1e9fcfbb27fb5dcbf61765cf92c683254e03d5fe52f3e5e6dffe7cf0bbebc89b",
  "owner": {
    "gpu": "0",
    "gpu_uuid": "GPU-d98acf69-eea9-96f1-4f5e-d5af8e7190d1",
    "server": "qflex"
  },
  "schema_version": 1,
  "seq": 1,
  "spec": {
    "arm": "basic",
    "code_commit": "919cd245b965ac65de41dfcc13569db6e11c5639",
    "compatibility": {
      "min_memory_mb": 1000
    },
    "config": "configs/global/inbreast/final/stage-2/basic-f0-s13.json",
    "config_sha256": "e9807c1b937018caebd1b9a42aab9fce4761c8202ecdfd6781d46d7839a87392",
    "dataset": "inbreast",
    "dataset_version": "legacy-proposal-v1-converted",
    "dependencies": [],
    "fold": 0,
    "global": {
      "campaign_sha256": "b56e51504619d210d6dc936b7ecbbf4d11471b916de9adee4d7747226c661064",
      "candidate_id": null,
      "freeze_kind": "external_fixed",
      "freeze_sha256": "1bed679a2e52315cdb3050c031e4f6ebd7f2deaf12ce705f19d0fbb29567c740",
      "plan_sha256": null,
      "pools_sha256": "d85f4200c7ecce757cad046eace621fcde284f772d70d69d2879511f157eb00f",
      "search_space_sha256": null
    },
    "job_id": "1e9fcfbb27fb5dcbf61765cf92c683254e03d5fe52f3e5e6dffe7cf0bbebc89b",
    "phase": "final",
    "protocol_version": "global-final-v1",
    "recipe_hash": "1bed679a2e52315cdb3050c031e4f6ebd7f2deaf12ce705f19d0fbb29567c740",
    "split_hash": "0ec448dcff34b542ca0a0c3f8666b405e5c1f3e99db19be29bc0a9392e8a83df",
    "train_seed": 13
  },
  "state": "claimed",
  "token": "5a3e00fd938247c292a18244fd3c84f0",
  "version": 1
}
```
