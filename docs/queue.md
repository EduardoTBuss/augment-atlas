# Git queue protocol (prototype v1)

The `queue` branch contains `jobs/<sha256>.md`. Each Markdown record includes a
human summary and one validated JSON block. The JSON is authoritative; consumers
must not infer machine state from the summary. No patient IDs, private paths,
credentials or raw medical filenames belong in job specs, owners or events.

`GitQueue(checkout, remote, branch="queue")` uses a dedicated mutable coordination
checkout. Keep the training checkout pinned to `spec.code_commit` in another
directory. Git and the Python standard library are sufficient for queue tests.

## Stable identity and dependencies

`prepare_job(spec)` derives a SHA256 ID from canonical JSON containing exactly
`protocol_version`, `dataset`, `dataset_version`, `split_hash`, `recipe_hash`, `phase`,
`fold`, `train_seed`, `arm`, and `code_commit`. Fold and train seed are separate
nonnegative integers. Phase must explicitly be `smoke`, `hpo` or `final`.
Hashes and versions must refer to the actual validated artifacts. Arbitrary
fixture strings in unit tests are synthetic examples, not production hashes.
The dataset ID is required independently of its version: two datasets with
identical version labels and split/recipe hashes must still receive different
job IDs. All ten identity fields are required.

Additional immutable spec fields can contain recipe, HPO origin, configuration
references and `compatibility`. Dependencies are a list of prerequisite job IDs;
all must exist, form a DAG and reach `succeeded` before claim. The runtime checks
local data, phase, image and GPU compatibility before calling claim. Queue claim
enforces dependency completion but cannot inspect remote GPU/data capabilities.
Reusing the same identity with a different immutable spec is rejected.

## API and ownership

```python
queue = GitQueue("/coordination/worker-gpu0", "git@github.com:PROFILE/REPO.git")
records = queue.seed([spec])
record = queue.claim(records[0]["job_id"], {"server": "host-alias", "gpu": "GPU-alias"})
if record is not None:
    record = queue.update(record["job_id"], record["attempt_id"], record["token"],
                          record["version"],
                          {"event_id": "durable-unique-event-id", "state": "running", "epoch": 0})
```

`claim` returns `None` when another owner won or dependencies are incomplete.
It returns a record only after a fresh remote read confirms its generated
attempt ID and token. Failed verification raises and **must not launch training**.
An accepted reservation with unavailable confirmation may remain reserved; this
conservative orphan requires manual inspection. Ownership tokens are public
concurrency guards, not credentials or an authorization boundary. Remote write
access is controlled by Git hosting permissions.

Every mutation reads the remote branch, creates a direct child commit and uses
an ordinary push. Divergent concurrent children are rejected. A losing writer
fetches and revalidates the state before constructing another commit. There is no
force push, blind rebase or merge. After every push, even an error/ambiguous
response, a new fetch decides what reached the remote. Unrelated working files
and the normal Git index are preserved; temporary indexes build queue commits.
Keep one checkout per worker process: the Python lock protects threads within a
single `GitQueue`, not two processes sharing one `.git/FETCH_HEAD`.

## Versioned events and outage replay

The transitions are `pending -> claimed -> running -> succeeded|failed|cancelled|interrupted`.
An owned claimed attempt may fail/cancel/interrupt before training starts.
Running progress can remain `running`. Success requires a running observation.
An explicit operator-controlled resume can transition `interrupted -> running`
within the same attempt using `event.manual_resume: true`, the current token and
exact version. The runtime must first inspect and confirm that the old child is
stopped and the compatible checkpoint belongs to this attempt. A resumed attempt
retains IDs, output directories and checkpoint lineage. An old unacknowledged
progress event fails its version check; previously acknowledged IDs remain
idempotent. No other terminal state accepts a fresh running event.
Each event carries a durable unique `event_id` and target `state`; epoch,
artifacts and sanitized summary fields can be added. The queue stores the event
payload, server UTC observation time and monotonic queue sequence in `history`.
The `at` timestamp records queue publication time. Persist `observed_at_utc` in
the original event to retain its true observation time through a network outage;
the Markdown summary prefers it when supplied. Metadata keys `at`, `seq`,
`attempt_id`, `token`, `payload_hash` and `kind` are reserved to the queue.
The runtime must independently validate successful artifacts before publishing
success: the Git queue cannot inspect heavyweight private files.

Updates require the exact attempt ID, token and expected state version. An old
attempt raises `StaleAttempt`. A version mismatch raises `QueueConflict` and
requires the caller to reconcile remote history. Replaying an acknowledged
identical event is idempotent, including after the record became terminal; it
returns current state without regression. Reusing an event ID with different
content raises. Fresh updates to terminal states are rejected except for the
explicit interrupted-attempt resume described above.

During network loss, an already-running process continues and checkpoints.
The runtime persists ordered events in a local spool before publication, keeps
the same IDs on retries and advances expected versions only after confirmation.
It must drain the oldest event first and reconcile acknowledged IDs before
publishing the next event. The queue itself is synchronous and has no hidden
background spool. A new claim always requires live remote confirmation.

## Manual recovery and restart

No timeout or heartbeat age automatically reassigns work. First inspect the old
coordinator and child/container, preserve checkpoint/logs and confirm whether the
old execution is alive. Reattach an existing live child using runtime state;
never launch another child merely because the coordinator restarted.

After confirming the old execution stopped, its current owner may publish an
`interrupted`, `failed` or `cancelled` event. An operator can then call
`retry(job_id, expected_version, reason)` to return that terminal job to pending.
Successful jobs cannot be retried. The recovery reason and old attempt ID remain
in history; the next claim receives a fresh attempt and token. An uncertain live
claimed/running attempt remains reserved until the operator resolves it.
Operators with Git write access can repair a lost-owner record manually only
after confirming the old process stopped, preserving the audit history and
version invariants. No exactly-once execution guarantee is claimed across host
or network failures.

Records intentionally keep their complete event history for this short smoke
prototype. Long campaigns will need an explicit history retention design before
the queue branch grows large.
