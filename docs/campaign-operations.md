# Campaign operations

These host-side tools carry the operational serial driver and thermal watchdog
into the repository. They use Bash, Python 3.11+ stdlib, Docker CLI, nvidia-smi,
lm-sensors (sensors -j), and df -Pk; no Python packages are needed. They do not
change queue ordering, scientific policy, campaign identities or retry policy.
ISIC and RSNA use the same interfaces with their own private settings and runs
roots. The examples are placeholders, not instructions to start an approved run.

## Thermal limits and scope

| Control | Default |
| --- | --- |
| Worker GPU stop | 87 C (a stricter settings.max_temperature_c is preserved) |
| Watchdog alert, GPU or CPU package | 85 C |
| Watchdog GPU trip | 89 C |
| Watchdog CPU package trip | 90 C |
| Sample interval | 10 seconds |
| Consecutive missing GPU **or** CPU samples before fail-safe trip | 3 |
| docker stop grace | 120 seconds |

Trip and worker limits must remain at or below Eduardo's recorded 90 C ceiling.
All boundaries are inclusive. nvidia-smi samples all GPUs, so heat on a different
GPU also trips the campaign. sensors reads the highest coretemp Package id N
input, rather than individual cores. If the host exposes no coretemp package,
this is unavailable telemetry and the fail-safe applies; validate host sensor
support before operation. The tools do not configure fans, clocks or power.

The watchdog requires --label KEY=VALUE (exact match) or --label KEY (presence).
It filters docker ps and independently verifies docker inspect labels. If any
--runs-root is supplied, the container must additionally have an /output bind
mount whose resolved source is inside one of those roots. An unlabelled
container is never selected, even when mounted under the runs root. Other label
values and sibling paths such as runs-other are excluded.

The existing worker applies org.augment-atlas.attempt=<container-name>. For this
worker use --label org.augment-atlas.attempt **and** a dedicated campaign runs
root. The root is mandatory for this shared label and separates campaigns. Do not share a
runs root between campaigns that must be thermally stopped independently. The
watchdog also supports exact campaign labels for containers that already carry
them; it does not add labels or relabel existing containers. No worker/runtime
interface change is needed. The driver's --label is the operator's ledger tag,
not an instruction to add a Docker label.

## Start the watchdog first

Keep settings, logs, pidfiles and runs outside the public checkout. Use absolute
paths in worker settings (code, runs and other operational paths). For example:

~~~bash
STATE=/srv/campaign-private/isic/gpu-0
RUNS=/srv/campaign-private/isic/runs
mkdir -p "$STATE" "$RUNS"
umask 077
python3 scripts/campaign/watchdog.py \
  --label org.augment-atlas.attempt --runs-root "$RUNS" \
  --log "$STATE/watchdog.jsonl" --summary "$STATE/watchdog-summary.json" \
  --pidfile "$STATE/watchdog.pid" --stop-file "$STATE/watchdog.stop" \
  --interval 10 --alert-c 85 --gpu-trip-c 89 --cpu-trip-c 90 \
  --stop-timeout 120 &
~~~

Use a supervised session or service for persistence. The watchdog creates its
pidfile exclusively, containing its PID. A pre-existing marker is an error;
verify the old process has exited before removing a stale marker. On trip it
writes PID 0 **before** docker stop so a completing worker cannot start another
turn while the watchdog waits out the grace. It removes the marker at exit.
The driver treats a missing, invalid, zero or dead PID as a closed gate. A
pidfile is a local process-liveness gate, not an authenticated service identity;
check stale files and possible PID reuse before restarting.

JSONL contains samples with readings, reasons, selected containers and errors,
plus TRIP and killed_after_grace events. The summary includes thresholds,
observed maxima, failures, exit reason and per-container post-stop state.
--max-seconds can bound the watchdog lifetime. Exit 0 means signal, stop file or
time limit; 3 means thermal/sensor trip; 1 means a runtime error; invalid arguments
exit 2. After a trip inspect each stopped entry: stop_exit, confirmed_stopped,
state_after and error. A failed Docker command is not evidence of a stopped job;
resolve unconfirmed stops before restarting anything. A failure for one selected
container does not prevent stop attempts for the others.

## Grace expiry and recovery

Containers must remain inspectable (do not use automatic removal). After
successful docker stop -t GRACE, a stopped container with ExitCode 137 and
OOMKilled=false produces an explicit killed_after_grace JSONL event and summary
flag. Its operator_action says: failed: retry; do not use --resume. The event
retains the container ID, grace and inspected state. Exit 137 is conservative
SIGKILL evidence: Docker inspect alone cannot distinguish grace expiry from an
external concurrent SIGKILL or a program exiting 137. This inference is marked
inferred_from_exit_137=true. OOMKilled=true is not labelled as grace expiry.
Missing post-stop inspection is recorded as an error, not a confirmed clean stop.

A grace-expiry kill can leave the job failed. Recover through the queue's
explicit retry procedure and the campaign's existing retry policy; --resume is
for an interrupted attempt with a valid checkpoint. Neither tool retries or
resumes jobs automatically. See [x299-runbook.md](x299-runbook.md), C2.1, for the
existing recovery distinction. Retain the summary and JSONL for operator review.

## Run the serial driver

~~~bash
bash scripts/campaign/driver.sh \
  --label isic --gpu 0 --runs-root "$RUNS" --state-dir "$STATE" \
  --watchdog-pidfile "$STATE/watchdog.pid" --stop-file "$STATE/driver.stop" \
  --settings /srv/campaign-private/isic/hpo-worker.json \
  --settings /srv/campaign-private/isic/final-worker.json \
  --reserve-gib 20 --cap-gpu-s 28800 --default-estimate-s 900 \
  --worker-stop-c 87
~~~

For RSNA replace the campaign tag, root, state directory and settings with that
campaign's values; the same applies to INbreast. Supply exactly one GPU index or
UUID to --gpu. The driver runs worker.sh --settings FILE --phase PHASE --once,
waits for it to finish, then rotates settings in argument order. The worker
selects the job using its existing claim order. There is no background worker
fan-out. A per-GPU exclusive marker under runs prevents overlapping drivers on
that root; the worker also retains its existing GPU UUID lock. Use a consistent
GPU identifier (avoid mixing index and UUID aliases) and one state directory per
campaign/GPU. An ungraceful driver kill can leave a stale marker: verify no
worker/driver remains before manually removing it.

Source settings are never modified. Private temporary copies apply the selected
GPU and the minimum of --worker-stop-c and the existing worker threshold. Their
runs value must resolve to --runs-root, and phase must be smoke, hpo or final.
Temporary copies are removed after each turn, including ordinary failures. Set
PYTHON to an interpreter path if the host command is not python3.

Before **each turn**, the driver checks the stop file, live watchdog PID,
filesystem reserve using df on the actual runs root, and predictive GPU budget.
A stop-file request allows an active worker to finish; the next turn will not
start. The watchdog supplies active thermal stopping. To pause the driver:

~~~bash
touch "$STATE/driver.stop"
~~~

Keep the watchdog running until the worker has finished or all its containers
are confirmed stopped. To stop the watchdog afterward, create watchdog.stop or
send SIGTERM/SIGINT. A watchdog restart requires its stop file to be absent; a
driver restart requires its stop file to be absent and a live watchdog marker.
These stop files are independent.

The persistent gpu-ledger.jsonl records turns with new attempt directories.
GPU-s uses container StartedAt to FinishedAt from docker inspect, summed across
new attempts. If any timing is unavailable, the whole turn's wall time is used
and marked wall_fallback; missing timing is never silently charged as zero.
Whole seconds round upward. The cap checks used GPU-s plus the larger of the
configured estimate floor and prior same-phase job GPU-s. The first job uses
the default 900 s floor. This is an estimate, not a hard runtime timeout: an
unexpectedly long running job can exceed the budget before the next check.

Reuse the same state directory and ledger after restarting. Malformed ledger
rows, negative/nonfinite durations, a different ledger campaign/GPU or failed
filesystem probes stop the driver. Do not delete or truncate the ledger to
resume. CAP_REACHED is an informational marker; the ledger controls the cap.
Separate GPU state directories carry separate budgets.

The driver exits on any nonzero worker status, propagating ordinary statuses
1..125 (other abnormal statuses map to 1). It also exits after a full settings
cycle with no new attempts. Expected stop gates return 0; lock/argument errors
return 2; runtime/configuration failures return 1. driver.log always retains:

~~~text
turn N done rc=... wall=...s gpu=...s
~~~

worker.log contains worker output. No settings contents are written to driver.log;
all operational outputs stay in the private state directory.

## Offline verification

~~~bash
python3 -m unittest discover -s tests -p test_campaign_watchdog.py -v
bash -n scripts/campaign/driver.sh
~~~

Only the new test module runs. It creates synthetic settings and executable
nvidia-smi, sensors, docker and df stubs at the front of PATH, plus a fake
worker.sh. Every external tool resolution is asserted to point into its fixture.
No Docker daemon, GPU, real worker, queue or dataset is used. Fixtures are created
inside the checkout and removed after tests. Windows uses Python and Git Bash;
Linux uses the same Bash/stdlib paths. Live thermal response and real container
checkpoint recovery remain operator validation tasks.
