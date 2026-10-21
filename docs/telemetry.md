# Operational telemetry

Raw JSONL stays in each private attempt directory. UTC and monotonic seconds,
job/attempt, alias, GPU UUID, commit, resolved image ID, config hash and available
epoch/sample counters identify observations. Public events contain numeric
progress and artifact checksums only. No patient-level metrics are published.

Linux `/proc`: RAM/swap bytes, uptime seconds, CPU cumulative ticks and load
average, interval CPU utilization percent, disk cumulative counters (kernel sectors/operations), network byte and
packet counters. CPU utilization over an interval can be derived from CPU tick
deltas; the first observation has null utilization. Load average is not utilization.
Optional sysfs collectors record frequency kHz, thermal zones Celsius with domain,
and accessible RAPL energy microjoules with package name/range. Missing values are
null. No administrative privileges are requested. Host counters cover all jobs.

Optional `nvidia-smi`: temperature Celsius, utilization percent, memory MiB, power
watts, clocks MHz, configured power limit watts and exposed performance/throttle
reasons. Unsupported fields remain absent/null. Collector errors do not stop a
job. GPU temperature at the configured ceiling requests a graceful checkpoint.
Temperatures and load do not measure energy savings. GPU/package energy is not
whole-machine energy; concurrent jobs complicate attribution.

Telemetry defaults to 30 seconds, independent of hourly progress publications.
Actual intervals come from monotonic timestamps; `collection_s` measures collector
duration. Disable optional collectors with `optional_collectors:false`. Evaluate
collector time/interval and total private file size during each bounded smoke.
Do not infer emissions, kWh or scientific efficiency from these observations.

Collection and synchronous Git operations currently share one monitor loop.
Separate settings do not guarantee separate scheduling: slow Git can delay
telemetry and temperature checks, and coordinator loss stops host sampling while
the child continues. The compressed one-second acceptance test observed regular
gaps of about 11–14 seconds; recovery included a longer gap. Do not treat this
monitor as continuous thermal protection. Precise cadence needs a dedicated
collector. Collector duration does not measure causal training slowdown; see
[acceptance](acceptance.md) for actual coverage.

## Trainer telemetry inside the attempt (`atlas.train_telemetry.v1`)

`train.py` (legacy `run` and global `_run_global`) writes, through
`train_telemetry.py`, two private files in a `telemetry/` subdirectory of the
attempt output. They are **not** required artifacts: `completed.json` lists the
same 8/9 names as before, and `validate_artifacts`/`validate_global_completion`
read only those names, so the extra subdirectory changes no validation. The
hooks never touch RNG streams, batch order, loss arithmetic, checkpoints or
`metrics.json`; a toy CPU test compares telemetry on vs. off (`ENABLED=False`)
and requires identical metrics (minus `utc`), checkpoint states, `best.pt`,
`results.json` and `predictions.json`. Constants are internal (not in the
validated config): `BATCH_EVERY=10`, `MACHINE_INTERVAL_S=30`, fsync after 32
records or 5 s and on every epoch/session record.

`telemetry/train.jsonl` is append-only; every line carries `schema`, `type`,
`session_id` (random per process; a resume appends a new session), `seq`
(0, 1, … per session), `utc` and `mono_s` (comparable only within a session).
Missing values are `null` with the field in `null_reasons` (fleet protocol 5.2
enum plus `device_not_cuda`, `no_optimizer_step`, `non_finite`, `no_samples`,
`not_evaluated_per_epoch`, `zero_duration`, `unlimited`); never `NaN`.

| `type` | When | Fields |
|---|---|---|
| `session_start` | process start | `context` (path, phase, resume, start_epoch, epochs, device, batch/accumulation, train_images, batches_per_epoch, hashes, code_commit, image_id; global adds arm/fold/candidate_id/workers) |
| `batch` | every 10 batches and the last batch of the epoch | `epoch`, `step` (1-based batch in the epoch), `loss` (this batch), `window_loss_mean`, `lr`, `batch_s` (since the previous batch, data wait included), `images_per_s`, `window_s`, `window_images_per_s`, `grad_norm_preclip` (value returned by `clip_grad_norm_` at the latest optimizer step, i.e. before clipping), `grad_norm_step`, `grad_clipped`, `vram_{allocated,reserved,max_allocated,max_reserved}_bytes` (`torch.cuda.memory_*`, process peak since start) |
| `epoch` | after the epoch is committed (checkpoint + `metrics.json`) | `epoch_s`, `train_s`, `post_train_s` (selection pass, checkpoint, journal), `images`, `images_per_s`, `train_loss`, `learning_rate` (as in `metrics.json`: after the scheduler step), `samples_processed`, `optimizer_updates` (global), `selection` (legacy per-epoch selection only; `null` in global jobs) |
| `machine` | daemon thread, every 30 s | `gpus[]` from `nvidia-smi --query-gpu` (index, uuid, name, temp_c, util_pct, mem_used/total_mib, power_w, power_limit_w, sm_mhz, pstate, fan_pct), `process_cpu_pct` (trainer process, all threads; DataLoader workers excluded), `children_cpu_s` (reaped workers), `rss_bytes`/`rss_peak_bytes`, `load_1_5_15`, `host_cpu_util_pct`, cgroup memory, `collection_s` |
| `session_end` | `finally` of the trainer | `status` (`completed`/`interrupted`/`failed`) |
| `error` | writer failure | telemetry is disabled for the rest of the session; training continues |

Evaluated-partition metrics (external test, E_f, the final D_select pass) are
never written to telemetry, at any time; only the per-epoch selection metrics the
legacy trainer already journals appear in `epoch` records.

`telemetry/summary.json` (`atlas.train_telemetry_summary.v1`) is re-derived from
the whole JSONL at the end of every session (so it covers resumes): `status`,
`sessions`, `duration_s` (sum of session spans), `epochs_recorded`,
`throughput_img_s_mean`, VRAM peaks, `gpu_temp_max_c`, `gpu_power_max_w`,
`machine_samples`, and `energy_kwh`: trapezoid over `power_w` with monotonic
dt, gaps above 2.5 × 30 s not integrated (`energy.coverage_fraction`), status
`measured_integrated`, GPU board only, coarse 30 s sampling, not wall-plug energy.
Without power readings `energy_kwh` is `null` with the reason. On CPU/no
`nvidia-smi` (toy tests) VRAM is `device_not_cuda`, GPU fields and energy are
`tool_absent`; on Windows hosts `/proc` fields are `not_exposed`.

This instrumentation changes `src/`, hence the code identity: it must be in
the commit used for HPO/freeze (Q6), never added in between.
