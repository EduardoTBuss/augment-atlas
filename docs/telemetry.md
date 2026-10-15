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
