# Technical acceptance — local gates passed, remote gate blocked

Date: 2026-10-15. Scope: initial smoke prototype; no scientific performance
conclusion. **The complete requested remote acceptance has not been achieved.**

Local command: `PYTHONPATH=src python -m unittest discover -s tests -v`.
Observed: 56 tests, exit 0, 2 skips on Windows (Linux flock and symlink privilege).
No PyTorch installation or training was performed on the notebook. Coverage:

| Evidence | Result |
|---|---|
| 12 real disposable Git tests | Concurrent clones, push rejection/lost response, stale ownership, terminal state and explicit resume passed |
| 19 contract/data tests | Explicit smoke, six policies, persisted disjoint groups, hashes, paths/hardlink aliases passed; symlink skipped on Windows |
| 25 host runtime tests | Spool outage/order, acknowledgments, crash windows, preflight and artifact semantics passed; Docker/network mocked; flock skipped on Windows |

Independent review found defects in artifact validation, CLAHE dependencies,
resume crash windows, physical image aliases, environment identity and the example
dataset ID. These were fixed and the affected checks repeated. Smoke random
initialization is supported by the worker; ImageNet cache deployment remains
pending (the direct training entry accepts trusted cached weights).

Remote trial budget: at most one Titan X loaded at a time, batch 2, 64px images,
2 epochs per job, at most four short jobs, and 20 minutes of GPU load total.
Stop affected work on current instability or configured temperature ceiling.
No BIOS, driver, thermal or OS changes. Reuse a verified local image when available.
Raw logs/environment/checkpoints remain private; only sanitized summaries belong here.

## Remote preflight

The authorized host alias `x299` was inspected. Both Titan X GPUs were idle at
28/30 Celsius. The earlier mammography executor/watchdog were already stopped;
its logs recorded failure exits 137/139. No old training was cancelled or resumed
by this implementation, and its data, results, logs and checkpoints were untouched.
Exit 137 alone was not interpreted as OOM.

The existing environment image was resolved to its immutable ID privately. A
non-training container import preflight failed with **Docker exit 125 / containerd
EOF** before any CUDA operation. Further read-only diagnostics found current
crashes in unrelated host services. No new GPU workload was started; no driver,
BIOS, thermal or OS settings were changed. Raw operational evidence stays private.

Therefore these remain **unverified**: real ResNet-50 CUDA forward/backward,
Docker training, checkpoint/RNG continuation, live coordinator reattachment,
network outage while training, real INbreast image loading, telemetry overhead
under load, and the complete launcher-to-GitHub completion flow. Host stability
and host-side GitHub write authentication must be restored/verified before that
bounded trial. A compatible tag or passing mocks do not establish these results.
