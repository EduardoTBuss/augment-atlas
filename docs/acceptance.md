# Technical acceptance — local gates passed, remote gate blocked

Date: 2026-10-15. Scope: initial smoke prototype; no scientific performance
conclusion. **The complete requested remote acceptance has not been achieved.**

Local command: `PYTHONPATH=src python -m unittest discover -s tests -v`.
Observed: 56 tests, exit 0, 54 passed and 2 skipped on Windows (Linux flock and
symlink privilege). Both skipped checks were subsequently executed successfully
on Linux, as described below.
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

## Additional real checks after publication

Two independent local queue clients raced on the real public GitHub `queue`
branch: exactly one confirmed reservation, zero container launches. The winning
attempt was explicitly cancelled with an explanatory event. This proves the
transport/reservation path in that observation, not training execution or
connectivity between two training servers. No runnable job remains pending.

The reviewed training revision `e936d282fbd6259c1082568c35b8b115f178c040` was
cloned and pinned on x299. Only source and private settings were prepared; no
dataset or training stack was transferred or installed. `worker.sh --help` ran.
The real second-process GPU UUID lock test passed on Linux (0.127 seconds); the
real symlink alias rejection test also passed (0.003 seconds). These stdlib checks
did not initialize CUDA or start Docker containers. A first test-filter command
matched no tests and returned 5; the corrected selectors executed both checks.

Final inspected host state: no running Docker containers, no old mammography
executor/watchdog, no AugmentAtlas worker or training, both GPUs idle at 29/30 C.
Existing pilot files were preserved. No prior queue was restarted.

Prepared command on x299, **for use after host stability and Git write access
are restored**, is:

```bash
cd ~/augment-atlas
./scripts/worker.sh --settings ~/augment-atlas-private/worker.json --phase smoke --once
```

The private settings select the observed immutable environment image and the
reviewed code revision. They are not committed. Add one explicit synthetic job
from that same pinned checkout with the README `seed` command (using private
state/data paths); the cancelled transport job will not train. The full startup
command has not passed its Docker preflight in this session.
