# Bounded GPU endurance test — 16 October 2026

This is a technical test authorized for approximately five to six hours, including
training, checkpoint recovery and Markdown communication. All configurations are
`phase=smoke`, `scientific_status=provisional`. No scientific HPO or full campaign
is authorized by these files. The orchestration and independent resource guardian
are private operational artifacts; the configurations alone do not enforce time,
temperature or disk limits.

## Available data and chosen workload

Only the existing INbreast data are presently available. The newly discussed
clinical datasets have not been acquired and have access/storage prerequisites.
Codex therefore chose the user's alternative of a small complete battery: **one
clinical dataset plus synthetic load**, not two clinical datasets. This choice
must be recorded in Brain for Eduardo's review.

1. Run the ten `configs/smoke/endurance-v1/inbreast/*.json` arm configurations,
   sequentially, five epochs each, using the same persisted outer fold 0 and
   selection partition. Expected image counts are 248 train, 62 selection and
   77 test. The full existing manifest contains 387 images.
2. If those jobs pass and the resource checks permit, run
   `configs/smoke/endurance-v1/synthetic/none.json` as a timed soak: 1,024 generated
   groups/images, 616 train, 204 selection and 204 test. Its 10,000-epoch setting is
   a ceiling to avoid finishing a fast synthetic workload prematurely; it is not
   authorization to run 10,000 epochs without a deadline. It remains subordinate
   to the external time budget and independent six-hour hard guardian. Stop at the authorized
   deadline rather than extend the run to finish those epochs.

Eleven jobs are planned, seeded **one at a time**. There is no automatic training
retry. A checkpoint resume used for an explicit recovery test is recorded as a
recovery event, not hidden as a successful first attempt. If the clinical battery
fails, do not proceed to the synthetic soak. Prior experiment outputs remain
untouched.

Both stages use one GPU (GPU0), ResNet-50, cached `IMAGENET1K_V2` weights checked
against their full SHA256, 224px inputs, batch size 2, seed 3 and deterministic
FP32. The ten-arm set preserves the exact `candidate-opencv-v2` augmentation
parameters and legacy basic operations from the earlier acceptance. There is no
new parameter search. `recipe.hpo.status=pending`, origin
`technical-endurance-no-search`, and `protocol=within-outer-fold` are retained
technical metadata, not an execution or reversal of the separately documented
global-development HPO proposal.

## Identity and interpretation

The INbreast configuration uses the read-only hashed manifest
`atlas_scientific_v2/inbreast_images_hashed.csv`, exact dataset version
`inbreast-existing-classification-integrity-20261016` and persisted
`atlas_scientific_v2/fold-0.json` (`stratified-group-outer-v1`, split seed 17).
Image hashes are required. `image_dir` is omitted because manifest paths already
include their image-directory prefix. The worker must provide the private data
root; no host-specific clinical directory belongs in a public configuration.

Existing groups are derived from a filename token and have **not** been verified
against PatientID. Fold disjointness therefore establishes group separation,
not proven patient separation. These pilot data have already been inspected;
this run does not create an untouched scientific test set. All arms explicitly
use the existing technical `evaluation_scope=test` contract. Metrics and scores
remain private diagnostics and support no clinical or augmentation ranking claim.

## Duration and independent limits

The planned target is **5h30 from the controller start**, with a **six-hour hard
ceiling** covering the complete battery and soak. There must be an independent
guardian that can act without Git communication, targets a one-second cadence
for GPU0 checks (actual gaps must be recorded), and stops only the positively
identified experiment containers at 75°C,
less than 20GiB project free space, a missing controller or the hard deadline.
The deadline and safe-stop mechanism must be armed before training starts.
The soak is intentionally interrupted when its budget expires; report its actual
terminal state and recovered checkpoint, not a fabricated completed scientific
run. Do not prolong idle time solely to reach a duration.

Operational communication uses progress publication every 300 seconds,
telemetry every 30 seconds and local monitor checks every two seconds. All
callbacks, failed publication attempts and recovery actions must be distinguishable
from model metrics. At completion, fetch the authoritative remote Markdown and
compare it to the local accepted job record; checking a local file alone is
insufficient.

## Planned recovery and communication exercises

- About two minutes into the synthetic soak, request graceful interruption and
  resume from its persisted checkpoint, verifying identity and epoch continuity.
- About 30 minutes into the soak, exercise worker reattachment/restart while
  preserving the active container and durable state. Verify that no duplicate
  training container or new automatic training attempt is created.
- About 60 minutes into the soak, suspend **only the worker's own Git transport**
  for seven minutes, then restore it. Capture publication errors and eventual
  Markdown convergence while the independent guardian remains live.

These times are relative to the soak start and are subordinate to safety and
the overall deadline. They do not authorize an actual server reboot, a global
network outage, changes to SSH access or interruption of unrelated processes.
If a planned exercise cannot run safely, mark it skipped with its reason.
The controller must log actual timestamps, job/attempt identity and observed
outcomes; this document is a plan and does not claim the exercises have passed.

## Acceptance evidence and privacy

Before launch, validate all eleven contracts, confirm the exact manifest/split
and image/weights hashes in the server container, and audit class and group
coverage. Confirm the pinned code revision, single-GPU assignment, no unrelated
workload conflict, resource reserves and armed independent guardian.

Completed clinical jobs require exit status, expected epoch count, artifact
checksums and authoritative remote Markdown reconciliation. The soak additionally
requires checkpoint/resume continuity, reattachment and communication outcomes,
an explicit reason for stopping, and confirmation that its containers are gone.
Record actual elapsed time, peak sampled temperature, minimum free disk,
sampling gaps and every deviation. A short launch observation is not proof that
the five-hour test has completed or that an old server crash has been explained.

Raw clinical data, image/group lists, logs, private settings and model checkpoints
stay on private storage. Public reporting is limited to aggregate technical
status and non-identifying provenance. A final Brain checkpoint must distinguish
planned, running, passed, interrupted and skipped work, with evidence paths and
the choices made by Codex for Eduardo to review.
