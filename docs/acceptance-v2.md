# Candidate-v2 technical acceptance — 16 October 2026

Scope: ten arms, separate candidate augmentation backend, pretrained cache and
persisted grouped outer folds. Scientific status remains **provisional**.
The earlier five real training jobs are documented in [acceptance](acceptance.md).
They do not establish this new backend's GPU acceptance.

## Executed local and CPU-container checks

Independent local integration review accepted the source with no remaining
functional blocker after correcting OpenCV distribution preflight and a resized
fixture. The initial full local suite ran108tests:100passed,8skipped. Focused
checks covered the subsequent224px and metadata fixes.

A separate actual CPU container on `x299` executed **110 tests, all passed,
zero skips**, in17.004seconds. Both `ATLAS_REQUIRE_OPENCV=1` and
`ATLAS_CONTAINER_CPU_TESTS=1` were enabled. This executed the actual OpenCV,
PIL/NumPy, CPU dataset integration, Linux lock/symlink and sklearn split cases.
The final preflight amendment was then checked with **37 runtime tests,
all passed, zero skips**, in1.915seconds. Docker/GPU/network boundaries in the
host runtime unit tests remain explicitly mocked; numerical OpenCV and Linux
filesystem behavior in these gates are real.

- Ten old-style arm examples preserve the six original configurations and
  algorithms. Four isolated examples remove only basic.
- Ten `candidate-v2` examples use ImageNet weights,224px,batch2,two epochs and
  the provisional gamma/CLAHE parameters accepted for today's tests.
- Named RNG pairs each operation between isolated/combined policies; adding
  basic cannot shift the intensity draw. Legacy sequential RNG is preserved.
- Fixed-domain gamma is tested at pixel boundaries. Gray and RGB Lab CLAHE are
  checked against OpenCV. At64px/grid8×8, clip2/4 quantize to the same integer
  histogram limit; the numerical regression confirms equality there and a
  distinction at224px on its fixture. This does not prove universal monotonic
  strength. Codex chose224px for this technical trial, subject to user review.
- Cached ImageNet file SHA256 is checked before claim/create and before loading;
  cache is mounted read-only, with no download or random-init fallback.
- Candidate preflight requires cv2 4.12.0 plus `opencv-python-headless`4.12.0.88;
  a different or missing distribution is rejected before reservation.
- Five outer grouped folds and a single inner holdout per fold are generated,
  persisted and audited for hashes, class coverage, mixed-label groups and one
  external appearance per image. Missing/partial artifacts fail explicitly.
  These functions prepare splits; no HPO has run. The existing smoke trainer
  evaluates its technical test at completion and is not yet a selection-only
  objective for scientific HPO candidates.
- Validated final epoch metadata now reaches the completion event, so a fast
  job can show2/2 in Markdown even without an intermediate progress publication.

## Candidate image

The derived image adds only `opencv-python-headless==4.12.0.88` with
`--no-deps --only-binary=:all:` to the previously observed local base. No
framework, driver or host OS upgrade was performed.

```text
image: sha256:75f24f0e788291fd4005054774994b3945eac91573a8f01834cf35d5e30c434a
cv2 4.12.0; headless distribution4.12.0.88
torch2.5.1+cu124; torchvision0.20.1+cu124; NumPy2.1.3
Pillow11.0.0; scikit-image0.24.0
```

## Planned real GPU and Markdown gate

**Not yet executed at this report revision.** The accepted ceiling is ten jobs,
two epochs each, one GPU and30minutes of training-container lifetime, with no
automatic retry. A private independent guardian checks GPU temperature/disk/time
without depending on Git progress publication. Proposed operational stop values
chosen by Codex:75C,20GiB free space and a one-hour acceptance wall limit.
The lifetime abort trigger is25minutes, reserving margin for confirmed cleanup
within the30-minute budget; signalling/inspection have latency.
Only verified containers belonging to this code/image/attempt may be signalled;
stop must be confirmed. Existing mammography outputs remain preserved.

The real gate will fetch each authoritative remote Markdown blob, parse it,
verify the succeeded state, one claim and2/2 epoch total, validate all required
artifact hashes and the actual cached-weight/OpenCV identity, and check final
container/journal state. CPU success above is not relabelled GPU training.

Raw logs, private manifests, checkpoints and sensor samples stay outside the
public repository. No600-run campaign, new dataset acquisition, HPO outcome,
scientific performance or long-term host-stability claim is made.
