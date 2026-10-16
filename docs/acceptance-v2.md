# Candidate-v2 technical acceptance — 16 October 2026

Scope: ten arms, separate candidate augmentation backend, pretrained cache and
persisted grouped outer folds. Scientific status remains **provisional**.
The earlier five real training jobs are documented in [acceptance](acceptance.md).
The separate candidate acceptance below passed ten additional real jobs.

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

After the GPU trial, the selection-only objective contract was added. The final
complete CPU-container suite ran **127 tests, all passed, zero skips**, in
17.065seconds, using source snapshot SHA256
`9470eaa00380667823ec009a220b527c917ed2975245c74e6b65e23b57e43f0d`.
An independent reviewer checked that the four changed source/test files match
this snapshot and accepted the change; their local rerun passed14tests with
three container-only cases skipped. The127-test run includes those three:
tiny real CPU training loops substitute a small model for the ResNet constructor
to check selection-only, default test evaluation and resume. This is CPU evidence,
not an additional GPU ResNet-50 trial.

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
  These functions prepare splits; no HPO has run. A subsequent selection-only
  objective mode requires no augmentation and an inner-fold HPO protocol.
  It never constructs a test dataset or computes test predictions/metrics;
  integrity checks may still read test files for hashes. Scope is part of
  checkpoint identity and artifact validation. No search service is implemented.
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

## Executed real GPU and Markdown gate

**PASS, completed16October2026 at04:22:45UTC**, using source commit
`39f890e654505d82a1459a3cb57b1effd0d1424c` and the image above. All ten candidate
arms trained a real ImageNet-initialized ResNet-50 in Docker/CUDA on GPU0:
synthetic12groups,224px,batch2,seed3,two epochs. Each had one claim, one attempt,
exit0 and eight artifact hashes validated. These are technical data, with no
clinical-performance interpretation. No automatic retry occurred.

The gate fetched each authoritative remote Markdown blob from Git, parsed it
back to the queue record and confirmed `succeeded` plus2/2. A separate final
reconciliation at09:44UTC repeated all80artifact checks, compared the fetched
Markdown hashes with the accepted copies, inspected container exit/code/image
identity and verified cached weights/OpenCV identity. No active attempt journal
or running container remained. The combined historical queue had15succeeded and
one cancelled job, with no pending job. The host boot time was unchanged and
no crash report had been modified since this acceptance started.

| Bounded trial observation | Value |
|---|---|
| Training containers | 10, sequential on one GPU |
| Sum of container Created→FinishedAt | 288.320444s, below1800s |
| Independent guardian samples | 774 |
| Largest observed sample gap | 2.162s |
| Highest sampled GPU0 temperature | 49°C |
| Lowest sampled project free space | 43,892,981,760bytes |

The lifetime sum includes paused gaps and excludes short CUDA preflight probes;
it is not measured GPU active time or energy. The guardian used Git-independent
checks and Codex-chosen stops at75°C,20GiB free,25minutes lifetime trigger and
one-hour trial wall limit. Four focused mock tests exercised stop confirmation
and launch guards. No real overtemperature fault was induced. This trial-specific
guardian is not a general long-term thermal protection service.

The accepted ceiling was ten jobs, two epochs each, one GPU and30minutes, with
zero automatic retries. No further GPU training was started for the subsequent
selection-only change. Its separate evidence is the127-test CPU gate above.
Existing mammography outputs remain preserved and their queue was not restarted.

## Existing INbreast manifest: audit without training

The private387-image/103-group manifest received streaming SHA256 checks for
every image and five persisted outer folds with one internal selection holdout
each. Both classes appear in every partition; group intersections are empty;
every image is in an outer test exactly once. No scores or HPO were computed.
An additional CPU audit decoded all387existing896×896images and found zero exact
file duplicates and zero exact grayscale pixel duplicates. It also loaded each
of the five split artifacts through the new hashed-manifest config and checked
persisted resume identity. No exclusions were made. Near duplicates were not
assessed; this is no proof of patient identity.
The current group key comes from a filename token, without a verified patient
column. Patient identity remains a prerequisite before calling these partitions
patient independent. This audit does not undo exposure in the earlier pilot.

Raw logs, private manifests, checkpoints and sensor samples stay outside the
public repository. No600-run campaign, new dataset acquisition, HPO outcome,
scientific performance or long-term host-stability claim is made.
