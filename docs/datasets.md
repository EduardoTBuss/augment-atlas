# Local classification data

This first version is a technical smoke prototype. ResNet-50 is the only
supported model. Synthetic runs validate infrastructure; they cannot establish
scientific performance or validate medical labels. INbreast has already been
examined in the earlier pilot.

No dataset image, patient identifier, real medical filename or private manifest
belongs in Git. Mount the local dataset read-only at `/data`, the code read-only
at `/code`, and the attempt output read-write at `/output`. `ATLAS_DATA_ROOT`
overrides `dataset.root`; all manifest, image and supplied split paths must be
relative to that root. Absolute paths, traversal and escaping symlinks fail.

The smoke `dataset` section is a closed set of keys (`contracts.SMOKE_DATASET_KEYS`);
an unknown key such as `max_images` fails validation instead of being ignored.
Subsetting requires a separate group-closed private manifest.

## Adapters

`synthetic` constructs a small deterministic image per synthetic group. Its
content depends on image identity and dataset version, independently of train
seed and split seed. `dataset.synthetic_groups` defaults to 12. There is no
download or dataset preparation service.

`manifest_csv` reads a private UTF-8 CSV with required columns:

```csv
path,group,label
images/synthetic-example-001.png,synthetic-group-a,0
images/synthetic-example-002.png,synthetic-group-b,1
```

These names are illustrative. `label` is a zero-based integer class index;
`group` represents the independent unit (patient where appropriate), and `path`
is unique within the manifest. Empty groups, missing files, duplicate images,
invalid labels and an empty manifest fail before training. An optional `sha256`
column verifies each referenced image's bytes. Set
`dataset.require_image_hashes: true` to require full image-hash coverage.
`dataset.manifest_sha256` optionally pins the original CSV bytes. The canonical
record hash always pins actual paths, labels, groups and supplied image hashes.
Without the image-hash column, existence and manifest identity are checked, but
the individual image bytes are not pinned. The environment manifest records
image-hash coverage explicitly.

`inbreast_csv` uses the existing pilot columns
`processed_image_filename,group_key,classification`, with optional
`dataset.image_dir`. It intentionally requires no lesion boxes for a
classification task. `configs/smoke/inbreast.json` describes the expected
private mount layout. A short adapter test should use a deliberately small
private CSV subset containing at least six independent groups and both classes,
without copying images. This adapter consumes already prepared 8-bit PNGs; raw
DICOM conversion and label-policy selection remain outside this prototype.

## Partitions and privacy

`splits.split_seed` is independent of `training.seed`. `group-hash-v1` sorts
groups by a SHA-256 of split seed, fold and group, then assigns approximately
60% train / 20% selection / 20% test (integer group counts). This is an
**unstratified technical partition**, not the proposed five-fold scientific
protocol. It can produce single-class selection/test partitions, so the smoke
runner reports accuracy and cross-entropy, not scientific AUC estimates.

The attempt's private `splits.json` persists the exact image lists, exact group
lists, dataset version, canonical manifest hash, algorithm version, fold,
split seed and document hash. A resumed run validates this existing file and
never regenerates it. Set `dataset.split_file` to reuse a validated private
`splits.json` from local read-only data. Supplied lists must match the manifest
and configured identity. All images must occur exactly once, every partition
must be nonempty, and groups must be disjoint between train, selection and test.
The current adapter does not support unassigned development patients in the
same manifest; provide the intended evaluation population as its own manifest.

Keep these lists private. Publish only their hashes and synthetic descriptions.
Pair the same persisted lists across augmentation arms and train seeds. Changing
folds or seeds does not undo HPO contamination. Do not describe the existing
INbreast pilot as an unseen independent test.

## Pixels and augmentation

The input policy is explicit: `grayscale_to_rgb` opens an image as 8-bit PIL `L`,
applies grayscale augmentation, then replicates into RGB. `rgb` preserves three
channels and applies basic/gamma transformations there. Legacy CLAHE is
accepted only for `grayscale_to_rgb`. Fixed ImageNet mean/std normalize the
result after bilinear resize and conversion to a float32 tensor in `[0,1]`.
No preprocessing statistic is learned from selection or test.

Operations run only for train, in the configured order. Basic applies a
horizontal flip with probability 0.5, then brightness and contrast with a
shared probability 0.8 and independently sampled factors in `[0.9,1.1]`.
Unlike the multitask pilot, this classification prototype does not crop/zoom:
without lesion boxes it cannot apply the pilot's lesion-preservation rejection.
Gamma follows basic, uses extrema-preserving `lo+(hi-lo)*((x-lo)/(hi-lo))**g`,
and leaves constant channels constant. RGB extrema are per channel. CLAHE
follows basic, using scikit-image `equalize_adapthist`, 256 bins and a kernel
of `max(8, round(window_fraction * image_width))` before final resize.

Every sample gets a deterministic RNG from train seed, epoch and image identity;
loader order also depends only on train seed and epoch. Shared basic operations
consume the same draws across arms. Augmentation never consults class labels.
See `provenance.md` for the ten provisional examples and their limitations.
For isolated gamma/CLAHE arms, basic is absent. The current sequential RNG gives
repeatable draws within an arm but does not pair the exact intensity realization
between isolated and combined arms. Named per-operation streams are proposed
for the scientific protocol; the legacy implementation is preserved here.

## Running and restarting

```sh
python -m augment_atlas.contracts --config configs/smoke/synthetic.json --validate
python -m augment_atlas.train --config configs/smoke/synthetic.json --output /output --preflight
python -m augment_atlas.train --config configs/smoke/synthetic.json --output /output --stop-after-epoch 1
python -m augment_atlas.train --config configs/smoke/synthetic.json --output /output --resume-if-exists
```

Validation/preflight uses only the Python standard library. Runtime requires
the pinned torch/torchvision stack plus NumPy/Pillow and scikit-image for CLAHE.
The legacy smoke examples use `weights: none`. Candidate-v2 uses
`weights: IMAGENET1K_V2` and a full `model.weights_sha256`. The host setting
`torch_cache` points to a prepared Torch home containing
`hub/checkpoints/resnet50-11ad3fa6.pth`; the worker validates the file before
claiming and mounts it read-only as `/torch-cache`. The trainer rechecks before
loading. Downloads are disabled; no random-initialization fallback is allowed.

Candidate-v2 resizes first and uses named per-operation RNG. Gamma is a direct
power on the fixed uint8 domain; CLAHE uses OpenCV fixed clipping/grid, in gray
or RGB Lab luminance. It requires cv2 4.12.0 and the headless distribution
4.12.0.88 before reservation. These semantics are separate from the legacy
extrema-preserving gamma and scikit-image CLAHE described above.

`stratified-group-outer-v1` accepts only pre-generated private split artifacts.
The `augment_atlas.scientific_splits` CLI prepares five outer folds and one
inner selection holdout per fold, with group/class/hash and external-coverage
audits. It does not execute HPO or full inner CV. Missing artifacts fail instead
of generating a new hash holdout. Persisted lists define resumption.

The runner uses FP32, disables autocast and TF32, enables deterministic torch
algorithms, and records real package/CUDA/GPU versions and numeric flags in
`environment.json`. It records nominal effective batch size; a partial final
accumulation window is normalized by its actual sample count. BatchNorm is
frozen by default for tiny batches. This is not a promise of identical bits
across GPU families.

SIGTERM/SIGINT received after training initialization finish the current epoch,
commit a checkpoint and return 143/130. `--stop-after-epoch` returns 75 to
support a controlled interruption. Checkpoints contain model, selected model,
optimizer, scheduler, Python/NumPy/torch/CUDA RNG states, actual epoch journal,
config hash and split hash. An epoch interrupted by a hard crash is replayed
from the previous committed boundary. Resume uses the same attempt directory;
an existing checkpoint requires explicit resume. Checkpoints are trusted
private artifacts and must not be loaded from arbitrary untrusted sources.

Selection loss chooses the epoch. The default `training.evaluation_scope: "test"`
evaluates the selected model on the external test once, after training finishes.
For a future inner-fold HPO objective, set
`training.evaluation_scope: "selection-only"`, `augmentation.arm: "none"`,
empty augmentation operations and `recipe.hpo.protocol: "within-outer-fold"`.
This mode reports `results.selection` and never constructs a test dataset or
computes test predictions/metrics. Manifest integrity checks still cover every
listed file, including hashes of test files when supplied. The scope participates
in config/recipe identity and resume validation; switching it cannot reuse an
existing checkpoint. Completion validation rejects any `results.test` in this
mode. This contract prepares an objective; it does not implement a search service,
choose its budget or authorize scientific HPO. There is no threshold HPO.
`completed.json` is written last and lists SHA-256/size for every required
artifact: checkpoint, selected model, config, environment, splits, metrics,
results and progress. The coordinator must verify those hashes; exit zero
alone does not establish completion. `progress.json` is atomically replaced
each epoch. Detailed logs, `splits.json` and all outputs are private by default.
