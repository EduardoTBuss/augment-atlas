# Technical provenance and provisional scientific choices

The current phase is `smoke`, with `scientific_status: provisional`. All six
arm configurations are examples for integration tests. They are not optimized
hyperparameters, literature-backed ranges or approval to launch a final
campaign. The scientific code and parameters will be reviewed again before
that campaign. Only ResNet-50 classification is implemented here; the old
multitask module is never imported.

The implementation was written for AugmentAtlas after inspecting these paths
in the existing `mamografia-estudo` working tree, read-only, on 2026-10-15:

| Component | Source | What is retained or changed |
| --- | --- | --- |
| Classification CSV adapter | `src/mamografia_exp/dados.py` | `processed_image_filename`, `group_key`, `classification`; no boxes needed |
| ResNet-50 path | `codigo/multitarefa.py` | torchvision ResNet-50 only; new standard linear classification head, no old shared-module import |
| Basic image operations | `src/mamografia_exp/aumentos.py`, `codigo/multitarefa.py` | flip and brightness/contrast probabilities/ranges retained; zoom omitted because class-only manifests have no lesion boxes |
| Light gamma | `configs/exp03_gamma.yaml` | `p=0.5`, uniform `[0.9,1.1]`, extrema-preserving intensity formula |
| Heavy gamma | `configs/exp03b_gamma_forte.yaml` | `p=0.5`, uniform `[2/3,1.5]`; already marked optional in the pilot |
| Light CLAHE | `configs/exp07_clahe.yaml`, `src/mamografia_exp/aumentos.py` | `p=0.5`, clip `0.01`, window fraction `256/3328`, 256 bins |
| Heavy CLAHE | No approved pilot value found | **Demonstration gap:** clip `0.02` merely exercises a distinct configuration; it is not an approved candidate range |

Intensity implementations adapt the pilot's formulas. Augmentation is applied
before resizing, only for train. CLAHE was experimental in the pilot and can
amplify noise or alter texture. The basic classification arm differs from the
full pilot policy; do not claim numeric equivalence with old multitask results.
ImageNet normalization is fixed; batch optimizer settings in smoke examples
are deliberately small technical defaults, not a transferred scientific recipe.

The original working tree contained local edits and untracked work. It was
neither altered nor mirrored. No historical checkpoint, patient data, private
manifest or old experiment result was copied into the new repository. No
scientific conclusion is inferred from these infrastructure tests.

## Recipe and HPO boundaries

Each config records a frozen, versioned technical recipe, HPO status, protocol,
origin and dependency identifiers. `pending` + `undecided` is intentional.
An `external` origin requires a protocol choice and artifact hash. These are
validation contracts, not an executed HPO service: the current version neither
searches nor resolves global-development versus within-outer-fold HPO. Stage
dependency execution belongs to the coordinator; a recipe listing dependencies
does not itself prove they completed.

Future literature work will define candidate ranges, then an explicit HPO
protocol will select a training recipe, initially without augmentation. Budget,
search space, optimizer, final datasets/tasks and patient development budget
remain open. A single internal selection split is not a full inner CV. Changing
folds or train seeds after HPO does not remove contamination. Development
groups used for selection must stay separate from an independent test. INbreast
has already been examined in the pilot. See `datasets.md` for the deliberately
technical unstratified split used by this first version.

## Retained third-party license

The source project identifies its origin as `guilhermedallm4/mamografia-estudo`
and supplies the following MIT license. This notice applies to reused or
adapted substantial portions and must remain with redistributed copies:

```text
MIT License

Copyright (c) 2026 Guilherme de Lima

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

PyTorch, torchvision, NumPy, Pillow and scikit-image retain their respective
licenses. Dependencies and ImageNet weights are not vendored here; a cached
weight file's checksum and actual runtime versions are recorded privately.
