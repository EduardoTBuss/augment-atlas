# Dataset and augmentation discussion — 2026-10-15

This is a bounded primary-source survey, not a final protocol, exhaustive review,
or certification of a global state-of-the-art ranking. It compares manageable
datasets with larger references and records their actual use in recent work.
No new dataset was downloaded and no scientific augmentation range was approved.
The existing technical smoke settings remain provisional.

Discussion preference recorded on 2026-10-15: compare manageable and larger
options, prioritizing evidence of use in recent competitive work. For skin,
prioritize **ISIC dermoscopy**, with **HAM10000** as the smaller alternative.
The exact ISIC release and scientific task remain to be frozen. PAD-UFES-20 and
ISIC2024 total-body crops are contextual alternatives, not the selected acquisition.

## Datasets used in recent competitive work

| Modality | Manageable option | Larger reference | Evidence and decision to resolve |
|---|---|---|---|
| Diabetic retinopathy | APTOS-2019: 3,662 labelled images; IDRiD: 516 | EyePACS/Kaggle-2015: 35,126 documented training images | APTOS, IDRiD and Messidor-2 are downstream DR benchmarks in a 2026 foundation-model study. EyePACS is a pretraining source in RETFound-DE (2025). Confirm the release, paired-eye/patient linkage and duplicate audit; image IDs alone do not prove independent patients. |
| Dermoscopic skin lesions | HAM10000: 10,015 images | ISIC-2019: 25,331 training images | PanDerm (2025) evaluates HAM10000. ISIC2019 remains a recent classification benchmark. HAM has repeated lesions, and ISIC2019 includes HAM: these are not independent external test sets for each other. Confirm patient linkage; lesion IDs alone do not group different lesions of one patient. |
| Chest radiographs | VinDr-CXR: 18,000 images, official 15,000/3,000 split | CheXpert: 224,316 radiographs; MIMIC-CXR: 377,110 images / 227,835 studies / 65,379 patients in the original release | Ark+ (2025) and RayDINO (2026) use these benchmarks. RayDINO pretrains on CheXpert/MIMIC and evaluates on VinDr outside its pretraining domain. VinDr's released image IDs do not establish patient linkage. CheXpert/MIMIC have repeated studies/views, uncertain report-derived labels and access considerations. |

Primary evidence: [retinal pretraining comparison, Nature Communications 2026](https://www.nature.com/articles/s41467-026-70077-z),
[RETFound-DE, data availability and supplementary tables, 2025](https://www.nature.com/articles/s41551-025-01365-0),
[PanDerm, Methods/datasets, 2025](https://www.nature.com/articles/s41591-025-03747-y),
[Ark+, data availability and supplement, 2025](https://www.nature.com/articles/s41586-025-09079-8).
[RayDINO, Methods/Data, 20 August 2026](https://www.nature.com/articles/s41467-026-76076-4)
provides a newer chest-radiograph reference: ViT-L/DINOv2 pretraining followed
by frozen-encoder adapters, not the proposed ResNet-50 fine-tuning protocol.
Different backbones, pretraining populations and resolutions prevent interpreting
their reported scores as a controlled augmentation comparison with ResNet-50.

For clinical smartphone skin photographs, PAD-UFES-20 is a different acquisition
option: 2,298 images, 1,641 lesions and 1,373 patients, with patient and lesion IDs.
Its patient grouping is useful, but it is not a smaller dermoscopy release.
ISIC2024/SLICE-3D uses crops from total-body photography, also distinct from
dermoscopy. Changing between these datasets changes acquisition and population,
not just sample size. Sources: [PAD-UFES-20 original paper](https://arxiv.org/abs/2007.00478),
[official ISIC challenge releases](https://challenge.isic-archive.com/data/).

## Access, grouping and the meaning of the baseline

- Check the actual dataset terms and release before acquisition. Official Kaggle
  pages did not expose enough text in this survey to confirm every licence and
  download size. No account terms or data-use agreement were accepted.
- PhysioNet [VinDr-CXR](https://physionet.org/content/vindr-cxr/1.0.0/) and
  [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) have credential
  and data-use requirements. Do not infer data rights from a code licence.
- The MIMIC-CXR-JPG conversion includes histogram equalization. This is not proof
  of CLAHE. [HAM10000](https://www.nature.com/articles/sdata2018161) also reports
  manual histogram corrections for some images. A baseline using these releases
  already contains upstream processing; it is not an untouched-image baseline.
- DR is usually ordinal grading; skin tasks may be multiclass or binary; CheXpert
  and MIMIC tasks are generally multilabel. Primary metrics, label uncertainty,
  losses and study/view aggregation must be chosen explicitly. The current
  smoke trainer implements mutually exclusive classes, not a ready multilabel
  chest-X-ray training recipe.

## Published augmentation parameters: examples, not chosen ranges

| Source | What was actually specified | Interpretation boundary |
|---|---|---|
| Schaudt et al., Scientific Reports 2023, Table 3 | Resize 224; a `OneOf` group with probability 0.9 includes CLAHE, brightness/contrast and `RandomGamma(80,120)` | In Albumentations 1.3, that gamma notation maps to exponent 0.8–1.2. Group probability is not an unconditional 0.9 probability for gamma. Paper and official code differ in details; the installed library version was not pinned. |
| Huang et al., Diagnostics 2023, DR/ResNet-50 | CLAHE preprocessing with clipping 3 and grid 8; flips, geometry and colour adjustments also examined | CLAHE did not improve DR grading in that comparison. Deterministic preprocessing and stochastic train-only augmentation are different interventions. |
| Cino et al., Journal of Imaging 2025, ISIC2019 | Horizontal/vertical flips 0.5; brightness/contrast 0.2; hue/saturation 0.05; rotations 0–359 degrees and scale 0.7–1.7 | Other operation probabilities/order were incompletely specified. Do not transfer wide rotations or vertical flips to chest/mammography without semantic justification. |

Sources: [Schaudt et al.](https://doi.org/10.1038/s41598-023-45532-2),
[Huang et al.](https://doi.org/10.3390/diagnostics13101664),
[Cino et al.](https://doi.org/10.3390/jimaging11010015),
[versioned Albumentations gamma/CLAHE implementation](https://github.com/albumentations-team/albumentations/blob/1.3.0/albumentations/augmentations/transforms.py).
The gamma/CLAHE examples recovered here are from 2023–2025; a complete comparable
2026 gamma/CLAHE protocol was not verified. RayDINO describes large 512px and
small 224px crops with area thresholds at 15%, referring other settings to
DINOv2 Fast Setup. That is a self-supervised pretraining intervention, not a
drop-in train-only ResNet-50 gamma/CLAHE recipe.

CLAHE parameters require a library, version, channel, dtype, resize order and
region definition. OpenCV tile-grid counts and scikit-image kernel pixel sizes
are different; their clipping scales are not interchangeable. A scalar clip
limit may also represent a sampled interval in a wrapper library. These sources
do not define this project's future "light" and "heavy" levels. The pilot's
extrema-preserving gamma formula also differs from a fixed 0–255-domain power.

[MedAugment](https://arxiv.org/abs/2306.17466) and
[MediAug](https://arxiv.org/abs/2504.18983) provide additional contemporary
comparators. They use different operation spaces, mixing or data-expansion
budgets; MedAugment does not use gamma/CLAHE. Adding them would expand the
proposed ten-arm design and needs a separate budget/protocol decision.

## ISIC release clarification — 16 October 2026

The official [ISIC data portal](https://challenge.isic-archive.com/data/) and
[ISIC2020 dataset descriptor](https://doi.org/10.1038/s41597-021-00815-z) establish
33,126training dermoscopic images from2,056patients,584melanomas, patient and
lesion identifiers, and425declared pixel duplicates. The declared duplicate
list must be reconciled before splitting. This makes2020 a practical candidate
for patient grouping **if the task is melanoma versus the remaining lesions**.
That is a different task from HAM's seven classes or ISIC2019's eight classes.
The2020official test has10,982images without public ground truth; do not assume
its labels are available for a local external test. ISIC2019 lesion identifiers
alone do not establish patient grouping. Codex recommends evaluating2020 for the
binary protocol; this does not silently change the study task or acquire data.

Next decisions: choose acquisition/task and data release; verify groups and
access; freeze preprocessing and metrics; freeze the chosen inner-fold HPO
search space and training budget; then fix candidate augmentation settings.
None of these choices is replaced by a successful infrastructure smoke test.
