# AugmentAtlas

Code for the paper **"How Stable Are Gamma and CLAHE Effects in Medical Image
Classification?"** (submitted to ESANN 2027).

We trained a ResNet-50 **450 times**: three public datasets × ten augmentation
policies × five group-disjoint test folds × three training settings. Each
treated policy adds light or heavy gamma correction or CLAHE to a basic
augmentation (horizontal flip, brightness and contrast), and is compared with
that basic policy trained on the same split, initialization and sample order.

| Dataset | Task | Images / groups |
|---|---|---|
| INbreast (mammography) | BI-RADS 4a–6 vs 1–2 | 387 / 103 |
| ISIC 2020 (dermoscopy) | melanoma vs other lesions | 32,693 / 2,056 |
| RSNA 2018 (chest radiographs) | lung opacity vs normal / other | 29,678 / 12,232 |

## Main findings

- Subtracting the matched basic AUROC reduced dispersion across doses, folds and
  settings on RSNA, but increased it on INbreast and ISIC.
- On ISIC, both transform families looked beneficial at the primary seed
  (Holm *p* < 0.001) because one basic fit collapsed to a constant score;
  without that fold, neither mean is positive.
- On INbreast, both gamma doses were negative at the primary seed, while their
  means across the three settings differed in sign.

## Exact code used

The tag [`esann2027`](https://github.com/EduardoTBuss/augment-atlas/tree/esann2027)
is the code that produced the reported runs. `train.py`, `augment.py` and
`analysis_sap_v1.py` are identical on every campaign branch. The analysis plan
was sealed on 2026-10-28 (tag `sap-v1-sealed-20261028`, [plan](docs/analysis-plan-sap-v1-sealed.md),
[amendment A1](docs/analysis-plan-sap-v1-amendment-20261030.md)). Commit hashes
recorded before 2026-11-25 are mapped in [docs/commit-hash-map.md](docs/commit-hash-map.md).

## Setup

| Item | Value |
|---|---|
| Model | ResNet-50, ImageNet-1K V2 weights, frozen BatchNorm statistics, 224 px input |
| Training | AdamW, batch 16, cosine decay, 20 epochs, last-epoch evaluation |
| Policies | none, basic, gamma U(0.8, 1.2) / U(0.67, 1.5), CLAHE clip 2 / 4 (8×8 tiles), each alone or after basic; added transforms with p = 0.3 |
| Settings | seed 11 on 2× GTX TITAN X; seeds 13 and 19 on one RTX A4500 |
| Runtime | PyTorch 2.5.1, CUDA 12.4, FP32, deterministic algorithms |
| Analysis | image-level AUROC per fit; paired group bootstrap (10,000 resamples) within folds; Holm over the two primary family contrasts per dataset |

## Repository layout

| Path | Content |
|---|---|
| `src/augment_atlas/train.py`, `augment.py` | training loop and the ten augmentation policies |
| `src/augment_atlas/acquisition/` | ISIC 2020 and RSNA 2018 acquisition and group-disjoint splits |
| `src/augment_atlas/legacy_pools.py` | INbreast: conversion of the fixed pilot pools and folds |
| `src/augment_atlas/global_*.py`, `queue.py`, `cli.py` | campaign planning, Git job queue, GPU workers, collection and reports |
| `src/augment_atlas/analysis_sap_v1.py` | sealed analysis: paired contrasts, bootstrap intervals, Holm |
| `src/augment_atlas/paper_results.py` | tables and panels from the analysis outputs |
| `docker/`, `scripts/`, `configs/` | container image, worker and campaign scripts, configuration examples |
| `docs/` | protocol, analysis plan and operations ([index](docs/README.md)) |
| `tests/` | unit and integration tests (synthetic data only) |

## Reproducing

1. **Environment.** Linux, Python 3.11+, Docker with the NVIDIA Container
   Toolkit; build the image from `docker/`. Run the tests:
   `PYTHONPATH=src python3 -m unittest discover -s tests -v`.
2. **Data.** Images are not redistributed. Obtain
   [INbreast](https://doi.org/10.1016/j.acra.2011.09.014) from its authors,
   [ISIC 2020](https://doi.org/10.34970/2020-ds01) (CC BY-NC 4.0) and the
   [RSNA Pneumonia Detection Challenge 2018](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
   data, then build the group-disjoint pools
   ([ISIC](docs/isic-acquisition.md), [RSNA](docs/rsna2018-acquisition.md),
   [data contracts](docs/datasets.md)).
3. **Training campaign.** Plan, seed and run the jobs with the global CLI and
   the queue workers ([global CLI](docs/global-cli.md),
   [infrastructure](docs/infrastructure.md),
   [campaign operations](docs/campaign-operations.md)).
4. **Analysis.** Collect the predictions and run the sealed analysis
   ([analysis](docs/analysis-sap-v1.md), [paper tables](docs/paper-results.md)):

   ```bash
   PYTHONPATH=src python3 -m augment_atlas.analysis_sap_v1 \
     --predictions <dir-or-csv> --dataset <id> --out <dir> --seed 20261028 --B 10000
   ```

Not included: images, trained weights and per-image predictions. The
paper-specific aggregation behind Tables 2–3 and Figure 1 (sum-of-squares
shares and the fold-by-setting grid) is not yet in this repository.

## Branches

| Branch | Content |
|---|---|
| `main` | code and documentation |
| `campaign/*` | per-dataset campaign code and records |
| `queue` | job queue state written by the workers |
| `board` | automatically generated progress board |

Some design and operations notes in `docs/` are in Portuguese.

## Citation

See [CITATION.cff](CITATION.cff). The paper is under review; this section will
be updated with the proceedings reference.

## License

MIT ([LICENSE](LICENSE)); third-party attribution in [docs/provenance.md](docs/provenance.md).
