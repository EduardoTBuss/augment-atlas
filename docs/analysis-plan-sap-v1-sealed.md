# AugmentAtlas — Statistical Analysis Plan v1 (SEALED 2026-10-28)

Sealed by Eduardo T. Buss on 2026-10-28, before any prediction on an evaluation partition (E) of
any dataset existed. Scope: ESANN 2027 submission; exploratory/pilot evidence; datasets INbreast,
ISIC 2020 and RSNA Pneumonia 2018. Code at seal: `main` = `f41a992`.
Analysis script: `src/augment_atlas/analysis_sap_v1.py`, SHA-256
`bb3bb94a76705f2499f3d32c3778a0c3d212abc314d92ec08678d6ea1d2c2458` (implementation spec:
`docs/analysis-sap-v1.md`). Any later change is a dated amendment; the sealed analysis is still reported.

## Chosen options (all [A] of the v0 draft)
1. Primary family per dataset: H1 = mean(basic_gamma_light, basic_gamma_heavy) − basic;
   H2 = mean(basic_clahe_light, basic_clahe_heavy) − basic. Two-sided, H0: ΔAUROC = 0.
   Secondary (CI only, exploratory, unadjusted): S1/S2 families vs none, S3 basic − none, S4 eight
   per-intensity contrasts, S5 family × basic interaction (descriptive).
2. Estimand: paired ΔAUROC at image level (same folds, images, seed, initialisation); AUROC per E
   fold → mean of 5 folds → mean over seeds. Pooled-E AUROC and group-level AUROC are sensitivities.
3. Multiplicity: Holm, m = 2 within each dataset, α = 0.05; 97.5% simultaneous CI reported for H1/H2.
4. Uncertainty: paired group bootstrap over out-of-fold E predictions (same resample for all arms
   and seeds), within-fold resampling stratified by (group has a positive image), multiplicity-weighted
   AUROC; B = 10,000, fixed seed, 95% percentile CI; p_boot = min(1, 2(min(#Δ*≤0, #Δ*≥0)+1)/(B+1))
   for H1/H2 only. Folds with < 8 positive groups emit a coverage warning.
5. Seeds: primary analysis uses training seed 1; extra seeds only as a sensitivity and only if all
   10 arms × 5 folds complete; the decision to run them is made on time/budget, never on results.
6. ISIC 2020 (Q9): full dataset (all 2,056 patients). The 150 patient groups whose pixel statistics
   were inspected on 2026-10-28 (selection: SHA-256("20261028||group"), first 150; labels never
   consulted) are forced into the development pool D (`global-dev-eval-v1+forced-d-v1`).
   Pre-registered fallback ("rule b"): if forcing exceeds any stratum quota
   (`forced_development_quota_exceeded`), the pools are generated with the plain v1 algorithm without
   forcing, the overlap of explored groups with E is declared, and a sensitivity analysis excluding
   those groups is reported. No reseed, no other fallback.
7. INbreast: primary = campaign `external_fixed` (fixed external recipe Q2); the HPO-D campaign, if
   run, is descriptive only. All INbreast results are exploratory (prior pilot exposure; group_key
   not verified as patient identity). Pools: the 16/10 lists converted without resampling
   (`legacy-proposal-v1-converted`).
8. No equivalence margin; wide intervals are reported as inconclusive, never as "no effect".
9. Calibration (Brier) not reported.
10. Infrastructure failures: rerun the same job (same seed/config) up to 2 retries; divergence (NaN,
    collapse) is a result and is not rerun. A contrast is computed only if both arms have all 5 folds;
    otherwise "not evaluable". Isolation violations invalidate the affected reading and are reported.
11. Seal record: this file + git tag `sap-v1-sealed-20261028` + Brain note (no OSF).
12. Gamma and CLAHE are described as different interventions in grayscale vs RGB (chromatic in RGB);
    no cross-dataset modality effect is claimed. No pooling or ranking across datasets.

## Fixed training rules
Last epoch fixed (20, Q2); no early stopping; no per-epoch evaluation on E; threshold 0.5 only for
descriptive metrics (macro-F1, balanced accuracy, sensitivity, specificity). Secondary metric:
ΔAP with CI, no p-value.

## Dataset definitions
- INbreast: BI-RADS 1–2 (0) vs 4a–6 (1), 387 images / 103 groups (manifest sha256 3836b029…80915).
- ISIC 2020: melanoma vs other, 32,693 images / 2,056 patients after duplicate exclusion
  (manifest sha256 8c5042f3…3e09e10a).
- RSNA Pneumonia 2018 (amendment 2026-10-28, replaces CheXpert for this paper): target = "Lung Opacity"
  in labelGroup "Calculated"; "Normal" and "No Lung Opacity / Not Normal" = 0; group = NIH patient
  (8-digit prefix of img_id via the official mapping); exclusions: images without a Calculated label
  and all pixel-duplicate images; derived PNG 8-bit 512×512 (`configs/acquisition/rsna2018-v1.json`).

## Pre-specified sensitivities
Pooled-E AUROC; group-level AUROC; INbreast per-breast effective n; unstratified bootstrap;
3 seeds if complete; ISIC without explored groups (if any are in E); INbreast external_fixed vs HPO-D;
CLAHE/black-background note for INbreast.

## Engineering amendment (2026-10-28)
Gate V4 + 5h30 endurance replaced by the lean technical gate (B) (PASS 2026-10-28): real queue/worker,
forced interruption + resume with bit-identical best checkpoint, no E data, external thermal watchdog.
This affects operations only, not the statistical analysis.
