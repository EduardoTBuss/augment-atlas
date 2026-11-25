# Commit hash map (2026-11-25)

On 2026-11-25 the commit metadata of this repository was normalized (author
identity and commit-message trailers). File contents did not change: every
branch tip and tag has exactly the same tree as before.

Commit hashes recorded before that date (campaign manifests, container image
tags, run records) are in the `old` column; `new` is the same commit in the
current history. Commits not listed kept their hash.

The annotated tag `sap-v1-sealed-20261028` was recreated on the rewritten
commit with its original tagger date and message (old tag object 3ec5261e,
old commit 7e04e5b7).

| old | new | date | subject |
|---|---|---|---|
| `86f1296febfb` | `26cea0caf501` | 2026-10-21 | Add global HPO producer: search space, plan, D-view and queue evidence collector (S2) |
| `12b4ee3b3ebf` | `9f51336fdd3a` | 2026-10-21 | Close remaining minor review findings: fail-closed durations, symlinked source dirs, mount |
| `b8d45f59a196` | `f78194f22d39` | 2026-10-21 | Enforce job timeout, close retry/confinement gaps and tighten S2 tests after second review |
| `8d25e052df70` | `02f2f8adf99e` | 2026-10-21 | Harden S2 collector and D-view per independent review: retries, GPU cap, confinement, cand |
| `0756187027bd` | `82871e819fc4` | 2026-10-21 | Merge remote-tracking branch 'origin/feature/fleet-20261019' into tarefa-c-merge-branches- |
| `7ebe3e53bcad` | `8d10c142d8df` | 2026-10-21 | Merge remote-tracking branch 'origin/feature/global-protocol-20261016' into tarefa-c-merge |
| `34627c4d2ce7` | `12e7379fa002` | 2026-10-21 | Merge remote-tracking branch 'origin/feature/isic-acquisition-20261019' into tarefa-c-merg |
| `10b52e0773e5` | `cb77341a7488` | 2026-10-21 | Merge remote-tracking branch 'origin/fix/cpu-isa-avx2-20261020' into tarefa-c-merge-branch |
| `46679aac9125` | `10952dd23152` | 2026-10-21 | Merge remote-tracking branch 'origin/wip/s4-telemetry-20261021' into tarefa-c-merge-branch |
| `98b413c70f1f` | `4851013c6a63` | 2026-10-21 | S4 review round 2: closed reason codes in check_report, absolute article root, tighter com |
| `2960a5e0eacf` | `e75ef068766c` | 2026-10-21 | S4 review round 3: typed errors and closed status set for matrix cells in check_report |
| `02c5d904f8e8` | `c5256631ad2a` | 2026-10-21 | S4/telemetry review round 1: reason codes, comparison checks, explicit article root, lazy  |
| `7a96d83edca6` | `34aee600b59d` | 2026-10-22 | Add docs/cloud-context.md: what was done, pending decisions and next steps |
| `e6aa7f391df7` | `030e1c587690` | 2026-10-22 | Add the D/E pools generator global-dev-eval-v1 and the make-pools CLI step |
| `f16679981906` | `0fecd47ee9b3` | 2026-10-22 | Add the HPO path to the global CLI (decision B1) with end-to-end synthetic tests |
| `b5ee1c9e35b1` | `ddd60d0c20d9` | 2026-10-22 | Add the X299 runbook; document B1, Q6, S1/S2 and the HPO commands |
| `4f8831eb54d5` | `870e7c3f748f` | 2026-10-22 | Add the global CLI and final-job collector for the external fixed recipe path |
| `8425f6f4ee87` | `68ab090a2bd8` | 2026-10-22 | Address independent review: checksummed results, explicit validation mode, leak scan |
| `7510b38af5fa` | `9a658f1c68b0` | 2026-10-22 | Address second review: pin pools generation to the pre-registered config |
| `4f3fd095ed04` | `55d78d69edc4` | 2026-10-22 | Address the integrated audit: freeze only from the current complete evidence, pin image an |
| `5bbae4111954` | `36afd26e70bc` | 2026-10-22 | Address the re-review: binding first HPO collection, hpo_d anchor re-check, published amen |
| `9d8125c01015` | `9eb66129b2ba` | 2026-10-22 | Address the third review: refuse failed HPO attempts with scores, pin the amendment to ori |
| `324d4b9c17f8` | `561dd961e8cc` | 2026-10-22 | B1: collect HPO jobs at the configs commit under Q6 evidence; allow hpo/final workers |
| `9ae9c0292bbc` | `f0261d6748a2` | 2026-10-22 | ISIC: pre-registered split amendment (S1/S2) and the eight extra duplicate pairs |
| `c4aefd2bd579` | `049141cef38c` | 2026-10-22 | Report a bare tool missing from PATH as tool_absent even when exec says EACCES |
| `0fe04b6f1d54` | `281fb6605cbf` | 2026-10-22 | Update docs/cloud-context.md for the second session of 22/10/2026 |
| `bcf27ca48fc6` | `15d798605122` | 2026-10-22 | cloud-context: record the integrated audit, its fixes and the test totals at 55d78d6 |
| `47402d5dd98b` | `82c8200d7aae` | 2026-10-22 | cloud-context: record the re-review, its fixes and the totals at 36afd26 |
| `c537b3c68bbd` | `7982cfe38b0a` | 2026-10-22 | cloud-context: third review, final commits and totals at 9eb6612, open decisions (retry po |
| `c2ff5b3190be` | `23b389fbe1f0` | 2026-10-23 | HPO collector: refuse non-succeeded attempts with scores before the timeout and cap shortc |
| `4b590cb2c1f5` | `485aff6f3d6b` | 2026-10-23 | HPO worker: pin image and probe to plan.runtime.environment_sha256 before claim and create |
| `22fe609a10a3` | `dcef079307a9` | 2026-10-23 | Runbook: separate the manual HPO retry from --resume and document the retry procedure (F3) |
| `1dabfd87f35c` | `fd351ca2e42e` | 2026-10-23 | cloud-context: record the 23/10 review findings F1, F2, F3, V1 and the pending decision R |
| `5efb412c6237` | `857274d46050` | 2026-10-24 | HPO collector: an earlier attempt with a dangling predictions.json link also blocks a retr |
| `8913ad328620` | `80d29e6375b8` | 2026-10-24 | cloud-context: record the independent review, its three findings and the final test totals |
| `f23135d00134` | `ca7a70e91356` | 2026-10-28 | Add a timing-independent backpressure test for the ISIC acquisition (V1) |
| `0a49e0beee40` | `d9f311244bb8` | 2026-10-28 | C2 optional forced-development groups for make-pools |
| `426cf1279a45` | `2d2cb8f69a2c` | 2026-10-28 | C2b forced-development: record and re-verify canonical hash of the forced group list |
| `5563c8e020ed` | `492149814d08` | 2026-10-28 | C5 SAP v1 analysis script (grouped paired bootstrap, Holm, forest plot) with synthetic tes |
| `81d4245ea220` | `41ee453f1dc5` | 2026-10-28 | C5b apply stats review: no forest title, <8 positive-group warning, provenance, realistic  |
| `b1ed2c62831d` | `7639e74461cf` | 2026-10-28 | C6 convert INbreast 16/10 legacy pools to the v1 pools schema without redraw |
| `60a9e00743a4` | `c9d413a996c9` | 2026-10-28 | C7 RSNA Pneumonia 2018 offline acquisition (NIH patient groups, Calculated labels, approve |
| `7df91a967476` | `ea143113dd22` | 2026-10-28 | Campaign driver and thermal watchdog as reviewed repo scripts (label-filtered stop, config |
| `9ef0042b7ae0` | `8549c06ca6be` | 2026-10-28 | Merge c2b-forced-development-canonical-20261028 into integration/esann-20261028 |
| `b9041f87058e` | `20b00ab08a9e` | 2026-10-28 | Merge c5b-analysis-sap-v1-review-20261028 into integration/esann-20261028 |
| `fd3996cbbd1f` | `d7f036db9ef6` | 2026-10-28 | Merge c6-inbreast-legacy-pools-20261028 into integration/esann-20261028 (doc conflict: kee |
| `f41a992bb3ef` | `aa11cc8de15b` | 2026-10-28 | Merge c7-rsna2018-acquisition-20261028 into integration/esann-20261028 |
| `33a8f3b3ed82` | `63c96569e631` | 2026-10-28 | Merge fix/rsna-split-spec-20261029 into integration/esann-20261029 |
| `eeecf951f95a` | `0e28343d11e1` | 2026-10-28 | Merge origin/ops/campaign-driver-watchdog-20261029 into integration/esann-20261029 |
| `835c70e59e5b` | `18c9021920f0` | 2026-10-28 | Merge origin/paper/results-generator-20261029 into integration/esann-20261029 |
| `4914016e21b5` | `ba1348b48774` | 2026-10-28 | Merge origin/sap/seal-20261028 into integration/esann-20261029 |
| `a22833f329e8` | `876998748f54` | 2026-10-28 | Never record code/image provenance as a silent "unavailable" |
| `9c7dac45a345` | `017e279096c5` | 2026-10-28 | Paper results generator over the sealed SAP v1 output (tables, forest panels, provenance); |
| `570ad4cef20b` | `3220c12e5ccc` | 2026-10-28 | Pin engineering-smoke contract for gray and RGB inputs; record gaps |
| `4b28830fcfcd` | `1d412d3938a6` | 2026-10-28 | Pin the F1 residual: an earlier attempt with only results.json is not refused |
| `a9b3d4274339` | `b108ffb04eda` | 2026-10-28 | RSNA 2018: accept the observed patients_included divergence (12,232 vs approximate 12,233) |
| `ae56e6d99c63` | `695e08123c81` | 2026-10-28 | RSNA 2018: pre-register the global-dev-eval-v1 split amendment (seeds 20261029/20261029/20 |
| `a7754db879ae` | `c4e49d7fe3be` | 2026-10-28 | RSNA 2018: split-spec/check_amendment accept the rsna2018 acquisition layout |
| `fa5f9af6a020` | `271e422e36d2` | 2026-10-28 | Refuse unknown smoke dataset keys instead of ignoring them |
| `7e04e5b7bffd` | `687803ba377a` | 2026-10-28 | Seal SAP v1 for the ESANN 2027 exploratory campaigns (INbreast, ISIC 2020, RSNA 2018) |
| `48d95341936e` | `86e3284c3708` | 2026-10-28 | inbreast: HPO configs |
| `0feacec6f0a2` | `e4127a79e4d5` | 2026-10-28 | inbreast: drop final configs (a) from the tree |
| `0381775ce9ae` | `aa6f1454c347` | 2026-10-28 | inbreast: final configs, campaign (a) |
| `b8cabee2f4cb` | `bc105302084d` | 2026-10-28 | inbreast: final configs, campaign (b) |
| `d717ea915f3a` | `dc9ceefa3c69` | 2026-10-28 | isic2020: HPO-D configs (plan 3f9ba41e, search space 0b142415) |
| `5aad98f6fa45` | `681efec58cd6` | 2026-10-29 | Queue board: derived BOARD.md (totals, per-campaign progress, running/stale, failures, ETA |
| `ce1d45e1930c` | `556bfbf79ee9` | 2026-10-29 | Seed requests control plane: control/seed-requests.md parser, plan/apply with node affinit |
| `32444c227ef2` | `a8665777dae4` | 2026-10-29 | authorize-stage: sealed, anchored authorization of a later pre-declared seed stage (additi |
| `775704ff7ff3` | `e20f35046688` | 2026-10-29 | isic2020: final configs, campaign hpo_d (freeze baa20807, stage seed 11) |
| `19b727b54333` | `ec03e5efbca4` | 2026-10-29 | rsna2018: HPO-D configs (plan 42118088, search space 2c3ee069) |
| `d89fbe87239e` | `b4c9ccb036b0` | 2026-10-29 | rsna2018: final configs, campaign hpo_d (freeze bb130dc0, stage seed 11) |
| `05bb6675385c` | `1ab8883b0bd5` | 2026-10-30 | A1: stage 2 may run while the campaign's stage 1 is still running (authorize-stage --allow |
| `2eb042cdfb5e` | `23f886e60cef` | 2026-10-30 | D9: declared release of the primary_workers_active gate for one campaign's stage authoriza |
| `5039562c9f4c` | `534388fac726` | 2026-10-30 | Merge feature/stage-authorization-20261029 (C9b) into integration/seeds-20261030 (X5) with |
| `d270e34739b6` | `c4e447e5135b` | 2026-10-30 | Runbook: known limitations accepted after re-review R2 (gates at production entry points o |
| `27b4a1f14f0d` | `bc5ce420c911` | 2026-10-30 | SAP v1 amendment A1 (2026-10-30): pre-declared seeds 13/19 may run in parallel with the ca |
| `826e76f67b4b` | `f833caedc273` | 2026-10-30 | Stage gates after review R1: seed eligibility by authorized seeds, primary-workers publica |
| `bc5b20b26479` | `701235862e73` | 2026-10-30 | Tests: adapt three C9b stage tests to the R1 gates and make the cross-environment job choi |
| `919cd245b965` | `045cd91b32aa` | 2026-10-30 | inbreast (a): add declared stage 2 sensitivity configs (seeds 13,19) |
| `b005e8414ef8` | `31202c644f7d` | 2026-10-30 | isic2020: add declared stage 2 sensitivity configs (seeds 13,19), SAP amendment A1 |
| `f58b8d4b29b3` | `0dc285508b7b` | 2026-10-30 | rsna2018: add declared stage 2 sensitivity configs (seeds 13,19), SAP amendment A1 |
