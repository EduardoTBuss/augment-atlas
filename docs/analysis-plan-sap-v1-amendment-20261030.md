# SAP v1 amendment A1 (2026-10-30): timing of the pre-declared later seeds

Status: recorded before execution. The sealed SAP (`analysis-plan-sap-v1-sealed.md`, tag
`sap-v1-sealed-20261028`) is not edited. This file amends one rule, item 5, and keeps the remaining items.

## What changes

SAP item 5 said that extra seeds run only after all 10 arms x 5 folds of stage 1 are complete. With A1,
the pre-declared stage 2 of a campaign (seeds 13 and 19, `--stages "11;13,19"`, sealed on 2026-10-28/29)
**may run in parallel with that campaign's stage 1 on a different node**. Two conditions apply:

1. Each later stage runs entirely on one node. If that node differs from the campaign environment, the
   difference is declared (`cross_environment`, same image), so hardware varies only between seeds.
   Comparisons between arms stay paired within each seed.
2. The primary analysis still uses only training seed 11 and still requires the **complete** stage-1
   matrix. Its report and collection must reflect the final state of every stage-1 cell. A later
   authorization must never freeze the primary at a partial state.

## What does not change

- Hypotheses H1/H2, estimand, contrasts, multiplicity, bootstrap and decision rules (items 1–4 and 6–12).
- The seeds themselves. Seeds 13 and 19 were declared before any result; no seed is added.
- The role of seeds 13 and 19: sensitivity only ("3 seeds if complete"); they never enter the primary.
- Infrastructure retries (item 10) and the treatment of divergence or collapse as a result.

## Provenance of the decision

- 2026-10-29: Eduardo ordered seeds 13 and 19 for all datasets (INbreast first, then ISIC/RSNA) after the
  seed-11 runs, on time/budget grounds.
- 2026-10-30, before any ISIC 2020 or RSNA 2018 result was viewed: Eduardo decided that the later seeds
  run in parallel on a second node (Q-FLEX) without waiting for stage 1 to finish ("seeds paralelas em
  máquinas diferentes não têm problema").
- 2026-10-30, after that decision: at the PI's request, a descriptive interim look at the partial ISIC
  seed-11 results took place. It changed no decision. Eduardo then confirmed the parallel execution and
  authorized the code changes needed for it.
- The decision concerns only the timing and placement of pre-declared seeds. It does not depend on
  any result.

## Classification

Results of seeds 13/19 stay labelled as sensitivity, with environment provenance per seed. The INbreast
stage-2 decision was taken after viewing the INbreast primary results, and its authorization records
`decided_after_primary_results: true`. ISIC 2020 and RSNA 2018 authorizations record the decision dates
above and mention the interim look in their rationale.
