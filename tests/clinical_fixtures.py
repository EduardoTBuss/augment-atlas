"""Synthetic-only clinical evaluation fixtures (no patient data, no torch).

Not collected by unittest discovery (name lacks the test prefix); test modules
load it by path, matching the existing test_runtime fixture-loading pattern.
"""
from copy import deepcopy
from pathlib import Path
import tempfile

from augment_atlas.contracts import canonical_hash, load_config, validate_config
from augment_atlas.data import load_records, prepare_splits
from augment_atlas.evaluation import clinical_environment_block, evaluation_partition

ROOT = Path(__file__).resolve().parents[1]
CODE_COMMIT = "b" * 40

BINARY = dict(schema_version=1, task="binary", classes=[0, 1], score_kind="softmax_probability",
              frozen=True, positive_class=1, threshold=0.5, comparison="ge")
MULTICLASS = dict(schema_version=1, task="multiclass", classes=["nevus", "melanoma", "keratosis"],
                  score_kind="softmax_probability", frozen=True, tie_break="first")
ORDINAL = dict(schema_version=1, task="ordinal", classes=[0, 1, 2], score_kind="softmax_probability",
               frozen=True, tie_break="first")


def clinical_config(evaluation=BINARY, *, groups=30, scope=None):
    """Synthetic smoke config plus an explicit evaluation opt-in."""
    cfg = load_config(ROOT / "configs/smoke/synthetic.json")
    cfg["model"]["num_classes"] = len(evaluation["classes"])
    cfg["dataset"]["synthetic_groups"] = groups
    if scope is not None:
        cfg["training"]["evaluation_scope"] = scope
        if scope == "selection-only":
            cfg["recipe"]["hpo"]["protocol"] = "within-outer-fold"
    cfg["evaluation"] = deepcopy(evaluation)
    return validate_config(cfg)


def splits_for(cfg, records, root=None):
    with tempfile.TemporaryDirectory(prefix="atlas-clinical-fixture-") as directory:
        return prepare_splits(cfg, records, directory, root=root)


def environment_for(cfg, splits, **extra):
    """The fields the trainer writes that clinical validation depends on."""
    environment = {"schema_version": 1, "source": "synthetic-fixture", "utc": "2026-10-19T00:00:00+00:00",
                   "config_sha256": canonical_hash(cfg), "splits_sha256": splits["sha256"],
                   "evaluation_scope": cfg["training"].get("evaluation_scope", "test"),
                   "clinical_evaluation": clinical_environment_block(cfg),
                   "code_commit": CODE_COMMIT, "image_id": "sha256:fixture-image"}
    environment.update(extra)
    return environment


def partition_rows(cfg, splits, records):
    lookup = {record["id"]: record for record in records}
    return [lookup[item] for item in splits["lists"][evaluation_partition(cfg)]]


def peaked_scores(rows, width, *, wrong=()):
    """Dyadic softmax-like rows: 3/4 (binary) or 1/2 (3 classes) on one column.

    Row i peaks on its true class unless i is in ``wrong`` (then on the next class).
    Dyadic values keep sums exactly one and JSON round trips exact.
    """
    peak = 0.75 if width == 2 else 0.5
    rest = (1 - peak) / (width - 1)
    output = []
    for index, row in enumerate(rows):
        column = (row["label"] + 1) % width if index in wrong else row["label"]
        output.append([peak if j == column else rest for j in range(width)])
    return output


def build_inputs(evaluation=BINARY, *, groups=30, scope=None):
    cfg = clinical_config(evaluation, groups=groups, scope=scope)
    records = load_records(cfg)
    splits = splits_for(cfg, records)
    return cfg, records, splits, environment_for(cfg, splits)
