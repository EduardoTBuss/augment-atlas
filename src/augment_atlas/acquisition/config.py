"""Acquisition config: load, validate, and split into identity vs operational parts.

The *identity* (sources, names, documented counts, preprocessing, duplicate policy,
split parameters) is pinned in the ledger on the first run; changing it afterwards is
a divergence (exit 3). Operational knobs (transfer, disk, canary) and the list of
accepted divergences may change between sessions.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from urllib.parse import urlsplit

from ..contracts import canonical_hash

SCHEMA = "augment-atlas/acquisition-config/v1"
GIB = 1 << 30
MIB = 1 << 20
SOURCE_ROLES = ("images_zip", "ground_truth_v1", "ground_truth_v2", "duplicates")
CSV_ROLES = ("ground_truth_v1", "ground_truth_v2", "duplicates")
OPERATIONAL_KEYS = ("transfer", "disk", "canary", "accepted_divergences")
REQUIRED_COLUMNS = {
    "ground_truth_v1": ("image_name", "patient_id", "target"),
    "ground_truth_v2": ("image_name", "patient_id", "lesion_id", "diagnosis", "benign_malignant", "target"),
    "duplicates": ("image_name_1", "image_name_2"),
}
DOCUMENTED_KEYS = ("images", "patients", "melanoma_images", "duplicate_pairs", "duplicate_distinct_images")
_FILENAME = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class ConfigError(ValueError):
    """Invalid config file (treated as a usage error by the CLI)."""


def _require(condition, message):
    if not condition:
        raise ConfigError(message)


def _positive_int(value, name):
    _require(isinstance(value, int) and not isinstance(value, bool) and value > 0, f"{name} must be a positive int")


def _number(value, name, minimum=0.0, maximum=math.inf):
    _require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
             and minimum <= value <= maximum, f"{name} must be a number in [{minimum}, {maximum}]")


def _regex(value, name, group=None):
    _require(isinstance(value, str) and value, f"{name} must be a non-empty regex string")
    try:
        compiled = re.compile(value)
    except re.error as exc:
        raise ConfigError(f"{name} is not a valid regex: {exc}") from exc
    if group:
        _require(group in compiled.groupindex, f"{name} must define the named group (?P<{group}>...)")
    return compiled


@dataclass(frozen=True)
class AcquisitionConfig:
    raw: dict
    sha256: str            # canonical hash of the whole document
    identity_sha256: str   # canonical hash without operational keys

    # convenience accessors -------------------------------------------------
    @property
    def dataset(self):
        return self.raw["dataset"]

    def source(self, role):
        return self.raw["sources"][role]

    def filename(self, role):
        return Path(urlsplit(self.source(role)["url"]).path).name

    @property
    def member_regex(self):
        return re.compile(self.raw["names"]["member_regex"])

    @property
    def image_name_regex(self):
        return re.compile(self.raw["names"]["image_name_regex"])

    @property
    def ignored_member_regexes(self):
        return [re.compile(item) for item in self.raw["names"]["ignored_member_regexes"]]

    @property
    def group_prefix(self):
        return self.raw["names"]["group_prefix"]

    @property
    def documented(self):
        return self.raw["documented_counts"]

    @property
    def preprocessing(self):
        return self.raw["preprocessing"]

    @property
    def splits(self):
        return self.raw["splits"]

    @property
    def privacy_patterns(self):
        return list(self.raw["privacy"]["patterns"])

    @property
    def accepted_divergences(self):
        return list(self.raw.get("accepted_divergences", []))

    def disk_bytes(self, key):
        return int(round(self.raw["disk"][key] * GIB))

    @property
    def transfer(self):
        return self.raw["transfer"]

    @property
    def canary(self):
        return self.raw["canary"]


def identity_of(raw):
    return canonical_hash({key: value for key, value in raw.items() if key not in OPERATIONAL_KEYS})


def validate(raw):
    _require(isinstance(raw, dict), "config must be a JSON object")
    _require(raw.get("schema") == SCHEMA, f"config schema must be {SCHEMA!r}")
    dataset = raw.get("dataset")
    _require(isinstance(dataset, dict) and isinstance(dataset.get("name"), str) and dataset["name"], "dataset.name required")
    sources = raw.get("sources")
    _require(isinstance(sources, dict) and set(sources) == set(SOURCE_ROLES), f"sources must have exactly {SOURCE_ROLES}")
    for role in SOURCE_ROLES:
        source = sources[role]
        _require(isinstance(source, dict), f"sources.{role} must be an object")
        parts = urlsplit(source.get("url", ""))
        _require(parts.scheme in ("http", "https") and parts.hostname and not parts.query and not parts.fragment,
                 f"sources.{role}.url must be http(s) without query/fragment")
        _require(_FILENAME.match(Path(parts.path).name or ""), f"sources.{role}.url must end in a plain file name")
        _positive_int(source.get("bytes"), f"sources.{role}.bytes")
        _require(isinstance(source.get("etag"), str) and source["etag"] and '"' not in source["etag"],
                 f"sources.{role}.etag must be the unquoted ETag")
        if role in CSV_ROLES:
            header = source.get("header")
            _require(isinstance(header, list) and header and all(isinstance(c, str) and c for c in header)
                     and len(set(header)) == len(header), f"sources.{role}.header must list unique column names")
            missing = [column for column in REQUIRED_COLUMNS[role] if column not in header]
            _require(not missing, f"sources.{role}.header lacks required columns {missing}")
    names = raw.get("names")
    _require(isinstance(names, dict), "names required")
    _regex(names.get("member_regex"), "names.member_regex", group="image")
    _regex(names.get("image_name_regex"), "names.image_name_regex")
    ignored = names.get("ignored_member_regexes")
    _require(isinstance(ignored, list), "names.ignored_member_regexes must be a list")
    for index, item in enumerate(ignored):
        _regex(item, f"names.ignored_member_regexes[{index}]")
    _require(isinstance(names.get("group_prefix"), str) and re.fullmatch(r"[a-z0-9-]{1,32}", names["group_prefix"]),
             "names.group_prefix must match [a-z0-9-]{1,32}")
    documented = raw.get("documented_counts")
    _require(isinstance(documented, dict) and set(documented) == set(DOCUMENTED_KEYS),
             f"documented_counts must have exactly {DOCUMENTED_KEYS}")
    for key in DOCUMENTED_KEYS:
        _positive_int(documented[key], f"documented_counts.{key}")
    _require(isinstance(raw.get("informational_counts", {}), dict), "informational_counts must be an object")
    pre = raw.get("preprocessing")
    _require(isinstance(pre, dict) and isinstance(pre.get("id"), str) and pre["id"], "preprocessing.id required")
    _positive_int(pre.get("short_side"), "preprocessing.short_side")
    _require(pre.get("upscale") is False and pre.get("exif_transpose") is False, "preprocessing: upscale/exif_transpose must be false")
    _require(pre.get("resample") == "LANCZOS" and pre.get("reducing_gap") is None, "preprocessing: LANCZOS, reducing_gap null")
    _require(pre.get("format") == "JPEG" and pre.get("subsampling") == "4:4:4", "preprocessing: JPEG 4:4:4")
    _require(isinstance(pre.get("quality"), int) and 1 <= pre["quality"] <= 100, "preprocessing.quality in 1..100")
    for flag in ("progressive", "optimize", "keep_icc", "keep_exif"):
        _require(pre.get(flag) is False, f"preprocessing.{flag} must be false")
    disk = raw.get("disk")
    _require(isinstance(disk, dict), "disk required")
    for key in ("reserve_gib", "derived_cap_gib", "spool_cap_gib", "precheck_margin_gib", "write_margin_gib"):
        _number(disk.get(key), f"disk.{key}")
    _require(disk["spool_cap_gib"] > 0 and disk["derived_cap_gib"] > 0, "disk caps must be positive")
    transfer = raw.get("transfer")
    _require(isinstance(transfer, dict), "transfer required")
    _number(transfer.get("max_rate_mib_s"), "transfer.max_rate_mib_s", 1e-6)
    _number(transfer.get("timeout_s"), "transfer.timeout_s", 1e-3)
    _positive_int(transfer.get("max_attempts"), "transfer.max_attempts")
    _number(transfer.get("backoff_base_s"), "transfer.backoff_base_s")
    _number(transfer.get("backoff_max_s"), "transfer.backoff_max_s")
    _require(isinstance(transfer.get("workers"), int) and 0 <= transfer["workers"] <= 2,
             "transfer.workers must be 0 (inline) .. 2 (plan: 2 cores)")
    canary = raw.get("canary")
    _require(isinstance(canary, dict), "canary required")
    _positive_int(canary.get("limit"), "canary.limit")
    _number(canary.get("projection_factor"), "canary.projection_factor", 1.0)
    _positive_int(canary.get("projection_every"), "canary.projection_every")
    _require((raw.get("duplicates") or {}).get("policy") == "keep-min-image-name-v1",
             "duplicates.policy must be keep-min-image-name-v1")
    splits = raw.get("splits")
    _require(isinstance(splits, dict) and splits.get("algorithm") == "global-dev-eval-v1", "splits.algorithm must be global-dev-eval-v1")
    _require(isinstance(splits.get("split_seed"), int) and not isinstance(splits["split_seed"], bool), "splits.split_seed must be an int")
    _number(splits.get("dev_fraction"), "splits.dev_fraction", 0.0, 1.0)
    _number(splits.get("select_fraction"), "splits.select_fraction", 0.0, 1.0)
    _positive_int(splits.get("eval_folds"), "splits.eval_folds")
    _require(isinstance(splits.get("min_select_positive_groups"), int) and splits["min_select_positive_groups"] >= 1,
             "splits.min_select_positive_groups must be >= 1")
    _require(isinstance(splits.get("min_eval_fold_positive_groups"), int) and splits["min_eval_fold_positive_groups"] >= 1,
             "splits.min_eval_fold_positive_groups must be >= 1")
    privacy = raw.get("privacy")
    _require(isinstance(privacy, dict) and isinstance(privacy.get("patterns"), list) and privacy["patterns"],
             "privacy.patterns must be a non-empty list")
    for index, item in enumerate(privacy["patterns"]):
        _regex(item, f"privacy.patterns[{index}]")
    accepted = raw.get("accepted_divergences", [])
    _require(isinstance(accepted, list), "accepted_divergences must be a list")
    for index, item in enumerate(accepted):
        _require(isinstance(item, dict) and isinstance(item.get("field"), str) and "observed" in item
                 and isinstance(item.get("resolution"), str) and item["resolution"]
                 and isinstance(item.get("decided_by"), str) and item["decided_by"],
                 f"accepted_divergences[{index}] needs field, observed, resolution, decided_by")
    return AcquisitionConfig(raw=raw, sha256=canonical_hash(raw), identity_sha256=identity_of(raw))


def load(path):
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    return validate(raw)


__all__ = ["AcquisitionConfig", "ConfigError", "CSV_ROLES", "GIB", "MIB", "SCHEMA", "SOURCE_ROLES", "identity_of", "load",
           "validate"]
