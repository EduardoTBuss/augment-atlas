"""End-to-end fixtures for the ISIC acquisition tool: a local multi-object server, a synthetic
release (ZIP64 + three CSVs) and stdlib-only test derivers. Only 127.0.0.1 is contacted.

Synthetic identifiers only (``SYNTH_0000001``, ``PSYNTH_0001``, ``LSYNTH_0000001``); nothing here
matches the real ISIC ID patterns.
"""

import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

from acquisition_helpers import LocalRangeServer, ServerState, _Handler, build_zip

GIB = 1 << 30
V1_HEADER = ["image_name", "patient_id", "sex", "age_approx", "anatom_site_general_challenge", "diagnosis",
             "benign_malignant", "target"]
V2_HEADER = ["image_name", "patient_id", "lesion_id", "sex", "age_approx", "anatom_site_general_challenge",
             "diagnosis", "benign_malignant", "target"]
DUP_HEADER = ["image_name_1", "image_name_2"]
ZIP_DIR = "SYNTH_Training_JPEG/train/"
PATHS = {"images_zip": "/challenges/2020/SYNTH_Training_JPEG.zip",
         "ground_truth_v1": "/challenges/2020/SYNTH_Training_GroundTruth.csv",
         "ground_truth_v2": "/challenges/2020/SYNTH_Training_GroundTruth_v2.csv",
         "duplicates": "/challenges/2020/SYNTH_Training_Duplicates.csv"}
ROOT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "acquisition" / "isic2020-v1.json"


# ---------------------------------------------------------------------------
# multi-object server
# ---------------------------------------------------------------------------

class _MultiHandler(_Handler):
    states = None

    @property
    def state(self):
        return self.states.get(urlsplit(self.path).path)

    def _missing(self):
        if self.state is not None:
            return False
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        return True

    def do_HEAD(self):
        if not self._missing():
            super().do_HEAD()

    def do_GET(self):
        if not self._missing():
            super().do_GET()


class MultiObjectServer(LocalRangeServer):
    """Serves ``{path: (bytes, etag)}``; each object has its own ServerState (fault knobs)."""

    def __init__(self, objects):
        from http.server import ThreadingHTTPServer
        import threading
        self.states = {path: ServerState(data, etag) for path, (data, etag) in objects.items()}
        handler = type("Handler", (_MultiHandler,), {"states": self.states})
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)

    def base(self):
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def url(self, role):
        return self.base() + PATHS[role]

    def state(self, role):
        return self.states[PATHS[role]]

    def ranged_gets(self, role="images_zip"):
        return [rng for method, rng, _ir in self.state(role).requests if method == "GET" and rng]


# ---------------------------------------------------------------------------
# synthetic release
# ---------------------------------------------------------------------------

def _payload(seed, size=700):
    block = hashlib.sha256(f"synthetic-{seed}".encode()).digest()
    return (block * (size // 32 + 1))[:size]


def image_name(i):
    return f"SYNTH_{i:07d}"


def patient_name(i):
    return f"PSYNTH_{i:04d}"


class SyntheticRelease:
    """A small but structurally faithful release.

    * ``n_patients`` patients with ``images_per_patient`` images each; the first image of
      every ``positive_every``-th patient is a melanoma.
    * duplicate pairs: images whose bytes are identical; one pair crosses patients.
    * ZIP64 with directory entry, ``__MACOSX`` and ``.DS_Store`` junk, deflate + stored,
      data descriptors.
    """

    def __init__(self, n_patients=40, images_per_patient=3, positive_every=3, jpeg_bytes=None):
        self.rows = []
        index = 0
        for p in range(1, n_patients + 1):
            count = images_per_patient + (p % 2)  # 3 or 4 images: varied group sizes for quartiles
            for k in range(count):
                index += 1
                positive = (p % positive_every == 0 and k == 0)
                self.rows.append({"image_name": image_name(index), "patient_id": patient_name(p),
                                  "lesion_id": f"LSYNTH_{index:07d}", "sex": "female" if p % 2 else "male",
                                  "age_approx": str(30 + p % 40), "anatom_site_general_challenge": "torso",
                                  "diagnosis": "melanoma" if positive else "nevus",
                                  "benign_malignant": "malignant" if positive else "benign", "target": "1" if positive else "0"})
        self.data = {row["image_name"]: (jpeg_bytes(i) if jpeg_bytes else _payload(i)) for i, row in enumerate(self.rows)}
        names = [row["image_name"] for row in self.rows]
        # pair 1: same patient (images 1,2); pair 2: crosses patients 1 and 2 (images 3 and 5)
        self.pairs = [(names[1], names[2]), (names[3], names[5])]
        for a, b in self.pairs:
            self.data[b] = self.data[a]
            self.row(b)["target"] = self.row(a)["target"]
            self.row(b)["diagnosis"] = self.row(a)["diagnosis"]
            self.row(b)["benign_malignant"] = self.row(a)["benign_malignant"]
        self.zip_extra_members = []      # (name, bytes) appended (e.g. image without label)
        self.drop_from_zip = set()       # images whose member is omitted (label without image)
        self.corrupt = set()             # images whose bytes start with CORRUPT (fake deriver skips)

    def row(self, image):
        return next(r for r in self.rows if r["image_name"] == image)

    def members(self):
        out = [{"name": ZIP_DIR, "data": b"", "method": 0}]
        for i, row in enumerate(self.rows):
            image = row["image_name"]
            if image in self.drop_from_zip:
                continue
            data = (b"CORRUPT" + self.data[image]) if image in self.corrupt else self.data[image]
            out.append({"name": f"{ZIP_DIR}{image}.jpg", "data": data, "method": 8 if i % 3 else 0,
                        "descriptor": i % 4 == 1})
            if i == 4:
                out.append({"name": f"__MACOSX/{ZIP_DIR}._{image}.jpg", "data": b"\x00\x05\x16\x07junk", "method": 0})
            if i == 9:
                out.append({"name": f"{ZIP_DIR}.DS_Store", "data": b"Bud1" + b"\x00" * 60, "method": 8})
        for name, data in self.zip_extra_members:
            out.append({"name": name, "data": data, "method": 8})
        return out

    def zip_bytes(self):
        data, size = build_zip(self.members(), zip64=True)
        assert size == len(data)
        return data

    def csv_bytes(self, role, *, header=None, rows=None, line_end="\n"):
        if role == "duplicates":
            header = header or DUP_HEADER
            body_rows = rows if rows is not None else [list(pair) for pair in self.pairs]
        else:
            header = header or (V1_HEADER if role == "ground_truth_v1" else V2_HEADER)
            source = rows if rows is not None else self.rows
            body_rows = [[row.get(column, "") for column in header] for row in source]
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, lineterminator=line_end)
        writer.writerow(header)
        writer.writerows(body_rows)
        return buffer.getvalue().encode("utf-8")

    def objects(self, overrides=None):
        """``{path: (bytes, etag)}``; CSV ETags are their MD5 (simple S3 ETag), ZIP gets a multipart-like ETag."""
        overrides = overrides or {}
        out = {}
        for role in PATHS:
            body = overrides.get(role)
            if body is None:
                body = self.zip_bytes() if role == "images_zip" else self.csv_bytes(role)
            etag = (hashlib.md5(body).hexdigest() + "-3") if role == "images_zip" else hashlib.md5(body).hexdigest()
            out[PATHS[role]] = (body, etag)
        return out

    def counts(self):
        labelled = [r for r in self.rows]
        return {"images": len(labelled), "patients": len({r["patient_id"] for r in labelled}),
                "melanoma_images": sum(int(r["target"]) for r in labelled), "duplicate_pairs": len(self.pairs),
                "duplicate_distinct_images": len({n for p in self.pairs for n in p})}

    def identifiers(self):
        ids = set()
        for row in self.rows:
            ids |= {row["image_name"], row["patient_id"], row["lesion_id"]}
        return ids


def make_config(server, release, path, *, documented=None, objects=None, **section_overrides):
    """Write a test config derived from the versioned one: same structure, local URLs/sizes/ETags,
    synthetic name patterns, workers=0 (inline) unless overridden, tiny backoff."""
    cfg = json.loads(ROOT_CONFIG.read_text(encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    objects = objects or {PATHS[role]: (server.state(role).data, server.state(role).etag) for role in PATHS}
    for role in PATHS:
        body, etag = objects[PATHS[role]]
        cfg["sources"][role].update(url=server.url(role), bytes=len(body), etag=etag)
    cfg["names"]["member_regex"] = r"^(?:[A-Za-z0-9_.-]+/)*(?P<image>SYNTH_\d{7})\.jpg$"
    cfg["names"]["image_name_regex"] = r"^SYNTH_\d{7}$"
    cfg["privacy"]["patterns"] = [r"SYNTH_\d{7}", r"PSYNTH_\d{4}", r"LSYNTH_\d{7}"]
    cfg["documented_counts"] = documented or release.counts()
    cfg["transfer"].update(workers=0, backoff_base_s=0.0, backoff_max_s=0.0, timeout_s=10, max_rate_mib_s=1024)
    cfg["splits"].update(dev_fraction=0.4, select_fraction=0.5, min_select_positive_groups=1,
                         min_eval_fold_positive_groups=1)
    for section, values in section_overrides.items():
        if isinstance(values, dict) and isinstance(cfg.get(section), dict):
            cfg[section].update(values)
        else:
            cfg[section] = values
    Path(path).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg


# ---------------------------------------------------------------------------
# stdlib test derivers (Pillow-free); top-level so a process pool can pickle them
# ---------------------------------------------------------------------------

def fake_derive(src, dst_tmp, params):
    data = Path(src).read_bytes()
    log = os.environ.get("ACQ_TEST_CALL_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as stream:
            stream.write(Path(src).name + "\n")
    if data.startswith(b"CORRUPT"):
        return {"status": "skipped", "reason": "decode-failed:FakeCorrupt"}
    delay = float(os.environ.get("ACQ_TEST_DERIVE_DELAY", "0") or 0)
    if delay:
        time.sleep(delay)  # makes the derive side slower than the stream (backpressure tests)
    kill_after = int(os.environ.get("ACQ_TEST_KILL_AFTER", "0") or 0)
    if kill_after and log:
        calls = len(Path(log).read_text(encoding="utf-8").splitlines())
        if calls >= kill_after:
            with open(dst_tmp, "wb") as partial:
                partial.write(b"PARTIAL")
            os._exit(137)  # abrupt: no finally blocks, no sqlite commit
    out = b"DERIVED\n" + hashlib.sha256(data).digest() + bytes([len(data) % 251]) * (len(data) // 10)
    with open(dst_tmp, "wb") as sink:
        sink.write(out)
        sink.flush()
        os.fsync(sink.fileno())
    digest = hashlib.sha256(out).hexdigest()
    return {"status": "ok",
            "original": {"format": "FAKE", "mode": "RGB", "width": len(data), "height": 1, "exif_orientation": 1, "icc": False},
            "derived": {"width": len(out), "height": 1, "pixel_sha256": digest, "icc": False, "exif": False,
                        "bytes": len(out), "sha256": digest},
            "short_side_lt_target": True}


class FreeSpace:
    """Simulated ``free_fn``: ``plenty`` bytes until ``fail_after`` calls, then ``low`` bytes."""

    def __init__(self, plenty=40 * GIB, low=None, fail_after=None):
        self.plenty, self.low, self.fail_after, self.calls = plenty, low, fail_after, 0

    def __call__(self, _path):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            return self.low
        return self.plenty
