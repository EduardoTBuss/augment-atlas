"""Durable per-member state (sqlite3, WAL, synchronous=FULL, one transaction per transition).

State machine per ZIP member (plan 3.4 step 5):

    PENDING --stream+CRC+SHA256, spool file durable--> FETCHED --derived file durable--> DERIVED
    PENDING --(junk / unexpected / unlabeled, after verification)--> SKIPPED
    FETCHED --(decoder refused)--> SKIPPED

Directory entries are SKIPPED at insertion (they carry no data). Members are streamed in
local-header order and committed in that order, so the non-directory members that are
still PENDING always form a suffix of the order: resuming means opening a range at the
first PENDING member, and everything before it is never re-read (checked on open).

Only the main process writes; derive workers return results to it. No thread shares the
connection.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from .errors import DivergenceError, IntegrityError

STATES = ("PENDING", "FETCHED", "DERIVED", "SKIPPED")
_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS member(
  idx INTEGER PRIMARY KEY,
  position INTEGER NOT NULL UNIQUE,
  kind TEXT NOT NULL CHECK (kind IN ('dir', 'junk', 'image', 'unexpected')),
  image TEXT UNIQUE,
  labelled INTEGER NOT NULL CHECK (labelled IN (0, 1)),
  crc32 INTEGER NOT NULL,
  csize INTEGER NOT NULL,
  usize INTEGER NOT NULL,
  header_offset INTEGER NOT NULL,
  state TEXT NOT NULL CHECK (state IN ('PENDING', 'FETCHED', 'DERIVED', 'SKIPPED')),
  sha256_original TEXT,
  reason TEXT,
  info TEXT,
  updated_utc TEXT NOT NULL
);
-- Resume and progress queries filter by state and walk in stream order.
CREATE INDEX IF NOT EXISTS member_state_position ON member(state, position);
CREATE TABLE IF NOT EXISTS session(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  mode TEXT NOT NULL,
  start_utc TEXT NOT NULL,
  end_utc TEXT,
  start_position INTEGER,
  end_position INTEGER,
  exit INTEGER,
  note TEXT
);
"""


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), isolation_level=None, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(_SCHEMA)

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _tx(self, statements):
        cursor = self.conn.cursor()
        cursor.execute("BEGIN IMMEDIATE")
        try:
            for sql, params in statements:
                cursor.execute(sql, params)
            cursor.execute("COMMIT")
        except BaseException:
            cursor.execute("ROLLBACK")
            raise

    # -- identity pinning ------------------------------------------------------
    def meta(self):
        return {row["key"]: row["value"] for row in self.conn.execute("SELECT key, value FROM meta")}

    def pin(self, values):
        """First session stores ``values``; later sessions must match them exactly (exit 3 otherwise)."""
        current = self.meta()
        new = {key: str(value) for key, value in values.items() if key not in current}
        for key, value in values.items():
            if key in current and current[key] != str(value):
                raise DivergenceError(f"ledger pins {key}={current[key]!r}, this session has {value!r}; "
                                      "the object or config identity changed after acquisition started")
        if new:
            self._tx([("INSERT INTO meta(key, value) VALUES (?, ?)", (key, value)) for key, value in sorted(new.items())])

    # -- members -----------------------------------------------------------------
    def init_members(self, entries, classified, labelled_images):
        """Insert all CD members on the first session; afterwards verify they are unchanged."""
        rows = []
        for entry, member in zip(entries, classified):
            if entry.index != member["index"]:
                raise ValueError("entries and classification are misaligned")
            rows.append((entry.index, member["position"], member["kind"], member["image"],
                         int(member["image"] in labelled_images) if member["image"] else 0,
                         entry.crc32, entry.compressed_size, entry.uncompressed_size, entry.local_header_offset))
        existing = self.conn.execute("SELECT idx, position, kind, image, labelled, crc32, csize, usize, header_offset "
                                     "FROM member ORDER BY idx").fetchall()
        if not existing:
            now = utc_now()
            self._tx([("INSERT INTO member(idx, position, kind, image, labelled, crc32, csize, usize, header_offset, state, "
                       "reason, updated_utc) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       row + (("SKIPPED", "directory-entry") if row[2] == "dir" else ("PENDING", None)) + (now,))
                      for row in rows])
            return
        if [tuple(row) for row in existing] != sorted(rows):
            raise IntegrityError("ledger members differ from the central directory of this session")

    def first_pending_position(self, total):
        row = self.conn.execute("SELECT MIN(position) AS p FROM member WHERE state='PENDING'").fetchone()
        first = total if row["p"] is None else row["p"]
        late = self.conn.execute("SELECT COUNT(*) AS n FROM member WHERE position > ? AND kind != 'dir' AND state != 'PENDING'",
                                 (first,)).fetchone()["n"]
        if late:
            raise IntegrityError(f"ledger invariant broken: {late} members after the first PENDING are not PENDING")
        return first

    def member(self, idx):
        return self.conn.execute("SELECT * FROM member WHERE idx=?", (idx,)).fetchone()

    def members(self, state=None, kind=None):
        sql, params = "SELECT * FROM member WHERE 1=1", []
        if state:
            sql += " AND state=?"
            params.append(state)
        if kind:
            sql += " AND kind=?"
            params.append(kind)
        return self.conn.execute(sql + " ORDER BY position", params).fetchall()

    def _transition(self, idx, expected, new, **fields):
        assignments = ", ".join(f"{key}=?" for key in fields)
        sql = f"UPDATE member SET state=?, updated_utc=?{', ' + assignments if assignments else ''} WHERE idx=? AND state=?"
        params = (new, utc_now(), *fields.values(), idx, expected)
        cursor = self.conn.cursor()
        cursor.execute("BEGIN IMMEDIATE")
        try:
            cursor.execute(sql, params)
            if cursor.rowcount != 1:
                raise IntegrityError(f"member {idx}: expected state {expected} for transition to {new}")
            cursor.execute("COMMIT")
        except BaseException:
            cursor.execute("ROLLBACK")
            raise

    def mark_fetched(self, idx, sha256):
        self._transition(idx, "PENDING", "FETCHED", sha256_original=sha256)

    def mark_skipped(self, idx, reason, *, from_state="PENDING", sha256=None, info=None):
        fields = {"reason": reason}
        if sha256 is not None:
            fields["sha256_original"] = sha256
        if info is not None:
            fields["info"] = json.dumps(info, sort_keys=True)
        self._transition(idx, from_state, "SKIPPED", **fields)

    def mark_derived(self, idx, info):
        self._transition(idx, "FETCHED", "DERIVED", info=json.dumps(info, sort_keys=True))

    def counts(self):
        out = {state: 0 for state in STATES}
        for row in self.conn.execute("SELECT state, COUNT(*) AS n FROM member GROUP BY state"):
            out[row["state"]] = row["n"]
        return out

    def derived_stats(self):
        """``(count, total derived bytes)`` over DERIVED members."""
        count, total = 0, 0
        for row in self.conn.execute("SELECT info FROM member WHERE state='DERIVED'"):
            count += 1
            total += json.loads(row["info"])["derived"]["bytes"]
        return count, total

    # -- sessions ------------------------------------------------------------------
    def start_session(self, mode, start_position):
        cursor = self.conn.execute("INSERT INTO session(mode, start_utc, start_position) VALUES (?, ?, ?)",
                                   (mode, utc_now(), start_position))
        return cursor.lastrowid

    def end_session(self, session_id, end_position, exit_code, note=None):
        self.conn.execute("UPDATE session SET end_utc=?, end_position=?, exit=?, note=? WHERE id=?",
                          (utc_now(), end_position, exit_code, note, session_id))

    def sessions(self):
        return [dict(row) for row in self.conn.execute("SELECT * FROM session ORDER BY id")]


__all__ = ["Ledger", "STATES", "utc_now"]
