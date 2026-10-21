"""Disk guard (reserve + margin, measured on the real filesystem) and a single-instance lock.

``free_fn(path) -> int`` is injectable so tests can simulate a filling disk. The
default is ``shutil.disk_usage(path).free``, which on POSIX is ``f_bavail * f_frsize``
(space available to an unprivileged user, as the plan requires).
"""

import os
from pathlib import Path
import shutil

from .errors import DiskPaused, LockBusy


def default_free_fn(path):
    return shutil.disk_usage(path).free


def existing_anchor(path):
    """Nearest existing ancestor of ``path`` (the root may not exist yet)."""
    path = Path(path).absolute()
    while not path.exists():
        if path.parent == path:
            break
        path = path.parent
    return path


class DiskGuard:
    """Refuse any write that would leave less than ``reserve + margin`` free."""

    def __init__(self, root, *, reserve_bytes, write_margin_bytes, free_fn=None):
        self.root = Path(root)
        self.reserve = int(reserve_bytes)
        self.write_margin = int(write_margin_bytes)
        self.free_fn = free_fn or default_free_fn
        self.checks = 0
        self.min_headroom = None  # smallest (free - s - reserve) observed; logged for audit

    def free(self):
        return int(self.free_fn(existing_anchor(self.root)))

    def precheck(self, needed_bytes):
        """Plan 3.4 step 1: ``free - reserve >= needed`` before anything is written."""
        free = self.free()
        if free - self.reserve < needed_bytes:
            raise DiskPaused(f"precheck: free {free} B - reserve {self.reserve} B < required {needed_bytes} B")
        return free

    def before_write(self, size_bytes, what="write"):
        """Plan 3.4 step 9: ``free - s - reserve >= write_margin`` before each write of ``s`` bytes."""
        self.checks += 1
        free = self.free()
        headroom = free - int(size_bytes) - self.reserve
        self.min_headroom = headroom if self.min_headroom is None else min(self.min_headroom, headroom)
        if headroom < self.write_margin:
            raise DiskPaused(f"disk guard before {what} of {size_bytes} B: free {free} B leaves {headroom} B "
                             f"above the {self.reserve} B reserve (< margin {self.write_margin} B)")
        return free


class ProcessLock:
    """Exclusive, non-blocking advisory lock on ``path`` (flock on POSIX, msvcrt on Windows)."""

    def __init__(self, path):
        self.path = Path(path)
        self._handle = None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self.path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise LockBusy(f"another acquisition holds {self.path}") from exc
        self._handle = handle
        return self

    def release(self):
        if self._handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                self._handle.seek(0)
                try:
                    msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            else:
                import fcntl
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()


__all__ = ["DiskGuard", "ProcessLock", "default_free_fn", "existing_anchor"]
