"""Error taxonomy for acquisition; each class maps to a process exit code.

Exit codes follow docs/dataset-acquisition-plan.md section 3.4:
0 = READY, 75 = paused/resumable (network or disk), 2 = integrity failure
(no automatic retry), 3 = document or remote object diverged (human decision).
"""


class AcquisitionError(Exception):
    """Base class; ``exit_code`` is what the CLI should return."""

    exit_code = 1


class TransientError(AcquisitionError):
    """Network-level failure that may succeed on retry (timeouts, resets, 5xx)."""

    exit_code = 75


class RetriesExhausted(TransientError):
    """Bounded retry budget consumed; the session pauses and can be resumed later."""


class IntegrityError(AcquisitionError):
    """Bytes or structure failed verification (CRC, size, hostile ZIP layout)."""

    exit_code = 2


class DivergenceError(AcquisitionError):
    """Remote object or server behaviour differs from what the config pins."""

    exit_code = 3


class SpoolPaused(AcquisitionError):
    """Spool cap reached; caller must drain the spool and resume from ``next_index``."""

    exit_code = 75

    def __init__(self, message, next_index):
        super().__init__(message)
        self.next_index = next_index
