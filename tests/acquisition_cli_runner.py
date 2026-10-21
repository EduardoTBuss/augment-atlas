"""Subprocess entry point for kill/resume tests: the real CLI with the Pillow-free test deriver.

``python tests/acquisition_cli_runner.py <cli args>``. Environment knobs (tests only):
* ``ACQ_TEST_KILL_AFTER`` / ``ACQ_TEST_CALL_LOG``: ``fake_derive`` kills its process abruptly
  (``os._exit(137)``) on the N-th derivation, leaving a partial ``.jpg.tmp`` behind;
* ``ACQ_TEST_FREE_KILL_AFTER``: the disk probe kills the process on its (N+1)-th call, i.e.
  right before a spool or derived write is admitted by the disk guard.
"""

import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "src"), str(HERE)]

from acquisition_e2e_helpers import FreeSpace, fake_derive  # noqa: E402

from augment_atlas.acquisition import cli, pipeline  # noqa: E402


def _free_fn():
    free = FreeSpace()
    kill_after = int(os.environ.get("ACQ_TEST_FREE_KILL_AFTER", "0") or 0)
    if not kill_after:
        return free

    def killing(path):
        if free.calls >= kill_after:
            os._exit(137)
        return free(path)
    return killing


def main():
    deps = pipeline.Deps(deriver=fake_derive, free_fn=_free_fn(), sleep=lambda _s: None)
    return cli.main(sys.argv[1:], deps)


if __name__ == "__main__":
    sys.exit(main())
