"""``python -m augment_atlas.acquisition probe|canary|full|verify --config <json> --root <dir>``.

Exit codes (plan 3.4): 0 = done (probe clean / canary within budget / READY written),
75 = paused, resumable (disk guard, network retries exhausted, lock busy, incomplete, SIGTERM/SIGINT),
2 = integrity failure (no automatic retry), 3 = divergence needing a human decision.
64 = usage or invalid config (argparse's own 2 would collide with "integrity").

stdout gets one JSON document with counts only (no identifiers); diagnostics go to stderr.
"""

import argparse
import contextlib
import errno
import json
import signal
import sys
import threading

from . import config as config_mod
from .errors import AcquisitionError, Interrupted
from .pipeline import Deps, cmd_probe, cmd_run, cmd_verify

EXIT_USAGE = 64


def _raise_interrupted(signum, _frame):
    raise Interrupted(f"received signal {signum}; stopping cleanly (resume with the same command)")


@contextlib.contextmanager
def _signals_as_pause():
    """Turn SIGTERM/SIGINT into Interrupted (exit 75) so the ledger session is closed properly.

    As PID 1 in a container Python would otherwise ignore SIGTERM and be SIGKILLed by
    ``docker stop`` after its grace period (still consistent, but the session is not recorded).
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for sig in previous:
            signal.signal(sig, _raise_interrupted)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        raise SystemExit(EXIT_USAGE)


def build_parser():
    parser = _Parser(prog="python -m augment_atlas.acquisition", description="ISIC 2020 acquisition (plan 3.4)")
    sub = parser.add_subparsers(dest="command", required=True, parser_class=_Parser)
    for name, help_text in (("probe", "download/verify CSVs, HEAD the ZIP, retain the central directory"),
                            ("canary", "stream + derive the first --limit labelled images, project the size"),
                            ("full", "stream + derive everything (resumes from the ledger), then verify"),
                            ("verify", "offline checks; writes manifest/*, READY.json, splits, public summary")):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--config", required=True, help="acquisition config JSON")
        command.add_argument("--root", required=True, help="private dataset root, e.g. ~/augment-atlas-private/datasets/isic2020")
        if name == "canary":
            command.add_argument("--limit", type=int, default=None, help="labelled images to derive (default: config canary.limit)")
        if name in ("canary", "full"):
            command.add_argument("--lenient", action="store_true",
                                 help="stream despite pending count divergences (they stay recorded and block READY)")
    return parser


def main(argv=None, deps=None):
    args = build_parser().parse_args(argv)
    deps = deps or Deps()
    try:
        cfg = config_mod.load(args.config)
    except config_mod.ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    if getattr(args, "limit", None) is not None and args.limit <= 0:
        print("--limit must be positive", file=sys.stderr)
        return EXIT_USAGE
    try:
        with _signals_as_pause():
            if args.command == "probe":
                result = cmd_probe(cfg, args.root, deps)
            elif args.command == "verify":
                result = cmd_verify(cfg, args.root, deps)
            else:
                result = cmd_run(cfg, args.root, deps, mode=args.command, limit=getattr(args, "limit", None),
                                 strict=not args.lenient)
    except AcquisitionError as exc:
        print(f"{type(exc).__name__} (exit {exc.exit_code}): {exc}", file=sys.stderr)
        return exc.exit_code
    except OSError as exc:
        if exc.errno in (errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC)):
            # The guard should prevent this; if another process filled the disk first, still pause.
            print(f"DiskPaused (exit 75): the filesystem reported {errno.errorcode[exc.errno]}: {exc}", file=sys.stderr)
            return 75
        raise
    print(json.dumps({"command": args.command, "exit": 0, "result": _brief(args.command, result)}, sort_keys=True))
    return 0


def _brief(command, result):
    if command == "probe":
        return result
    if command == "canary":
        return {key: result[key] for key in ("limit", "derived", "derived_bytes", "projection", "counts", "timing")}
    return {"counts": result["counts"], "duplicates": result["duplicates"], "splits_passed": result["splits"]["passed"],
            "hashes": result["hashes"]}


__all__ = ["EXIT_USAGE", "build_parser", "main"]
