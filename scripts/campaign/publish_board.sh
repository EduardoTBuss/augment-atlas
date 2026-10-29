#!/usr/bin/env bash
# Dedicated single writer: queue is fetched; only refs/heads/board is pushed.
set -euo pipefail
umask 077
SOURCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$SOURCE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python3}" - "$@" <<'PY'
import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from augment_atlas.queue_board import read_snapshot, render_board


def git(clone, *args, input=None, env=None, check=True):
    return subprocess.run(["git", "-C", str(clone), *args], input=input,
                          env=env, capture_output=True, text=True, encoding="utf-8",
                          timeout=120, check=check)


def publish(clone, output, env):
    # Fetch only queue. Heartbeat refs are deliberately out of scope.
    git(clone, "fetch", "--no-tags", "origin",
        "refs/heads/queue:refs/remotes/origin/queue", env=env)
    commit, records, anchors = read_snapshot(clone)
    content = render_board(commit, records, anchors, datetime.now(timezone.utc))
    temporary = output.with_suffix(".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(output)

    # An absent board is an independent root; an existing board is a direct
    # parent. A competing push fails with no force/rebase.
    exists = git(clone, "ls-remote", "--exit-code", "--heads", "origin",
                 "refs/heads/board", env=env, check=False)
    if exists.returncode not in (0, 2):
        raise RuntimeError("could not determine remote board state")
    parent = None
    if exists.returncode == 0:
        git(clone, "fetch", "--no-tags", "origin",
            "refs/heads/board:refs/remotes/origin/board", env=env)
        parent = git(clone, "rev-parse", "--verify", "refs/remotes/origin/board^{commit}", env=env).stdout.strip()
    blob = git(clone, "hash-object", "-w", "--stdin", input=content, env=env).stdout.strip()
    # A board-only tree, without touching a checkout, index or queue tree.
    tree = git(clone, "mktree", input=f"100644 blob {blob}\tBOARD.md\n", env=env).stdout.strip()
    if parent and git(clone, "rev-parse", parent + "^{tree}", env=env).stdout.strip() == tree:
        print("board unchanged", flush=True)
        return
    args = ["commit-tree", tree]
    if parent:
        args += ["-p", parent]
    new_commit = git(clone, *args, input=f"Update queue board from {commit}\n", env=env).stdout.strip()
    git(clone, "push", "origin", new_commit + ":refs/heads/board", env=env)
    print(f"published board from queue {commit}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Publish BOARD.md to the separate board branch")
    parser.add_argument("--repo", required=True, help="SSH Git remote; authentication inherited via GIT_SSH")
    parser.add_argument("--state-dir", required=True, type=Path, help="dedicated publisher state, never a worker checkout")
    parser.add_argument("--interval", type=float, default=900, help="seconds, default 900 (15 min)")
    parser.add_argument("--stop-file", type=Path, help="default STATE_DIR/board.stop")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval must be finite and positive")
    state = args.state_dir.resolve()
    state.mkdir(parents=True, exist_ok=True)
    stop = args.stop_file.resolve() if args.stop_file else state / "board.stop"
    lock = state / "publisher.lock"
    try:
        with lock.open("x", encoding="ascii") as handle:
            handle.write(str(os.getpid()) + "\n")
    except FileExistsError:
        print("publisher lock exists; verify the previous writer exited before removing it", file=sys.stderr)
        return 2
    try:
        if stop.exists():
            return 0
        clone = state / "board-publisher.git"
        env = os.environ.copy()  # GIT_SSH and authentication remain inherited.
        env["GIT_TERMINAL_PROMPT"] = "0"
        for key, value in {
            "GIT_AUTHOR_NAME": "AugmentAtlas board", "GIT_AUTHOR_EMAIL": "board@example.invalid",
            "GIT_COMMITTER_NAME": "AugmentAtlas board", "GIT_COMMITTER_EMAIL": "board@example.invalid",
        }.items():
            env.setdefault(key, value)
        if not clone.exists():
            clone.mkdir()
            git(clone, "init", "--bare", env=env)
            git(clone, "remote", "add", "origin", args.repo, env=env)
        if (git(clone, "rev-parse", "--is-bare-repository", env=env).stdout.strip() != "true"
                or git(clone, "remote", "get-url", "origin", env=env).stdout.strip() != args.repo):
            raise ValueError("state clone must be bare and belong to --repo")
        while not stop.exists():
            publish(clone, state / "BOARD.md", env)
            if args.once:
                return 0
            deadline = time.monotonic() + args.interval
            while not stop.exists() and time.monotonic() < deadline:
                time.sleep(min(1, max(0, deadline - time.monotonic())))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        # Do not echo arguments, remote addresses or document contents.
        print("board publisher failed: " + type(error).__name__ + "; inspect the dedicated state clone", file=sys.stderr)
        return 1
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
PY
