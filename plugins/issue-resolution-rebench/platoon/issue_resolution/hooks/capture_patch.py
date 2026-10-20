"""Save unique cumulative patches and file contents using only the stdlib.

Run synchronously after tools finish. A private index leaves the agent's staging
and working files untouched, even when the agent commits during the rollout.
"""

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path


def decode_source(data: bytes) -> str:
    # Match the localization processor's UTF-8-sig, universal-newline reads.
    with io.TextIOWrapper(io.BytesIO(data), encoding="utf-8-sig") as stream:
        return stream.read()


def capture(repo: Path, base_commit: str, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    index = output_dir / "index"
    git_env = dict(os.environ, GIT_INDEX_FILE=str(index.resolve()), LC_ALL="C")

    def git(*args):
        result = subprocess.run(
            ["git", "-C", str(repo), *args], env=git_env,
            capture_output=True,
        )
        if result.returncode:
            error = result.stderr.decode(errors="replace").strip()
            raise RuntimeError(f"git {' '.join(args)}: {error}")
        return result.stdout

    if not index.exists():
        git("read-tree", base_commit)
    git("add", "-A")
    # Match the ORM diff format without deleting working files.
    # The baseline never follows agent commits.
    patch = git("diff", "--no-color", "--cached", base_commit)
    patch_hash = hashlib.sha256(patch).hexdigest()
    result = {"patch_hash": patch_hash}
    destination = output_dir / "snapshots.jsonl"
    if destination.exists():
        with destination.open() as stream:
            if any(json.loads(line)["patch_hash"] == patch_hash for line in stream):
                return result

    # Include both paths of renames; localization decides which sources it needs.
    names = git("diff", "--cached", "--name-only", "--no-renames", "-z", base_commit)
    before, after, source_errors = {}, {}, {}
    for name in names.split(b"\0")[:-1]:
        path = os.fsdecode(name)
        for sources, ref in ((before, f"{base_commit}:{path}"), (after, f":{path}")):
            try:
                sources[path] = decode_source(git("show", ref))
            except Exception as error:
                # Missing/unreadable sources matter only if localization needs them.
                source_errors[ref] = str(error)
    record = {**result, "patch": patch.decode("utf-8", errors="replace"),
              "before_sources": before, "after_sources": after}
    if source_errors:
        record["source_errors"] = source_errors
    with destination.open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/hooks/captures"))
    args = parser.parse_args()
    try:
        result = capture(args.repo, args.base_commit, args.output_dir)
    except Exception as error:
        print(json.dumps({"capture_error": str(error)}), file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(result))


if __name__ == "__main__":
    main()
