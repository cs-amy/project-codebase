import argparse
import os
import re
import sys
import uuid
from pathlib import Path
from typing import List


def natural_sort_key(s: str):
    """Split a string into list of str and int for natural sorting."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


def rename_sequence(dir_path: Path, start: int) -> int:
    """
    Rename all files in dir_path sequentially starting at `start`.
    Returns the number of files renamed.
    """
    if not dir_path.exists():
        raise FileNotFoundError(f'Directory "{dir_path}" does not exist.')
    if not dir_path.is_dir():
        raise NotADirectoryError(f'"{dir_path}" is not a directory.')

    # Gather only files, sorted naturally
    files: List[Path] = sorted(
        (p for p in dir_path.iterdir() if p.is_file()),
        key=lambda p: natural_sort_key(p.name)
    )

    if not files:
        return 0

    # Two‑pass rename: use a random prefix to avoid collisions
    tmp_prefix = f".__tmp_{uuid.uuid4().hex}__"
    # First pass → tmp names
    for idx, p in enumerate(files):
        seq = start + idx
        new_name = f"{tmp_prefix}{seq}{p.suffix}"
        p.rename(dir_path / new_name)

    # Second pass → final names
    count = 0
    for p in dir_path.iterdir():
        if not p.name.startswith(tmp_prefix):
            continue
        # strip prefix, then split off suffix
        base = p.name[len(tmp_prefix):]
        seq_str, ext = os.path.splitext(base)
        final_name = f"{seq_str}{ext}"
        p.rename(dir_path / final_name)
        count += 1

    return count


def parse_args():
    p = argparse.ArgumentParser(
        description="Sequentially rename all files in a directory."
    )
    p.add_argument(
        "directory",
        type=Path,
        help="Path to the target directory"
    )
    p.add_argument(
        "start",
        type=int,
        help="Non-negative integer to start numbering from"
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.start < 0:
        print(f"ERROR: start must be a non-negative integer; got {args.start}.",
              file=sys.stderr)
        sys.exit(1)

    try:
        renamed = rename_sequence(args.directory, args.start)
    except (FileNotFoundError, NotADirectoryError, PermissionError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)

    if renamed == 0:
        print("No files to rename.")
    else:
        print(f"Renamed {renamed} file(s) starting from {args.start}.")


if __name__ == "__main__":
    main()
