"""Command-line tools for saved memories. Run from the project folder:

    python -m companion.memory_admin list
    python -m companion.memory_admin dedupe            (dry run: shows duplicates only)
    python -m companion.memory_admin dedupe --apply    (backs up, then removes newer duplicates)
    python -m companion.memory_admin forget 3-5        (dry run: shows what would be removed)
    python -m companion.memory_admin forget 3-5 --apply

Every change needs --apply. A backup copy of the database is made first, in the backups folder
next to the memory database. The companion does not need to be stopped for list or dry runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from companion.config import get_config
from companion.memory.intent import MAX_FORGET_IDS, parse_id_spec
from companion.memory.store import MemoryStore, memory_line


def _open(db_path: Path) -> MemoryStore:
    return MemoryStore(db_path)


def _list(store: MemoryStore) -> int:
    memories = store.all()
    if not memories:
        print("No saved memories.")
        return 0
    print(f"{len(memories)} saved memories:")
    for memory in memories:
        print(memory_line(memory))
    return 0


def _dedupe(db_path: Path, apply: bool) -> int:
    store = _open(db_path)
    try:
        groups = store.duplicate_groups()
        if not groups:
            print("No duplicate memories found.")
            return 0
        to_remove: list[int] = []
        for group in groups:
            keep, extra = group[0], group[1:]
            print(f"Keep {memory_line(keep)}")
            for memory in extra:
                print(f"  remove {memory_line(memory)}")
            to_remove.extend(memory.id for memory in extra)
        if not apply:
            print(f"Dry run: {len(to_remove)} duplicate(s) would be removed. Add --apply to do it.")
            return 0
        backup = store.backup(db_path.parent / "backups")
        print(f"Backup written to {backup}")
        removed = store.delete(to_remove)
        print(f"Removed {removed} duplicate memor{'y' if removed == 1 else 'ies'}.")
    finally:
        store.close()
    # Reopening lets the store create the unique index that was blocked by the duplicates.
    reopened = _open(db_path)
    try:
        if reopened.has_norm_index():
            print("Unique index is in place.")
        else:
            print("Warning: the unique index could not be created. Check the log.")
            return 1
    finally:
        reopened.close()
    return 0


def _forget(db_path: Path, spec: str, apply: bool) -> int:
    ids = parse_id_spec(spec)
    if ids is None:
        print(f'Error: "{spec}" is not a list of memory numbers. Example: 3 or 2-4 or 2,5.')
        return 2
    if len(ids) > MAX_FORGET_IDS:
        print(f"Error: that names {len(ids)} memories. Remove at most {MAX_FORGET_IDS} at once.")
        return 2
    store = _open(db_path)
    try:
        found = store.get_many(ids)
        found_ids = {memory.id for memory in found}
        for number in ids:
            if number not in found_ids:
                print(f"No memory {number}.")
        if not found:
            return 0
        for memory in found:
            print(f"remove {memory_line(memory)}")
        if not apply:
            print(f"Dry run: {len(found)} memory(ies) would be removed. Add --apply to do it.")
            return 0
        backup = store.backup(db_path.parent / "backups")
        print(f"Backup written to {backup}")
        removed = store.delete([memory.id for memory in found])
        print(f"Removed {removed} memor{'y' if removed == 1 else 'ies'}.")
    finally:
        store.close()
    return 0


def main(argv: list[str] | None = None, *, db_path: Path | None = None) -> int:
    parser = argparse.ArgumentParser(prog="companion.memory_admin", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="print every saved memory with its number")
    dedupe = sub.add_parser("dedupe", help="remove memories whose text is the same")
    dedupe.add_argument("--apply", action="store_true", help="make the change")
    forget = sub.add_parser("forget", help="remove memories by number, such as 3 or 2-4")
    forget.add_argument("spec", help="memory numbers, e.g. 3, 2-4, or 2,5,7")
    forget.add_argument("--apply", action="store_true", help="make the change")
    args = parser.parse_args(argv)

    path = db_path if db_path is not None else get_config().memory.db_path
    if args.command == "list":
        store = _open(path)
        try:
            return _list(store)
        finally:
            store.close()
    if args.command == "dedupe":
        return _dedupe(path, args.apply)
    return _forget(path, args.spec, args.apply)


if __name__ == "__main__":
    sys.exit(main())
