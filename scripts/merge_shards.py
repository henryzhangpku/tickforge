#!/usr/bin/env python3
"""Merge CSV output from sweep shards into one file ordered by task index.

    merge_shards.py shard-0.log shard-1.log ... > sweep.csv

Inputs may contain other log lines (scheduler noise, stderr); only the CSV
header and rows that start with a task index are kept. Fails if a task index
appears twice or the indices are not 0..N-1, so a lost or duplicated shard
cannot go unnoticed.
"""

import re
import sys

ROW = re.compile(r"^(\d+),")


def main(paths):
    header = None
    rows = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\r\n")
                if line.startswith("index,"):
                    if header not in (None, line):
                        sys.exit(f"{path}: header differs from earlier shards")
                    header = line
                    continue
                m = ROW.match(line)
                if not m:
                    continue
                idx = int(m.group(1))
                if idx in rows:
                    sys.exit(f"task {idx} appears twice (duplicated shard?)")
                rows[idx] = line
    if header is None:
        sys.exit("no CSV header found")
    missing = sorted(set(range(len(rows))) - set(rows))
    if missing or (rows and max(rows) != len(rows) - 1):
        sys.exit(f"missing task indices, e.g. {missing[:5]} (lost shard?)")
    print(header)
    for idx in sorted(rows):
        print(rows[idx])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
