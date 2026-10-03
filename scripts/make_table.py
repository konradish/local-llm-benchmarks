#!/usr/bin/env python3
"""Render rows of data/hydra-ledger.jsonl as a markdown table, so result pages are generated from the recorded runs instead of retyped.

usage: make_table.py LABEL [LABEL ...]            rows in the order given (a missing label is an error)
       make_table.py --prefix C64- [--prefix H-]  every label starting with a prefix, in ledger order

Columns: layers on CPU (--n-cpu-moe), micro-batch (-ub), KV types, prompt reading at 4,096 tokens, decode at depth 0 / 8K / 65K, peak VRAM, and a status.
A run that could not allocate its context (out of VRAM) is shown as 'no fit' instead of silently dropped.
"""
import json, re, sys
from pathlib import Path

LEDGER = Path(__file__).resolve().parent.parent / "data" / "hydra-ledger.jsonl"


def flag(flags, name, default=""):
    m = re.search(rf"(?:^|\s){re.escape(name)}\s+(\S+)", flags)
    return m.group(1) if m else default


def val(x):
    return "" if x is None else f"{x[0]:.1f}"


def row(e):
    f = e["flags"]
    r = e["result"]
    rc = e.get("rc") or []
    if isinstance(rc, int):
        rc = [rc]
    ctk, ctv = flag(f, "-ctk", "f16"), flag(f, "-ctv", "f16")
    kv = ctk if ctk == ctv else f"K {ctk} / V {ctv}"
    # rc = [prompt pass, decode pass (0 and 8K deep), 64K pass]; a non-zero code means that pass could not allocate its context (out of VRAM)
    names = ["4K prompt", "8K context", "64K context"]
    failed = [names[i] for i, c in enumerate(rc[:3]) if c != 0]
    if not any(r.values()):
        status = "no fit"
    elif failed:
        status = "does not fit: " + ", ".join(failed)
    else:
        status = "ok"
    peak = e.get("peak_vram_mib")
    return (f"| `{e['label']}` | {e['model'].replace('Qwen3.6-35B-A3B-', '').replace('.gguf', '')} | {flag(f, '--n-cpu-moe')} | {flag(f, '-ub', '512')} | {kv} | "
            f"{val(r.get('pp4096'))} | {val(r.get('tg128'))} | {val(r.get('tg128_d8192'))} | {val(r.get('tg64_d65000'))} | {peak if peak else ''} | {status} |")


def main(argv):
    entries = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    by = {}
    for e in entries:
        by[e["label"]] = e                       # last run of a label wins
    prefixes = [argv[i + 1] for i, a in enumerate(argv) if a == "--prefix"]
    labels = [a for i, a in enumerate(argv) if not a.startswith("--") and (i == 0 or argv[i - 1] != "--prefix")]
    picked = []
    for p in prefixes:
        picked += [e for e in entries if e["label"].startswith(p) and by[e["label"]] is e]
    for l in labels:
        if l not in by:
            sys.exit(f"label not in ledger: {l}")
        picked.append(by[l])
    print("| Run | Quant | CPU layers | `-ub` | KV | Read (4,096) | Write | Write @ 8K | Write @ 65K | Peak VRAM (MiB) | Status |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for e in picked:
        print(row(e))


if __name__ == "__main__":
    main(sys.argv[1:])
