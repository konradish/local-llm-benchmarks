#!/usr/bin/env python3
"""Build results pages from templates (results/*.md.tmpl) by replacing table markers with tables rendered from data/hydra-ledger.jsonl.

Markers, one per line:
  {{table: LABEL LABEL ...}}        rows in the order given
  {{table-prefix: C64-}}            every label with that prefix (last run of each label)
Run:  python3 scripts/build_pages.py        (writes results/<name>.md next to each <name>.md.tmpl)
"""
import re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAKE = ROOT / "scripts" / "make_table.py"


def render(args):
    out = subprocess.run([sys.executable, str(MAKE), *args], capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"make_table failed for {args}: {out.stderr.strip() or out.stdout.strip()}")
    return out.stdout.rstrip("\n")


def build(tmpl):
    text = tmpl.read_text()

    def sub(m):
        kind, rest = m.group(1), m.group(2).split()
        return render(["--prefix", *rest]) if kind == "table-prefix" else render(rest)

    done = re.sub(r"^\{\{(table|table-prefix):\s*([^}]*)\}\}\s*$", sub, text, flags=re.M)
    dest = tmpl.with_suffix("")          # foo.md.tmpl -> foo.md
    dest.write_text(done)
    print("built", dest.relative_to(ROOT))


if __name__ == "__main__":
    for t in sorted((ROOT / "results").glob("*.md.tmpl")):
        build(t)
