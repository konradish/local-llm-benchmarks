#!/usr/bin/env python3
"""Render the MTP A/B (data/mtp-ab.jsonl) as the two markdown tables used on the 35B results page. The control is the same MTP-capable file with speculation switched off.
Write speed is the median of 3 requests at production sampling (temperature 1.0, top-p 0.95, top-k 20); the 30K-prompt read is the FIRST (uncached) request, because the repeat hits the server's prompt cache."""
import json, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rows = [json.loads(l) for l in (ROOT / "data" / "mtp-ab.jsonl").read_text().splitlines() if l.strip()]
LOG = (ROOT / "data" / "mtp-ab.log").read_text()
NAMES = {"M0-off": "MTP off (control), 36 CPU layers", "M1-mtp-n36": "MTP on, 36 CPU layers", "M2-mtp-n38": "MTP on, 38 CPU layers",
         "M3-mtp-n38-nographs": "MTP on, 38 CPU layers, CUDA graphs off"}


def med(cfg, prompt, mode, key):
    v = [r[key] for r in rows if r.get("ok") and r["config"] == cfg and r["prompt"] == prompt and r["mode"] == mode and r.get(key) is not None]
    return statistics.median(v) if v else None


def accept(cfg, prompt, mode):
    a = [r for r in rows if r.get("ok") and r["config"] == cfg and r["prompt"] == prompt and r["mode"] == mode and r.get("draft_n")]
    return 100 * sum(r["draft_accepted"] for r in a) / sum(r["draft_n"] for r in a) if a else None


def vram_after(cfg):
    block = LOG.split(f"=== {cfg} ")[1].split("===")[0] if f"=== {cfg} " in LOG else ""
    import re
    m = re.findall(r"VRAM after the matrix: (\d+) MiB", block)
    return m[-1] if m else ""


print("| Config | Code | Thinking | Prose |")
print("|---|---|---|---|")
for c, name in NAMES.items():
    cells = []
    for p in ("code", "think", "prose"):
        w, base, a = med(c, p, "prod", "write_tps"), med("M0-off", p, "prod", "write_tps"), accept(c, p, "prod")
        if w is None:
            cells.append("")
        elif c == "M0-off":
            cells.append(f"{w:.1f}")
        else:
            cells.append(f"**{w:.1f}** ({(w / base - 1) * 100:+.0f}%, {a:.0f}% drafts accepted)")
    print(f"| {name} | " + " | ".join(cells) + " |")
print()
print("| Config | Read, 30K prompt | Read, 59K prompt | Peak VRAM after the run (MiB) |")
print("|---|---|---|---|")
for c, name in NAMES.items():
    r30, r59 = med(c, "long", "greedy", "read_tps"), med(c, "fit", "greedy", "read_tps")
    base = med("M0-off", "long", "greedy", "read_tps")
    d = "" if c == "M0-off" else f" ({(r30 / base - 1) * 100:+.0f}%)"
    print(f"| {name} | {r30:.0f}{d} | {r59:.0f} | {vram_after(c)} |")
