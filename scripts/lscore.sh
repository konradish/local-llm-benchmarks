#!/bin/bash
# Run on the inference box. Hill-climb scoreboard v3 for llama.cpp-family builds (2026-10-03): ONE configuration per call, 3 repeats, appended to an append-only ledger.
# usage: lscore.sh LABEL BINDIR MODEL.gguf [llama-bench flags ...]        e.g. lscore.sh B1 ~/llama/main/bin ~/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -ngl 99 --n-cpu-moe 32 -fa on -ctk q8_0 -ctv q8_0 -t 14 -lm none
# v3 (a >= 64K context is required for the coding profile; the chat profile can use a smaller window): for labels starting C64- a THIRD pass allocates a 64K context (-d 65000 -n 64, one rep, ~2-3 min) and records decode at 65K depth + peak VRAM with the full 64K KV.
# Three llama-bench passes (so each gets its own context): PP = pp512 and pp4096 at depth 0; TG = tg128 at depth 0 and at depth 8192 (decode while 8K tokens of context exist).
# v2 over v1: pp4096 (pp512 cannot see a batch-size effect), peak VRAM sampled at 1 Hz DURING the run (v1 read it after exit = 17 MiB, useless), and a clear 'CTX FAIL' when the
# context cannot be created (a bigger -ub needs a bigger compute buffer; add --n-cpu-moe layers).
# Refuses to run if a model server or another bench is alive (one GPU, one job). Ledger: ~/hc/ledger.jsonl ; raw output: ~/hc/raw/LABEL.{jsonl,pp.err,tg.err,vram}
set -u
LABEL=$1; BINDIR=$2; MODEL=$3; shift 3
mkdir -p "$HOME/hc/raw"
for p in llama-server strata llama-bench; do
  if pgrep -x "$p" >/dev/null; then echo "REFUSED: $p is running (one GPU, one job)"; exit 4; fi
done
load=$(cut -d' ' -f1 /proc/loadavg)
export LD_LIBRARY_PATH="$BINDIR:${LD_LIBRARY_PATH:-}"
RAW="$HOME/hc/raw/$LABEL.jsonl"; VR="$HOME/hc/raw/$LABEL.vram"
: > "$RAW"; : > "$VR"
echo "== $LABEL | $(basename "$MODEL") | $* | load $load | governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) | $(date +%T)"
( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$VR" 2>/dev/null; sleep 1; done ) &
SPID=$!
"$BINDIR/llama-bench" -m "$MODEL" "$@" -p 512,4096 -n 0 -d 0 -r 3 -o jsonl >> "$RAW" 2> "$HOME/hc/raw/$LABEL.pp.err"; rc1=$?
"$BINDIR/llama-bench" -m "$MODEL" "$@" -p 0 -n 128 -d 0,8192 -r 3 -o jsonl >> "$RAW" 2> "$HOME/hc/raw/$LABEL.tg.err"; rc2=$?
# The 64K pass only runs for CODING-profile labels (C64-*): >= 64K context is required for coding/agent use; the chat profile may use a smaller window, so chat runs skip it.
rc3=0
case "$LABEL" in C64-*) "$BINDIR/llama-bench" -m "$MODEL" "$@" -p 0 -n 64 -d 65000 -r 1 -o jsonl >> "$RAW" 2> "$HOME/hc/raw/$LABEL.deep.err"; rc3=$? ;; esac
kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
peak=$(sort -n "$VR" | tail -1)
python3 - "$LABEL" "$MODEL" "$RAW" "$rc1" "$rc2" "$rc3" "$load" "$peak" "$*" <<'PY'
import json, os, sys, time
label, model, raw, rc1, rc2, rc3, load, peak, flags = sys.argv[1:10]
rows = []
for line in open(raw):
    line = line.strip()
    if line.startswith("{"):
        try: rows.append(json.loads(line))
        except Exception: pass
def pick(npr, ngen, depth):
    for r in rows:
        if r.get("n_prompt") == npr and r.get("n_gen") == ngen and r.get("n_depth", 0) == depth:
            return round(r["avg_ts"], 2), round(r.get("stddev_ts", 0), 2)
    return None
res = {"pp512": pick(512, 0, 0), "pp4096": pick(4096, 0, 0), "tg128": pick(0, 128, 0), "tg128_d8192": pick(0, 128, 8192), "tg64_d65000": pick(0, 64, 65000)}
gov = open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read().strip()
line = {"v": 3, "ts": time.strftime("%F %T"), "label": label, "model": model.split("/")[-1], "flags": flags, "rc": [int(rc1), int(rc2), int(rc3)], "load1": float(load),
        "governor": gov, "peak_vram_mib": int(peak) if peak.isdigit() else None, "result": res}
open(os.path.expanduser("~/hc/ledger.jsonl"), "a").write(json.dumps(line) + "\n")
print(f"   pp512 {res['pp512']}  pp4096 {res['pp4096']}  tg128 {res['tg128']}  tg128@8K {res['tg128_d8192']}  tg64@65K {res['tg64_d65000']}  (mean, sd tok/s)   peak VRAM {peak} MiB   rc {rc1}/{rc2}/{rc3}")
for tag, rc in (("pp", rc1), ("tg", rc2), ("deep", rc3)):
    if rc != "0":
        err = open(os.path.expanduser(f"~/hc/raw/{label}.{tag}.err"), errors="replace").read()
        why = "CTX FAIL (probably out of VRAM at this context/ubatch: add --n-cpu-moe layers; the deep pass is the 64K check)" if "failed to create context" in err else "see ~/hc/raw/" + label + "." + tag + ".err"
        print(f"   {tag.upper()} pass failed: {why}")
PY
