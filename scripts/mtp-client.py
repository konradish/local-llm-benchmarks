#!/usr/bin/env python3
"""Run ON HYDRA. Measure a llama-server on 127.0.0.1:18080 with a fixed prompt matrix and append one JSON line per request to ~/hc/mtp-ab.jsonl.
usage: mtp-client.py CONFIG_NAME
Prompts: code, prose, think (thinking on), long (30K-token prompt), fit (59K-token prompt, checks the full 64K context fits). Modes: greedy (temperature 0) and prod
(temperature 1.0, top_p 0.95, top_k 20: the model card's thinking-mode settings, what real use runs). Short prompts x3 repetitions, long/fit x1.
Reads the server's own `timings` (prompt_per_second, predicted_per_second, draft_n, draft_n_accepted when speculation is on)."""
import json, os, sys, time, urllib.request

CFG = sys.argv[1]
URL = "http://127.0.0.1:18080/v1/chat/completions"
OUT = os.path.expanduser("~/hc/mtp-ab.jsonl")
DOC = open(os.path.expanduser("~/hc/doc.md (any long text; I used the public Qwen3.6-35B-A3B model card)")).read()

PROMPTS = {
    "code": ("Write a Python function merge_intervals(intervals) that merges overlapping [start, end] lists (touching ones merge), with a docstring and three doctest examples. Reply with one code block.", 400, False, 3),
    "prose": ("Explain in about 300 words how a hash map handles collisions and resizing, and why load factor matters.", 400, False, 3),
    "think": ("A farmer has 17 sheep and all but 9 run away. Then he buys twice as many sheep as he has left. How many does he have now? Think it through.", 400, True, 3),
    "long": ((DOC * 4)[:85000] + "\n\nIn two sentences: what is this document about?", 128, False, 1),
    "fit": ((DOC * 4)[:168000] + "\n\nIn one sentence: what is this document about?", 32, False, 1),
}
MODES = {"greedy": {"temperature": 0.0}, "prod": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0}}


def call(prompt, max_tokens, think, sampling):
    body = {"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "stream": False,
            "chat_template_kwargs": {"enable_thinking": think}, **sampling}
    t = time.perf_counter()
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900))
    return r, time.perf_counter() - t


with open(OUT, "a") as f:
    for pname, (prompt, mt, think, reps) in PROMPTS.items():
        modes = ["greedy"] if pname == "fit" else list(MODES)
        for mode in modes:
            for rep in range(reps):
                row = {"config": CFG, "prompt": pname, "mode": mode, "rep": rep, "ts": time.strftime("%F %T")}
                try:
                    r, wall = call(prompt, mt, think, MODES[mode])
                    tm = r.get("timings") or {}
                    row.update(ok=True, wall_s=round(wall, 2), prompt_tokens=(r.get("usage") or {}).get("prompt_tokens"), completion_tokens=(r.get("usage") or {}).get("completion_tokens"),
                               read_tps=round(tm.get("prompt_per_second", 0), 1), write_tps=round(tm.get("predicted_per_second", 0), 2),
                               draft_n=tm.get("draft_n"), draft_accepted=tm.get("draft_n_accepted"), finish=r["choices"][0]["finish_reason"])
                except Exception as e:
                    row.update(ok=False, error=repr(e)[:200])
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"{CFG} {pname:5} {mode:6} rep{rep}: " + (f"write {row['write_tps']} read {row['read_tps']} draft {row.get('draft_accepted')}/{row.get('draft_n')}" if row["ok"] else row["error"]), flush=True)
                if not row["ok"]:
                    sys.exit(3)       # the server died or refused: stop this config
