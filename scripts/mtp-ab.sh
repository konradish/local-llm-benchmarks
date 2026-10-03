#!/bin/bash
# Run on the inference box, detached. MTP A/B through a real llama-server (llama-bench cannot run speculation), 2026-10-03.
# Same MTP-capable GGUF in every config, so the ONLY difference between M0 and M1-M3 is the speculation flags (and the layer split the MTP head forces).
#   M0  speculation OFF                         n-cpu-moe 36   (control; the coding-profile split)
#   M1  --spec-type draft-mtp --spec-draft-n-max 2   n-cpu-moe 36
#   M2  same as M1                              n-cpu-moe 38   (the MTP head costs ~1.2 GB VRAM: give it room)
#   M3  M2 + GGML_CUDA_DISABLE_GRAPHS=1         (a report says MTP + CPU offload re-captures CUDA graphs every step)
# One GPU: the script unloads llama-swap's model first and refuses to start if anything else holds the card. It runs its OWN server on 127.0.0.1:18080 and kills it by exact PID.
# Results: ~/hc/mtp-ab.jsonl ; log: ~/hc/mtp-ab.log ; marker file ~/hc/mtp-ab.done when finished.
set -u
MODEL=$HOME/models/Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf
BIN=$HOME/llama/main/bin
LOG=$HOME/hc/mtp-ab.log
CLIENT="$(cd "$(dirname "$0")" && pwd)/mtp-client.py"
rm -f "$HOME/hc/mtp-ab.done"; : > "$LOG"
say() { echo "$(date +%T) $*" | tee -a "$LOG"; }

curl -s -m 10 http://127.0.0.1:8080/unload > /dev/null 2>&1
for i in $(seq 1 30); do pgrep -x llama-server > /dev/null || break; sleep 2; done
if pgrep -x llama-server > /dev/null || pgrep -x strata > /dev/null || pgrep -x llama-bench > /dev/null; then say "REFUSED: another model process holds the GPU"; echo refused > "$HOME/hc/mtp-ab.done"; exit 4; fi

run_cfg() {   # name ncmoe extra_args... ; env via ENV_EXTRA
  local name=$1 ncmoe=$2; shift 2
  say "=== $name (n-cpu-moe $ncmoe $* ${ENV_EXTRA:-})"
  env LD_LIBRARY_PATH=$BIN ${ENV_EXTRA:-} "$BIN/llama-server" -m "$MODEL" --host 127.0.0.1 --port 18080 -ngl 99 --n-cpu-moe "$ncmoe" -fit off -fa on \
      -c 65536 -b 4096 -ub 2048 -ctk f16 -ctv f16 -t 14 -np 1 --load-mode none --jinja --reasoning-format deepseek "$@" > "$HOME/hc/raw/mtp-$name.server.log" 2>&1 &
  local pid=$!
  local up=0
  for i in $(seq 1 150); do
    kill -0 $pid 2>/dev/null || break
    curl -s -m 2 http://127.0.0.1:18080/health 2>/dev/null | grep -q '"ok"' && { up=1; break; }
    sleep 2
  done
  if [ $up = 1 ]; then
    say "server up (pid $pid); VRAM $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
    python3 "$CLIENT" "$name" 2>&1 | tee -a "$LOG"
    say "VRAM after the matrix: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  else
    say "SERVER FAILED TO START for $name: $(grep -i -E 'error|out of memory|failed|abort' "$HOME/hc/raw/mtp-$name.server.log" | tail -2 | cut -c1-200 | tr '\n' ' ')"
    echo "{\"config\":\"$name\",\"server_failed\":true}" >> "$HOME/hc/mtp-ab.jsonl"
  fi
  kill $pid 2>/dev/null; for i in $(seq 1 30); do kill -0 $pid 2>/dev/null || break; sleep 1; done; kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
  sleep 3
}

ENV_EXTRA= run_cfg M0-off 36
ENV_EXTRA= run_cfg M1-mtp-n36 36 --spec-type draft-mtp --spec-draft-n-max 2
ENV_EXTRA= run_cfg M2-mtp-n38 38 --spec-type draft-mtp --spec-draft-n-max 2
ENV_EXTRA=GGML_CUDA_DISABLE_GRAPHS=1 run_cfg M3-mtp-n38-nographs 38 --spec-type draft-mtp --spec-draft-n-max 2
say "ALL CONFIGS DONE"
echo ok > "$HOME/hc/mtp-ab.done"
