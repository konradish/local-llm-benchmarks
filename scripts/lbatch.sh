#!/bin/bash
# Run ON HYDRA, detached. Run a queue of hill-climb experiments ONE AFTER ANOTHER through lscore.sh (one GPU, never in parallel).
# usage: lbatch.sh QUEUEFILE      each non-comment line:  LABEL BINDIR MODEL [llama-bench flags ...]
# Writes ~/hc/batch-<basename>.done when finished and ~/hc/batch-<basename>.out with every lscore line. Safe to re-run: labels are appended to the ledger, never overwritten.
Q=$1
OUT="$HOME/hc/batch-$(basename "$Q").out"
DONE="$HOME/hc/batch-$(basename "$Q").done"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HOME/hc"
rm -f "$DONE"
: > "$OUT"
while read -r label bindir model rest; do
  [ -z "${label:-}" ] && continue
  case "$label" in \#*) continue;; esac
  # shellcheck disable=SC2086
  "$HERE/lscore.sh" "$label" "${bindir/#\~/$HOME}" "${model/#\~/$HOME}" $rest < /dev/null 2>&1 | tee -a "$OUT"
done < "$Q"
echo "BATCH DONE $(date +%T)" | tee -a "$OUT"
echo ok > "$DONE"
