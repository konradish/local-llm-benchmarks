# Methodology

## What is measured

All llama.cpp numbers come from `llama-bench` (upstream commit `fc07d78`), driven by [`scripts/lscore.sh`](scripts/lscore.sh), one configuration per call, three repetitions, appended to an append-only ledger ([`data/hydra-ledger.jsonl`](data/hydra-ledger.jsonl)). Results are mean tokens/second.

| Column | What it is |
|---|---|
| **pp512 / pp4096** | Prompt processing ("reading") speed on a 512- and a 4,096-token prompt. pp512 cannot show a batch-size effect (the prompt is smaller than the micro-batch), so pp4096 is the one to read. |
| **tg128** | Token generation ("writing", decode) speed, generating 128 tokens from an empty context. |
| **tg128 @ 8K** | The same, with 8,192 tokens of context already in the cache. Decode slows as context grows. |
| **tg64 @ 65K** | Coding profile only: decode with 65,000 tokens already in the cache, in a real 64K context allocation. One repetition. |
| **Peak VRAM** | Maximum GPU memory used during the run, sampled once a second. |

## Two profiles

- **Coding / agent profile:** at least a 64K context (coding agents carry long sessions). Prompt reading matters as much as decode, because an agent loop re-reads a lot. Runs are labeled `C64-*` and include the 65K-depth pass.
- **Chat profile:** a small window is fine, so VRAM that would have held context goes to keeping more expert layers on the GPU. Runs are labeled `H-*`.

## How the climb was run

- **One change per run**, compared against a named baseline. A run that fails to fit (usually a context that cannot be allocated in 8 GB) is recorded as a failure, not skipped.
- **Noise floor measured, not assumed.** Two identical baseline runs differed by 0.25 tok/s on decode (about 0.5%) and by under 1 tok/s on prompt reading. The single-repetition 65K-depth number repeats to about +/- 1 tok/s (about 3%). Differences smaller than that are called noise in the write-ups.
- **One GPU, one job.** The runner refuses to start if another model server or benchmark is alive.
- **Downloads verified.** Every model file was checked against the SHA-256 published by Hugging Face before use.
- **Research agents proposed, measurements decided.** Three web-research agents ranked ideas from published reports; none of their numbers are used as results. Only runs in the ledger count.

## What this does not cover

- **Quality.** These are speed benchmarks. Quant quality figures cited anywhere come from third parties and are labeled as such.
- **Server behavior.** `llama-bench` has no prompt cache, no multi-user queueing, no speculative decoding. A server adds its own effects; where a result depends on one (for example multi-token-prediction speculation), it is tested through a server and says so.
- **Variance across machines.** One example of each machine. Another 2080 box with different RAM would land elsewhere.
- **The 4080 rows are older and looser.** See [`results/rtx4080-desktop.md`](results/rtx4080-desktop.md): collected over several months from lab notes, single runs, different methods, desktop shared. They are labeled as collected, not as controlled measurements.
