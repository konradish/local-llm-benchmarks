# Local LLM benchmarks: an 8 GB box and a 16 GB desktop

How fast do big open models actually run on consumer GPUs, and which settings matter? Measurements from two machines I run at home:

- **hydra**, a *dedicated* headless server: Threadripper 1950X, 62 GiB DDR4-3200 (quad channel), **RTX 2080 8 GB**, Debian 13.
- **koni9**, my *shared* desktop: i9-14900K, 62 GiB DDR5-4800, **RTX 4080 16 GB**, Debian 13, KDE Plasma on the same GPU.

Full specs and the software versions behind each number: [hardware.md](hardware.md). How everything was measured, including the noise floor and what is *not* covered: [methodology.md](methodology.md).

## Headline numbers

Tokens per second. "Write" is generation (decode); "read" is prompt processing.

| Model | Machine | Setting | Write | Read | Notes |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B Q4_K_M (35B MoE) | RTX 2080 8 GB | coding profile, **64K context**, stock llama.cpp | **48.2** | **1,040** | 38.1 write with 65,000 tokens already in the context |
| Qwen3.6-35B-A3B Q4_K_M | RTX 2080 8 GB | chat profile, small window | **52.5** | 470 | 30 expert layers on the CPU, the fewest that fit |
| Qwen3.6-35B-A3B **Q8_0** | RTX 2080 8 GB | coding profile, 64K context | 33.8 | 712 | the highest quant that fits; BF16 does not |
| Qwen3.8-Flash-Next (**125B** MoE, IQ3_XXS) | RTX 2080 8 GB | 64K context, Strata engine | **39.2** | **460** | 76 GB of weights, experts in RAM |
| Qwen3.6-35B-A3B Q4_K_M (community fine-tune) | RTX 4080 16 GB | 22 expert layers on the CPU | 73 | ~950 | collected earlier, not controlled |
| Bonsai 2 27B (ternary, 1.75 bit) | RTX 4080 16 GB | 64K context, q4_0 KV | 73.9 | 408 | collected earlier, not controlled |

The 4080 rows are older single measurements from my lab notes; see [results/rtx4080-desktop.md](results/rtx4080-desktop.md) for their dates, caveats and what they do and do not show.

## What I learned

1. **Prompt reading is a micro-batch setting.** `-ub 2048` instead of the default 512 took the 35B model from ~450 to ~1,090 tokens/s of prompt reading with no loss in write speed. It costs about 1.5 GB of VRAM, which is the same as three expert layers. A 512-token test prompt cannot see it.
2. **At long context, the KV cache type decides write speed.** With a 64K context in use, an f16 KV cache wrote at 38.1 tokens/s against about 31 for q8_0 (+20%), at the cost of 0.6 GB. Mixing the two (q8_0 K, f16 V) was the slowest of all (26.6).
3. **The usual tuning knobs did almost nothing.** CPU governor and thread count (12 to 16): no difference. `--poll 100`: slightly worse. Thread pinning gave about 2% at best. Moving an expert layer to the GPU is worth about 1% each.
4. **Each step up in quant costs 6 to 14% of write speed.** Q4_K_M to Q8_0 is 30% slower writing for a third-party KL-divergence gain that is mostly already there at Q6_K.
5. **A 125B-parameter model is usable on an 8 GB card** if the engine keeps the experts in RAM and caches the hot ones in VRAM: 39 write and 460 read tokens/s at 64K context, after tuning an engine whose *defaults* regressed prompt reading by 26% on this hardware.
6. **PCIe decides whether a GPU-side expert cache pays off.** On the 4080's PCIe 4.0 link it nearly tripled prompt reading and lifted write speed; on the 2080's PCIe 3.0 it only helped reading.

## Results

- [Qwen3.6-35B-A3B on an 8 GB GPU: a hill-climb](results/hydra-rtx2080-35b-a3b.md): coding and chat profiles, micro-batch, KV type, layers, threads, governor, the quant curve
- [A 125B-parameter MoE on an 8 GB GPU](results/hydra-rtx2080-flash-next-125b.md): Qwen3.8-Flash-Next on the Strata engine, before and after tuning
- [RTX 4080 desktop: collected results](results/rtx4080-desktop.md)

## Reproduce it

The harness is two small shell scripts, plus the exact queue files I ran:

```bash
# one configuration, three repetitions, appended to ~/hc/ledger.jsonl
scripts/lscore.sh LABEL ~/llama/main/bin ~/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
    -ngl 99 --n-cpu-moe 36 -fa on -ctk f16 -ctv f16 -t 14 -lm none -b 4096 -ub 2048

# a queue of them, one after another (one GPU, one job)
scripts/lbatch.sh experiments/q-phase2.txt
```

Labels starting `C64-` also run the 65K-depth pass. `scripts/build_pages.py` regenerates the result tables from [`data/hydra-ledger.jsonl`](data/hydra-ledger.jsonl), so no table is hand-typed. The scripts expect upstream llama.cpp (`llama-bench`) at the commit listed in [hardware.md](hardware.md).

## Honest limits

- **One example of each machine.** Another 2080 with slower RAM would land elsewhere.
- **Speed only.** The only quality figures are third-party and labeled as such.
- **llama-bench is not a server.** No prompt cache, no multi-user queueing, no speculative decoding. Speculative decoding and the `ik_llama.cpp` fork are researched and queued but **not yet run**.
- **The 4080 numbers are looser** (older, single runs, a shared GPU) and are kept apart for that reason.

## Credits and licenses

Measurements built on [llama.cpp](https://github.com/ggml-org/llama.cpp), the [Strata](https://github.com/Niko1221/Strata) engine, quantized weights from [Unsloth](https://huggingface.co/unsloth), models from Qwen (Alibaba) and [PrismML](https://huggingface.co/prism-ml). No model weights are redistributed here; their licenses are their authors'. The scripts and write-ups in this repository are MIT licensed (see [LICENSE](LICENSE)).
