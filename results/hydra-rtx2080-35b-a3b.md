# Qwen3.6-35B-A3B on an 8 GB GPU: a hill-climb

A 35B mixture-of-experts model with about 3B parameters active per token, run by stock llama.cpp on the dedicated 2080 box ([hardware](../hardware.md)): the experts stay in system RAM, attention and a few expert layers sit on the 8 GB GPU. The question was how fast it can go, with two different jobs in mind:

- **Coding / agent profile:** at least a **64K context**, because coding agents carry long sessions, and fast prompt reading, because an agent loop re-reads a lot.
- **Chat profile:** a small window is fine, so spare VRAM goes to keeping more layers on the GPU.

All numbers are tokens/second from `llama-bench` (3 repetitions, mean) on the unsloth `UD-Q4_K_M` file unless a row says otherwise. How they were measured, and the noise floor, are in [methodology](../methodology.md). Every row is in [`data/hydra-ledger.jsonl`](../data/hydra-ledger.jsonl).

## Headline

| Profile | Before (September 28 profiles) | After |
|---|---|---|
| **Coding**, 64K context | 41 write, ~1,100 read, **32K** context (GPU expert-cache fork) | **48.2 write, 1,040 read, and 38.1 write with 65,000 tokens already in the context**, stock llama.cpp |
| **Chat**, small window | 45 to 49 write, 32K context | **52.5 write**, 470 read (740 read with `-ub 1024` at 51.2 write) |

**Highest quant that fits:** Q8_0 runs with a full 64K context at **33.8 write / 712 read** (30% slower write than Q4); see [section 5](#5-how-high-a-quant-fits-and-what-it-costs).

Three things did the work: a **bigger micro-batch** (`-ub 2048`) more than doubled prompt reading for free; an **f16 KV cache** instead of q8_0 raised long-context write speed by about 20%; and a handful of **layers moved to the GPU** for the chat profile added about 5%. Everything else I tried was noise or worse.

## 1. Micro-batch: the free 2.4x on prompt reading

Raising the micro-batch from the default 512 to 2,048 changes prompt reading from ~450 to ~1,090 tokens/s and costs essentially nothing on writing. The catch is VRAM: the bigger compute buffer takes about 1.5 GB, which is the same as giving up about three expert layers.

| Run | Quant | CPU layers | `-ub` | KV | Read (4,096) | Write | Write @ 8K | Write @ 65K | Peak VRAM (MiB) | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| `B1-n32-ub512` | UD-Q4_K_M | 32 | 512 | q8_0 | 452.7 | 50.0 | 47.6 |  | 6604 | ok |
| `N33-ub512` | UD-Q4_K_M | 33 | 512 | q8_0 | 444.6 | 49.4 | 47.3 |  | 6138 | ok |
| `N34-ub512` | UD-Q4_K_M | 34 | 512 | q8_0 | 435.1 | 48.5 | 47.0 |  | 5680 | ok |
| `U-n33-ub2048` | UD-Q4_K_M | 33 | 2048 | q8_0 | 1086.9 | 49.4 | 46.2 |  | 7662 | ok |
| `U-n34-ub2048` | UD-Q4_K_M | 34 | 2048 | q8_0 | 1069.2 | 48.9 | 46.5 |  | 7286 | ok |
| `U-n34-ub4096` | UD-Q4_K_M | 34 | 4096 | q8_0 |  | 48.7 |  |  | 5602 | does not fit: 4K prompt, 8K context |
| `U-n36-ub4096` | UD-Q4_K_M | 36 | 4096 | q8_0 |  | 48.0 |  |  | 4642 | does not fit: 4K prompt, 8K context |
- Pair at the same layer count: `N33-ub512` to `U-n33-ub2048`: read **445 to 1,087 (+144%)**, write 49.4 to 49.4, VRAM 6.1 to 7.7 GB.
- Each expert layer moved off the GPU frees about 466 MiB and costs about 1% of write speed (50.0, 49.4, 48.5 for 32, 33, 34 CPU layers).
- `-ub 4096` does not fit at 34 or even 36 CPU layers.
- A 512-token benchmark prompt cannot see any of this, because it is smaller than the micro-batch. Early runs that used one showed no difference; the 4,096-token prompt is what exposed it.

## 2. Coding profile: a real 64K context

Every row below allocates a full 64K context and measures write speed with 65,000 tokens already in it (one repetition; repeats of the same configuration land within about 1 tok/s).

| Run | Quant | CPU layers | `-ub` | KV | Read (4,096) | Write | Write @ 8K | Write @ 65K | Peak VRAM (MiB) | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| `C64-n35-ub2048` | UD-Q4_K_M | 35 | 2048 | q8_0 | 1054.5 | 48.1 | 45.5 | 31.3 | 7306 | ok |
| `C64-n36-ub2048` | UD-Q4_K_M | 36 | 2048 | q8_0 | 1037.0 | 47.6 | 44.9 | 32.1 | 6846 | ok |
| `C64-n34-ub2048` | UD-Q4_K_M | 34 | 2048 | q8_0 | 1069.1 | 48.0 | 46.3 |  | 7286 | does not fit: 64K context |
| `C64-n34-ub1536` | UD-Q4_K_M | 34 | 1536 | q8_0 | 843.6 | 48.1 | 45.4 | 32.0 | 7296 | ok |
| `C64-n35-ub1024` | UD-Q4_K_M | 35 | 1024 | q8_0 | 690.5 | 48.1 | 46.3 | 32.7 | 6292 | ok |
| `C64-n36-ub2048-kvf16` | UD-Q4_K_M | 36 | 2048 | f16 | 1039.8 | 48.2 | 47.1 | 38.1 | 7446 | ok |
| `C64-n36-ub2048-k8v16` | UD-Q4_K_M | 36 | 2048 | K q8_0 / V f16 | 1039.1 | 48.1 | 45.3 | 26.6 | 7146 | ok |
| `C64-n36-ub2048-repeat` | UD-Q4_K_M | 36 | 2048 | q8_0 | 1036.6 | 47.9 | 45.4 | 31.2 | 6846 | ok |
| `C64-G-n36-ub2048-kvf16` | UD-Q4_K_M | 36 | 2048 | f16 | 1041.3 | 48.3 | 46.8 | 38.4 | 7446 | ok |
- **Write speed collapses with depth, and the KV type decides how much.** With the usual q8_0 KV cache, write speed falls from 48 to about 31 at 65K tokens of context; with an **f16 KV cache it falls only to 38.1**. That is +20% at long context, and no loss at short context. The price is about 0.6 GB of VRAM, which costs one more CPU layer (36 instead of 35).
- **Do not mix K and V types.** q8_0 for K with f16 for V was the slowest at depth (26.6), worse than q8_0 for both (31.2 to 32.2). Matched types stay on the fast path.
- **Layer count barely moves long-context write speed.** It sits at 31 to 33 (q8_0) across 34 to 36 CPU layers, which is what pointed at the KV cache instead.
- **Prompt reading follows the micro-batch:** 1,040 at `-ub 2048`, 844 at 1,536, 690 at 1,024.
- **34 CPU layers do not fit a 64K context at `-ub 2048`** (the 8K-context pass fits, the 64K pass runs out of VRAM); they do fit at `-ub 1536`, at the cost of slower prompt reading.
- **Recommended coding setting:** `--n-cpu-moe 36 -b 4096 -ub 2048 -ctk f16 -ctv f16`, peak VRAM 7.4 GB of the card's 7.8.

## 3. Chat profile

| Run | Quant | CPU layers | `-ub` | KV | Read (4,096) | Write | Write @ 8K | Write @ 65K | Peak VRAM (MiB) | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| `B1-n32-ub512` | UD-Q4_K_M | 32 | 512 | q8_0 | 452.7 | 50.0 | 47.6 |  | 6604 | ok |
| `H-n31-ub512` | UD-Q4_K_M | 31 | 512 | q8_0 | 460.9 | 50.5 | 48.0 |  | 7066 | ok |
| `H-n30-ub512` | UD-Q4_K_M | 30 | 512 | q8_0 | 470.2 | 52.5 | 49.8 |  | 7530 | ok |
| `H-n29-ub512` | UD-Q4_K_M | 29 | 512 | q8_0 |  | 51.4 |  |  | 7576 | does not fit: 4K prompt, 8K context |
| `H-n31-ub1024` | UD-Q4_K_M | 31 | 1024 | q8_0 | 739.8 | 51.2 | 48.3 |  | 7574 | ok |
| `H-n30-kvf16` | UD-Q4_K_M | 30 | 512 | f16 | 470.9 | 51.7 | 50.4 |  | 7608 | ok |
- **30 CPU layers is the most the card holds** (29 runs out of VRAM): write 52.5 against 50.0 at 32 layers (+5%).
- **A bigger micro-batch helps chat too:** 31 CPU layers with `-ub 1024` reads at 740 instead of 470, with write at 51.2.
- **KV type does not matter at short context** (f16 vs q8_0 at 30 layers: 51.7 vs 52.5, inside the noise).

## 4. Things that did not help

| Idea | Result | Rows |
|---|---|---|
| CPU governor `performance` instead of `schedutil` | **No difference** on either profile (coding 38.4 vs 38.1 at 65K; chat 52.0 and 52.3 vs 52.5) | `C64-G-n36-ub2048-kvf16`, `H-G-n30-ub512`, `H-G-n30-ub512-repeat` |
| More or fewer threads: 12, 14, 16 | Flat: 52.1, 52.5, 53.0 | `H-n30-t12`, `H-n30-ub512`, `H-n30-t16` |
| Pinning threads to physical cores | 16 threads pinned: **53.9** (about +2%, borderline); 14 pinned: 52.5 (no change) | `H-n30-t16-pin`, `H-n30-t14-pin` |
| `--poll 100` | Slightly worse (49.7 vs 50.4 and 50.6, about -1.4%) | `P1-poll100`, `B0-q4-prod`, `B0b-q4-prod` |
| Mixed K/V cache types | Worse at depth (26.6 vs 31 to 32) | `C64-n36-ub2048-k8v16` |

| Run | Quant | CPU layers | `-ub` | KV | Read (4,096) | Write | Write @ 8K | Write @ 65K | Peak VRAM (MiB) | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| `H-n30-t12` | UD-Q4_K_M | 30 | 512 | q8_0 | 470.3 | 52.1 | 49.7 |  | 7530 | ok |
| `H-n30-t16` | UD-Q4_K_M | 30 | 512 | q8_0 | 470.5 | 53.0 | 50.3 |  | 7530 | ok |
| `H-n30-t14-pin` | UD-Q4_K_M | 30 | 512 | q8_0 | 470.4 | 52.5 | 49.7 |  | 7530 | ok |
| `H-n30-t16-pin` | UD-Q4_K_M | 30 | 512 | q8_0 | 470.2 | 53.9 | 51.1 |  | 7530 | ok |
| `H-G-n30-ub512` | UD-Q4_K_M | 30 | 512 | q8_0 | 470.2 | 52.0 | 49.6 |  | 7530 | ok |
| `H-G-n30-ub512-repeat` | UD-Q4_K_M | 30 | 512 | q8_0 | 471.1 | 52.2 | 49.0 |  | 7530 | ok |
Why the box behaves like this (my arithmetic from the model's published config, not a measurement): decode reads about 0.5 GB of expert weights per token, so at ~50 tokens/s it uses roughly a quarter of the quad-channel DDR4's peak bandwidth. That points at CPU work and per-operation overhead, not memory bandwidth, as the limit, which would explain why moving layers to the GPU helps a little and why governor and thread changes do nothing.

## 5. How high a quant fits, and what it costs

**The highest quants that fit are Q8_0 (34.4 GiB) and Unsloth's UD-Q8_K_XL (35.8 GiB): both run a full 64K context on the 8 GB card**, with the experts in system RAM. BF16 (64.6 GiB) does not fit. The price is write speed, and it is steeper than I expected from an earlier Q4-to-Q6 comparison.

Coding profile (64K context, f16 KV, `-ub 2048`), best fitting split of each quant:

| Run | Quant | CPU layers | `-ub` | KV | Read (4,096) | Write | Write @ 8K | Write @ 65K | Peak VRAM (MiB) | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| `C64-n36-ub2048-kvf16` | UD-Q4_K_M | 36 | 2048 | f16 | 1039.8 | 48.2 | 47.1 | 38.1 | 7446 | ok |
| `C64-Q5XL-n37` | UD-Q5_K_XL | 37 | 2048 | f16 | 909.6 | 42.1 | 40.5 | 34.7 | 7456 | ok |
| `C64-Q5XL-n38` | UD-Q5_K_XL | 38 | 2048 | f16 | 899.6 | 38.7 | 41.5 | 33.7 | 6896 | ok |
| `C64-Q6K-n37` | UD-Q6_K | 37 | 2048 | f16 | 850.2 | 39.5 | 40.3 | 34.3 | 7656 | ok |
| `C64-Q6K-n38` | UD-Q6_K | 38 | 2048 | f16 | 839.5 | 38.8 | 38.9 | 34.8 | 7028 | ok |
| `C64-Q80-n38` | Q8_0 | 38 | 2048 | f16 | 711.7 | 33.8 | 33.5 | 31.8 | 7278 | ok |
| `C64-Q80-n39` | Q8_0 | 39 | 2048 | f16 | 698.8 | 33.8 | 33.4 | 30.5 | 6704 | ok |
| `C64-Q8XL-n38` | UD-Q8_K_XL | 38 | 2048 | f16 | 669.6 | 34.2 | 32.6 |  | 7800 | does not fit: 64K context |
| `C64-Q8XL-n39` | UD-Q8_K_XL | 39 | 2048 | f16 | 655.9 | 33.6 | 32.3 | 28.4 | 7080 | ok |
Chat profile (small window, q8_0 KV, `-ub 512`):

| Run | Quant | CPU layers | `-ub` | KV | Read (4,096) | Write | Write @ 8K | Write @ 65K | Peak VRAM (MiB) | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| `H-n30-ub512` | UD-Q4_K_M | 30 | 512 | q8_0 | 470.2 | 52.5 | 49.8 |  | 7530 | ok |
| `H-Q5XL-n32` | UD-Q5_K_XL | 32 | 512 | q8_0 | 389.6 | 44.9 | 43.7 |  | 7594 | ok |
| `H-Q5XL-n33` | UD-Q5_K_XL | 33 | 512 | q8_0 | 382.4 | 45.5 | 43.2 |  | 7032 | ok |
| `H-Q6K-n34` | UD-Q6_K | 34 | 512 | q8_0 | 344.0 | 42.4 | 39.3 |  | 6888 | ok |
| `H-Q6K-n35` | UD-Q6_K | 35 | 512 | q8_0 | 336.4 | 41.0 | 39.2 |  | 6198 | ok |
| `H-Q80-n36` | Q8_0 | 36 | 512 | q8_0 | 270.7 | 34.0 | 33.1 |  | 6186 | ok |
| `H-Q80-n37` | Q8_0 | 37 | 512 | q8_0 | 264.8 | 35.1 | 34.4 |  | 5372 | ok |
- **Each step up costs write speed:** 48.2 (Q4_K_M) to 42.1 (Q5) to 39.5 (Q6) to **33.8 (Q8_0)**, a 30% drop end to end; prompt reading falls from 1,040 to 910, 850 and 712. At 65K tokens of depth the gap narrows: 38.1, 34.7, 34.3, 31.8 (-17%).
- **More CPU layers are needed as the quant grows:** 36 for Q4, 37 for Q5 and Q6 (38 for headroom), 38 for Q8_0, 39 for UD-Q8_K_XL, to leave room for the bigger expert weights and a 64K context.
- **UD-Q8_K_XL buys nothing here.** Same write speed as plain Q8_0 (33.6 vs 33.8), slower at depth (28.4 vs 31.8), slower reading (656 vs 712), and one more CPU layer. One concern raised in other users' reports (which I did not verify) was that its BF16 tensors could be slow on a GPU with no BF16 support; I do not see that in write speed on the 2080, but I do see the other costs.
- **Chat profile follows the same curve:** 52.5 (Q4, 30 layers), about 45 (Q5), 42.4 (Q6), 35.1 (Q8_0, 37 layers).
- One odd row: the 38-layer Q5 run writes slower at depth 0 (38.7) than at 8K (41.5), which is backwards. I treat the 37-layer Q5 row as the reference.

**Quality, from a third party** (not measured here). Mean KL divergence from the BF16 model on wikitext (lower is closer to the original), from [AesSedai's published table](https://huggingface.co/AesSedai/Qwen3.6-35B-A3B-GGUF) for the same model family:

| Quant | Mean KL divergence | Perplexity vs BF16 |
|---|---|---|
| Q8_0 | 0.0060 | +0.04% |
| Q6_K | 0.0065 | -0.01% |
| Q5_K_M | 0.0083 | +0.18% |
| Q4_K_M | 0.0137 | +0.33% |
| IQ4_XS | 0.0329 | +2.36% |

These are that author's own quant builds, not the Unsloth files I benchmarked, so read them as the shape of the curve. The shape matters for the speed table: Q6_K to Q8_0 reduces the divergence by only about 8% while costing about 14% of write speed here. By these figures the quality knee sits around Q5_K_M to Q6_K, and Q8_0 is for people who want the last sliver and can pay for it.

## What this does not cover

- Quality: these are speed numbers for a 4-bit quant. The quant comparison below uses third-party quality figures, not mine.
- Speculative decoding (multi-token prediction) and the `ik_llama.cpp` fork: researched and queued, **not yet run**. The research reports were mixed on whether speculation helps when the experts live in system RAM, so I am not guessing.
- Server behavior: `llama-bench` has no prompt cache or queueing.
