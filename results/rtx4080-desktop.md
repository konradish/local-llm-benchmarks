# RTX 4080 16 GB (desktop workstation): collected results

**Read this first.** These are the numbers I had on file for the 4080 box (see [hardware](../hardware.md)), collected between May and September 2026. They are **not** a controlled set like the [2080 results](hydra-rtx2080-35b-a3b.md):

- single runs, taken with different tools and flags over several months;
- the desktop shares the GPU, so the free VRAM changed from day to day;
- quoted from my lab notes, not re-measured for this repo;
- rows marked "est." or "TBD" in the notes are left out.

Treat them as "about this fast, in this configuration", not as a ranking. A matched re-run of the same model and flags on both machines is the obvious next step.

## Measured rows

| Model | Quant | Setup | Decode tok/s | Prefill tok/s | VRAM | Date | Build |
|---|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B (MoE, 3B active) | Q4_K_M | 22 expert layers on CPU (`--n-cpu-moe 22`), TurboQuant-3 KV, 64K ctx, TurboQuant llama.cpp fork | **73** short / 70 at an 11.6K prompt | ~950 | ~15.0 GB | 2026-09-07 | community fine-tune ¹ |
| Qwen3.6-35B-A3B | Q4_K_M | Same file, **GPU expert cache** fork (`moe-cache` branch), 112 cached expert slabs, `-ub 4096`, f16 KV; 11,637-token prompt + 128 generated | **85** | **2,630** | ~15.0 GB | 2026-09-07 | community fine-tune ¹ |
| Qwen3.6-35B-A3B | Q4_K_M | "daily" profile: 30 expert layers on CPU, 32K ctx | ~54 | n/a | 8.4 GB | 2026-09-09 | community fine-tune ¹ |
| Qwen3.8-27B (dense) | UD-IQ3_XXS (10.9 GB file) | All on GPU, 64K ctx, with vision projector | ~51 | n/a | 13.6 GB | 2026-09-06 | stock |
| Bonsai 2 27B (ternary Qwen3.8-27B) | PTQ1_0, 1.75 bits/weight | PrismML llama.cpp fork, 64K ctx, q4_0 KV | **73.9** | 408 | 8.3 GB | 2026-09-18 | stock |
| gpt-oss-20b | IQ4_NL | All on GPU, q8_0 KV, 32K ctx | 188 end-to-end (197 in a 16K test) | n/a | n/a | 2026-05 ² | community fine-tune ¹ |
| Gemma 4 26B-A4B | Q4_K_S | All on GPU, TurboQuant-3 K + f16 V, 16K ctx | 80 short / 55 on a 500-token generation | 126 short / 96 | n/a | 2026-05 ² | community fine-tune ¹ |

¹ An "abliterated" community fine-tune (refusals removed) of the named model. Same architecture, tensor shapes and file size as the stock model, so it should run at about the same speed, but I did not measure the stock build on this card; I ran these because they were what my local tooling used. The stock-model rows on the 2080 box are not abliterated.
² Measured under Windows 11 + WSL2 (driver 591.86), before the machine moved to bare-metal Linux in September. Leaving WSL freed about 3 GB of VRAM, so the later rows are not comparable to these.

## What these rows say

- **A GPU-side expert cache pays off on the 4080 and mostly does not on the 2080 box.** On a PCIe 4.0 x16 link the cache fork took the 35B model from 950 to 2,630 tokens/s of prompt reading and from 70 to 85 on decode. The same trick on the 2080 box (PCIe 3.0, September 28 measurement, not in the ledger) gave about 1,100 prompt tokens/s but **slower** decode (41 vs 45 to 49). Moving experts across the bus is the whole trick, so the benefit tracks the PCIe link, not the GPU.
- **A ternary 27B is the fastest "big dense" option on 16 GB.** The 1.75-bit Bonsai 2 build of Qwen3.8-27B decodes at 73.9 tokens/s in 8.3 GB, against ~51 for the 3-bit IQ3_XXS quant of the same model, which needs 13.6 GB.
- **Sharing the GPU with a desktop costs real headroom.** The 27B IQ3_XXS quant at 13.6 GB cannot coexist with the usual desktop tenants; it needs the card cleared. This is why the dedicated 2080 box is still useful despite having half the VRAM.

## Same model on both machines (read with care)

Qwen3.6-35B-A3B Q4_K_M, decode tokens/s, **not** matched settings:

| Machine | Expert layers on CPU | Decode | Prefill | Notes |
|---|---|---|---|---|
| RTX 4080 desktop | 22 | 70 to 73 | ~950 | TurboQuant fork, TurboQuant-3 KV, 64K context |
| RTX 2080 dedicated | 30 (chat profile) | 52.5 | 470 | stock llama.cpp, q8_0 KV, small window ([details](hydra-rtx2080-35b-a3b.md)) |
| RTX 2080 dedicated | 36 (coding profile, 64K ctx) | 48.2 | 1,040 | stock llama.cpp, f16 KV, `-ub 2048` |
