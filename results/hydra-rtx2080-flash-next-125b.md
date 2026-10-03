# A 125B-parameter MoE on an 8 GB GPU

**Qwen3.8-Flash-Next** (a 125B-total mixture-of-experts model) at IQ3_XXS, run by the open-source [Strata](https://github.com/Niko1221/Strata) engine on the dedicated 2080 box ([hardware](../hardware.md)). 64K context, int8 KV cache. The experts (about 76 GB on disk, roughly 43 GiB resident in RAM) are computed on the CPU with AVX2; attention and the dense layers run on the 8 GB GPU.

## Result

Speeds in tokens/second, one model server, one request at a time. "12.5K" and "47K" are prompts of that many tokens.

| Setup | Write (short chat) | Read (12.5K prompt) | Write at 12.5K context | Read (47K prompt) | Write at 47K context |
|---|---|---|---|---|---|
| Strata 0.1.27 (the version I started on) | 31.0 | 336.8 | 31.2 | 321 | 30.2 |
| Strata 0.1.31, **default settings** | 31.4 | **248.1** | 34.9 | not run | not run |
| **Strata 0.1.31, tuned** | **39.2** | **459.9** | **37.9** | **476** | **37.5** |

Against the version I started on, tuned is **+26% write and +37% read**, with a 64K context and the same model file. Against the new version's own defaults it is +25% write and **+85% read**: the 0.1.31 defaults read *slower* than 0.1.27 on this card (248 vs 337), a regression the tuning first had to climb out of. All rows ran on the same box and driver (615.71). Raw rows for every experiment: [`data/strata-scoreboard.csv`](../data/strata-scoreboard.csv).

Two things to keep in mind:

- The table is **greedy decoding**. Sampling at the model card's settings (temperature 1.0, top-p 0.95, top-k 20) makes the draft guesses miss more often: write speed measured 41.7 greedy against 38.4 sampled (four alternating requests, about 8% lower), reads unchanged. Real use is the sampled number.
- Stock llama.cpp on the same model and card, measured earlier (2026-09-29, before the tuning): about 20 write and 175 to 180 read at 64K context. Tuned Strata is roughly 2x faster on write and 2.5x on read.

## What moved the numbers

Each row is a pair of runs from the scoreboard that differ only by the listed settings (checked against the recorded engine command lines). The gains are **not additive**; the settings interact.

| Change | Pair | Effect |
|---|---|---|
| Prompt-chunk ring of 16 (`STRATA_PREFILL_RING=16`) | `E0-new` to `E1-ring16` | read 248 to 371 (+50%), write flat |
| English-only draft vocabulary (`--mtp .../rt-en`) | `E0-new` to `E1-vocab-en` | write 31.4 to 32.8 (+4%) |
| Stricter draft threshold (`--spec-min-p 0.8`) plus a smaller PCIe share (`--pcie-frac 0.2`) | `E4-ref-v16` to `E3-d2-mp08-pf02` | write 32.7 to 38.4 (+17%), read flat |
| VRAM headroom (`--kv-resident 20480`, `--vram-reserve-mib 500`) | `E4-ref-v16` to `E3-v-kv20-res500` | read 370 to 460 (+24%), write +5%: more cached experts (1,069 vs 860 slots) let the engine pick a 3,072-token chunk instead of 2,048 |

(`E4-ref-v16` is the default plus the ring and the English vocabulary, the starting point for the later runs.)

Final settings: `--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.8 --max-context 65536 --kv int8 --kv-resident 20480 --vram-reserve-mib 500 --pcie-frac 0.2`, English draft vocabulary, plus `STRATA_PREFILL_RING=16`.

## Traps worth knowing

- **Never lower the VRAM reserve below 500 MiB on this card.** 400 ran with almost no headroom; 300 crashed with out-of-memory on the first request.
- **There is no prebuilt Strata for Linux**; it is compiled from source (about a minute). A new engine version's setup moves the prepared model pack out of an older install's data folder, so run it in an isolated config directory.
- **Greedy decoding plus thinking can loop**, and agent-style context compaction needs a bigger reply cap than the usual 8,192 tokens once the running summary grows. Both are server and client settings, not engine speed.

## What this does not show

Quality. This is a speed test of a heavily quantized (3-bit) model. How good IQ3_XXS Flash-Next is at real tasks is a separate question; I ran a small set of coding tasks against it, but they are not part of this repo.
