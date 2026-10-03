# Hardware

Two machines, one lab. Every number in this repo says which one it came from.

## hydra: dedicated inference box, RTX 2080 8 GB

A headless server whose only job is serving local models. Nothing else uses the GPU or the RAM.

| | |
|---|---|
| Role | Dedicated, headless (no display manager, `multi-user.target`) |
| CPU | AMD Ryzen Threadripper 1950X, 16 cores / 32 threads (Zen 1), AVX2, **no AVX-512** |
| Memory | 62 GiB usable: 4 x 16 GB DDR4-3200, one DIMM per channel (quad channel), configured at 3200 MT/s; one NUMA node |
| GPU | GeForce RTX 2080, **8 GB**, Turing (sm_75), PCIe 3.0 x16 |
| GPU driver | NVIDIA 615.71.09 (open kernel modules) |
| Storage | 1 TB SATA SSD |
| OS | Debian 13 "trixie", Linux 6.12 |
| CPU governor | `schedutil` (the default); `performance` where a table says so (it made no measurable difference) |

Software used for the numbers in `results/`:

- **llama.cpp** upstream commit `fc07d78` (2026-09-29), built with GCC 14.2 for CUDA architecture 75 (`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_NATIVE=ON`), CUDA 12 runtime.
- **Strata** engine 0.1.31 (commit `9259cad`, 2026-10-01), built against CUDA 13.1, for the 125B model.

## koni9: desktop workstation, RTX 4080 16 GB

My daily-driver desktop. **Not dedicated:** the same GPU also drives the desktop, so the models only see part of the card.

| | |
|---|---|
| Role | Daily-driver desktop (KDE Plasma on Wayland, browsers, chat apps, Docker services). Not dedicated. |
| CPU | Intel Core i9-14900K, 24 cores / 32 threads, AVX2, no AVX-512 |
| Memory | 62 GiB usable: 2 x 32 GB DDR5-4800 (dual channel) |
| GPU | GeForce RTX 4080, **16 GB**, Ada (sm_89), PCIe 4.0 x16 |
| GPU driver | NVIDIA 615.71.09 today; the May 2026 rows were measured under Windows + WSL2 with driver 591.86 |
| Desktop VRAM tenant | Roughly 3 to 7 GB of the 16 GB at any moment (compositor, browser, chat apps), so a model gets about 9 to 12 GB |
| OS | Debian 13 "trixie", Linux 6.12, bare metal since early September 2026 (Windows 11 + WSL2 before that) |

## Why these two

They bracket the realistic "one consumer GPU at home" range. The 4080 has twice the VRAM, about 1.6x the memory bandwidth, and a PCIe link twice as fast. The 2080 box has more CPU memory bandwidth (quad-channel DDR4 vs dual-channel DDR5) and nothing competing for the card.
