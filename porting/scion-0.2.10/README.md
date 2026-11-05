# Scion 176B port for Strata v0.2.10

- Base: upstream tag **v0.2.10** (`ffa80d73`, the tip of `upstream/main`)
- Branch: **`scion-0.2.10`**
- Port commits: `b33aaba9` (the port) and `2e021b95` (the 32-lane PTQ1_0 fixes)
- Patch: [`0001-176b-scion.patch`](0001-176b-scion.patch), the diff `v0.2.10..2e021b95`
  (sha256 `666cb9210146bd7e0d7713c89d3d891bd9d7fdd2e0ff4e7abc6124554e1da41d`)

The patch is the reviewable form of the branch. It is a 3-way port of the 176B work from
`scion-0.1.44`; it is not a rebase of the 33-commit line.

## Logical stages

| Stage | Paths | Purpose |
|---|---|---|
| `0001-artifact-formats` | `third_party/ggml/ggml-common.h` | PTQ1_0 type 143: the 128-value / 28-byte block struct the kernels decode. Upstream already names and sizes it in the GGUF readers. |
| `0002-ptq1-experts` | `include/strata/kernels/cpu/ptq1_0.hpp`, `src/kernels/cpu/ptq1_0.cpp`, `src/kernels/cpu/native_expert.cpp`, `src/kernels/cuda/iq_kernels.cu`, `src/kernels/ptq1_0_parity.cpp` | CPU base-3 decode + Q8_K dot, native-expert routing, CUDA dot/split/prompt dequant, parity tool. |
| `0003-corrections` | `include/strata/core/corrections.hpp`, `src/core/corrections.cpp`, `include/strata/kernels/lora_corr.hpp`, `src/kernels/cuda/lora_corr.cu`, `src/core/layer.cpp`, `src/core/verify.cpp`, `src/prefill/prefill.cpp`, `src/prefill/kernels.cu`, `src/kernels/lora_corr_parity.cpp` | `out += B(Ax)` at the three block outputs; f32 post-gate path for QSA. |
| `0004-qwen4exp-runtime` | `src/program/generate.cpp`, `include/strata/prefill/kernels.hpp`, `src/prefill/kernels.cu` | Startup validation with the real geometry; correction dispatch in chunk-major prefill. |
| `0005-packaging-tests` | `CMakeLists.txt`, `src/kernels/ple_q4_parity.cpp`, `src/kernels/f16_hw_check.cu`, `src/kernels/lora_corr_bench.cu` | Targets and checks. PLE Q4_0 is upstream already; the parity tool proves the reader. |

The patch is one file because the stages share files (`CMakeLists.txt`, `prefill.cpp`, `iq_kernels.cu`),
so splitting it into independently applying patches would be fiction. The stage table is the review map.

## Pack note: the chat template

The Q6_K release's GGUF does **not** carry `tokenizer.chat_template`. A pack built from it without a
template can end up with a one-line placeholder (`{{ messages[-1].content }}`), and thinking mode then
degenerates into repeating the prompt. Build the pack with the Qwen3.8 template explicitly:

```sh
python3 tools/iq_pack.py --gguf "$SCION_MODELS/Scion-FlashNext-176B-A6B-Q6_K.gguf" \
  --out "$SCION_MODELS/pack-q6k" --compat-bf16 \
  --chat-template "$SCION_MODELS/pack-q2-ab/tokenizer/chat_template.jinja"
```

`--chat-template` overrides a template in the GGUF and also replaces an existing pack's placeholder;
`tools/strata_tokenizer.py --template FILE` does the same for a tokenizer-only extraction.

## Validation on this box

Build: HIP gfx1100, ROCm 7.2, `STRATA_BUILD_TESTS=ON`, 392 targets, exit 0.

The 176B Q6K GGUF is not on disk, so the three real-artifact tools were run against synthetic GGUFs
with the release's exact shapes and container types:

| Tool | Fixture | Result |
|---|---|---|
| `ptq1_0_parity` | `[2560, 640, 2]` type-143 gate/up/down, random valid blocks | CPU gu `1.31e-10`, down `9.31e-10`; GPU decode-once vs per-entry bitwise equal; swiglu `9.76e-08`; down `3.79e-08`; `iq_mmvq` `6.09e-08`; dequant f32/f16 zero differences. **0 failures.** |
| `lora_corr_parity` | three F16 pairs: `[2560,512]/[512,2560]` and `[6144,512]/[512,2560]` | 0 of 7680 values differ on each pair, `max|err| 0.000e+00`. **3 pairs, 0 failures.** |
| `ple_q4_parity` | 20,000,004-row sparse Q4_0 table | `max_abs 0.000e+00`. **PASS.** |

These prove the container geometry, decode, CUDA kernels and correction arithmetic. They do **not**
prove the real 176B artifact: that still needs the Q6K/Q4_0 release and the 30-task comparison.

### Upstream test failures that are not ours

On this build, two ctest entries fail for reasons already present in v0.2.10:

- `expert_parity`: `cannot read expert 0 (layer 0) from pack/full/experts.bin` — a missing fixture.
- `isa_guard_check`: flags `iq512x_rows` and `iq512_reference_gu_rows` in `iq_avx512.cpp` as unlisted
  public wide-ISA functions. Both functions exist in v0.2.10 and are absent from
  `tools/check_isa_guard.py`'s `WIDE_PUBLIC` set. This port does not touch either file.

Everything else in the targeted run passed, including `ggml_tq_overlay`, `arch_defaults_test`,
`bf16_bits_test`, `dequant_q5_1_test`, `shared_expert_parity`, `tqp_cpu_test`, `hip_device_selftest`
and `ple_reader_selftest`.

## Fixture note

The synthetic GGUFs live outside the repo (`/tmp/opencode/scion-fixtures/`) and are not committed.
Regenerate them with a minimal GGUF v3 writer; the parity tools need no metadata, only the tensor
names, shapes and types listed above.

## F16 teacher control (not run on this box)

The F16 body is the fidelity variant, but it is not a native quantized body. A pack built from it
starts in Strata, and the native expert layout loads, but prefill rejects
`blk.0.attn_qkv.weight` with `no native GGUF blocks`: the native prefill/verify path expects native
dense/head/BF16 projections, and the F16 body supplies none. Running it needs engine work for a
packed body inside a native expert pack, or the Prism `qwen4exp` fork on a host with at least 64 GB
RAM (this box has ~30 GB). The F16 GGUF is kept; its 35 GB derived pack was removed.

