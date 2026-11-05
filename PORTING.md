# Porting Scion support onto a new Strata release

This repository is a downstream fork of [`Niko1221/Strata`](https://github.com/Niko1221/Strata).
It carries one downstream feature set: the **Scion** GGUF containers and the runtime pieces they need.

The fork is deliberately split so an upstream update is a port, not a merge of unrelated histories.

## Branch policy

| Ref | Meaning |
|---|---|
| `main` | A clean mirror of `upstream/main`. No Scion commits. |
| `scion-<version>` | One Scion port on top of the upstream tag of that version. |
| `archive/scion-main-0.1.x` | The old `main` head (`4f7fb6b5`), preserved. |
| `scion-0.1.44` | The last working 0.1.44 port (`2e52e930`), frozen reference. |
| `naming/tq1-q2-g128` | The canonical-name pass on the 0.1.44 line (`5bcefc73`), preserved. |

`upstream` is fetch-only. Its push URL is disabled on purpose:

```sh
git remote -v
# upstream  DISABLED: read-only mirror (push)
```

Do not force-update tags. The local `v0.1.37` tag differs from upstream's, so a plain
`git fetch upstream --tags` refuses to clobber it. Fetch the tags you need explicitly, and never
rewrite a published tag.

## Current port

- Upstream tag: **v0.2.10** (`ffa80d73`, also the tip of `upstream/main`)
- Branch: **`scion-0.2.10`**
- Source of the port: `319cf7f5` + `42bef63a` from the `scion-0.1.44` line
- Scope: the **176B FlashNext direct path** only

The 35B/qwen35 work (`3d34f904` onward: `Q2_0_g128` experts, g128 corrections, q36 decode
optimisations, MTP, decision head) is **not** part of this port. It stays on `scion-0.1.44`
until it is ported as its own series.

## The patch series

The logical series is kept in [`porting/scion-0.2.10/`](porting/scion-0.2.10/). The branch is the
source of truth; the patches are for review and for the next port.

| # | Series | What it adds |
|---|---|---|
| 0001 | `artifact-formats` | `PTQ1_0` (type 143) block geometry in the vendored `ggml-common.h`; GGUF reader name/geometry (upstream already reserves 143 as `PTQ1_0`, 128/28). |
| 0002 | `ptq1-experts` | CPU base-3 decode and Q8_K dot; `native_expert` routing; CUDA `Fmt<143>` / `Split<143>` / `vec_dot_ptq1_0`; prompt-path dequant; `ptq1_0_parity`. |
| 0003 | `corrections` | `lora_corr` A/B kernels; `core/corrections` validation and dispatch; decode, prefill and verify hooks; `gate_attn` f32 post-gate output; `lora_corr_parity`. |
| 0004 | `qwen4exp-runtime` | `generate` startup validation and the correction branches on the real engine path; PLE Q4_0 (upstream v0.2.10 already has the format, so no PLE patch is needed). |
| 0005 | `packaging-tests` | CMake targets, parity tools, and this document. |

## What upstream already provides

v0.2.10 already has, and the port does **not** re-add:

- type 143 named `PTQ1_0`, 128 values / 28 bytes, in the GGUF reader and tools;
- PLE `Q4_0` rows (`PleFormat::Q4_0`);
- `answer_room`, `thinking_budget_tokens`, `fit_max_tokens`;
- the terminal archive and Docker packaging.

The port only adds the kernels and the correction plumbing upstream does not have.

## Deferred container types

Do **not** reuse types 144–149. Upstream assigns them to Agention's Gyro/APR family:

| Type | Upstream name |
|---:|---|
| 144 | `TQ2_T` |
| 145 | `TQK6` |
| 146 | `TQK7` |
| 147 | `TQP5` |
| 148 | `TQP6` |
| 149 | `TQP7` |

The old Scion line used 144/145/146 for the PLE `Q3X32`/`Q2X32`/`Q4X160` rows. Those are
deferred until they are moved to free IDs. The direct 176B release uses PLE `Q4_0`, so this does
not block it.

The 35B containers also stay deferred:

- `Q2_0_g128` (formerly `PQ2_0`) used type 142;
- `Q1_0_g128` (type 43) corrections.

Type 142 and 43 are free upstream, but they belong to the 35B series, not this one.

## Known conflict surface

The 176B series touches these files, which upstream also moves quickly:

```
CMakeLists.txt
include/strata/artifact/gguf_reader.hpp
include/strata/core/corrections.hpp
include/strata/core/layer.hpp
include/strata/core/verify.hpp
include/strata/kernels/cpu/ptq1_0.hpp
include/strata/kernels/lora_corr.hpp
include/strata/prefill/kernels.hpp
src/core/corrections.cpp
src/core/layer.cpp
src/core/verify.cpp
src/kernels/cpu/native_expert.cpp
src/kernels/cpu/ptq1_0.cpp
src/kernels/cuda/iq_kernels.cu
src/kernels/cuda/lora_corr.cu
src/kernels/ngram.cpp
src/kernels/ptq1_0_parity.cpp
src/kernels/lora_corr_parity.cpp
src/kernels/ple_q4_parity.cpp
src/prefill/kernels.cu
src/prefill/prefill.cpp
src/program/generate.cpp
third_party/ggml/ggml-common.h
tools/gguf_reader.py
```

Expected conflicts on the next port: the `Fmt`/`Split` format lists and the `STRATA_GU/D/MMVQ_FMTS`
macros in `iq_kernels.cu`, the layer image layout in `layer.cpp` / `verify.cpp`, the prefill
buffer carve in `prefill.cpp`, and the CMake source and target lists.

## Two runtime restrictions in this port

1. **Layer-major prompt prefill and corrections are not mixed.** The layer-major walk
   (`STRATA_PREFILL_LAYER_MAJOR=1`) has no correction hooks in this port. It is off by default on
   HIP and on a layer split. When corrections are present it must stay off; the port prints a
   reason and falls back to the chunk-major path instead of producing uncorrected logits.
2. **The fused QSA attention epilogue is disabled on layers with an `attn_output` correction.**
   The fused epilogue writes only the f16 o-projection input; the correction needs the post-gate
   value in f32. Those 12 QSA layers take `gate_attn` instead. The other layers keep the fusion.

## Build

A HIP build on gfx1100, matching the local 2×7900 XT box:

```sh
cmake -S . -B build-hip-0.2.10 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_PREFILL_MMQ=ON \
  -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON
cmake --build build-hip-0.2.10 -j 16
```

## Tests

The parity tools take the real 176B GGUF:

```sh
./build-hip-0.2.10/ptq1_0_parity   /path/to/Scion-FlashNext-176B-A6B-Q6K.gguf
./build-hip-0.2.10/lora_corr_parity /path/to/Scion-FlashNext-176B-A6B-Q6K.gguf
./build-hip-0.2.10/ple_q4_parity    /path/to/Scion-FlashNext-176B-A6B-Q6K.gguf
```

Unit and format tests:

```sh
ctest --test-dir build-hip-0.2.10 --output-on-failure
```

Python tools (the repo's venv has pytest; the system Python does not):

```sh
../hivebench/.venv/bin/python -m pytest -q tools/test_iq_pack.py tools/test_strata_inspect.py
```
