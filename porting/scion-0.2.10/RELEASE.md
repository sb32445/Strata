# Release checklist: `scion-v0.2.10-r1`

Do not publish, tag, or push until the direct Q6_K comparison has passed and the owner approves.

## What ships

| Piece | Source | Notes |
|---|---|---|
| Engine component | `scion-0.2.10` build (`build-hip-0.2.10/strata`) | HIP gfx1100 first; other backends only when built and tested |
| Terminal runtime | upstream `packaging/package_terminal.py` | self-contained Python + `serve/` + `strata_cli`; no engine binary, no weights |
| Model pack | `tools/iq_pack.py` output (`pack-q6k`) | dense weights + index + `native_experts.txt`; experts stay in the GGUF |
| Model file | `Scion-FlashNext-176B-A6B-Q6_K.gguf` | external; Hive owns the path and SHA-256 |

The Q2 bridge is **not** part of this release. It stays a local comparison baseline until the
direct result is accepted.

## Build

```sh
# Engine component (local, unpublished)
python3 packaging/package_engine.py \
  --src build-hip-0.2.10 \
  --backend hip --platform linux-x86_64 --arch gfx1100 \
  --version 0.2.10-scion-r1 \
  --out release/engine

# Terminal archive
python3 packaging/package_terminal.py \
  --out release/artifacts --target linux-x86_64 \
  --version 0.2.10-scion-r1

# Container (optional)
docker/build.sh
```

The HIP package needs the ROCm libraries the binary loads (`libamdhip64`, `libhipblas`,
`librocblas`, the gfx1100 kernels). `package_engine.py --verify` must hash-check the finished
folder; do not publish a package that fails it.

## Pin in Hive

1. Put the engine package where the stack's `exe` points, or set the config `exe` to its absolute
   path.
2. Update `stacks/manifests/scion-flashnext-176b-a6b.json` with the final GGUF SHA-256 and the
   engine release id.
3. Run `hive stacks validate scion-flashnext-strata` with `STRATA_REPO`, `SCION_MODELS` and
   `STRATA_DATA` set.
4. Start the stack, run one smoke request, then the 30-task direct arm.

## Naming

- Engine tag: `scion-v0.2.10-r1` (versioned branch `scion-0.2.10`, upstream tag `v0.2.10`).
- Do not call it `strata-lite`: that name belongs to a different fork.
- Container names stay `TQ1_0_g128` / `Q2_0_g128`, with the legacy aliases accepted.

## Rollback

The Q2 bridge stack (`stacks/scion-flashnext-strata-q2.json`) and its pack remain on disk.
Switching back is a stack change, not a rebuild.
