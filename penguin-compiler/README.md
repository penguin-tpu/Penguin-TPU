# penguin-compiler

Python package for the hyperspecialized compile flow:

- inspect one target PyTorch model
- validate fixed shapes and parameters
- emit Penguin assembly
- emit a sidecar symbol-table JSON5 file next to the `.S` program
- emit `manifest.json5`
- emit packed binary constant blobs

The output is a program bundle consumable by `penguin-model`, RTL testbenches, and FPGA
host loaders.

## Current Status

The current compiler-side milestone is a bundle-packaging CLI plus the shared bundle
data structures:

- `BundleSymbol`
- `BundleSymbolTable`
- `BundleManifest`
- `ExecutableBundle`
- `write_executable_bundle(...)`
- `penguin-compile bundle ...`

`penguin-compile bundle ...` packages a checked-in or user-authored assembly source into
the executable-bundle contract consumed by `penguin-model`. If the source program has an
adjacent `*.symbols.json5` sidecar, the CLI reuses it and carries forward any referenced
file-backed payloads it can find next to that sidecar.

Still deferred for a later milestone:

- direct PyTorch-to-Penguin model export
- automatic generation of workload-specific `constants.bin` contents from model weights

## Bundle Layout

The expected output bundle is:

```text
program.S
program.symbols.json5
manifest.json5
constants.bin
```

The symbol-table sidecar records the IMEM base of the program entry plus any named
input, output, or scratch regions used by the workload.

## Example Usage

```bash
uv run penguin-compile bundle \
  --program tests/vectors/programs/tensor/examples/matmul_large.S \
  --output-dir /tmp/penguin-matmul-large
```
