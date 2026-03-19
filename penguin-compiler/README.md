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
data structures, plus a first direct fixed-model export path:

- `BundleSymbol`
- `BundleSymbolTable`
- `BundleManifest`
- `ExecutableBundle`
- `write_executable_bundle(...)`
- `penguin-compile bundle ...`
- `export_pytorch_model_package(...)`
- `penguin-compile export-model ...`
- `write_verilog_rom_init(...)`
- `penguin-compile rtl-rom ...`

`penguin-compile bundle ...` packages a checked-in or user-authored assembly source into
the executable-bundle contract consumed by `penguin-model`. If the source program has an
adjacent `*.symbols.json5` sidecar, the CLI reuses it and carries forward any referenced
file-backed payloads it can find next to that sidecar.

`penguin-compile rtl-rom ...` assembles a checked-in or user-authored scalar program and
writes a Verilog include containing `imem[index] = 32'h...;` assignments for FPGA/RTL ROM
initialization.

`penguin-compile export-model ...` captures one supported fixed PyTorch Gemma-style
module through `torch.export`, validates the exported graph against the current
architecture-visible `vmat*` / VPU / XLU baseline, and emits a staged model package made
of executable bundles plus a package-level `model_package.json5`.

Still deferred:

- broader model coverage beyond the current fixed Gemma attention / MLP / decoder slices
- manifest-driven runtime mapping for `constants.bin`

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

The staged model-package layout used by `export-model` is:

```text
model_package.json5
model_input.bin
expected_output.bin
stages/
  <stage-name>/
    <program>.S
    <program>.symbols.json5
    manifest.json5
    constants.bin
    <static input payloads>.bin
```

## Example Usage

```bash
uv run penguin-compile bundle \
  --program tests/vectors/programs/tensor/examples/matmul_large.S \
  --output-dir /tmp/penguin-matmul-large
```

```bash
uv run penguin-compile rtl-rom \
  --program tests/vectors/programs/scalar/rtl/uart_mmio_hello.S \
  --output rtl/penguin_tpu/scalar/penguin_scalar_uart_hello_program_init.vh
```

```bash
uv run penguin-compile export-model \
  --model gemma_attention \
  --output-dir /tmp/penguin-gemma-attention
```
