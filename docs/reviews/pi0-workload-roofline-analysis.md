# PI0 Workload Roofline Analysis

Status: Working Baseline

## 1. Purpose

This note records how the Penguin roofline model is grounded in a real workload rather
than synthetic tile-only kernels.

The target workload is the PyTorch PI0 policy implementation from OpenPI, configured as
in the external `Understanding-PI0` example script and profiled by the external
`flops.log` report.

Source references:

- OpenPI model code:
  `https://github.com/Physical-Intelligence/openpi/tree/main/src/openpi/models_pytorch`
- `Understanding-PI0` example configuration:
  `https://github.com/ucb-bar/Understanding-PI0/blob/main/scripts/run_example.py`
- `Understanding-PI0` FLOP report:
  `https://github.com/ucb-bar/Understanding-PI0/blob/main/reports/flops.log`
- `Understanding-PI0` README model printout:
  `https://github.com/ucb-bar/Understanding-PI0/blob/main/README.md`

## 2. Workload Understanding

The relevant sequence geometry for the profiled workload is:

- 3 RGB images of size `224 x 224`
- SigLIP patch size `14 x 14`, which produces `16 x 16 = 256` patch tokens per image
- 48 language tokens
- 50 action tokens
- 1 state token in the action expert

From the OpenPI preprocessing and model structure, the two main token domains are:

- prefix length `816 = 3 x 256 + 48`
- suffix length `51 = 1 + 50`

Those lengths explain the dominant matrix shapes in the profiler report:

- large prefix MLP and attention projections at `816 x 2048` scale
- SigLIP vision blocks at `256 x 1152` scale
- action-expert projections at `51 x 1024` scale
- attention score/context batched matmuls with `8` heads

## 3. Experiment Plan

The workload roofline experiment uses the following steps:

1. identify the dominant kernels directly from the external profiler output
2. keep only the dominant kernels whose cumulative FLOP share is materially complete
3. map those kernels onto the current Penguin tensor contract
4. compute DRAM arithmetic intensity from dense workload operand bytes per kernel call
5. compute VMEM arithmetic intensity from Penguin whole-tile traffic using the current
   `64 x 32 @ 32 x 16 -> 64 x 16` MXU contract and whole-register VMEM transfers
6. project those intensities onto the current DRAM roof, VMEM roof, and MXU peak
7. plot both hierarchy views on one roofline and report the aggregate dominant-kernel
   point

## 4. Modeling Assumptions

The current roofline model makes the following explicit assumptions:

- all dominant PI0 matrix kernels are mapped onto the Penguin MXU path
- Penguin MXU uses `FP8_e4m3 x FP8_e4m3 -> BF16`, so DRAM-side operand bytes use
  1 byte for activation/weight inputs and 2 bytes for BF16 outputs
- `aten::addmm` is treated as MXU-dominated matmul plus non-fused bias behavior
- DRAM arithmetic intensity uses dense kernel operands:
  - activation bytes
  - weight bytes
  - output bytes
  - bias bytes for `addmm`
- VMEM arithmetic intensity uses current Penguin tile traffic:
  - one full-register `vload` per activation tile
  - one `mxu.push.*` per weight tile
  - one full-register `vstore` per output tile
  - one additional full-register bias `vload` per output tile for `addmm`
- VMEM traffic is intentionally schedule-shaped, not ideal-cache-shaped, because the
  goal is to reflect the current Penguin execution model and tile granularity

## 5. Dominant Kernel Set

The dominant set used by the model is the top 16 kernels from `flops.log`.

That set covers `99.82%` of the reported `4,354,614,038,072` total FLOPs.

The largest entries are:

| Kernel | FLOP share | Interpretation |
|---|---:|---|
| `vlm_mlp_gate_up` | `45.27%` | Gemma prefix MLP expansion projections |
| `vlm_mlp_down` | `22.64%` | Gemma prefix MLP contraction |
| `vlm_attention_q_o` | `5.66%` | Gemma prefix attention projection cluster |
| `siglip_attention_proj` | `5.06%` | SigLIP 1152-wide projection-heavy blocks |
| `siglip_mlp_fc1` | `4.72%` | SigLIP MLP expansion |
| `siglip_mlp_fc2` | `4.72%` | SigLIP MLP contraction |
| `expert_mlp_up` | `3.54%` | Action-expert MLP expansion |

Elementwise kernels are not part of the dominant set because each is far below the top
matrix terms in FLOP contribution.

## 6. Findings

Using the current Penguin hardware parameters:

- DRAM roof bandwidth is `2 B/cycle`
- VMEM roof bandwidth is `16 B/cycle`
- total MXU peak is `2048 ops/cycle`

For the dominant-kernel aggregate:

- DRAM arithmetic intensity is `426.18 ops/byte`
- VMEM arithmetic intensity is `24.27 ops/byte`
- DRAM ceiling is `852.36 ops/cycle`
- VMEM ceiling is `388.36 ops/cycle`

This means the dominant PI0 workload is memory-bound on both roofs under the current
Penguin parameters, and more strongly constrained by VMEM-side tile traffic than by the
dense DRAM operand footprint.

Representative kernel observations:

- the largest prefix MLP expansion kernel (`816 x 2048 @ 2048 x 16384`) reaches
  `883.75 ops/byte` on DRAM and `24.81 ops/byte` on VMEM
- that kernel is still DRAM-bound on the current roofline (`1767.51 ops/cycle`) and
  much more strongly VMEM-bound (`396.90 ops/cycle`)
- the action-expert kernels are substantially smaller and lose more intensity to tile
  padding in the `51`-row dimension

## 7. Implication For Penguin

For this workload, improving only raw MXU peak would not move the aggregate point much.

The more relevant levers for PI0-style inference are:

- higher DRAM bandwidth
- higher VMEM bandwidth
- better VMEM-side tile reuse and less padding waste for irregular matrix shapes
- software scheduling that reduces repeated whole-register traffic for narrow and
  short-sequence kernels
