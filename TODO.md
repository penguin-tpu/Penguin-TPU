# Penguin-TPU TODO

This file is the single source of truth for active pending work, undecided items, and
repo-level TODO tracking. Keep historical context and completed-change notes in
`SOUL.md`, but keep open work here.

## Repo-Wide

- Keep `TODO.md` as the only place for active TODOs, pending tasks, and to-be-decided
  items. Other docs may reference this file, but should not maintain their own open-work
  lists.
- Configure the repo or org `OPENAI_API_KEY` Actions secret and allow GitHub Actions to
  create pull requests so `.github/workflows/codex-autofix-ci.yml` can open CI repair
  PRs.
- Define the formal tensor layout and packing specification, including padding and
  tail-handling rules.
- Define the 32-bit binary/text assembly encoding specification used across compiler,
  model, and RTL.
- Define the host control and CSR map for launch, halt, done, and error reporting.
- Finalize tensor-era ISA decisions that affect multiple stacks:
  - whether VMEM is logically unified or internally partitioned by traffic class
  - halt behavior while long-chime tensor operations are in flight

## Compiler

- Extend the executable-package contract so `constants.bin` has a defined runtime memory
  mapping instead of being bundle collateral only.
- Broaden the direct exporter beyond the current fixed Gemma attention / MLP / decoder
  model-package flow.

## Modeling

- Add manifest-driven runtime staging for `constants.bin` once the bundle contract defines
  its memory mapping.
- Refine the cycle-accurate tensor hazard / overlap model beyond the current architectural
  scoreboard if RTL-visible pipeline details require it.

## RTL

- Keep the scalar RTL slice aligned with the frozen decode/control contract in
  `docs/specs/microarchitecture-spec.md`:
  - preserve the full scalar decode control record (`valid`, `illegal`, `format_class`,
    `scalar_op_class`, `alu_fn`, register fields, immediates, and control flags)
  - keep reserved custom opcodes classified distinctly from fully illegal encodings
  - keep module responsibilities coherent across `penguin_scalar_defs.vh`, decoder,
    regfile, ALU, branch unit, LSU, controller, and core top
- Finish the first scalar RTL milestone:
  - fetch and execute the frozen scalar subset from `IMEM`
  - prove `x0` hardwiring
  - prove two-delay-slot control flow
  - prove misaligned target / scalar-memory halts
  - report distinct `ecall` and `ebreak` outcomes
  - run the scalar RTL regression path in automation
- Build the RTL execution/testbench flow around the same executable package artifacts used
  by the software stack.
- Connect compiler- or vector-generated assembly/bundle artifacts into cocotb-based RTL
  regression runs.
- Reuse the same executable-package artifacts for on-hardware FPGA validation after RTL
  execution is in place.
- Reuse the checked-in scalar vectors for decoder, ALU/register-file, branch/jump,
  scalar-memory, and end-to-end core verification before widening accelerator scope.
- Decide the scalar RTL bring-up details that remain open:
  - whether non-`x0` scalar registers power up as zero in RTL or are explicitly
    initialized only by testbench/setup flow
  - whether the first scalar IMEM/VMEM interface stays simple and blocking or adopts a
    request/response handshake immediately
  - whether encoded scalar binaries are checked in beside source vectors or generated on
    demand during tests
  - whether `fence` remains a distinct controller-visible op or is normalized to a
    generic no-op immediately after decode
- Grow the preliminary BF16 VPU RTL beyond the current single-lane `vadd` slice:
  - replace the MMIO-seeded one-lane mreg file with architecturally meaningful tensor
    register storage
  - decide whether the first real hardware VPU register view is BF16-only or shared with
    FP8-oriented tensor paths
  - add additional VPU ops only after the `vadd` path and encoding are stabilized

## Spec Clarifications

- Resolve the current MXU `VM`-format documentation mismatch:
  - `architecture-spec.md` / `microarchitecture-spec.md` currently model `vmatmul.*`
    with only `vs1` plus `wsel`, with all other `VM` bits reserved zero
  - `greencard.md` currently shows an `e_sel` field and scaled-matmul semantics
  - the Python functional/perf model currently follows the architecture and
    microarchitecture specs and treats `e0` as hardwired unity only
