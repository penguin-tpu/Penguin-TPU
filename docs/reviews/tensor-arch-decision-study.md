# Tensor-Architecture Decision Study

Status: Research note

## 1. Scope

This document studies three open Penguin-TPU architecture questions:

1. whether Penguin should add a separate architectural 1D vector-register file
2. where scaled-matmul scale factors should live
3. whether Penguin should keep a rectangular MXU or return to a square MXU

The intent is not to freeze the architecture in this document. The intent is to separate:

- facts directly supported by primary sources
- inferences from those sources
- Penguin-specific recommendations

## 2. Source Set

Primary sources used here:

- Google Cloud TPU architecture overview:
  - https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm
- Google Cloud TPU product architecture pages:
  - https://docs.cloud.google.com/tpu/docs/v4
  - https://docs.cloud.google.com/tpu/docs/v5e
  - https://docs.cloud.google.com/tpu/docs/v6e
  - https://docs.cloud.google.com/tpu/docs/tpu7x
- TPU v1 paper:
  - https://arxiv.org/abs/1704.04760
- TPU v4 paper landing page:
  - https://arxiv.org/abs/2304.01433
- Google Cloud BF16 guidance:
  - https://docs.cloud.google.com/tpu/docs/bfloat16
- Google AQT materials:
  - https://cloud.google.com/blog/products/compute/accurate-quantized-training-aqt-for-tpu-v5e
  - https://github.com/google/aqt
  - https://github.com/google/qwix
- JAX scaled matmul API:
  - https://docs.jax.dev/en/latest/_autosummary/jax.nn.scaled_matmul.html

## 3. Directly Supported Facts

### 3.1 What Google publicly says about TensorCores

Google’s TPU system-architecture documentation says each TensorCore contains one or more
MXUs, a vector unit, and a scalar unit. It also says the vector unit is used for general
computation such as activations and softmax, while the scalar unit handles control flow
and address calculation. It does not publicly describe a separate architectural
vector-register file analogous to Penguin’s current `m`-register concept.

Google’s public TPU docs also show that MXU shape has been square across generations:

- TPU versions prior to v6e use `128 x 128` systolic arrays
- TPU v6e and TPU7x use `256 x 256` systolic arrays

The same public doc also states that the MXU multiplies reduced-precision inputs and
accumulates in a wider type. For the BF16 path it states: multiplies take BF16 inputs and
accumulations are performed in FP32.

The TPU v4, v5e, v6e, and TPU7x product pages consistently describe a TensorCore as:

- one or more MXUs
- one vector unit
- one scalar unit

They do not describe a second programmer-visible 1D vector storage class.

### 3.2 What Google publicly says about memory and precision

The TPU v1 paper describes the TPU as pairing a large matrix unit with a large
software-managed on-chip memory. The emphasis is on keeping the matrix unit busy and
feeding it efficiently rather than on exposing many orthogonal programmer-visible storage
classes.

Google’s BF16 guidance says reduced-precision storage matters partly because some
operations are memory-bandwidth-bound, so storing operands and outputs in a narrower
format reduces the amount of data moved.

This matters for Penguin because it separates two concerns:

- compute-array geometry
- memory bandwidth and storage format

Google’s public materials treat these as related but not identical design decisions.

### 3.3 What Google publicly says about quantized / scaled matmul software surfaces

Google’s AQT materials are software-level, but they are still informative because they
show how Google exposes quantized tensor ops in production JAX stacks.

The AQT README says quantization is applied to tensor operations such as `dot_general`,
and that weights can be stored in quantized form for serving. It also shows that
quantization configuration is specified per tensor side of a tensor op.

The newer Google `qwix` repository is even more explicit: quantized weights are stored as
a structure with separate `qvalue` and `scale` arrays. The README example shows INT8
weights with a separate FP32 `scale` tensor.

The JAX `jax.nn.scaled_matmul` API similarly models scales as separate array operands,
`lhs_scales` and `rhs_scales`, with their own shapes. This is especially important:
Google’s current software-visible scaled-matmul API does not model the scales as hidden
scalar registers. It models them as separate tensors or tensor-like side arrays.

## 4. Question 1: Should Penguin Add 1D Vector Registers?

### 4.1 Why this idea is attractive

A dedicated 1D vector register file is attractive for at least four real reasons:

- bias vectors are naturally 1D
- row/column reduction results are naturally 1D
- elementwise epilogues often want a narrow operand reused across many rows
- it avoids wasting full `m` registers on logically small values

If Penguin keeps only whole-tile `m` registers, then several common patterns become
wasteful:

- storing bias as a replicated tile
- storing reduction outputs in a mostly empty tile
- moving narrow metadata-like values through the same storage as dense tiles

### 4.2 What the TPU references imply

The public TPU materials do not expose a dedicated programmer-visible vector-register
class. Instead, Google exposes:

- MXUs for dense tensor contraction
- a vector unit for elementwise and reduction-like work
- a scalar unit for control and address generation

That is not proof that no narrow internal storage exists. It only means public TPU
abstractions do not make it a first-class architectural operand class.

That choice appears deliberate. Inference from the sources:

- Google seems to prefer keeping the architecture surface centered on a small number of
  execution/storage abstractions
- when a workload class is dominant enough to justify a special path, Google adds a
  dedicated unit such as SparseCore for embedding-heavy models rather than adding many
  small architectural storage classes everywhere

This inference is consistent with the official documentation that calls out SparseCore as
specialized support for embedding-dominated workloads, while keeping the ordinary
TensorCore programming model relatively stable.

### 4.3 Penguin tradeoff analysis

Benefits of a full vector-register file:

- efficient bias and reduction storage
- simpler VPU operand shapes for 1D data
- avoids wasting `m`-register capacity
- can reduce VMEM traffic for narrow reused operands

Costs of a full vector-register file:

- another architected storage namespace
- more move instructions and more compiler bookkeeping
- more bypass / arbitration complexity if VPU and XLU can read both `m` and `v` files
- more instruction-encoding pressure
- more questions about broadcasting rules, reduction destinations, and interaction with
  matmul epilogues

The most important problem is not storage area. It is architectural surface area. Once
Penguin adds a full `v` register file, many follow-on questions appear immediately:

- Can MXU read `v` registers directly?
- Can VPU binary ops combine `m` and `v` operands?
- Do reductions write `v` by default?
- Do broadcasts from `v` implicitly expand across rows or columns?
- Are `vload` / `vstore` independent from `mload` / `mstore`?

That is a large policy surface for a first tensor slice.

### 4.4 Recommendation

Recommendation: do not add a full 64-entry architectural vector-register file in the
first tensor revision.

Instead, prefer one of these narrower steps:

1. Add broadcast semantics to selected VPU / MXU-epilogue instructions so a narrow VMEM
   vector can be streamed or expanded without becoming a first-class register file.
2. Add reduction instructions that write into a conventional `m` register using a
   canonical packed layout.
3. If measurements later prove a true narrow on-chip operand store is necessary, add a
   small dedicated side storage, for example 8 to 16 vector registers, not a second
   register file that mirrors the full `m` namespace.

Rationale:

- it captures most of the bias/reduction benefit
- it avoids prematurely doubling the tensor operand model
- it preserves the simpler “one main tensor register file” compiler story

## 5. Question 2: Where Should Scaled-Matmul Scale Factors Live?

### 5.1 The decision is broader than “scalar vs tensor”

The right answer depends on scaling granularity.

There are at least three meaningful regimes:

- one scale per entire tensor
- one scale per row / column / channel
- one scale per block along the contracting dimension

The official JAX scaled-matmul API already assumes the third class exists. Its
`lhs_scales` and `rhs_scales` have their own shapes and represent block-structured side
data, not just one scalar.

That immediately weakens the “put scales in scalar `x` registers” option. Scalar
registers only make sense for the coarsest possible scaling policy.

### 5.2 What Google’s software surfaces imply

The Google AQT and Qwix materials show quantized operands as:

- quantized values
- separate scale tensors

The JAX scaled-matmul API makes scales explicit array operands.

Inference from these sources:

- Google’s software stack treats scales as tensor-side metadata, not hidden architectural
  scalar state
- scales are expected to support per-channel and sub-channel structure
- the design center is “scale arrays travel with quantized tensors,” not “one magic scale
  register per matmul”

### 5.3 Penguin option analysis

#### Option A: Store scales in scalar `x` registers

Pros:

- simplest initial implementation
- works for a single global scale per operand
- trivial instruction encoding

Cons:

- fundamentally incompatible with block-scaled or per-channel scaling
- ties a tensor datapath concern to the scalar control file
- creates awkward pressure and lifetime coupling with address-generation state

Conclusion:

- viable only for a very limited “one scale per operand” bring-up mode
- not a good long-term design center

#### Option B: Store scales in ordinary tensor `m` registers

Pros:

- immediately supports per-channel or blockwise scaling
- scales use the same movement path as tensors
- no new storage class required

Cons:

- extremely wasteful when the scale tensor is much smaller than a data tile
- pollutes the main tensor register file with metadata-like values
- makes the cost model ugly for small scale payloads

Conclusion:

- acceptable as a software emulation or temporary compiler fallback
- poor architectural steady state

#### Option C: Add dedicated scale storage

Pros:

- matches what scaled matmul actually wants: small structured side data
- supports per-channel and block scaling cleanly
- keeps control scalars and dense tiles separate
- avoids wasting full tensor registers

Cons:

- introduces another storage class and instructions
- requires careful scope control to avoid recreating a second full tensor register file

Conclusion:

- best long-term architectural answer if Penguin wants scaled FP8 matmul to be a real
  feature rather than a toy extension

### 5.4 Recommendation

Recommendation: do not use scalar `x` registers as the architectural home for scaled
matmul factors, except possibly in an extremely restricted bring-up mode.

Recommendation: also do not use full `m` registers as the steady-state home for scales.

Preferred direction:

- add dedicated architectural scale storage
- keep that scale storage distinct from both scalar `x` registers and tensor `m`
  registers
- let MXU instructions reference scale operands explicitly

The architecture has since chosen a simpler first cut than the ideal block-scale design
discussed above:

- `32` architectural `e` registers
- each `e` register stores one `FP8_E8M0` scale
- each `e` register applies to one whole tensor operand
- scalar-side instructions load `e` registers from immediate or memory

That choice is less flexible than a packed block-scale buffer, but it still captures the
main architectural point of this study: scale values deserve their own storage class
rather than living in scalar `x` registers or full tensor `m` registers.

## 6. Question 3: Rectangular or Square MXU?

### 6.1 Why Penguin’s current rectangular idea is appealing

Penguin’s current rectangular shape was motivated by byte symmetry:

- FP8 inputs are 1 byte each
- BF16 outputs are 2 bytes each
- therefore feeding 32 FP8 values and producing 16 BF16 values matches byte bandwidth

That is a defensible local optimization if the only target mode is:

- FP8 inputs
- BF16 outputs
- fixed operand shape

The problem is that the geometry becomes entangled with one datatype pairing.

As soon as Penguin adds:

- FP8 output mode
- INT8 mode
- different accumulation/storage policies
- different epilogues

the rectangular geometry starts imposing awkward tiling rules.

### 6.2 What Google publicly does

Google publicly documents square MXUs:

- `128 x 128` before v6e
- `256 x 256` for v6e and TPU7x

At the same time, Google also publicly documents mixed-precision behavior:

- BF16 inputs with FP32 accumulation
- TPU7x exposes both BF16 and FP8 peak rates

The public evidence therefore points to an important architectural principle:

- TPU compute-array geometry is not chosen to match output byte width to input byte width
- reduced-precision storage and mixed-precision accumulation are handled without changing
  the systolic-array aspect ratio

This is the strongest single argument against Penguin’s rectangular rationale.

### 6.3 Why square arrays remain attractive

A square MXU has several practical benefits:

- simpler tiling and mental model
- cleaner transposition symmetry
- easier reuse across precisions
- easier compiler reasoning for `M`, `N`, and `K` blocking
- easier future scaling of the array size

Most importantly, square geometry decouples:

- compute topology
- result storage format

That decoupling matters because result type and accumulator type often differ. Google’s
TPU docs are already an existence proof of this pattern: reduced-precision inputs and
wider accumulators do not force a rectangular array.

### 6.4 How to handle different precisions without a rectangular array

The cleanest approach is:

- keep the compute array square
- let accumulation happen in the architecturally preferred accumulation type
- make output-format conversion an explicit epilogue or store-time decision

In other words:

- compute geometry should be chosen for dataflow and utilization
- storage geometry should be chosen for bandwidth and software layout

Those are different layers.

For Penguin, that suggests:

- a square MXU for contraction
- BF16 or wider accumulation as the default architecturally visible result
- optional post-matmul conversion to FP8 if and when FP8 output mode is added

That conversion can live in one of three places:

1. a VPU epilogue instruction
2. an MXU-integrated “convert-and-write” mode
3. a store-side pack/quantize instruction

From an architecture cleanliness standpoint, option 1 or 3 is better than baking too
many output modes into the core systolic array semantics.

### 6.5 Recommendation

Recommendation: move Penguin back toward a square MXU.

Rationale:

- it matches the public TPU design trend
- it avoids coupling array geometry to one datatype pair
- it scales better to FP8 output support
- it simplifies the long-term compiler and ISA story

Recommended precision policy:

- choose one square compute tile shape first
- keep one canonical accumulation type first
- treat narrower result storage as an explicit conversion / packing step

If Penguin wants a very simple first tensor slice, the best baseline is:

- square FP8-input MXU
- BF16 accumulation and BF16 architectural MXU result
- FP8 output added later through an explicit conversion path rather than through a
  rectangular MXU

## 7. Consolidated Recommendations

### 7.1 Recommended near-term direction

For Penguin’s next tensor-architecture revision:

1. Keep one main tensor register file. Do not add a full 64-entry architectural vector
   register file yet.
2. Add narrow-data support through semantics first:
   - broadcast-friendly VPU / epilogue operations
   - reduction-to-packed-tile conventions
   - possibly a very small side vector store later if profiling demands it
3. Introduce dedicated scale storage for scaled matmul rather than reusing scalar `x`
   registers or full `m` registers. The adopted first cut is a `32`-entry `e`-register
   file with one `FP8_E8M0` whole-tensor scale per register.
4. Move the MXU plan back toward square geometry and decouple result-format conversion
   from compute-array shape. The adopted first cut is a `64 x 64` FP8 MXU whose `64 x 64`
   BF16 result is written back as two consecutive `64 x 32` BF16 tensor-register halves.

### 7.2 Suggested concrete Penguin policy

If the goal is a practical and stable first tensor ISA:

- `m` registers remain the only general tensor register class
- reductions write into canonical packed rows of `m` registers
- bias application uses broadcast semantics from packed rows or from VMEM streams
- scales are loaded into dedicated `e` registers from immediate or VMEM
- MXU remains square at `64 x 64` and produces BF16 results first
- one full BF16 MXU result occupies two consecutive `m` registers
- FP8 result emission is a later explicit conversion step

## 8. Confidence and Gaps

### 8.1 High-confidence conclusions

These are directly supported by official/public sources:

- Google publicly documents TensorCores as MXU + vector unit + scalar unit
- Google publicly documents square MXUs across generations
- Google publicly documents mixed-precision accumulation without changing MXU geometry
- Google’s current quantized / scaled matmul software surfaces model scales as separate
  arrays, not as hidden scalar registers

### 8.2 Lower-confidence conclusions

These are informed inferences, not explicit public statements from Google:

- that Google intentionally avoids exposing a separate vector-register file because the
  added architectural surface is not worth it for common TensorCore workloads
- that scale metadata should map to a dedicated small architectural store rather than a
  full tensor register file
- that Penguin’s rectangular MXU should be retired rather than kept as a special-case
  optimization

I judge these inferences to be strong because they align with both:

- Google’s publicly exposed programming model
- the current pain points in Penguin’s own proposed ISA and tiling model

## 9. Bottom Line

If Penguin wants the cleanest path forward:

- do not add a full vector-register file yet
- do add dedicated scale storage if scaled matmul is real
- do not let byte-bandwidth symmetry force a rectangular MXU
- keep the compute array square and make output precision a conversion problem, not a
  geometry problem
