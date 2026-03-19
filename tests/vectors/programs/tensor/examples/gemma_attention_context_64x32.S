# Fixed-shape Gemma attention context tile: 64x32 FP8 probs @ 32x32 FP8 values -> 64x32 BF16 context.

    # Startup scalar-register scrub: software must not assume power-on register contents.
    addi x1, x0, 0
    addi x2, x0, 0
    addi x3, x0, 0
    addi x4, x0, 0
    addi x5, x0, 0
    addi x6, x0, 0
    addi x7, x0, 0
    addi x8, x0, 0
    addi x9, x0, 0
    addi x10, x0, 0
    addi x11, x0, 0
    addi x12, x0, 0
    addi x13, x0, 0
    addi x14, x0, 0
    addi x15, x0, 0
    addi x16, x0, 0
    addi x17, x0, 0
    addi x18, x0, 0
    addi x19, x0, 0
    addi x20, x0, 0
    addi x21, x0, 0
    addi x22, x0, 0
    addi x23, x0, 0
    addi x24, x0, 0
    addi x25, x0, 0
    addi x26, x0, 0
    addi x27, x0, 0
    addi x28, x0, 0
    addi x29, x0, 0
    addi x30, x0, 0
    addi x31, x0, 0

    addi x1, x0, 536870912
    addi x10, x0, 536875008
    addi x20, x0, 536879104

    seli e0, 0
    seli e1, 0
    vload m1, 0(x1)
    vload.weight.mxu0 w0, x10
    delay 63
    vmatmul.mxu0 m1, w0
    delay 63
    vmatpop.bf16.acc.mxu0 m2
    delay 127
    vstore m2, 0(x20)
