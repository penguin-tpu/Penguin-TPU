# Fixed-shape Gemma attention score tile: 64x32 FP8 query @ 32x64 FP8 key^T -> 64x64 BF16 scores.

    # Startup scalar-register scrub: software must not assume power-on register contents.
    saddi x1, x0, 0
    saddi x2, x0, 0
    saddi x3, x0, 0
    saddi x4, x0, 0
    saddi x5, x0, 0
    saddi x6, x0, 0
    saddi x7, x0, 0
    saddi x8, x0, 0
    saddi x9, x0, 0
    saddi x10, x0, 0
    saddi x11, x0, 0
    saddi x12, x0, 0
    saddi x13, x0, 0
    saddi x14, x0, 0
    saddi x15, x0, 0
    saddi x16, x0, 0
    saddi x17, x0, 0
    saddi x18, x0, 0
    saddi x19, x0, 0
    saddi x20, x0, 0
    saddi x21, x0, 0
    saddi x22, x0, 0
    saddi x23, x0, 0
    saddi x24, x0, 0
    saddi x25, x0, 0
    saddi x26, x0, 0
    saddi x27, x0, 0
    saddi x28, x0, 0
    saddi x29, x0, 0
    saddi x30, x0, 0
    saddi x31, x0, 0

    li x1, VMEM_BASE + 0x0000
    li x10, VMEM_BASE + 0x1000
    li x11, VMEM_BASE + 0x2000
    li x20, VMEM_BASE + 0x3000
    li x21, VMEM_BASE + 0x4000

    seli e0, 0
    seli e1, 0
    vload m1, 0(x1)
    mxu.push.mxu0 w0, 0(x10)
    mxu.push.mxu0 w1, 0(x11)

    matmul.mxu0 m2, m1, w0, e0, e1
    vstore m2, 0(x20)
    matmul.mxu0 m4, m1, w1, e0, e1
    vstore m4, 0(x21)
