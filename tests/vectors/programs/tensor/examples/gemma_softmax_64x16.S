# Fixed-shape Gemma softmax tile: row-wise BF16 softmax over one 64x16 score tile.

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
    li x20, VMEM_BASE + 0x2000

    vload m1, 0(x1)
    reduce.max.xlu m2, m1
    vsub m3, m1, m2
    vexp m4, m3
    reduce.sum.xlu m5, m4
    vrecip m6, m5
    vmul m7, m4, m6
    vstore m7, 0(x20)
