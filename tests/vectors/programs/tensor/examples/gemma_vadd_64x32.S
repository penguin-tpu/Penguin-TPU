# Fixed-shape Gemma residual tile: add two 64x32 BF16 tensors.

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
    li x2, VMEM_BASE + 0x0800
    li x3, VMEM_BASE + 0x1000
    li x4, VMEM_BASE + 0x1800
    li x20, VMEM_BASE + 0x2000
    li x21, VMEM_BASE + 0x2800

    vload m1, 0(x1)
    vload m2, 0(x2)
    vload m3, 0(x3)
    vload m4, 0(x4)

    vadd m5, m1, m3
    vstore m5, 0(x20)
    vadd m6, m2, m4
    vstore m6, 0(x21)
