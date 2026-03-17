# Tiled linear layer with bias: 128x64 input @ 64x128 weights + 128 bias -> 128x128 BF16 output.

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
    li x2, VMEM_BASE + 0x1000
    li x10, VMEM_BASE + 0x2000
    li x11, VMEM_BASE + 0x3000
    li x12, VMEM_BASE + 0x4000
    li x13, VMEM_BASE + 0x6000
    li x20, VMEM_BASE + 0xA000
    li x21, VMEM_BASE + 0xC000
    li x22, VMEM_BASE + 0xE000
    li x23, VMEM_BASE + 0x10000

    seli e0, 0
    seli e1, 0
    mxu.push.mxu0 w0, 0(x10)
    mxu.push.mxu0 w1, 0(x11)
    vload m10, 0(x12)
    vload m11, 0x1000(x12)
    vload m12, 0(x13)
    vload m13, 0x1000(x13)

    vload m1, 0(x1)
    matmul.mxu0 m20, m1, w0, e0, e1
    vadd m20, m20, m10
    vadd m21, m21, m11
    vstore m20, 0(x20)
    vstore m21, 0x1000(x20)
    matmul.mxu0 m22, m1, w1, e0, e1
    vadd m22, m22, m12
    vadd m23, m23, m13
    vstore m22, 0(x21)
    vstore m23, 0x1000(x21)

    vload m4, 0(x2)
    matmul.mxu0 m24, m4, w0, e0, e1
    vadd m24, m24, m10
    vadd m25, m25, m11
    vstore m24, 0(x22)
    vstore m25, 0x1000(x22)
    matmul.mxu0 m26, m4, w1, e0, e1
    vadd m26, m26, m12
    vadd m27, m27, m13
    vstore m26, 0(x23)
    vstore m27, 0x1000(x23)
