# Tiled linear layer with bias: 128x32 input @ 32x32 weights + 32 bias -> 128x32 BF16 output.

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
    li x10, VMEM_BASE + 0x1000
    li x11, VMEM_BASE + 0x1200
    li x12, VMEM_BASE + 0x1800
    li x13, VMEM_BASE + 0x2000
    li x20, VMEM_BASE + 0x3000
    li x21, VMEM_BASE + 0x3800
    li x22, VMEM_BASE + 0x4000
    li x23, VMEM_BASE + 0x4800

    mxu.push.mxu0 w0, 0(x10)
    mxu.push.mxu0 w1, 0(x11)
    vload m10, 0(x12)
    vload m11, 0(x13)

    vload m1, 0(x1)
    matmul.mxu0 m2, m1, w0
    vadd m2, m2, m10
    vstore m2, 0(x20)
    matmul.mxu0 m3, m1, w1
    vadd m3, m3, m11
    vstore m3, 0(x21)

    vload m4, 0(x2)
    matmul.mxu0 m5, m4, w0
    vadd m5, m5, m10
    vstore m5, 0(x22)
    matmul.mxu0 m6, m4, w1
    vadd m6, m6, m11
    vstore m6, 0(x23)
