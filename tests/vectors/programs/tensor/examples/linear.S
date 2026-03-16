# Tiled linear layer without bias: 128x32 input @ 32x32 weights -> 128x32 BF16 output.

    li x1, VMEM_BASE + 0x0000
    li x2, VMEM_BASE + 0x0800
    li x10, VMEM_BASE + 0x1000
    li x11, VMEM_BASE + 0x1200
    li x20, VMEM_BASE + 0x2000
    li x21, VMEM_BASE + 0x2800
    li x22, VMEM_BASE + 0x3000
    li x23, VMEM_BASE + 0x3800

    mxu.push.mxu0 w0, 0(x10)
    mxu.push.mxu0 w1, 0(x11)

    vload m1, 0(x1)
    matmul.mxu0 m2, m1, w0
    vstore m2, 0(x20)
    matmul.mxu0 m3, m1, w1
    vstore m3, 0(x21)

    vload m4, 0(x2)
    matmul.mxu0 m5, m4, w0
    vstore m5, 0(x22)
    matmul.mxu0 m6, m4, w1
    vstore m6, 0(x23)
