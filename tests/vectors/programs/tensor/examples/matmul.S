# Single MXU tile matmul: 64x32 activations @ 32x16 weights -> 64x16 BF16 output.

    li x1, VMEM_BASE + 0x0000
    li x2, VMEM_BASE + 0x1000
    li x3, VMEM_BASE + 0x2000

    vload m1, 0(x1)
    mxu.push.mxu0 w0, 0(x2)
    matmul.mxu0 m2, m1, w0
    vstore m2, 0(x3)
