# DMA-backed stripmined linear layer: 192x64 input @ 64x48 weights -> 192x48 BF16 output.

    li x1, DRAM_BASE + 0x01000000
    li x2, DRAM_BASE + 0x02000000
    li x3, DRAM_BASE + 0x03000000
    li x4, VMEM_BASE + 0x0000
    li x5, VMEM_BASE + 0x0800
    li x6, VMEM_BASE + 0x1000
    li x7, 0x00000003
    li x12, 0x00000800
    li x13, 0x00000200
    li x14, 0x00000600

m_loop:
    li x8, 0x00000003
    saddi x18, x2, 0x00000000
    saddi x19, x3, 0x00000000
n_loop:
    li x9, 0x00000002
    saddi x10, x1, 0x00000000
    saddi x11, x18, 0x00000000

    dma.load.ch0 x10, x4, x12
    dma.load.ch1 x11, x5, x13
    dma.wait.ch1
    dma.wait.ch0
    vload m1, 0(x4)
    mxu.push.mxu0 w0, 0(x5)
    matmul.mxu0 m2, m1, w0
    saddi x10, x10, 0x00000800
    saddi x11, x11, 0x00000600
    saddi x9, x9, -1
k_loop:
    sbeq x9, x0, k_done
    nop
    nop
    dma.load.ch0 x10, x4, x12
    dma.load.ch1 x11, x5, x13
    dma.wait.ch1
    dma.wait.ch0
    vload m1, 0(x4)
    mxu.push.mxu0 w0, 0(x5)
    matmul.add.mxu0 m2, m1, w0, m2
    saddi x10, x10, 0x00000800
    saddi x11, x11, 0x00000600
    saddi x9, x9, -1
    sjal x0, k_loop
    nop
    nop
k_done:
    vstore m2, 0(x6)
    dma.store.ch2 x19, x6, x12
    dma.wait.ch2
    saddi x18, x18, 0x00000200
    saddi x19, x19, 0x00000800
    saddi x8, x8, -1
    sbne x8, x0, n_loop
    nop
    nop
    saddi x1, x1, 0x00001000
    saddi x3, x3, 0x00001800
    saddi x7, x7, -1
    sbne x7, x0, m_loop
    nop
    nop
