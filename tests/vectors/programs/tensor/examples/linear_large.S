# DMA-backed stripmined linear layer: 192x128 input @ 128x192 weights + 192 bias -> 192x192 BF16 output.

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

    li x1, DRAM_BASE + 0x01000000
    li x2, DRAM_BASE + 0x02000000
    li x3, DRAM_BASE + 0x03000000
    li x4, VMEM_BASE + 0x0000
    li x5, VMEM_BASE + 0x1000
    li x6, VMEM_BASE + 0x2000
    li x7, 0x00000003
    li x12, 0x00001000
    li x13, 0x00001000
    li x14, 0x00002000
    li x15, VMEM_BASE + 0x4000

    seli e0, 0
    seli e1, 0

m_loop:
    li x8, 0x00000003
    saddi x17, x15, 0x00000000
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
    vload.weight.mxu0 w0, x5
    vmatmul.mxu0 m1, w0
    saddi x10, x10, 0x00001000
    saddi x11, x11, 0x00003000
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
    vload.weight.mxu0 w0, x5
    vmatmul.acc.mxu0 m1, w0
    saddi x10, x10, 0x00001000
    saddi x11, x11, 0x00003000
    saddi x9, x9, -1
    sjal x0, k_loop
    nop
    nop
k_done:
    vmatpop.bf16.acc.mxu0 m2
    vload m10, 0(x17)
    vload m11, 0x1000(x17)
    vadd m2, m2, m10
    vadd m3, m3, m11
    vstore m2, 0(x6)
    vstore m3, 0x1000(x6)
    dma.store.ch2 x19, x6, x14
    dma.wait.ch2
    saddi x17, x17, 0x00002000
    saddi x18, x18, 0x00001000
    saddi x19, x19, 0x00002000
    saddi x8, x8, -1
    sbne x8, x0, n_loop
    nop
    nop
    saddi x1, x1, 0x00002000
    saddi x3, x3, 0x00006000
    saddi x7, x7, -1
    sbne x7, x0, m_loop
    nop
    nop
