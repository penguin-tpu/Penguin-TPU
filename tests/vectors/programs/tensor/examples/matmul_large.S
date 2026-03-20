# DMA-backed stripmined matmul: 128x128 activation @ 128x128 weights -> 128x128 BF16 output.

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

    addi x1, x0, 2164260864
    addi x2, x0, 2181038080
    addi x3, x0, 2197815296
    addi x4, x0, 536870912
    addi x5, x0, 536875008
    addi x6, x0, 536879104
    addi x7, x0, 2
    addi x12, x0, 4096
    addi x13, x0, 4096
    addi x14, x0, 8192

    seli e0, 0
    seli e1, 0
    dma.config.ch0 x0

m_loop:
    addi x8, x0, 2
    addi x18, x2, 0
    addi x19, x3, 0
n_loop:
    addi x9, x0, 2
    addi x10, x1, 0
    addi x11, x18, 0

    dma.load.ch0 x4, x10, x12
    dma.load.ch1 x5, x11, x13
    dma.wait.ch1
    dma.wait.ch0
    vload m1, 0(x4)
    vload.weight.mxu0 w0, x5
    delay 63
    vmatmul.mxu0 m1, w0
    addi x10, x10, 4096
    addi x11, x11, 8192
    addi x9, x9, -1
k_loop:
    delay 0
    beq x9, x0, k_done
    addi x0, x0, 0
    addi x0, x0, 0
    dma.load.ch0 x4, x10, x12
    dma.load.ch1 x5, x11, x13
    dma.wait.ch1
    dma.wait.ch0
    vload m1, 0(x4)
    vload.weight.mxu0 w0, x5
    delay 63
    vmatmul.acc.mxu0 m1, w0
    addi x10, x10, 4096
    addi x11, x11, 8192
    addi x9, x9, -1
    jal x0, k_loop
    addi x0, x0, 0
    addi x0, x0, 0
k_done:
    delay 57
    vmatpop.bf16.acc.mxu0 m2
    delay 127
    vstore m2, 0(x6)
    vstore m3, 4096(x6)
    delay 63
    dma.store.ch2 x19, x6, x14
    dma.wait.ch2
    addi x18, x18, 4096
    addi x19, x19, 8192
    addi x8, x8, -1
    delay 0
    bne x8, x0, n_loop
    addi x0, x0, 0
    addi x0, x0, 0
    addi x1, x1, 8192
    addi x3, x3, 16384
    addi x7, x7, -1
    delay 0
    bne x7, x0, m_loop
    addi x0, x0, 0
    addi x0, x0, 0
