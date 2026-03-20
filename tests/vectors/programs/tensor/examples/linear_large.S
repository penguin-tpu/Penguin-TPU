# DMA-backed stripmined linear layer: 192x128 input @ 128x192 weights + 192 bias -> 192x192 BF16 output.

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
    addi x7, x0, 3
    addi x12, x0, 4096
    addi x13, x0, 4096
    addi x14, x0, 8192
    addi x15, x0, 536887296
    addi x16, x0, 2185232384
    addi x17, x0, 536895488
    addi x18, x0, 536903680
    addi x30, x0, 8192

    seli e0, 0
    seli e1, 0
    dma.config.ch0 x0
    dma.load.ch2 x15, x16, x30
    addi x16, x16, 8192
    delay 0
    dma.load.ch3 x17, x16, x30
    addi x16, x16, 8192
    delay 0
    dma.load.ch4 x18, x16, x30
    dma.wait.ch4
    dma.wait.ch3
    dma.wait.ch2

m_loop:
    addi x8, x0, 3
    addi x17, x15, 0
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
    vload m8, 0(x5)
    delay 63
    vmatpush.weight.mxu0 w0, m8
    delay 63
    vmatmul.mxu0 m1, w0
    addi x10, x10, 4096
    addi x11, x11, 12288
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
    vload m8, 0(x5)
    delay 63
    vmatpush.weight.mxu0 w0, m8
    delay 63
    vmatmul.acc.mxu0 m1, w0
    addi x10, x10, 4096
    addi x11, x11, 12288
    addi x9, x9, -1
    jal x0, k_loop
    addi x0, x0, 0
    addi x0, x0, 0
k_done:
    delay 57
    vmatpop.bf16.acc.mxu0 m2
    vload m10, 0(x17)
    vload m11, 4096(x17)
    delay 125
    vadd m2, m2, m10
    vadd m3, m3, m11
    delay 0
    vstore m2, 0(x6)
    vstore m3, 4096(x6)
    delay 63
    dma.store.ch2 x19, x6, x14
    dma.wait.ch2
    addi x17, x17, 8192
    addi x18, x18, 4096
    addi x19, x19, 8192
    addi x8, x8, -1
    delay 0
    bne x8, x0, n_loop
    addi x0, x0, 0
    addi x0, x0, 0
    addi x1, x1, 8192
    addi x3, x3, 24576
    addi x7, x7, -1
    delay 0
    bne x7, x0, m_loop
    addi x0, x0, 0
    addi x0, x0, 0
