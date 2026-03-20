# DRAM-backed tiled linear layer with bias:
# DRAM -> VMEM -> MREG -> compute -> MREG -> VMEM -> DRAM.

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
    addi x2, x0, 2164264960
    addi x10, x0, 2181038080
    addi x11, x0, 2181042176
    addi x12, x0, 2185232384
    addi x13, x0, 2185240576
    addi x4, x0, 536870912
    addi x5, x0, 536875008
    addi x6, x0, 536879104
    addi x7, x0, 536883200
    addi x14, x0, 536887296
    addi x15, x0, 536895488
    addi x20, x0, 2197815296
    addi x21, x0, 2197823488
    addi x22, x0, 2197831680
    addi x23, x0, 2197839872
    addi x24, x0, 536911872
    addi x25, x0, 536920064
    addi x26, x0, 536928256
    addi x27, x0, 536936448
    addi x28, x0, 4096
    addi x29, x0, 8192

    seli e0, 0
    seli e1, 0
    dma.config.ch0 x0
    dma.load.ch0 x4, x1, x28
    dma.load.ch1 x5, x2, x28
    dma.load.ch2 x6, x10, x28
    dma.load.ch3 x7, x11, x28
    dma.load.ch4 x14, x12, x29
    dma.load.ch5 x15, x13, x29
    dma.wait.ch5
    dma.wait.ch4
    dma.wait.ch3
    dma.wait.ch2
    dma.wait.ch1
    dma.wait.ch0
    vload m6, 0(x6)
    vload m7, 0(x7)
    vload m10, 0(x14)
    vload m11, 4096(x14)
    vload m12, 0(x15)
    vload m13, 4096(x15)

    vload m1, 0(x4)
    delay 63
    vmatpush.weight.mxu0 w0, m6
    vmatpush.weight.mxu0 w1, m7
    delay 63
    vmatmul.mxu0 m1, w0
    delay 63
    vmatpop.bf16.acc.mxu0 m20
    delay 127
    vadd m20, m20, m10
    vadd m21, m21, m11
    delay 0
    vstore m20, 0(x24)
    vstore m21, 4096(x24)
    vmatmul.mxu0 m1, w1
    delay 63
    vmatpop.bf16.acc.mxu0 m22
    delay 127
    vadd m22, m22, m12
    vadd m23, m23, m13
    delay 0
    vstore m22, 0(x25)
    vstore m23, 4096(x25)

    delay 63
    vload m4, 0(x5)
    delay 63
    vmatmul.mxu0 m4, w0
    delay 63
    vmatpop.bf16.acc.mxu0 m24
    delay 127
    vadd m24, m24, m10
    vadd m25, m25, m11
    delay 0
    vstore m24, 0(x26)
    vstore m25, 4096(x26)
    vmatmul.mxu0 m4, w1
    delay 63
    vmatpop.bf16.acc.mxu0 m26
    delay 127
    vadd m26, m26, m12
    vadd m27, m27, m13
    delay 0
    vstore m26, 0(x27)
    vstore m27, 4096(x27)

    delay 63
    dma.store.ch0 x20, x24, x29
    dma.store.ch1 x21, x25, x29
    dma.store.ch2 x22, x26, x29
    dma.store.ch3 x23, x27, x29
    dma.wait.ch3
    dma.wait.ch2
    dma.wait.ch1
    dma.wait.ch0
