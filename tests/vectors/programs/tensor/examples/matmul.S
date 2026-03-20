# DRAM-backed single MXU tile matmul:
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
    addi x2, x0, 2181038080
    addi x3, x0, 2197815296
    addi x4, x0, 536870912
    addi x5, x0, 536875008
    addi x6, x0, 536879104
    addi x12, x0, 4096
    addi x13, x0, 4096
    addi x14, x0, 8192

    seli e0, 0
    seli e1, 0
    dma.config.ch0 x0
    dma.load.ch0 x4, x1, x12
    dma.load.ch1 x5, x2, x13
    dma.wait.ch1
    dma.wait.ch0
    vload m1, 0(x4)
    vload.weight.mxu0 w0, x5
    delay 63
    vmatmul.mxu0 m1, w0
    delay 63
    vmatpop.bf16.acc.mxu0 m2
    delay 127
    vstore m2, 0(x6)
    vstore m3, 4096(x6)
    delay 63
    dma.store.ch2 x3, x6, x14
    dma.wait.ch2
