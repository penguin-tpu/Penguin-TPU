    li x10, 0x80000100
    li x11, 0x20000000
    li x12, 0x00000020  # 20 B
    dma.load.ch0 x10, x11, x12

    li x13, 0x80001000
    li x14, 0x20001000
    li x15, 0x00002000  # 8 kB
    dma.load.ch1 x13, x14, x15

    li x16, 0x80004000
    li x17, 0x20004000
    li x18, 0x00003000  # 12 kB
    dma.load.ch2 x16, x17, x18

    dma.wait.ch0
    li x19, 0x20000080
    sld x1, 0x00000000(x11)
    sst x1, 0x00000000(x19)
    sld x2, 0x00000004(x11)
    sst x2, 0x00000004(x19)
    sld x3, 0x00000008(x11)
    sst x3, 0x00000008(x19)
    sld x4, 0x0000000c(x11)
    sst x4, 0x0000000c(x19)
    sld x5, 0x00000010(x11)
    sst x5, 0x00000010(x19)
    sld x6, 0x00000014(x11)
    sst x6, 0x00000014(x19)
    sld x7, 0x00000018(x11)
    sst x7, 0x00000018(x19)
    sld x8, 0x0000001c(x11)
    sst x8, 0x0000001c(x19)

    dma.wait.ch1
    li x20, 0x20002000
    vload m0, 0x00000000(x14)
    vstore m0, 0x00000000(x20)
    vload m1, 0x00001000(x14)
    vstore m1, 0x00001000(x20)

    dma.wait.ch2
    li x21, 0x20008000
    vload m2, 0x00000000(x17)
    vstore m2, 0x00000000(x21)
    vload m3, 0x00001000(x17)
    vstore m3, 0x00001000(x21)
    vload m4, 0x00002000(x17)
    vstore m4, 0x00002000(x21)

    li x22, 0x80007000
    li x23, 0x00000020
    dma.store.ch0 x22, x19, x23

    li x24, 0x80008000
    li x25, 0x00002000
    dma.store.ch1 x24, x20, x25

    li x26, 0x8000b000
    li x27, 0x00003000
    dma.store.ch2 x26, x21, x27

    dma.wait.ch0
    dma.wait.ch1
    dma.wait.ch2
