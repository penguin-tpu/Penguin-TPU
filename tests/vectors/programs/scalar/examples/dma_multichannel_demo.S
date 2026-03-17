    li x10, 0x80000100
    li x11, 0x08000040
    li x12, 0x00000060
    dma.load.ch0 x10, x11, x12

    li x13, 0x80000400
    li x14, 0x08000180
    li x15, 0x00000020
    dma.store.ch1 x13, x14, x15

    li x16, 0x80000300
    li x17, 0x08000200
    li x18, 0x00000040
    dma.load.ch2 x16, x17, x18

    dma.wait.ch1
    li x19, 0x80000500
    li x20, 0x08000100
    li x21, 0x00000020
    dma.load.ch1 x19, x20, x21

    dma.wait.ch2
    li x22, 0x80000600
    li x23, 0x08000280
    li x24, 0x00000040
    dma.store.ch2 x22, x23, x24

    dma.wait.ch1
    sld x1, 0x00000000(x20)
    sst x1, 0x08000300(x0)

    dma.wait.ch2
    dma.wait.ch0

    sld x2, 0x00000000(x17)
    sst x2, 0x08000304(x0)

    sld x3, 0x00000000(x11)
    sst x3, 0x08000308(x0)
