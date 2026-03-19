# Fixed-shape Gemma softmax tile: row-wise BF16 softmax over one 64x16 score tile.

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

    addi x1, x0, 536870912
    addi x20, x0, 536879104

    vload m1, 0(x1)
    delay 63
    reduce.max.xlu m2, m1
    delay 3
    vsub m3, m1, m2
    delay 1
    vexp m4, m3
    delay 7
    reduce.sum.xlu m5, m4
    delay 3
    vrecip m6, m5
    delay 7
    vmul m7, m4, m6
    delay 1
    vstore m7, 0(x20)
