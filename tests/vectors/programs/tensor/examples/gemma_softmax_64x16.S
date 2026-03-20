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
    addi x21, x0, 536887296
    addi x22, x0, 536891392
    addi x23, x0, 536895488
    addi x24, x0, 536899584

    vload m1, 0(x1)
    delay 63
    vreduce.max.xlu m2, m1
    delay 3
    vstore m2, 0(x21)
    delay 63

    addi x2, x21, 0
    addi x3, x22, 0
    addi x4, x0, 64
max_row_loop:
    slh x5, 0(x2)
    addi x6, x0, 32
    addi x7, x3, 0
max_col_loop:
    ssh x5, 0(x7)
    addi x7, x7, 2
    addi x6, x6, -1
    sbne x6, x0, max_col_loop
    addi x0, x0, 0
    addi x0, x0, 0
    addi x2, x2, 64
    addi x3, x3, 64
    addi x4, x4, -1
    sbne x4, x0, max_row_loop
    addi x0, x0, 0
    addi x0, x0, 0

    vload m2, 0(x22)
    delay 63
    vsub.bf16 m3, m1, m2
    delay 2
    vexp m4, m3
    delay 8
    vreduce.sum.xlu m5, m4
    delay 3
    vstore m5, 0(x23)
    delay 63

    addi x8, x23, 0
    addi x9, x24, 0
    addi x10, x0, 64
sum_row_loop:
    slh x11, 0(x8)
    addi x12, x0, 32
    addi x13, x9, 0
sum_col_loop:
    ssh x11, 0(x13)
    addi x13, x13, 2
    addi x12, x12, -1
    sbne x12, x0, sum_col_loop
    addi x0, x0, 0
    addi x0, x0, 0
    addi x8, x8, 64
    addi x9, x9, 64
    addi x10, x10, -1
    sbne x10, x0, sum_row_loop
    addi x0, x0, 0
    addi x0, x0, 0

    vload m6, 0(x24)
    delay 63
    vrecip.bf16 m6, m6
    delay 8
    vmul.bf16 m7, m4, m6
    delay 2
    vstore m7, 0(x20)
