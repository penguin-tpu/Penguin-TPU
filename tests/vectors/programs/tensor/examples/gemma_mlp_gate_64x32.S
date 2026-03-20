# Fixed-shape Gemma MLP gating tile: tanh-approx GELU(gate) * up over one 64x32 BF16 tile pair.

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
    addi x3, x0, 536875008
    addi x5, x0, 536879104
    addi x6, x0, 536883200
    addi x7, x0, 536887296
    addi x8, x0, 536891392
    addi x9, x0, 536895488
    addi x10, x0, 536899584
    addi x20, x0, 536903680

    vload m1, 0(x1)
    vload m3, 0(x3)
    vload m10, 0(x5)
    vload m11, 0(x6)
    vload m12, 0(x7)
    vload m13, 0(x8)
    vload m14, 0(x9)
    vload m15, 0(x10)

    delay 63
    vmul m5, m1, m1
    delay 2
    vmul m6, m5, m1
    delay 2
    vmul m7, m6, m15
    delay 2
    vadd m8, m1, m7
    delay 2
    vmul m9, m8, m14
    delay 2
    vmul m16, m9, m12
    delay 2
    vsub m17, m10, m16
    delay 2
    vexp m18, m17
    delay 8
    vadd m19, m11, m18
    delay 2
    vrecip m20, m19
    delay 8
    vmul m21, m20, m12
    delay 2
    vsub m22, m21, m11
    delay 2
    vadd m23, m22, m11
    vmul m24, m1, m13
    delay 2
    vmul m25, m24, m23
    delay 2
    vmul m26, m25, m3
    delay 2
    vstore m26, 0(x20)
