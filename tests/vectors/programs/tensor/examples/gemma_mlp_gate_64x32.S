# Fixed-shape Gemma MLP gating tile: tanh-approx GELU(gate) * up over one 64x32 BF16 tile pair.

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

    li x1, VMEM_BASE + 0x0000
    li x3, VMEM_BASE + 0x1000
    li x5, VMEM_BASE + 0x2000
    li x6, VMEM_BASE + 0x3000
    li x7, VMEM_BASE + 0x4000
    li x8, VMEM_BASE + 0x5000
    li x9, VMEM_BASE + 0x6000
    li x10, VMEM_BASE + 0x7000
    li x20, VMEM_BASE + 0x8000

    vload m1, 0(x1)
    vload m3, 0(x3)
    vload m10, 0(x5)
    vload m11, 0(x6)
    vload m12, 0(x7)
    vload m13, 0(x8)
    vload m14, 0(x9)
    vload m15, 0(x10)

    vmul m5, m1, m1
    vmul m6, m5, m1
    vmul m7, m6, m15
    vadd m8, m1, m7
    vmul m9, m8, m14
    vmul m16, m9, m12
    vsub m17, m10, m16
    vexp m18, m17
    vadd m19, m11, m18
    vrecip m20, m19
    vmul m21, m20, m12
    vsub m22, m21, m11
    vadd m23, m22, m11
    vmul m24, m1, m13
    vmul m25, m24, m23
    vmul m26, m25, m3
    vstore m26, 0(x20)
