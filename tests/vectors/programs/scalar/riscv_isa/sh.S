# Flattened from upstream riscv-tests rv32ui/sh.S
#
# Penguin test-result convention for imported riscv-tests programs:
#   x10 = 0 on pass, 1 on fail
#   x3  = failing upstream test number on fail, 0 on pass
#   secall terminates the test and returns control to the host harness

test_2:
    saddi x3, x0, 2
    li x2, VMEM_BASE + 0x100
    saddi x1, x0, 170
    sh x1, 0(x2)
    lh x14, 0(x2)
    saddi x7, x0, 170
    sbne x14, x7, fail
    nop
    nop
test_3:
    saddi x3, x0, 3
    li x2, VMEM_BASE + 0x100
    slui x1, 0xffffb
    saddi x1, x1, -1536
    sh x1, 2(x2)
    lh x14, 2(x2)
    slui x7, 0xffffb
    saddi x7, x7, -1536
    sbne x14, x7, fail
    nop
    nop
test_4:
    saddi x3, x0, 4
    li x2, VMEM_BASE + 0x100
    slui x1, 0xbeef1
    saddi x1, x1, -1376
    sh x1, 4(x2)
    lw x14, 4(x2)
    slui x7, 0xbeef1
    saddi x7, x7, -1376
    sbne x14, x7, fail
    nop
    nop
test_5:
    saddi x3, x0, 5
    li x2, VMEM_BASE + 0x100
    slui x1, 0xffffa
    saddi x1, x1, 10
    sh x1, 6(x2)
    lh x14, 6(x2)
    slui x7, 0xffffa
    saddi x7, x7, 10
    sbne x14, x7, fail
    nop
    nop
test_6:
    saddi x3, x0, 6
    li x2, VMEM_BASE + 0x10e
    saddi x1, x0, 170
    sh x1, -6(x2)
    lh x14, -6(x2)
    saddi x7, x0, 170
    sbne x14, x7, fail
    nop
    nop
test_7:
    saddi x3, x0, 7
    li x2, VMEM_BASE + 0x10e
    slui x1, 0xffffb
    saddi x1, x1, -1536
    sh x1, -4(x2)
    lh x14, -4(x2)
    slui x7, 0xffffb
    saddi x7, x7, -1536
    sbne x14, x7, fail
    nop
    nop
test_8:
    saddi x3, x0, 8
    li x2, VMEM_BASE + 0x10e
    slui x1, 0x00001
    saddi x1, x1, -1376
    sh x1, -2(x2)
    lh x14, -2(x2)
    slui x7, 0x00001
    saddi x7, x7, -1376
    sbne x14, x7, fail
    nop
    nop
test_9:
    saddi x3, x0, 9
    li x2, VMEM_BASE + 0x10e
    slui x1, 0xffffa
    saddi x1, x1, 10
    sh x1, 0(x2)
    lh x14, 0(x2)
    slui x7, 0xffffa
    saddi x7, x7, 10
    sbne x14, x7, fail
    nop
    nop
test_12:
    saddi x3, x0, 12
    saddi x4, x0, 0
sh_test_12_loop_1:
    slui x13, 0xffffd
    saddi x13, x13, -803
    li x12, VMEM_BASE + 0x100
    sh x13, 0(x12)
    lh x14, 0(x12)
    slui x7, 0xffffd
    saddi x7, x7, -803
    sbne x14, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sh_test_12_loop_1
    nop
    nop
test_13:
    saddi x3, x0, 13
    saddi x4, x0, 0
sh_test_13_loop_2:
    slui x13, 0xffffc
    saddi x13, x13, -819
    li x12, VMEM_BASE + 0x100
    nop
    sh x13, 2(x12)
    lh x14, 2(x12)
    slui x7, 0xffffc
    saddi x7, x7, -819
    sbne x14, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sh_test_13_loop_2
    nop
    nop
test_14:
    saddi x3, x0, 14
    saddi x4, x0, 0
sh_test_14_loop_3:
    slui x13, 0xffffc
    saddi x13, x13, -1076
    li x12, VMEM_BASE + 0x100
    nop
    nop
    sh x13, 4(x12)
    lh x14, 4(x12)
    slui x7, 0xffffc
    saddi x7, x7, -1076
    sbne x14, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sh_test_14_loop_3
    nop
    nop
test_15:
    saddi x3, x0, 15
    saddi x4, x0, 0
sh_test_15_loop_4:
    slui x13, 0xffffb
    saddi x13, x13, -1092
    nop
    li x12, VMEM_BASE + 0x100
    sh x13, 6(x12)
    lh x14, 6(x12)
    slui x7, 0xffffb
    saddi x7, x7, -1092
    sbne x14, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sh_test_15_loop_4
    nop
    nop
test_16:
    saddi x3, x0, 16
    saddi x4, x0, 0
sh_test_16_loop_5:
    slui x13, 0xffffb
    saddi x13, x13, -1349
    nop
    li x12, VMEM_BASE + 0x100
    nop
    sh x13, 8(x12)
    lh x14, 8(x12)
    slui x7, 0xffffb
    saddi x7, x7, -1349
    sbne x14, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sh_test_16_loop_5
    nop
    nop
test_17:
    saddi x3, x0, 17
    saddi x4, x0, 0
sh_test_17_loop_6:
    slui x13, 0xffffe
    saddi x13, x13, -1365
    nop
    nop
    li x12, VMEM_BASE + 0x100
    sh x13, 10(x12)
    lh x14, 10(x12)
    slui x7, 0xffffe
    saddi x7, x7, -1365
    sbne x14, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sh_test_17_loop_6
    nop
    nop
test_18:
    saddi x3, x0, 18
    saddi x4, x0, 0
sh_test_18_loop_7:
    li x2, VMEM_BASE + 0x100
    slui x1, 0x00002
    saddi x1, x1, 563
    sh x1, 0(x2)
    lh x14, 0(x2)
    slui x7, 0x00002
    saddi x7, x7, 563
    sbne x14, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sh_test_18_loop_7
    nop
    nop
test_19:
    saddi x3, x0, 19
    saddi x4, x0, 0
sh_test_19_loop_8:
    li x2, VMEM_BASE + 0x100
    slui x1, 0x00001
    saddi x1, x1, 547
    nop
    sh x1, 2(x2)
    lh x14, 2(x2)
    slui x7, 0x00001
    saddi x7, x7, 547
    sbne x14, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sh_test_19_loop_8
    nop
    nop
test_20:
    saddi x3, x0, 20
    saddi x4, x0, 0
sh_test_20_loop_9:
    li x2, VMEM_BASE + 0x100
    slui x1, 0x00001
    saddi x1, x1, 290
    nop
    nop
    sh x1, 4(x2)
    lh x14, 4(x2)
    slui x7, 0x00001
    saddi x7, x7, 290
    sbne x14, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sh_test_20_loop_9
    nop
    nop
test_21:
    saddi x3, x0, 21
    saddi x4, x0, 0
sh_test_21_loop_10:
    li x2, VMEM_BASE + 0x100
    nop
    saddi x1, x0, 274
    sh x1, 6(x2)
    lh x14, 6(x2)
    saddi x7, x0, 274
    sbne x14, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sh_test_21_loop_10
    nop
    nop
test_22:
    saddi x3, x0, 22
    saddi x4, x0, 0
sh_test_22_loop_11:
    li x2, VMEM_BASE + 0x100
    nop
    saddi x1, x0, 17
    nop
    sh x1, 8(x2)
    lh x14, 8(x2)
    saddi x7, x0, 17
    sbne x14, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sh_test_22_loop_11
    nop
    nop
test_23:
    saddi x3, x0, 23
    saddi x4, x0, 0
sh_test_23_loop_12:
    li x2, VMEM_BASE + 0x100
    nop
    nop
    slui x1, 0x00003
    saddi x1, x1, 1
    sh x1, 10(x2)
    lh x14, 10(x2)
    slui x7, 0x00003
    saddi x7, x7, 1
    sbne x14, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sh_test_23_loop_12
    nop
    nop
    slui x10, 0x0000c
    saddi x10, x10, -273
    li x11, VMEM_BASE + 0x100
    sh x10, 6(x11)

test_pass:
    saddi x3, x0, 0
    saddi x10, x0, 0
    secall

fail:
    saddi x10, x0, 1
    secall
