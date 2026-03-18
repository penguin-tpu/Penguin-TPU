# Flattened from upstream riscv-tests rv32ui/sb.S
#
# Penguin test-result convention for imported riscv-tests programs:
#   x10 = 0 on pass, 1 on fail
#   x3  = failing upstream test number on fail, 0 on pass
#   secall terminates the test and returns control to the host harness

test_2:
    saddi x3, x0, 2
    li x2, VMEM_BASE + 0x100
    slui x1, 0x100000
    saddi x1, x1, -86
    sb x1, 0(x2)
    lb x14, 0(x2)
    slui x7, 0x100000
    saddi x7, x7, -86
    sbne x14, x7, fail
    nop
    nop
test_3:
    saddi x3, x0, 3
    li x2, VMEM_BASE + 0x100
    saddi x1, x0, 0
    sb x1, 1(x2)
    lb x14, 1(x2)
    saddi x7, x0, 0
    sbne x14, x7, fail
    nop
    nop
test_4:
    saddi x3, x0, 4
    li x2, VMEM_BASE + 0x100
    slui x1, 0xfffff
    saddi x1, x1, -96
    sb x1, 2(x2)
    lh x14, 2(x2)
    slui x7, 0xfffff
    saddi x7, x7, -96
    sbne x14, x7, fail
    nop
    nop
test_5:
    saddi x3, x0, 5
    li x2, VMEM_BASE + 0x100
    saddi x1, x0, 10
    sb x1, 3(x2)
    lb x14, 3(x2)
    saddi x7, x0, 10
    sbne x14, x7, fail
    nop
    nop
test_6:
    saddi x3, x0, 6
    li x2, VMEM_BASE + 0x107
    slui x1, 0x100000
    saddi x1, x1, -86
    sb x1, -3(x2)
    lb x14, -3(x2)
    slui x7, 0x100000
    saddi x7, x7, -86
    sbne x14, x7, fail
    nop
    nop
test_7:
    saddi x3, x0, 7
    li x2, VMEM_BASE + 0x107
    saddi x1, x0, 0
    sb x1, -2(x2)
    lb x14, -2(x2)
    saddi x7, x0, 0
    sbne x14, x7, fail
    nop
    nop
test_8:
    saddi x3, x0, 8
    li x2, VMEM_BASE + 0x107
    slui x1, 0x100000
    saddi x1, x1, -96
    sb x1, -1(x2)
    lb x14, -1(x2)
    slui x7, 0x100000
    saddi x7, x7, -96
    sbne x14, x7, fail
    nop
    nop
test_9:
    saddi x3, x0, 9
    li x2, VMEM_BASE + 0x107
    saddi x1, x0, 10
    sb x1, 0(x2)
    lb x14, 0(x2)
    saddi x7, x0, 10
    sbne x14, x7, fail
    nop
    nop
test_12:
    saddi x3, x0, 12
    saddi x4, x0, 0
sb_test_12_loop_1:
    slui x13, 0x100000
    saddi x13, x13, -35
    li x12, VMEM_BASE + 0x100
    sb x13, 0(x12)
    lb x14, 0(x12)
    slui x7, 0x100000
    saddi x7, x7, -35
    sbne x14, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sb_test_12_loop_1
    nop
    nop
test_13:
    saddi x3, x0, 13
    saddi x4, x0, 0
sb_test_13_loop_2:
    slui x13, 0x100000
    saddi x13, x13, -51
    li x12, VMEM_BASE + 0x100
    nop
    sb x13, 1(x12)
    lb x14, 1(x12)
    slui x7, 0x100000
    saddi x7, x7, -51
    sbne x14, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sb_test_13_loop_2
    nop
    nop
test_14:
    saddi x3, x0, 14
    saddi x4, x0, 0
sb_test_14_loop_3:
    slui x13, 0x100000
    saddi x13, x13, -52
    li x12, VMEM_BASE + 0x100
    nop
    nop
    sb x13, 2(x12)
    lb x14, 2(x12)
    slui x7, 0x100000
    saddi x7, x7, -52
    sbne x14, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sb_test_14_loop_3
    nop
    nop
test_15:
    saddi x3, x0, 15
    saddi x4, x0, 0
sb_test_15_loop_4:
    slui x13, 0x100000
    saddi x13, x13, -68
    nop
    li x12, VMEM_BASE + 0x100
    sb x13, 3(x12)
    lb x14, 3(x12)
    slui x7, 0x100000
    saddi x7, x7, -68
    sbne x14, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sb_test_15_loop_4
    nop
    nop
test_16:
    saddi x3, x0, 16
    saddi x4, x0, 0
sb_test_16_loop_5:
    slui x13, 0x100000
    saddi x13, x13, -69
    nop
    li x12, VMEM_BASE + 0x100
    nop
    sb x13, 4(x12)
    lb x14, 4(x12)
    slui x7, 0x100000
    saddi x7, x7, -69
    sbne x14, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sb_test_16_loop_5
    nop
    nop
test_17:
    saddi x3, x0, 17
    saddi x4, x0, 0
sb_test_17_loop_6:
    slui x13, 0x100000
    saddi x13, x13, -85
    nop
    nop
    li x12, VMEM_BASE + 0x100
    sb x13, 5(x12)
    lb x14, 5(x12)
    slui x7, 0x100000
    saddi x7, x7, -85
    sbne x14, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sb_test_17_loop_6
    nop
    nop
test_18:
    saddi x3, x0, 18
    saddi x4, x0, 0
sb_test_18_loop_7:
    li x2, VMEM_BASE + 0x100
    saddi x1, x0, 51
    sb x1, 0(x2)
    lb x14, 0(x2)
    saddi x7, x0, 51
    sbne x14, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sb_test_18_loop_7
    nop
    nop
test_19:
    saddi x3, x0, 19
    saddi x4, x0, 0
sb_test_19_loop_8:
    li x2, VMEM_BASE + 0x100
    saddi x1, x0, 35
    nop
    sb x1, 1(x2)
    lb x14, 1(x2)
    saddi x7, x0, 35
    sbne x14, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sb_test_19_loop_8
    nop
    nop
test_20:
    saddi x3, x0, 20
    saddi x4, x0, 0
sb_test_20_loop_9:
    li x2, VMEM_BASE + 0x100
    saddi x1, x0, 34
    nop
    nop
    sb x1, 2(x2)
    lb x14, 2(x2)
    saddi x7, x0, 34
    sbne x14, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sb_test_20_loop_9
    nop
    nop
test_21:
    saddi x3, x0, 21
    saddi x4, x0, 0
sb_test_21_loop_10:
    li x2, VMEM_BASE + 0x100
    nop
    saddi x1, x0, 18
    sb x1, 3(x2)
    lb x14, 3(x2)
    saddi x7, x0, 18
    sbne x14, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sb_test_21_loop_10
    nop
    nop
test_22:
    saddi x3, x0, 22
    saddi x4, x0, 0
sb_test_22_loop_11:
    li x2, VMEM_BASE + 0x100
    nop
    saddi x1, x0, 17
    nop
    sb x1, 4(x2)
    lb x14, 4(x2)
    saddi x7, x0, 17
    sbne x14, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sb_test_22_loop_11
    nop
    nop
test_23:
    saddi x3, x0, 23
    saddi x4, x0, 0
sb_test_23_loop_12:
    li x2, VMEM_BASE + 0x100
    nop
    nop
    saddi x1, x0, 1
    sb x1, 5(x2)
    lb x14, 5(x2)
    saddi x7, x0, 1
    sbne x14, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sb_test_23_loop_12
    nop
    nop
    saddi x10, x0, 239
    li x11, VMEM_BASE + 0x100
    sb x10, 3(x11)

test_pass:
    saddi x3, x0, 0
    saddi x10, x0, 0
    secall

fail:
    saddi x10, x0, 1
    secall
