# Flattened from upstream riscv-tests rv32ui/lbu.S
#
# Penguin test-result convention for imported riscv-tests programs:
#   x10 = 0 on pass, 1 on fail
#   x3  = failing upstream test number on fail, 0 on pass
#   secall terminates the test and returns control to the host harness

test_2:
    saddi x3, x0, 2
    saddi x15, x0, 255
    li x2, VMEM_BASE + 0x100
    lbu x14, 0(x2)
    saddi x7, x0, 255
    sbne x14, x7, fail
    nop
    nop
test_3:
    saddi x3, x0, 3
    saddi x15, x0, 0
    li x2, VMEM_BASE + 0x100
    lbu x14, 1(x2)
    saddi x7, x0, 0
    sbne x14, x7, fail
    nop
    nop
test_4:
    saddi x3, x0, 4
    saddi x15, x0, 240
    li x2, VMEM_BASE + 0x100
    lbu x14, 2(x2)
    saddi x7, x0, 240
    sbne x14, x7, fail
    nop
    nop
test_5:
    saddi x3, x0, 5
    saddi x15, x0, 15
    li x2, VMEM_BASE + 0x100
    lbu x14, 3(x2)
    saddi x7, x0, 15
    sbne x14, x7, fail
    nop
    nop
test_6:
    saddi x3, x0, 6
    saddi x15, x0, 255
    li x2, VMEM_BASE + 0x103
    lbu x14, -3(x2)
    saddi x7, x0, 255
    sbne x14, x7, fail
    nop
    nop
test_7:
    saddi x3, x0, 7
    saddi x15, x0, 0
    li x2, VMEM_BASE + 0x103
    lbu x14, -2(x2)
    saddi x7, x0, 0
    sbne x14, x7, fail
    nop
    nop
test_8:
    saddi x3, x0, 8
    saddi x15, x0, 240
    li x2, VMEM_BASE + 0x103
    lbu x14, -1(x2)
    saddi x7, x0, 240
    sbne x14, x7, fail
    nop
    nop
test_9:
    saddi x3, x0, 9
    saddi x15, x0, 15
    li x2, VMEM_BASE + 0x103
    lbu x14, 0(x2)
    saddi x7, x0, 15
    sbne x14, x7, fail
    nop
    nop
test_12:
    saddi x3, x0, 12
    saddi x4, x0, 0
lbu_test_12_loop_1:
    li x13, VMEM_BASE + 0x101
    lbu x14, 1(x13)
    saddi x6, x14, 0
    saddi x7, x0, 240
    sbne x6, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, lbu_test_12_loop_1
    nop
    nop
test_13:
    saddi x3, x0, 13
    saddi x4, x0, 0
lbu_test_13_loop_2:
    li x13, VMEM_BASE + 0x102
    lbu x14, 1(x13)
    nop
    saddi x6, x14, 0
    saddi x7, x0, 15
    sbne x6, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, lbu_test_13_loop_2
    nop
    nop
test_14:
    saddi x3, x0, 14
    saddi x4, x0, 0
lbu_test_14_loop_3:
    li x13, VMEM_BASE + 0x100
    lbu x14, 1(x13)
    nop
    nop
    saddi x6, x14, 0
    saddi x7, x0, 0
    sbne x6, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, lbu_test_14_loop_3
    nop
    nop
test_15:
    saddi x3, x0, 15
    saddi x4, x0, 0
lbu_test_15_loop_4:
    li x13, VMEM_BASE + 0x101
    lbu x14, 1(x13)
    saddi x7, x0, 240
    sbne x14, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, lbu_test_15_loop_4
    nop
    nop
test_16:
    saddi x3, x0, 16
    saddi x4, x0, 0
lbu_test_16_loop_5:
    li x13, VMEM_BASE + 0x102
    nop
    lbu x14, 1(x13)
    saddi x7, x0, 15
    sbne x14, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, lbu_test_16_loop_5
    nop
    nop
test_17:
    saddi x3, x0, 17
    saddi x4, x0, 0
lbu_test_17_loop_6:
    li x13, VMEM_BASE + 0x100
    nop
    nop
    lbu x14, 1(x13)
    saddi x7, x0, 0
    sbne x14, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, lbu_test_17_loop_6
    nop
    nop

test_pass:
    saddi x3, x0, 0
    saddi x10, x0, 0
    secall

fail:
    saddi x10, x0, 1
    secall
