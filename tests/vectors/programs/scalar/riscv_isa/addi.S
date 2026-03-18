# Flattened from upstream riscv-tests rv32ui/addi.S
#
# Penguin test-result convention for imported riscv-tests programs:
#   x10 = 0 on pass, 1 on fail
#   x3  = failing upstream test number on fail, 0 on pass
#   secall terminates the test and returns control to the host harness

test_2:
    saddi x3, x0, 2
    saddi x13, x0, 0
    saddi x14, x13, 0
    saddi x7, x0, 0
    sbne x14, x7, fail
    nop
    nop
test_3:
    saddi x3, x0, 3
    saddi x13, x0, 1
    saddi x14, x13, 1
    saddi x7, x0, 2
    sbne x14, x7, fail
    nop
    nop
test_4:
    saddi x3, x0, 4
    saddi x13, x0, 3
    saddi x14, x13, 7
    saddi x7, x0, 10
    sbne x14, x7, fail
    nop
    nop
test_5:
    saddi x3, x0, 5
    saddi x13, x0, 0
    saddi x14, x13, -2048
    slui x7, 0x100000
    saddi x7, x7, -2048
    sbne x14, x7, fail
    nop
    nop
test_6:
    saddi x3, x0, 6
    slui x13, 0x80000
    saddi x14, x13, 0
    slui x7, 0x80000
    sbne x14, x7, fail
    nop
    nop
test_7:
    saddi x3, x0, 7
    slui x13, 0x80000
    saddi x14, x13, -2048
    slui x7, 0x80000
    saddi x7, x7, -2048
    sbne x14, x7, fail
    nop
    nop
test_8:
    saddi x3, x0, 8
    saddi x13, x0, 0
    saddi x14, x13, 2047
    saddi x7, x0, 2047
    sbne x14, x7, fail
    nop
    nop
test_9:
    saddi x3, x0, 9
    slui x13, 0x80000
    saddi x13, x13, -1
    saddi x14, x13, 0
    slui x7, 0x80000
    saddi x7, x7, -1
    sbne x14, x7, fail
    nop
    nop
test_10:
    saddi x3, x0, 10
    slui x13, 0x80000
    saddi x13, x13, -1
    saddi x14, x13, 2047
    slui x7, 0x80000
    saddi x7, x7, 2046
    sbne x14, x7, fail
    nop
    nop
test_11:
    saddi x3, x0, 11
    slui x13, 0x80000
    saddi x14, x13, 2047
    slui x7, 0x80000
    saddi x7, x7, 2047
    sbne x14, x7, fail
    nop
    nop
test_12:
    saddi x3, x0, 12
    slui x13, 0x80000
    saddi x13, x13, -1
    saddi x14, x13, -2048
    slui x7, 0x7ffff
    saddi x7, x7, 2047
    sbne x14, x7, fail
    nop
    nop
test_13:
    saddi x3, x0, 13
    saddi x13, x0, 0
    saddi x14, x13, -1
    slui x7, 0x100000
    saddi x7, x7, -1
    sbne x14, x7, fail
    nop
    nop
test_14:
    saddi x3, x0, 14
    slui x13, 0x100000
    saddi x13, x13, -1
    saddi x14, x13, 1
    saddi x7, x0, 0
    sbne x14, x7, fail
    nop
    nop
test_15:
    saddi x3, x0, 15
    slui x13, 0x100000
    saddi x13, x13, -1
    saddi x14, x13, -1
    slui x7, 0x100000
    saddi x7, x7, -2
    sbne x14, x7, fail
    nop
    nop
test_16:
    saddi x3, x0, 16
    slui x13, 0x80000
    saddi x13, x13, -1
    saddi x14, x13, 1
    slui x7, 0x80000
    sbne x14, x7, fail
    nop
    nop
test_17:
    saddi x3, x0, 17
    saddi x11, x0, 13
    saddi x11, x11, 11
    saddi x7, x0, 24
    sbne x11, x7, fail
    nop
    nop
test_18:
    saddi x3, x0, 18
    saddi x4, x0, 0
addi_test_18_loop_1:
    saddi x1, x0, 13
    saddi x14, x1, 11
    saddi x6, x14, 0
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, addi_test_18_loop_1
    nop
    nop
    saddi x7, x0, 24
    sbne x6, x7, fail
    nop
    nop
test_19:
    saddi x3, x0, 19
    saddi x4, x0, 0
addi_test_19_loop_2:
    saddi x1, x0, 13
    saddi x14, x1, 10
    nop
    saddi x6, x14, 0
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, addi_test_19_loop_2
    nop
    nop
    saddi x7, x0, 23
    sbne x6, x7, fail
    nop
    nop
test_20:
    saddi x3, x0, 20
    saddi x4, x0, 0
addi_test_20_loop_3:
    saddi x1, x0, 13
    saddi x14, x1, 9
    nop
    nop
    saddi x6, x14, 0
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, addi_test_20_loop_3
    nop
    nop
    saddi x7, x0, 22
    sbne x6, x7, fail
    nop
    nop
test_21:
    saddi x3, x0, 21
    saddi x4, x0, 0
addi_test_21_loop_4:
    saddi x1, x0, 13
    saddi x14, x1, 11
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, addi_test_21_loop_4
    nop
    nop
    saddi x7, x0, 24
    sbne x14, x7, fail
    nop
    nop
test_22:
    saddi x3, x0, 22
    saddi x4, x0, 0
addi_test_22_loop_5:
    saddi x1, x0, 13
    nop
    saddi x14, x1, 10
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, addi_test_22_loop_5
    nop
    nop
    saddi x7, x0, 23
    sbne x14, x7, fail
    nop
    nop
test_23:
    saddi x3, x0, 23
    saddi x4, x0, 0
addi_test_23_loop_6:
    saddi x1, x0, 13
    nop
    nop
    saddi x14, x1, 9
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, addi_test_23_loop_6
    nop
    nop
    saddi x7, x0, 22
    sbne x14, x7, fail
    nop
    nop
test_24:
    saddi x3, x0, 24
    saddi x1, x0, 32
    saddi x7, x0, 32
    sbne x1, x7, fail
    nop
    nop
test_25:
    saddi x3, x0, 25
    saddi x1, x0, 33
    saddi x0, x1, 50
    saddi x7, x0, 0
    sbne x0, x7, fail
    nop
    nop

test_pass:
    saddi x3, x0, 0
    saddi x10, x0, 0
    secall

fail:
    saddi x10, x0, 1
    secall
