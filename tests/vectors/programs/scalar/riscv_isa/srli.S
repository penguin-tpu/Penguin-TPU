# Flattened from upstream riscv-tests rv32ui/srli.S
#
# Penguin test-result convention for imported riscv-tests programs:
#   x10 = 0 on pass, 1 on fail
#   x3  = failing upstream test number on fail, 0 on pass
#   secall terminates the test and returns control to the host harness

test_17:
    saddi x3, x0, 17
    slui x11, 0x80000
    ssrli x11, x11, 7
    slui x7, 0x01000
    sbne x11, x7, fail
    nop
    nop
test_18:
    saddi x3, x0, 18
    saddi x4, x0, 0
srli_test_18_loop_1:
    slui x1, 0x80000
    ssrli x14, x1, 7
    saddi x6, x14, 0
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, srli_test_18_loop_1
    nop
    nop
    slui x7, 0x01000
    sbne x6, x7, fail
    nop
    nop
test_19:
    saddi x3, x0, 19
    saddi x4, x0, 0
srli_test_19_loop_2:
    slui x1, 0x80000
    ssrli x14, x1, 14
    nop
    saddi x6, x14, 0
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, srli_test_19_loop_2
    nop
    nop
    slui x7, 0x00020
    sbne x6, x7, fail
    nop
    nop
test_20:
    saddi x3, x0, 20
    saddi x4, x0, 0
srli_test_20_loop_3:
    slui x1, 0x80000
    saddi x1, x1, 1
    ssrli x14, x1, 31
    nop
    nop
    saddi x6, x14, 0
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, srli_test_20_loop_3
    nop
    nop
    saddi x7, x0, 1
    sbne x6, x7, fail
    nop
    nop
test_21:
    saddi x3, x0, 21
    saddi x4, x0, 0
srli_test_21_loop_4:
    slui x1, 0x80000
    ssrli x14, x1, 7
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, srli_test_21_loop_4
    nop
    nop
    slui x7, 0x01000
    sbne x14, x7, fail
    nop
    nop
test_22:
    saddi x3, x0, 22
    saddi x4, x0, 0
srli_test_22_loop_5:
    slui x1, 0x80000
    nop
    ssrli x14, x1, 14
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, srli_test_22_loop_5
    nop
    nop
    slui x7, 0x00020
    sbne x14, x7, fail
    nop
    nop
test_23:
    saddi x3, x0, 23
    saddi x4, x0, 0
srli_test_23_loop_6:
    slui x1, 0x80000
    saddi x1, x1, 1
    nop
    nop
    ssrli x14, x1, 31
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, srli_test_23_loop_6
    nop
    nop
    saddi x7, x0, 1
    sbne x14, x7, fail
    nop
    nop
test_24:
    saddi x3, x0, 24
    ssrli x1, x0, 4
    saddi x7, x0, 0
    sbne x1, x7, fail
    nop
    nop
test_25:
    saddi x3, x0, 25
    saddi x1, x0, 33
    ssrli x0, x1, 10
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
