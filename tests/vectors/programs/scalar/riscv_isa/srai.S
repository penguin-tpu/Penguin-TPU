# Flattened from upstream riscv-tests rv32ui/srai.S
#
# Penguin test-result convention for imported riscv-tests programs:
#   x10 = 0 on pass, 1 on fail
#   x3  = failing upstream test number on fail, 0 on pass
#   secall terminates the test and returns control to the host harness

test_2:
    saddi x3, x0, 2
    saddi x13, x0, 0
    ssrai x14, x13, 0
    saddi x7, x0, 0
    sbne x14, x7, fail
    nop
    nop
test_3:
    saddi x3, x0, 3
    slui x13, 0x80000
    ssrai x14, x13, 1
    slui x7, 0xc0000
    sbne x14, x7, fail
    nop
    nop
test_4:
    saddi x3, x0, 4
    slui x13, 0x80000
    ssrai x14, x13, 7
    slui x7, 0xff000
    sbne x14, x7, fail
    nop
    nop
test_5:
    saddi x3, x0, 5
    slui x13, 0x80000
    ssrai x14, x13, 14
    slui x7, 0xfffe0
    sbne x14, x7, fail
    nop
    nop
test_6:
    saddi x3, x0, 6
    slui x13, 0x80000
    saddi x13, x13, 1
    ssrai x14, x13, 31
    slui x7, 0x100000
    saddi x7, x7, -1
    sbne x14, x7, fail
    nop
    nop
test_7:
    saddi x3, x0, 7
    slui x13, 0x80000
    saddi x13, x13, -1
    ssrai x14, x13, 0
    slui x7, 0x80000
    saddi x7, x7, -1
    sbne x14, x7, fail
    nop
    nop
test_8:
    saddi x3, x0, 8
    slui x13, 0x80000
    saddi x13, x13, -1
    ssrai x14, x13, 1
    slui x7, 0x40000
    saddi x7, x7, -1
    sbne x14, x7, fail
    nop
    nop
test_9:
    saddi x3, x0, 9
    slui x13, 0x80000
    saddi x13, x13, -1
    ssrai x14, x13, 7
    slui x7, 0x01000
    saddi x7, x7, -1
    sbne x14, x7, fail
    nop
    nop
test_10:
    saddi x3, x0, 10
    slui x13, 0x80000
    saddi x13, x13, -1
    ssrai x14, x13, 14
    slui x7, 0x00020
    saddi x7, x7, -1
    sbne x14, x7, fail
    nop
    nop
test_11:
    saddi x3, x0, 11
    slui x13, 0x80000
    saddi x13, x13, -1
    ssrai x14, x13, 31
    saddi x7, x0, 0
    sbne x14, x7, fail
    nop
    nop
test_12:
    saddi x3, x0, 12
    slui x13, 0x81818
    saddi x13, x13, 385
    ssrai x14, x13, 0
    slui x7, 0x81818
    saddi x7, x7, 385
    sbne x14, x7, fail
    nop
    nop
test_13:
    saddi x3, x0, 13
    slui x13, 0x81818
    saddi x13, x13, 385
    ssrai x14, x13, 1
    slui x7, 0xc0c0c
    saddi x7, x7, 192
    sbne x14, x7, fail
    nop
    nop
test_14:
    saddi x3, x0, 14
    slui x13, 0x81818
    saddi x13, x13, 385
    ssrai x14, x13, 7
    slui x7, 0xff030
    saddi x7, x7, 771
    sbne x14, x7, fail
    nop
    nop
test_15:
    saddi x3, x0, 15
    slui x13, 0x81818
    saddi x13, x13, 385
    ssrai x14, x13, 14
    slui x7, 0xfffe0
    saddi x7, x7, 1542
    sbne x14, x7, fail
    nop
    nop
test_16:
    saddi x3, x0, 16
    slui x13, 0x81818
    saddi x13, x13, 385
    ssrai x14, x13, 31
    slui x7, 0x100000
    saddi x7, x7, -1
    sbne x14, x7, fail
    nop
    nop
test_17:
    saddi x3, x0, 17
    slui x11, 0x80000
    ssrai x11, x11, 7
    slui x7, 0xff000
    sbne x11, x7, fail
    nop
    nop
test_18:
    saddi x3, x0, 18
    saddi x4, x0, 0
srai_test_18_loop_1:
    slui x1, 0x80000
    ssrai x14, x1, 7
    saddi x6, x14, 0
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, srai_test_18_loop_1
    nop
    nop
    slui x7, 0xff000
    sbne x6, x7, fail
    nop
    nop
test_19:
    saddi x3, x0, 19
    saddi x4, x0, 0
srai_test_19_loop_2:
    slui x1, 0x80000
    ssrai x14, x1, 14
    nop
    saddi x6, x14, 0
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, srai_test_19_loop_2
    nop
    nop
    slui x7, 0xfffe0
    sbne x6, x7, fail
    nop
    nop
test_20:
    saddi x3, x0, 20
    saddi x4, x0, 0
srai_test_20_loop_3:
    slui x1, 0x80000
    saddi x1, x1, 1
    ssrai x14, x1, 31
    nop
    nop
    saddi x6, x14, 0
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, srai_test_20_loop_3
    nop
    nop
    slui x7, 0x100000
    saddi x7, x7, -1
    sbne x6, x7, fail
    nop
    nop
test_21:
    saddi x3, x0, 21
    saddi x4, x0, 0
srai_test_21_loop_4:
    slui x1, 0x80000
    ssrai x14, x1, 7
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, srai_test_21_loop_4
    nop
    nop
    slui x7, 0xff000
    sbne x14, x7, fail
    nop
    nop
test_22:
    saddi x3, x0, 22
    saddi x4, x0, 0
srai_test_22_loop_5:
    slui x1, 0x80000
    nop
    ssrai x14, x1, 14
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, srai_test_22_loop_5
    nop
    nop
    slui x7, 0xfffe0
    sbne x14, x7, fail
    nop
    nop
test_23:
    saddi x3, x0, 23
    saddi x4, x0, 0
srai_test_23_loop_6:
    slui x1, 0x80000
    saddi x1, x1, 1
    nop
    nop
    ssrai x14, x1, 31
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, srai_test_23_loop_6
    nop
    nop
    slui x7, 0x100000
    saddi x7, x7, -1
    sbne x14, x7, fail
    nop
    nop
test_24:
    saddi x3, x0, 24
    ssrai x1, x0, 4
    saddi x7, x0, 0
    sbne x1, x7, fail
    nop
    nop
test_25:
    saddi x3, x0, 25
    saddi x1, x0, 33
    ssrai x0, x1, 10
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
