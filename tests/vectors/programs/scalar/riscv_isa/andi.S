# Flattened from upstream riscv-tests rv32ui/andi.S
#
# Penguin test-result convention for imported riscv-tests programs:
#   x10 = 0 on pass, 1 on fail
#   x3  = failing upstream test number on fail, 0 on pass
#   secall terminates the test and returns control to the host harness

test_2:
    saddi x3, x0, 2
    slui x13, 0xff010
    saddi x13, x13, -256
    sandi x14, x13, -241
    slui x7, 0xff010
    saddi x7, x7, -256
    sbne x14, x7, fail
    nop
    nop
test_3:
    saddi x3, x0, 3
    slui x13, 0x0ff01
    saddi x13, x13, -16
    sandi x14, x13, 240
    saddi x7, x0, 240
    sbne x14, x7, fail
    nop
    nop
test_4:
    saddi x3, x0, 4
    slui x13, 0x00ff0
    saddi x13, x13, 255
    sandi x14, x13, 1807
    saddi x7, x0, 15
    sbne x14, x7, fail
    nop
    nop
test_5:
    saddi x3, x0, 5
    slui x13, 0xf00ff
    saddi x13, x13, 15
    sandi x14, x13, 240
    saddi x7, x0, 0
    sbne x14, x7, fail
    nop
    nop
test_6:
    saddi x3, x0, 6
    slui x11, 0xff010
    saddi x11, x11, -256
    sandi x11, x11, 240
    saddi x7, x0, 0
    sbne x11, x7, fail
    nop
    nop
test_7:
    saddi x3, x0, 7
    saddi x4, x0, 0
andi_test_7_loop_1:
    slui x1, 0x0ff01
    saddi x1, x1, -16
    sandi x14, x1, 1807
    saddi x6, x14, 0
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, andi_test_7_loop_1
    nop
    nop
    saddi x7, x0, 1792
    sbne x6, x7, fail
    nop
    nop
test_8:
    saddi x3, x0, 8
    saddi x4, x0, 0
andi_test_8_loop_2:
    slui x1, 0x00ff0
    saddi x1, x1, 255
    sandi x14, x1, 240
    nop
    saddi x6, x14, 0
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, andi_test_8_loop_2
    nop
    nop
    saddi x7, x0, 240
    sbne x6, x7, fail
    nop
    nop
test_9:
    saddi x3, x0, 9
    saddi x4, x0, 0
andi_test_9_loop_3:
    slui x1, 0xf00ff
    saddi x1, x1, 15
    sandi x14, x1, -241
    nop
    nop
    saddi x6, x14, 0
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, andi_test_9_loop_3
    nop
    nop
    slui x7, 0xf00ff
    saddi x7, x7, 15
    sbne x6, x7, fail
    nop
    nop
test_10:
    saddi x3, x0, 10
    saddi x4, x0, 0
andi_test_10_loop_4:
    slui x1, 0x0ff01
    saddi x1, x1, -16
    sandi x14, x1, 1807
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, andi_test_10_loop_4
    nop
    nop
    saddi x7, x0, 1792
    sbne x14, x7, fail
    nop
    nop
test_11:
    saddi x3, x0, 11
    saddi x4, x0, 0
andi_test_11_loop_5:
    slui x1, 0x00ff0
    saddi x1, x1, 255
    nop
    sandi x14, x1, 240
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, andi_test_11_loop_5
    nop
    nop
    saddi x7, x0, 240
    sbne x14, x7, fail
    nop
    nop
test_12:
    saddi x3, x0, 12
    saddi x4, x0, 0
andi_test_12_loop_6:
    slui x1, 0xf00ff
    saddi x1, x1, 15
    nop
    nop
    sandi x14, x1, 1807
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, andi_test_12_loop_6
    nop
    nop
    saddi x7, x0, 15
    sbne x14, x7, fail
    nop
    nop
test_13:
    saddi x3, x0, 13
    sandi x1, x0, 240
    saddi x7, x0, 0
    sbne x1, x7, fail
    nop
    nop
test_14:
    saddi x3, x0, 14
    slui x1, 0x00ff0
    saddi x1, x1, 255
    sandi x0, x1, 1807
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
