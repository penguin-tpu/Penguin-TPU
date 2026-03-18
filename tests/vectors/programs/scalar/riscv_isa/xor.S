# Flattened from upstream riscv-tests rv32ui/xor.S
#
# Penguin test-result convention for imported riscv-tests programs:
#   x10 = 0 on pass, 1 on fail
#   x3  = failing upstream test number on fail, 0 on pass
#   secall terminates the test and returns control to the host harness

test_2:
    saddi x3, x0, 2
    slui x11, 0xff010
    saddi x11, x11, -256
    slui x12, 0x0f0f1
    saddi x12, x12, -241
    sxor x14, x11, x12
    slui x7, 0xf00ff
    saddi x7, x7, 15
    sbne x14, x7, fail
    nop
    nop
test_3:
    saddi x3, x0, 3
    slui x11, 0x0ff01
    saddi x11, x11, -16
    slui x12, 0xf0f0f
    saddi x12, x12, 240
    sxor x14, x11, x12
    slui x7, 0xff010
    saddi x7, x7, -256
    sbne x14, x7, fail
    nop
    nop
test_4:
    saddi x3, x0, 4
    slui x11, 0x00ff0
    saddi x11, x11, 255
    slui x12, 0x0f0f1
    saddi x12, x12, -241
    sxor x14, x11, x12
    slui x7, 0x0ff01
    saddi x7, x7, -16
    sbne x14, x7, fail
    nop
    nop
test_5:
    saddi x3, x0, 5
    slui x11, 0xf00ff
    saddi x11, x11, 15
    slui x12, 0xf0f0f
    saddi x12, x12, 240
    sxor x14, x11, x12
    slui x7, 0x00ff0
    saddi x7, x7, 255
    sbne x14, x7, fail
    nop
    nop
test_6:
    saddi x3, x0, 6
    slui x11, 0xff010
    saddi x11, x11, -256
    slui x12, 0x0f0f1
    saddi x12, x12, -241
    sxor x11, x11, x12
    slui x7, 0xf00ff
    saddi x7, x7, 15
    sbne x11, x7, fail
    nop
    nop
test_7:
    saddi x3, x0, 7
    slui x11, 0xff010
    saddi x11, x11, -256
    slui x12, 0x0f0f1
    saddi x12, x12, -241
    sxor x12, x11, x12
    slui x7, 0xf00ff
    saddi x7, x7, 15
    sbne x12, x7, fail
    nop
    nop
test_8:
    saddi x3, x0, 8
    slui x11, 0xff010
    saddi x11, x11, -256
    sxor x11, x11, x11
    saddi x7, x0, 0
    sbne x11, x7, fail
    nop
    nop
test_9:
    saddi x3, x0, 9
    saddi x4, x0, 0
xor_test_9_loop_1:
    slui x1, 0xff010
    saddi x1, x1, -256
    slui x2, 0x0f0f1
    saddi x2, x2, -241
    sxor x14, x1, x2
    saddi x6, x14, 0
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, xor_test_9_loop_1
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
xor_test_10_loop_2:
    slui x1, 0x0ff01
    saddi x1, x1, -16
    slui x2, 0xf0f0f
    saddi x2, x2, 240
    sxor x14, x1, x2
    nop
    saddi x6, x14, 0
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, xor_test_10_loop_2
    nop
    nop
    slui x7, 0xff010
    saddi x7, x7, -256
    sbne x6, x7, fail
    nop
    nop
test_11:
    saddi x3, x0, 11
    saddi x4, x0, 0
xor_test_11_loop_3:
    slui x1, 0x00ff0
    saddi x1, x1, 255
    slui x2, 0x0f0f1
    saddi x2, x2, -241
    sxor x14, x1, x2
    nop
    nop
    saddi x6, x14, 0
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, xor_test_11_loop_3
    nop
    nop
    slui x7, 0x0ff01
    saddi x7, x7, -16
    sbne x6, x7, fail
    nop
    nop
test_12:
    saddi x3, x0, 12
    saddi x4, x0, 0
xor_test_12_loop_4:
    slui x1, 0xff010
    saddi x1, x1, -256
    slui x2, 0x0f0f1
    saddi x2, x2, -241
    sxor x14, x1, x2
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, xor_test_12_loop_4
    nop
    nop
    slui x7, 0xf00ff
    saddi x7, x7, 15
    sbne x14, x7, fail
    nop
    nop
test_13:
    saddi x3, x0, 13
    saddi x4, x0, 0
xor_test_13_loop_5:
    slui x1, 0x0ff01
    saddi x1, x1, -16
    slui x2, 0xf0f0f
    saddi x2, x2, 240
    nop
    sxor x14, x1, x2
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, xor_test_13_loop_5
    nop
    nop
    slui x7, 0xff010
    saddi x7, x7, -256
    sbne x14, x7, fail
    nop
    nop
test_14:
    saddi x3, x0, 14
    saddi x4, x0, 0
xor_test_14_loop_6:
    slui x1, 0x00ff0
    saddi x1, x1, 255
    slui x2, 0x0f0f1
    saddi x2, x2, -241
    nop
    nop
    sxor x14, x1, x2
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, xor_test_14_loop_6
    nop
    nop
    slui x7, 0x0ff01
    saddi x7, x7, -16
    sbne x14, x7, fail
    nop
    nop
test_15:
    saddi x3, x0, 15
    saddi x4, x0, 0
xor_test_15_loop_7:
    slui x1, 0xff010
    saddi x1, x1, -256
    nop
    slui x2, 0x0f0f1
    saddi x2, x2, -241
    sxor x14, x1, x2
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, xor_test_15_loop_7
    nop
    nop
    slui x7, 0xf00ff
    saddi x7, x7, 15
    sbne x14, x7, fail
    nop
    nop
test_16:
    saddi x3, x0, 16
    saddi x4, x0, 0
xor_test_16_loop_8:
    slui x1, 0x0ff01
    saddi x1, x1, -16
    nop
    slui x2, 0xf0f0f
    saddi x2, x2, 240
    nop
    sxor x14, x1, x2
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, xor_test_16_loop_8
    nop
    nop
    slui x7, 0xff010
    saddi x7, x7, -256
    sbne x14, x7, fail
    nop
    nop
test_17:
    saddi x3, x0, 17
    saddi x4, x0, 0
xor_test_17_loop_9:
    slui x1, 0x00ff0
    saddi x1, x1, 255
    nop
    nop
    slui x2, 0x0f0f1
    saddi x2, x2, -241
    sxor x14, x1, x2
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, xor_test_17_loop_9
    nop
    nop
    slui x7, 0x0ff01
    saddi x7, x7, -16
    sbne x14, x7, fail
    nop
    nop
test_18:
    saddi x3, x0, 18
    saddi x4, x0, 0
xor_test_18_loop_10:
    slui x2, 0x0f0f1
    saddi x2, x2, -241
    slui x1, 0xff010
    saddi x1, x1, -256
    sxor x14, x1, x2
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, xor_test_18_loop_10
    nop
    nop
    slui x7, 0xf00ff
    saddi x7, x7, 15
    sbne x14, x7, fail
    nop
    nop
test_19:
    saddi x3, x0, 19
    saddi x4, x0, 0
xor_test_19_loop_11:
    slui x2, 0xf0f0f
    saddi x2, x2, 240
    slui x1, 0x0ff01
    saddi x1, x1, -16
    nop
    sxor x14, x1, x2
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, xor_test_19_loop_11
    nop
    nop
    slui x7, 0xff010
    saddi x7, x7, -256
    sbne x14, x7, fail
    nop
    nop
test_20:
    saddi x3, x0, 20
    saddi x4, x0, 0
xor_test_20_loop_12:
    slui x2, 0x0f0f1
    saddi x2, x2, -241
    slui x1, 0x00ff0
    saddi x1, x1, 255
    nop
    nop
    sxor x14, x1, x2
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, xor_test_20_loop_12
    nop
    nop
    slui x7, 0x0ff01
    saddi x7, x7, -16
    sbne x14, x7, fail
    nop
    nop
test_21:
    saddi x3, x0, 21
    saddi x4, x0, 0
xor_test_21_loop_13:
    slui x2, 0x0f0f1
    saddi x2, x2, -241
    nop
    slui x1, 0xff010
    saddi x1, x1, -256
    sxor x14, x1, x2
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, xor_test_21_loop_13
    nop
    nop
    slui x7, 0xf00ff
    saddi x7, x7, 15
    sbne x14, x7, fail
    nop
    nop
test_22:
    saddi x3, x0, 22
    saddi x4, x0, 0
xor_test_22_loop_14:
    slui x2, 0xf0f0f
    saddi x2, x2, 240
    nop
    slui x1, 0x0ff01
    saddi x1, x1, -16
    nop
    sxor x14, x1, x2
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, xor_test_22_loop_14
    nop
    nop
    slui x7, 0xff010
    saddi x7, x7, -256
    sbne x14, x7, fail
    nop
    nop
test_23:
    saddi x3, x0, 23
    saddi x4, x0, 0
xor_test_23_loop_15:
    slui x2, 0x0f0f1
    saddi x2, x2, -241
    nop
    nop
    slui x1, 0x00ff0
    saddi x1, x1, 255
    sxor x14, x1, x2
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, xor_test_23_loop_15
    nop
    nop
    slui x7, 0x0ff01
    saddi x7, x7, -16
    sbne x14, x7, fail
    nop
    nop
test_24:
    saddi x3, x0, 24
    slui x1, 0xff010
    saddi x1, x1, -256
    sxor x2, x0, x1
    slui x7, 0xff010
    saddi x7, x7, -256
    sbne x2, x7, fail
    nop
    nop
test_25:
    saddi x3, x0, 25
    slui x1, 0x00ff0
    saddi x1, x1, 255
    sxor x2, x1, x0
    slui x7, 0x00ff0
    saddi x7, x7, 255
    sbne x2, x7, fail
    nop
    nop
test_26:
    saddi x3, x0, 26
    sxor x1, x0, x0
    saddi x7, x0, 0
    sbne x1, x7, fail
    nop
    nop
test_27:
    saddi x3, x0, 27
    slui x1, 0x22222
    saddi x1, x1, 546
    saddi x2, x0, 0
    sxor x0, x1, x2
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
