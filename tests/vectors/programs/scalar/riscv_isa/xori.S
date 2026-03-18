# Flattened from upstream riscv-tests rv32ui/xori.S
#
# Penguin test-result convention for imported riscv-tests programs:
#   x10 = 0 on pass, 1 on fail
#   x3  = failing upstream test number on fail, 0 on pass
#   secall terminates the test and returns control to the host harness

test_2:
    saddi x3, x0, 2
    slui x13, 0x00ff1
    saddi x13, x13, -256
    sxori x14, x13, -241
    slui x7, 0xff00f
    saddi x7, x7, 15
    sbne x14, x7, fail
    nop
    nop
test_3:
    saddi x3, x0, 3
    slui x13, 0x0ff01
    saddi x13, x13, -16
    sxori x14, x13, 240
    slui x7, 0x0ff01
    saddi x7, x7, -256
    sbne x14, x7, fail
    nop
    nop
test_4:
    saddi x3, x0, 4
    slui x13, 0x00ff1
    saddi x13, x13, -1793
    sxori x14, x13, 1807
    slui x7, 0x00ff1
    saddi x7, x7, -16
    sbne x14, x7, fail
    nop
    nop
test_5:
    saddi x3, x0, 5
    slui x13, 0xf00ff
    saddi x13, x13, 15
    sxori x14, x13, 240
    slui x7, 0xf00ff
    saddi x7, x7, 255
    sbne x14, x7, fail
    nop
    nop
test_6:
    saddi x3, x0, 6
    slui x11, 0xff00f
    saddi x11, x11, 1792
    sxori x11, x11, 1807
    slui x7, 0xff00f
    saddi x7, x7, 15
    sbne x11, x7, fail
    nop
    nop
test_7:
    saddi x3, x0, 7
    saddi x4, x0, 0
xori_test_7_loop_1:
    slui x1, 0x0ff01
    saddi x1, x1, -16
    sxori x14, x1, 240
    saddi x6, x14, 0
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, xori_test_7_loop_1
    nop
    nop
    slui x7, 0x0ff01
    saddi x7, x7, -256
    sbne x6, x7, fail
    nop
    nop
test_8:
    saddi x3, x0, 8
    saddi x4, x0, 0
xori_test_8_loop_2:
    slui x1, 0x00ff1
    saddi x1, x1, -1793
    sxori x14, x1, 1807
    nop
    saddi x6, x14, 0
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, xori_test_8_loop_2
    nop
    nop
    slui x7, 0x00ff1
    saddi x7, x7, -16
    sbne x6, x7, fail
    nop
    nop
test_9:
    saddi x3, x0, 9
    saddi x4, x0, 0
xori_test_9_loop_3:
    slui x1, 0xf00ff
    saddi x1, x1, 15
    sxori x14, x1, 240
    nop
    nop
    saddi x6, x14, 0
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, xori_test_9_loop_3
    nop
    nop
    slui x7, 0xf00ff
    saddi x7, x7, 255
    sbne x6, x7, fail
    nop
    nop
test_10:
    saddi x3, x0, 10
    saddi x4, x0, 0
xori_test_10_loop_4:
    slui x1, 0x0ff01
    saddi x1, x1, -16
    sxori x14, x1, 240
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, xori_test_10_loop_4
    nop
    nop
    slui x7, 0x0ff01
    saddi x7, x7, -256
    sbne x14, x7, fail
    nop
    nop
test_11:
    saddi x3, x0, 11
    saddi x4, x0, 0
xori_test_11_loop_5:
    slui x1, 0x00ff1
    saddi x1, x1, -1
    nop
    sxori x14, x1, 15
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, xori_test_11_loop_5
    nop
    nop
    slui x7, 0x00ff1
    saddi x7, x7, -16
    sbne x14, x7, fail
    nop
    nop
test_12:
    saddi x3, x0, 12
    saddi x4, x0, 0
xori_test_12_loop_6:
    slui x1, 0xf00ff
    saddi x1, x1, 15
    nop
    nop
    sxori x14, x1, 240
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, xori_test_12_loop_6
    nop
    nop
    slui x7, 0xf00ff
    saddi x7, x7, 255
    sbne x14, x7, fail
    nop
    nop
test_13:
    saddi x3, x0, 13
    sxori x1, x0, 240
    saddi x7, x0, 240
    sbne x1, x7, fail
    nop
    nop
test_14:
    saddi x3, x0, 14
    slui x1, 0x00ff0
    saddi x1, x1, 255
    sxori x0, x1, 1807
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
