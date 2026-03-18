# Flattened from upstream riscv-tests rv32ui/sra.S
#
# Penguin test-result convention for imported riscv-tests programs:
#   x10 = 0 on pass, 1 on fail
#   x3  = failing upstream test number on fail, 0 on pass
#   secall terminates the test and returns control to the host harness

test_2:
    saddi x3, x0, 2
    slui x11, 0x80000
    saddi x12, x0, 0
    ssra x14, x11, x12
    slui x7, 0x80000
    sbne x14, x7, fail
    nop
    nop
test_3:
    saddi x3, x0, 3
    slui x11, 0x80000
    saddi x12, x0, 1
    ssra x14, x11, x12
    slui x7, 0xc0000
    sbne x14, x7, fail
    nop
    nop
test_4:
    saddi x3, x0, 4
    slui x11, 0x80000
    saddi x12, x0, 7
    ssra x14, x11, x12
    slui x7, 0xff000
    sbne x14, x7, fail
    nop
    nop
test_5:
    saddi x3, x0, 5
    slui x11, 0x80000
    saddi x12, x0, 14
    ssra x14, x11, x12
    slui x7, 0xfffe0
    sbne x14, x7, fail
    nop
    nop
test_6:
    saddi x3, x0, 6
    slui x11, 0x80000
    saddi x11, x11, 1
    saddi x12, x0, 31
    ssra x14, x11, x12
    slui x7, 0x100000
    saddi x7, x7, -1
    sbne x14, x7, fail
    nop
    nop
test_7:
    saddi x3, x0, 7
    slui x11, 0x80000
    saddi x11, x11, -1
    saddi x12, x0, 0
    ssra x14, x11, x12
    slui x7, 0x80000
    saddi x7, x7, -1
    sbne x14, x7, fail
    nop
    nop
test_8:
    saddi x3, x0, 8
    slui x11, 0x80000
    saddi x11, x11, -1
    saddi x12, x0, 1
    ssra x14, x11, x12
    slui x7, 0x40000
    saddi x7, x7, -1
    sbne x14, x7, fail
    nop
    nop
test_9:
    saddi x3, x0, 9
    slui x11, 0x80000
    saddi x11, x11, -1
    saddi x12, x0, 7
    ssra x14, x11, x12
    slui x7, 0x01000
    saddi x7, x7, -1
    sbne x14, x7, fail
    nop
    nop
test_10:
    saddi x3, x0, 10
    slui x11, 0x80000
    saddi x11, x11, -1
    saddi x12, x0, 14
    ssra x14, x11, x12
    slui x7, 0x00020
    saddi x7, x7, -1
    sbne x14, x7, fail
    nop
    nop
test_11:
    saddi x3, x0, 11
    slui x11, 0x80000
    saddi x11, x11, -1
    saddi x12, x0, 31
    ssra x14, x11, x12
    saddi x7, x0, 0
    sbne x14, x7, fail
    nop
    nop
test_12:
    saddi x3, x0, 12
    slui x11, 0x81818
    saddi x11, x11, 385
    saddi x12, x0, 0
    ssra x14, x11, x12
    slui x7, 0x81818
    saddi x7, x7, 385
    sbne x14, x7, fail
    nop
    nop
test_13:
    saddi x3, x0, 13
    slui x11, 0x81818
    saddi x11, x11, 385
    saddi x12, x0, 1
    ssra x14, x11, x12
    slui x7, 0xc0c0c
    saddi x7, x7, 192
    sbne x14, x7, fail
    nop
    nop
test_14:
    saddi x3, x0, 14
    slui x11, 0x81818
    saddi x11, x11, 385
    saddi x12, x0, 7
    ssra x14, x11, x12
    slui x7, 0xff030
    saddi x7, x7, 771
    sbne x14, x7, fail
    nop
    nop
test_15:
    saddi x3, x0, 15
    slui x11, 0x81818
    saddi x11, x11, 385
    saddi x12, x0, 14
    ssra x14, x11, x12
    slui x7, 0xfffe0
    saddi x7, x7, 1542
    sbne x14, x7, fail
    nop
    nop
test_16:
    saddi x3, x0, 16
    slui x11, 0x81818
    saddi x11, x11, 385
    saddi x12, x0, 31
    ssra x14, x11, x12
    slui x7, 0x100000
    saddi x7, x7, -1
    sbne x14, x7, fail
    nop
    nop
test_17:
    saddi x3, x0, 17
    slui x11, 0x81818
    saddi x11, x11, 385
    slui x12, 0x100000
    saddi x12, x12, -64
    ssra x14, x11, x12
    slui x7, 0x81818
    saddi x7, x7, 385
    sbne x14, x7, fail
    nop
    nop
test_18:
    saddi x3, x0, 18
    slui x11, 0x81818
    saddi x11, x11, 385
    slui x12, 0x100000
    saddi x12, x12, -63
    ssra x14, x11, x12
    slui x7, 0xc0c0c
    saddi x7, x7, 192
    sbne x14, x7, fail
    nop
    nop
test_19:
    saddi x3, x0, 19
    slui x11, 0x81818
    saddi x11, x11, 385
    slui x12, 0x100000
    saddi x12, x12, -57
    ssra x14, x11, x12
    slui x7, 0xff030
    saddi x7, x7, 771
    sbne x14, x7, fail
    nop
    nop
test_20:
    saddi x3, x0, 20
    slui x11, 0x81818
    saddi x11, x11, 385
    slui x12, 0x100000
    saddi x12, x12, -50
    ssra x14, x11, x12
    slui x7, 0xfffe0
    saddi x7, x7, 1542
    sbne x14, x7, fail
    nop
    nop
test_21:
    saddi x3, x0, 21
    slui x11, 0x81818
    saddi x11, x11, 385
    slui x12, 0x100000
    saddi x12, x12, -1
    ssra x14, x11, x12
    slui x7, 0x100000
    saddi x7, x7, -1
    sbne x14, x7, fail
    nop
    nop
test_22:
    saddi x3, x0, 22
    slui x11, 0x80000
    saddi x12, x0, 7
    ssra x11, x11, x12
    slui x7, 0xff000
    sbne x11, x7, fail
    nop
    nop
test_23:
    saddi x3, x0, 23
    slui x11, 0x80000
    saddi x12, x0, 14
    ssra x12, x11, x12
    slui x7, 0xfffe0
    sbne x12, x7, fail
    nop
    nop
test_24:
    saddi x3, x0, 24
    saddi x11, x0, 7
    ssra x11, x11, x11
    saddi x7, x0, 0
    sbne x11, x7, fail
    nop
    nop
test_25:
    saddi x3, x0, 25
    saddi x4, x0, 0
sra_test_25_loop_1:
    slui x1, 0x80000
    saddi x2, x0, 7
    ssra x14, x1, x2
    saddi x6, x14, 0
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sra_test_25_loop_1
    nop
    nop
    slui x7, 0xff000
    sbne x6, x7, fail
    nop
    nop
test_26:
    saddi x3, x0, 26
    saddi x4, x0, 0
sra_test_26_loop_2:
    slui x1, 0x80000
    saddi x2, x0, 14
    ssra x14, x1, x2
    nop
    saddi x6, x14, 0
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sra_test_26_loop_2
    nop
    nop
    slui x7, 0xfffe0
    sbne x6, x7, fail
    nop
    nop
test_27:
    saddi x3, x0, 27
    saddi x4, x0, 0
sra_test_27_loop_3:
    slui x1, 0x80000
    saddi x2, x0, 31
    ssra x14, x1, x2
    nop
    nop
    saddi x6, x14, 0
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sra_test_27_loop_3
    nop
    nop
    slui x7, 0x100000
    saddi x7, x7, -1
    sbne x6, x7, fail
    nop
    nop
test_28:
    saddi x3, x0, 28
    saddi x4, x0, 0
sra_test_28_loop_4:
    slui x1, 0x80000
    saddi x2, x0, 7
    ssra x14, x1, x2
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sra_test_28_loop_4
    nop
    nop
    slui x7, 0xff000
    sbne x14, x7, fail
    nop
    nop
test_29:
    saddi x3, x0, 29
    saddi x4, x0, 0
sra_test_29_loop_5:
    slui x1, 0x80000
    saddi x2, x0, 14
    nop
    ssra x14, x1, x2
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sra_test_29_loop_5
    nop
    nop
    slui x7, 0xfffe0
    sbne x14, x7, fail
    nop
    nop
test_30:
    saddi x3, x0, 30
    saddi x4, x0, 0
sra_test_30_loop_6:
    slui x1, 0x80000
    saddi x2, x0, 31
    nop
    nop
    ssra x14, x1, x2
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sra_test_30_loop_6
    nop
    nop
    slui x7, 0x100000
    saddi x7, x7, -1
    sbne x14, x7, fail
    nop
    nop
test_31:
    saddi x3, x0, 31
    saddi x4, x0, 0
sra_test_31_loop_7:
    slui x1, 0x80000
    nop
    saddi x2, x0, 7
    ssra x14, x1, x2
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sra_test_31_loop_7
    nop
    nop
    slui x7, 0xff000
    sbne x14, x7, fail
    nop
    nop
test_32:
    saddi x3, x0, 32
    saddi x4, x0, 0
sra_test_32_loop_8:
    slui x1, 0x80000
    nop
    saddi x2, x0, 14
    nop
    ssra x14, x1, x2
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sra_test_32_loop_8
    nop
    nop
    slui x7, 0xfffe0
    sbne x14, x7, fail
    nop
    nop
test_33:
    saddi x3, x0, 33
    saddi x4, x0, 0
sra_test_33_loop_9:
    slui x1, 0x80000
    nop
    nop
    saddi x2, x0, 31
    ssra x14, x1, x2
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sra_test_33_loop_9
    nop
    nop
    slui x7, 0x100000
    saddi x7, x7, -1
    sbne x14, x7, fail
    nop
    nop
test_34:
    saddi x3, x0, 34
    saddi x4, x0, 0
sra_test_34_loop_10:
    saddi x2, x0, 7
    slui x1, 0x80000
    ssra x14, x1, x2
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sra_test_34_loop_10
    nop
    nop
    slui x7, 0xff000
    sbne x14, x7, fail
    nop
    nop
test_35:
    saddi x3, x0, 35
    saddi x4, x0, 0
sra_test_35_loop_11:
    saddi x2, x0, 14
    slui x1, 0x80000
    nop
    ssra x14, x1, x2
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sra_test_35_loop_11
    nop
    nop
    slui x7, 0xfffe0
    sbne x14, x7, fail
    nop
    nop
test_36:
    saddi x3, x0, 36
    saddi x4, x0, 0
sra_test_36_loop_12:
    saddi x2, x0, 31
    slui x1, 0x80000
    nop
    nop
    ssra x14, x1, x2
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sra_test_36_loop_12
    nop
    nop
    slui x7, 0x100000
    saddi x7, x7, -1
    sbne x14, x7, fail
    nop
    nop
test_37:
    saddi x3, x0, 37
    saddi x4, x0, 0
sra_test_37_loop_13:
    saddi x2, x0, 7
    nop
    slui x1, 0x80000
    ssra x14, x1, x2
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sra_test_37_loop_13
    nop
    nop
    slui x7, 0xff000
    sbne x14, x7, fail
    nop
    nop
test_38:
    saddi x3, x0, 38
    saddi x4, x0, 0
sra_test_38_loop_14:
    saddi x2, x0, 14
    nop
    slui x1, 0x80000
    nop
    ssra x14, x1, x2
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sra_test_38_loop_14
    nop
    nop
    slui x7, 0xfffe0
    sbne x14, x7, fail
    nop
    nop
test_39:
    saddi x3, x0, 39
    saddi x4, x0, 0
sra_test_39_loop_15:
    saddi x2, x0, 31
    nop
    nop
    slui x1, 0x80000
    ssra x14, x1, x2
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sra_test_39_loop_15
    nop
    nop
    slui x7, 0x100000
    saddi x7, x7, -1
    sbne x14, x7, fail
    nop
    nop
test_40:
    saddi x3, x0, 40
    saddi x1, x0, 15
    ssra x2, x0, x1
    saddi x7, x0, 0
    sbne x2, x7, fail
    nop
    nop
test_41:
    saddi x3, x0, 41
    saddi x1, x0, 32
    ssra x2, x1, x0
    saddi x7, x0, 32
    sbne x2, x7, fail
    nop
    nop
test_42:
    saddi x3, x0, 42
    ssra x1, x0, x0
    saddi x7, x0, 0
    sbne x1, x7, fail
    nop
    nop
test_43:
    saddi x3, x0, 43
    slui x1, 0x00001
    saddi x1, x1, -2048
    saddi x2, x0, 0
    ssra x0, x1, x2
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
