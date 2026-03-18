# Flattened from upstream riscv-tests rv32ui/bne.S
#
# Penguin test-result convention for imported riscv-tests programs:
#   x10 = 0 on pass, 1 on fail
#   x3  = failing upstream test number on fail, 0 on pass
#   secall terminates the test and returns control to the host harness

    # Penguin note: upstream bne.S contains a no-delay-slot test case that is
    # architecturally incompatible with Penguin's two executed delay slots, so the
    # imported file keeps the instruction semantic cases and leaves delay-slot
    # behavior to the dedicated Penguin control-flow tests.

test_2:
    saddi x3, x0, 2
    saddi x1, x0, 0
    saddi x2, x0, 1
    sbne x1, x2, bne_test_2_after_forward_1
    nop
    nop
    sbne x0, x3, fail
    nop
    nop
bne_test_2_back_target_3:
    sbne x0, x3, bne_test_2_after_loop_2
    nop
    nop
bne_test_2_after_forward_1:
    sbne x1, x2, bne_test_2_back_target_3
    nop
    nop
    sbne x0, x3, fail
    nop
    nop
bne_test_2_after_loop_2:
test_3:
    saddi x3, x0, 3
    saddi x1, x0, 1
    saddi x2, x0, 0
    sbne x1, x2, bne_test_3_after_forward_4
    nop
    nop
    sbne x0, x3, fail
    nop
    nop
bne_test_3_back_target_6:
    sbne x0, x3, bne_test_3_after_loop_5
    nop
    nop
bne_test_3_after_forward_4:
    sbne x1, x2, bne_test_3_back_target_6
    nop
    nop
    sbne x0, x3, fail
    nop
    nop
bne_test_3_after_loop_5:
test_4:
    saddi x3, x0, 4
    slui x1, 0x100000
    saddi x1, x1, -1
    saddi x2, x0, 1
    sbne x1, x2, bne_test_4_after_forward_7
    nop
    nop
    sbne x0, x3, fail
    nop
    nop
bne_test_4_back_target_9:
    sbne x0, x3, bne_test_4_after_loop_8
    nop
    nop
bne_test_4_after_forward_7:
    sbne x1, x2, bne_test_4_back_target_9
    nop
    nop
    sbne x0, x3, fail
    nop
    nop
bne_test_4_after_loop_8:
test_5:
    saddi x3, x0, 5
    saddi x1, x0, 1
    slui x2, 0x100000
    saddi x2, x2, -1
    sbne x1, x2, bne_test_5_after_forward_10
    nop
    nop
    sbne x0, x3, fail
    nop
    nop
bne_test_5_back_target_12:
    sbne x0, x3, bne_test_5_after_loop_11
    nop
    nop
bne_test_5_after_forward_10:
    sbne x1, x2, bne_test_5_back_target_12
    nop
    nop
    sbne x0, x3, fail
    nop
    nop
bne_test_5_after_loop_11:
test_6:
    saddi x3, x0, 6
    saddi x1, x0, 0
    saddi x2, x0, 0
    sbne x1, x2, bne_test_6_label1_13
    nop
    nop
    sbne x0, x3, bne_test_6_label2_14
    nop
    nop
bne_test_6_label1_13:
    sbne x0, x3, fail
    nop
    nop
bne_test_6_label2_14:
    sbne x1, x2, bne_test_6_label1_13
    nop
    nop
bne_test_6_label3_15:
test_7:
    saddi x3, x0, 7
    saddi x1, x0, 1
    saddi x2, x0, 1
    sbne x1, x2, bne_test_7_label1_16
    nop
    nop
    sbne x0, x3, bne_test_7_label2_17
    nop
    nop
bne_test_7_label1_16:
    sbne x0, x3, fail
    nop
    nop
bne_test_7_label2_17:
    sbne x1, x2, bne_test_7_label1_16
    nop
    nop
bne_test_7_label3_18:
test_8:
    saddi x3, x0, 8
    slui x1, 0x100000
    saddi x1, x1, -1
    slui x2, 0x100000
    saddi x2, x2, -1
    sbne x1, x2, bne_test_8_label1_19
    nop
    nop
    sbne x0, x3, bne_test_8_label2_20
    nop
    nop
bne_test_8_label1_19:
    sbne x0, x3, fail
    nop
    nop
bne_test_8_label2_20:
    sbne x1, x2, bne_test_8_label1_19
    nop
    nop
bne_test_8_label3_21:
test_9:
    saddi x3, x0, 9
    saddi x4, x0, 0
bne_test_9_loop_22:
    saddi x1, x0, 0
    saddi x2, x0, 0
    sbne x1, x2, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, bne_test_9_loop_22
    nop
    nop
test_10:
    saddi x3, x0, 10
    saddi x4, x0, 0
bne_test_10_loop_23:
    saddi x1, x0, 0
    saddi x2, x0, 0
    nop
    sbne x1, x2, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, bne_test_10_loop_23
    nop
    nop
test_11:
    saddi x3, x0, 11
    saddi x4, x0, 0
bne_test_11_loop_24:
    saddi x1, x0, 0
    saddi x2, x0, 0
    nop
    nop
    sbne x1, x2, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, bne_test_11_loop_24
    nop
    nop
test_12:
    saddi x3, x0, 12
    saddi x4, x0, 0
bne_test_12_loop_25:
    saddi x1, x0, 0
    nop
    saddi x2, x0, 0
    sbne x1, x2, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, bne_test_12_loop_25
    nop
    nop
test_13:
    saddi x3, x0, 13
    saddi x4, x0, 0
bne_test_13_loop_26:
    saddi x1, x0, 0
    nop
    saddi x2, x0, 0
    nop
    sbne x1, x2, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, bne_test_13_loop_26
    nop
    nop
test_14:
    saddi x3, x0, 14
    saddi x4, x0, 0
bne_test_14_loop_27:
    saddi x1, x0, 0
    nop
    nop
    saddi x2, x0, 0
    sbne x1, x2, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, bne_test_14_loop_27
    nop
    nop
test_15:
    saddi x3, x0, 15
    saddi x4, x0, 0
bne_test_15_loop_28:
    saddi x1, x0, 0
    saddi x2, x0, 0
    sbne x1, x2, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, bne_test_15_loop_28
    nop
    nop
test_16:
    saddi x3, x0, 16
    saddi x4, x0, 0
bne_test_16_loop_29:
    saddi x1, x0, 0
    saddi x2, x0, 0
    nop
    sbne x1, x2, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, bne_test_16_loop_29
    nop
    nop
test_17:
    saddi x3, x0, 17
    saddi x4, x0, 0
bne_test_17_loop_30:
    saddi x1, x0, 0
    saddi x2, x0, 0
    nop
    nop
    sbne x1, x2, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, bne_test_17_loop_30
    nop
    nop
test_18:
    saddi x3, x0, 18
    saddi x4, x0, 0
bne_test_18_loop_31:
    saddi x1, x0, 0
    nop
    saddi x2, x0, 0
    sbne x1, x2, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, bne_test_18_loop_31
    nop
    nop
test_19:
    saddi x3, x0, 19
    saddi x4, x0, 0
bne_test_19_loop_32:
    saddi x1, x0, 0
    nop
    saddi x2, x0, 0
    nop
    sbne x1, x2, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, bne_test_19_loop_32
    nop
    nop
test_20:
    saddi x3, x0, 20
    saddi x4, x0, 0
bne_test_20_loop_33:
    saddi x1, x0, 0
    nop
    nop
    saddi x2, x0, 0
    sbne x1, x2, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, bne_test_20_loop_33
    nop
    nop

test_pass:
    saddi x3, x0, 0
    saddi x10, x0, 0
    secall

fail:
    saddi x10, x0, 1
    secall
