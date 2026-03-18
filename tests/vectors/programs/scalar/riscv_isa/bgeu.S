# Flattened from upstream riscv-tests rv32ui/bgeu.S
#
# Penguin test-result convention for imported riscv-tests programs:
#   x10 = 0 on pass, 1 on fail
#   x3  = failing upstream test number on fail, 0 on pass
#   secall terminates the test and returns control to the host harness

    # Penguin note: upstream bgeu.S contains a no-delay-slot test case that is
    # architecturally incompatible with Penguin's two executed delay slots, so the
    # imported file keeps the instruction semantic cases and leaves delay-slot
    # behavior to the dedicated Penguin control-flow tests.

test_2:
    saddi x3, x0, 2
    saddi x1, x0, 0
    saddi x2, x0, 0
    sbgeu x1, x2, bgeu_test_2_after_forward_1
    nop
    nop
    sbne x0, x3, fail
    nop
    nop
bgeu_test_2_back_target_3:
    sbne x0, x3, bgeu_test_2_after_loop_2
    nop
    nop
bgeu_test_2_after_forward_1:
    sbgeu x1, x2, bgeu_test_2_back_target_3
    nop
    nop
    sbne x0, x3, fail
    nop
    nop
bgeu_test_2_after_loop_2:
test_3:
    saddi x3, x0, 3
    saddi x1, x0, 1
    saddi x2, x0, 1
    sbgeu x1, x2, bgeu_test_3_after_forward_4
    nop
    nop
    sbne x0, x3, fail
    nop
    nop
bgeu_test_3_back_target_6:
    sbne x0, x3, bgeu_test_3_after_loop_5
    nop
    nop
bgeu_test_3_after_forward_4:
    sbgeu x1, x2, bgeu_test_3_back_target_6
    nop
    nop
    sbne x0, x3, fail
    nop
    nop
bgeu_test_3_after_loop_5:
test_4:
    saddi x3, x0, 4
    slui x1, 0x100000
    saddi x1, x1, -1
    slui x2, 0x100000
    saddi x2, x2, -1
    sbgeu x1, x2, bgeu_test_4_after_forward_7
    nop
    nop
    sbne x0, x3, fail
    nop
    nop
bgeu_test_4_back_target_9:
    sbne x0, x3, bgeu_test_4_after_loop_8
    nop
    nop
bgeu_test_4_after_forward_7:
    sbgeu x1, x2, bgeu_test_4_back_target_9
    nop
    nop
    sbne x0, x3, fail
    nop
    nop
bgeu_test_4_after_loop_8:
test_5:
    saddi x3, x0, 5
    saddi x1, x0, 1
    saddi x2, x0, 0
    sbgeu x1, x2, bgeu_test_5_after_forward_10
    nop
    nop
    sbne x0, x3, fail
    nop
    nop
bgeu_test_5_back_target_12:
    sbne x0, x3, bgeu_test_5_after_loop_11
    nop
    nop
bgeu_test_5_after_forward_10:
    sbgeu x1, x2, bgeu_test_5_back_target_12
    nop
    nop
    sbne x0, x3, fail
    nop
    nop
bgeu_test_5_after_loop_11:
test_6:
    saddi x3, x0, 6
    slui x1, 0x100000
    saddi x1, x1, -1
    slui x2, 0x100000
    saddi x2, x2, -2
    sbgeu x1, x2, bgeu_test_6_after_forward_13
    nop
    nop
    sbne x0, x3, fail
    nop
    nop
bgeu_test_6_back_target_15:
    sbne x0, x3, bgeu_test_6_after_loop_14
    nop
    nop
bgeu_test_6_after_forward_13:
    sbgeu x1, x2, bgeu_test_6_back_target_15
    nop
    nop
    sbne x0, x3, fail
    nop
    nop
bgeu_test_6_after_loop_14:
test_7:
    saddi x3, x0, 7
    slui x1, 0x100000
    saddi x1, x1, -1
    saddi x2, x0, 0
    sbgeu x1, x2, bgeu_test_7_after_forward_16
    nop
    nop
    sbne x0, x3, fail
    nop
    nop
bgeu_test_7_back_target_18:
    sbne x0, x3, bgeu_test_7_after_loop_17
    nop
    nop
bgeu_test_7_after_forward_16:
    sbgeu x1, x2, bgeu_test_7_back_target_18
    nop
    nop
    sbne x0, x3, fail
    nop
    nop
bgeu_test_7_after_loop_17:
test_8:
    saddi x3, x0, 8
    saddi x1, x0, 0
    saddi x2, x0, 1
    sbgeu x1, x2, bgeu_test_8_label1_19
    nop
    nop
    sbne x0, x3, bgeu_test_8_label2_20
    nop
    nop
bgeu_test_8_label1_19:
    sbne x0, x3, fail
    nop
    nop
bgeu_test_8_label2_20:
    sbgeu x1, x2, bgeu_test_8_label1_19
    nop
    nop
bgeu_test_8_label3_21:
test_9:
    saddi x3, x0, 9
    slui x1, 0x100000
    saddi x1, x1, -2
    slui x2, 0x100000
    saddi x2, x2, -1
    sbgeu x1, x2, bgeu_test_9_label1_22
    nop
    nop
    sbne x0, x3, bgeu_test_9_label2_23
    nop
    nop
bgeu_test_9_label1_22:
    sbne x0, x3, fail
    nop
    nop
bgeu_test_9_label2_23:
    sbgeu x1, x2, bgeu_test_9_label1_22
    nop
    nop
bgeu_test_9_label3_24:
test_10:
    saddi x3, x0, 10
    saddi x1, x0, 0
    slui x2, 0x100000
    saddi x2, x2, -1
    sbgeu x1, x2, bgeu_test_10_label1_25
    nop
    nop
    sbne x0, x3, bgeu_test_10_label2_26
    nop
    nop
bgeu_test_10_label1_25:
    sbne x0, x3, fail
    nop
    nop
bgeu_test_10_label2_26:
    sbgeu x1, x2, bgeu_test_10_label1_25
    nop
    nop
bgeu_test_10_label3_27:
test_11:
    saddi x3, x0, 11
    slui x1, 0x80000
    saddi x1, x1, -1
    slui x2, 0x80000
    sbgeu x1, x2, bgeu_test_11_label1_28
    nop
    nop
    sbne x0, x3, bgeu_test_11_label2_29
    nop
    nop
bgeu_test_11_label1_28:
    sbne x0, x3, fail
    nop
    nop
bgeu_test_11_label2_29:
    sbgeu x1, x2, bgeu_test_11_label1_28
    nop
    nop
bgeu_test_11_label3_30:
test_12:
    saddi x3, x0, 12
    saddi x4, x0, 0
bgeu_test_12_loop_31:
    slui x1, 0xf0000
    saddi x1, x1, -1
    slui x2, 0xf0000
    sbgeu x1, x2, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, bgeu_test_12_loop_31
    nop
    nop
test_13:
    saddi x3, x0, 13
    saddi x4, x0, 0
bgeu_test_13_loop_32:
    slui x1, 0xf0000
    saddi x1, x1, -1
    slui x2, 0xf0000
    nop
    sbgeu x1, x2, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, bgeu_test_13_loop_32
    nop
    nop
test_14:
    saddi x3, x0, 14
    saddi x4, x0, 0
bgeu_test_14_loop_33:
    slui x1, 0xf0000
    saddi x1, x1, -1
    slui x2, 0xf0000
    nop
    nop
    sbgeu x1, x2, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, bgeu_test_14_loop_33
    nop
    nop
test_15:
    saddi x3, x0, 15
    saddi x4, x0, 0
bgeu_test_15_loop_34:
    slui x1, 0xf0000
    saddi x1, x1, -1
    nop
    slui x2, 0xf0000
    sbgeu x1, x2, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, bgeu_test_15_loop_34
    nop
    nop
test_16:
    saddi x3, x0, 16
    saddi x4, x0, 0
bgeu_test_16_loop_35:
    slui x1, 0xf0000
    saddi x1, x1, -1
    nop
    slui x2, 0xf0000
    nop
    sbgeu x1, x2, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, bgeu_test_16_loop_35
    nop
    nop
test_17:
    saddi x3, x0, 17
    saddi x4, x0, 0
bgeu_test_17_loop_36:
    slui x1, 0xf0000
    saddi x1, x1, -1
    nop
    nop
    slui x2, 0xf0000
    sbgeu x1, x2, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, bgeu_test_17_loop_36
    nop
    nop
test_18:
    saddi x3, x0, 18
    saddi x4, x0, 0
bgeu_test_18_loop_37:
    slui x1, 0xf0000
    saddi x1, x1, -1
    slui x2, 0xf0000
    sbgeu x1, x2, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, bgeu_test_18_loop_37
    nop
    nop
test_19:
    saddi x3, x0, 19
    saddi x4, x0, 0
bgeu_test_19_loop_38:
    slui x1, 0xf0000
    saddi x1, x1, -1
    slui x2, 0xf0000
    nop
    sbgeu x1, x2, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, bgeu_test_19_loop_38
    nop
    nop
test_20:
    saddi x3, x0, 20
    saddi x4, x0, 0
bgeu_test_20_loop_39:
    slui x1, 0xf0000
    saddi x1, x1, -1
    slui x2, 0xf0000
    nop
    nop
    sbgeu x1, x2, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, bgeu_test_20_loop_39
    nop
    nop
test_21:
    saddi x3, x0, 21
    saddi x4, x0, 0
bgeu_test_21_loop_40:
    slui x1, 0xf0000
    saddi x1, x1, -1
    nop
    slui x2, 0xf0000
    sbgeu x1, x2, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, bgeu_test_21_loop_40
    nop
    nop
test_22:
    saddi x3, x0, 22
    saddi x4, x0, 0
bgeu_test_22_loop_41:
    slui x1, 0xf0000
    saddi x1, x1, -1
    nop
    slui x2, 0xf0000
    nop
    sbgeu x1, x2, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, bgeu_test_22_loop_41
    nop
    nop
test_23:
    saddi x3, x0, 23
    saddi x4, x0, 0
bgeu_test_23_loop_42:
    slui x1, 0xf0000
    saddi x1, x1, -1
    nop
    nop
    slui x2, 0xf0000
    sbgeu x1, x2, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, bgeu_test_23_loop_42
    nop
    nop

test_pass:
    saddi x3, x0, 0
    saddi x10, x0, 0
    secall

fail:
    saddi x10, x0, 1
    secall
