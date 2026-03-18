# Flattened from upstream riscv-tests rv32ui/auipc.S
#
# Penguin test-result convention for imported riscv-tests programs:
#   x10 = 0 on pass, 1 on fail
#   x3  = failing upstream test number on fail, 0 on pass
#   secall terminates the test and returns control to the host harness

test_2:
    saddi x3, x0, 2
    li x10, label_2_target + 9992
    sjal x11, label_2_target
    nop
    nop
label_2_target:
    ssub x10, x10, x11
    slui x7, 0x00002
    saddi x7, x7, 1802
    sbne x10, x7, fail
    nop
    nop
test_3:
    saddi x3, x0, 3
    li x10, label_3_target - 10008
    sjal x11, label_3_target
    nop
    nop
label_3_target:
    ssub x10, x10, x11
    slui x7, 0xffffe
    saddi x7, x7, -1814
    sbne x10, x7, fail
    nop
    nop

test_pass:
    saddi x3, x0, 0
    saddi x10, x0, 0
    secall

fail:
    saddi x10, x0, 1
    secall
