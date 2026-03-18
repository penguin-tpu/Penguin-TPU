# Flattened from upstream riscv-tests rv32ui/jalr.S
#
# Penguin test-result convention for imported riscv-tests programs:
#   x10 = 0 on pass, 1 on fail
#   x3  = failing upstream test number on fail, 0 on pass
#   secall terminates the test and returns control to the host harness

# Penguin note: upstream jalr.S contains a no-delay-slot case that does not
# apply to Penguin. Dedicated delay-slot tests cover that behavior separately.

test_2:
    saddi x3, x0, 2
    saddi x5, x0, 0
    li x6, target_2
    sjalr x5, x6, 0
linkaddr_2:
    nop
    nop
    sjal x0, fail
    nop
    nop
target_2:
    li x6, linkaddr_2
    sbne x5, x6, fail
    nop
    nop
test_3:
    saddi x3, x0, 3
    li x5, target_3
    sjalr x5, x5, 0
linkaddr_3:
    nop
    nop
    sjal x0, fail
    nop
    nop
target_3:
    li x6, linkaddr_3
    sbne x5, x6, fail
    nop
    nop
test_4:
    saddi x3, x0, 4
    saddi x4, x0, 0
jalr_bypass_4_loop:
    li x6, jalr_bypass_4_target
    sjalr x13, x6, 0
jalr_bypass_4_return:
    nop
    nop
    sjal x0, fail
    nop
    nop
jalr_bypass_4_target:
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, jalr_bypass_4_loop
    nop
    nop
test_5:
    saddi x3, x0, 5
    saddi x4, x0, 0
jalr_bypass_5_loop:
    li x6, jalr_bypass_5_target
    nop
    sjalr x13, x6, 0
jalr_bypass_5_return:
    nop
    nop
    sjal x0, fail
    nop
    nop
jalr_bypass_5_target:
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, jalr_bypass_5_loop
    nop
    nop
test_6:
    saddi x3, x0, 6
    saddi x4, x0, 0
jalr_bypass_6_loop:
    li x6, jalr_bypass_6_target
    nop
    nop
    sjalr x13, x6, 0
jalr_bypass_6_return:
    nop
    nop
    sjal x0, fail
    nop
    nop
jalr_bypass_6_target:
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, jalr_bypass_6_loop
    nop
    nop

test_pass:
    saddi x3, x0, 0
    saddi x10, x0, 0
    secall

fail:
    saddi x10, x0, 1
    secall
