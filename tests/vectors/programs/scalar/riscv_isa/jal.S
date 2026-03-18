# Flattened from upstream riscv-tests rv32ui/jal.S
#
# Penguin test-result convention for imported riscv-tests programs:
#   x10 = 0 on pass, 1 on fail
#   x3  = failing upstream test number on fail, 0 on pass
#   secall terminates the test and returns control to the host harness

# Penguin note: upstream jal.S contains a no-delay-slot case that does not
# apply to Penguin. Dedicated delay-slot tests cover that behavior separately.

test_2:
    saddi x3, x0, 2
    saddi x1, x0, 0
    sjal x4, target_2
linkaddr_2:
    nop
    nop
    sjal x0, fail
    nop
    nop
target_2:
    li x2, linkaddr_2
    sbne x2, x4, fail
    nop
    nop

test_pass:
    saddi x3, x0, 0
    saddi x10, x0, 0
    secall

fail:
    saddi x10, x0, 1
    secall
