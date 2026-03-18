# Flattened from upstream riscv-tests rv32ui/lui.S
#
# Penguin test-result convention for imported riscv-tests programs:
#   x10 = 0 on pass, 1 on fail
#   x3  = failing upstream test number on fail, 0 on pass
#   secall terminates the test and returns control to the host harness

test_2:
    saddi x3, x0, 2
    slui x1, 0x00000
    saddi x7, x0, 0
    sbne x1, x7, fail
    nop
    nop
test_3:
    saddi x3, x0, 3
    slui x1, 0xfffff
    ssrai x1, x1, 1
    slui x7, 0x100000
    saddi x7, x7, -2048
    sbne x1, x7, fail
    nop
    nop
test_4:
    saddi x3, x0, 4
    slui x1, 0x7ffff
    ssrai x1, x1, 20
    saddi x7, x0, 2047
    sbne x1, x7, fail
    nop
    nop
test_5:
    saddi x3, x0, 5
    slui x1, 0x80000
    ssrai x1, x1, 20
    slui x7, 0x100000
    saddi x7, x7, -2048
    sbne x1, x7, fail
    nop
    nop
test_6:
    saddi x3, x0, 6
    slui x0, 0x80000
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
