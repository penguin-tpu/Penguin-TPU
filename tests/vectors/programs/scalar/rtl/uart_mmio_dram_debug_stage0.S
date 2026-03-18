# Scalar-core DDR bring-up debug program.
#
# This version does not touch DDR. It only verifies that the scalar core,
# instruction ROM, clocking, reset release, and UART path are alive inside the
# DDR-integrated top-level.

    li x1, 0x00000100
    li x2, 0x00000104
    li x6, 0x00000108
    slui x7, 0x02FAF
    saddi x7, x7, 128

wait_a_init:
    sld x24, 0(x6)
wait_a:
    sld x3, 0(x1)
    sbeq x3, x0, wait_a_retry
    nop
    nop
    li x4, 65
    sst x4, 0(x2)

wait_cr_init:
    sld x24, 0(x6)
wait_cr:
    sld x3, 0(x1)
    sbeq x3, x0, wait_cr_retry
    nop
    nop
    li x4, 13
    sst x4, 0(x2)

wait_lf_init:
    sld x24, 0(x6)
wait_lf:
    sld x3, 0(x1)
    sbeq x3, x0, wait_lf_retry
    nop
    nop
    li x4, 10
    sst x4, 0(x2)

    sebreak

wait_a_retry:
    sld x25, 0(x6)
    ssub x26, x25, x24
    sbltu x26, x7, wait_a
    nop
    nop
    sjal x0, timeout_halt

wait_cr_retry:
    sld x25, 0(x6)
    ssub x26, x25, x24
    sbltu x26, x7, wait_cr
    nop
    nop
    sjal x0, timeout_halt

wait_lf_retry:
    sld x25, 0(x6)
    ssub x26, x25, x24
    sbltu x26, x7, wait_lf
    nop
    nop
    sjal x0, timeout_halt

timeout_halt:
    sebreak
