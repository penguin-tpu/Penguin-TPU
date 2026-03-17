# Scalar-core UART-MMIO VPU validation program.
#
# MMIO map used by the current FPGA top-level:
#   0x00000100 : UART status, bit0 = TX ready
#   0x00000104 : UART TX data, low byte transmitted on store
#   0x00000108 : free-running 32-bit cycle counter
#   0x00000200 : preliminary VPU BF16 mreg window, one 16-bit lane per word
#
# The current Nexys Video scalar-core top runs at 50 MHz, so a 1 Hz message
# cadence corresponds to 50,000,000 core cycles between message starts.

    li x1, 0x00000100
    li x2, 0x00000104
    li x5, 0x00000108
    li x10, 0x00000200
    slui x6, 0x02FAF
    saddi x6, x6, 128

    slui x11, 0x00004
    saddi x11, x11, -128
    sst x11, 0(x10)

    slui x11, 0x00004
    sst x11, 4(x10)

    vadd m2, m0, m1
    sld x12, 8(x10)

message_start:
    sld x7, 0(x5)

wait_v:
    sld x3, 0(x1)
    sbeq x3, x0, wait_v
    nop
    nop
    li x4, 118
    sst x4, 0(x2)

wait_a:
    sld x3, 0(x1)
    sbeq x3, x0, wait_a
    nop
    nop
    li x4, 97
    sst x4, 0(x2)

wait_d0:
    sld x3, 0(x1)
    sbeq x3, x0, wait_d0
    nop
    nop
    li x4, 100
    sst x4, 0(x2)

wait_d1:
    sld x3, 0(x1)
    sbeq x3, x0, wait_d1
    nop
    nop
    li x4, 100
    sst x4, 0(x2)

wait_equal:
    sld x3, 0(x1)
    sbeq x3, x0, wait_equal
    nop
    nop
    li x4, 61
    sst x4, 0(x2)

    ssrli x13, x12, 12
    sandi x13, x13, 15
    ssltiu x14, x13, 10
    sbeq x14, x0, hex0_alpha
    nop
    nop
    saddi x13, x13, 48
    sjal x0, hex0_emit
    nop
    nop
hex0_alpha:
    saddi x13, x13, 55
hex0_emit:
wait_hex0:
    sld x3, 0(x1)
    sbeq x3, x0, wait_hex0
    nop
    nop
    sst x13, 0(x2)

    ssrli x13, x12, 8
    sandi x13, x13, 15
    ssltiu x14, x13, 10
    sbeq x14, x0, hex1_alpha
    nop
    nop
    saddi x13, x13, 48
    sjal x0, hex1_emit
    nop
    nop
hex1_alpha:
    saddi x13, x13, 55
hex1_emit:
wait_hex1:
    sld x3, 0(x1)
    sbeq x3, x0, wait_hex1
    nop
    nop
    sst x13, 0(x2)

    ssrli x13, x12, 4
    sandi x13, x13, 15
    ssltiu x14, x13, 10
    sbeq x14, x0, hex2_alpha
    nop
    nop
    saddi x13, x13, 48
    sjal x0, hex2_emit
    nop
    nop
hex2_alpha:
    saddi x13, x13, 55
hex2_emit:
wait_hex2:
    sld x3, 0(x1)
    sbeq x3, x0, wait_hex2
    nop
    nop
    sst x13, 0(x2)

    sandi x13, x12, 15
    ssltiu x14, x13, 10
    sbeq x14, x0, hex3_alpha
    nop
    nop
    saddi x13, x13, 48
    sjal x0, hex3_emit
    nop
    nop
hex3_alpha:
    saddi x13, x13, 55
hex3_emit:
wait_hex3:
    sld x3, 0(x1)
    sbeq x3, x0, wait_hex3
    nop
    nop
    sst x13, 0(x2)

wait_carriage_return:
    sld x3, 0(x1)
    sbeq x3, x0, wait_carriage_return
    nop
    nop
    li x4, 13
    sst x4, 0(x2)

wait_newline:
    sld x3, 0(x1)
    sbeq x3, x0, wait_newline
    nop
    nop
    li x4, 10
    sst x4, 0(x2)

delay_wait:
    sld x8, 0(x5)
    ssub x9, x8, x7
    sbltu x9, x6, delay_wait
    nop
    nop

    sjal x0, message_start
    nop
    nop
