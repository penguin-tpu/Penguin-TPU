# Scalar-core UART-MMIO bring-up program.
#
# MMIO map used by the current FPGA top-level:
#   0x00000100 : UART status, bit0 = TX ready
#   0x00000104 : UART TX data, low byte transmitted on store

    li x1, 0x00000100
    li x2, 0x00000104

wait_00:
    sld x3, 0(x1)
    sbeq x3, x0, wait_00
    nop
    nop
    li x4, 104
    sst x4, 0(x2)

wait_01:
    sld x3, 0(x1)
    sbeq x3, x0, wait_01
    nop
    nop
    li x4, 101
    sst x4, 0(x2)

wait_02:
    sld x3, 0(x1)
    sbeq x3, x0, wait_02
    nop
    nop
    li x4, 108
    sst x4, 0(x2)

wait_03:
    sld x3, 0(x1)
    sbeq x3, x0, wait_03
    nop
    nop
    li x4, 108
    sst x4, 0(x2)

wait_04:
    sld x3, 0(x1)
    sbeq x3, x0, wait_04
    nop
    nop
    li x4, 111
    sst x4, 0(x2)

wait_05:
    sld x3, 0(x1)
    sbeq x3, x0, wait_05
    nop
    nop
    li x4, 44
    sst x4, 0(x2)

wait_06:
    sld x3, 0(x1)
    sbeq x3, x0, wait_06
    nop
    nop
    li x4, 32
    sst x4, 0(x2)

wait_07:
    sld x3, 0(x1)
    sbeq x3, x0, wait_07
    nop
    nop
    li x4, 116
    sst x4, 0(x2)

wait_08:
    sld x3, 0(x1)
    sbeq x3, x0, wait_08
    nop
    nop
    li x4, 104
    sst x4, 0(x2)

wait_09:
    sld x3, 0(x1)
    sbeq x3, x0, wait_09
    nop
    nop
    li x4, 105
    sst x4, 0(x2)

wait_10:
    sld x3, 0(x1)
    sbeq x3, x0, wait_10
    nop
    nop
    li x4, 115
    sst x4, 0(x2)

wait_11:
    sld x3, 0(x1)
    sbeq x3, x0, wait_11
    nop
    nop
    li x4, 32
    sst x4, 0(x2)

wait_12:
    sld x3, 0(x1)
    sbeq x3, x0, wait_12
    nop
    nop
    li x4, 105
    sst x4, 0(x2)

wait_13:
    sld x3, 0(x1)
    sbeq x3, x0, wait_13
    nop
    nop
    li x4, 115
    sst x4, 0(x2)

wait_14:
    sld x3, 0(x1)
    sbeq x3, x0, wait_14
    nop
    nop
    li x4, 32
    sst x4, 0(x2)

wait_15:
    sld x3, 0(x1)
    sbeq x3, x0, wait_15
    nop
    nop
    li x4, 112
    sst x4, 0(x2)

wait_16:
    sld x3, 0(x1)
    sbeq x3, x0, wait_16
    nop
    nop
    li x4, 101
    sst x4, 0(x2)

wait_17:
    sld x3, 0(x1)
    sbeq x3, x0, wait_17
    nop
    nop
    li x4, 110
    sst x4, 0(x2)

wait_18:
    sld x3, 0(x1)
    sbeq x3, x0, wait_18
    nop
    nop
    li x4, 103
    sst x4, 0(x2)

wait_19:
    sld x3, 0(x1)
    sbeq x3, x0, wait_19
    nop
    nop
    li x4, 117
    sst x4, 0(x2)

wait_20:
    sld x3, 0(x1)
    sbeq x3, x0, wait_20
    nop
    nop
    li x4, 105
    sst x4, 0(x2)

wait_21:
    sld x3, 0(x1)
    sbeq x3, x0, wait_21
    nop
    nop
    li x4, 110
    sst x4, 0(x2)

    sjal x0, wait_00
    nop
    nop
