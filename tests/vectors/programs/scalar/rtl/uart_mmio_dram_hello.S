# Scalar-core UART-MMIO DDR validation program.
#
# MMIO map used by the current FPGA top-level:
#   0x00000100 : UART status, bit0 = TX ready
#   0x00000104 : UART TX data, low byte transmitted on store
#   0x00000108 : free-running 32-bit cycle counter
#   0x0000010c : DRAM status
#                bit0 = !ui_clk_sync_rst
#                bit1 = init_calib_complete
#                bit2 = mig_mmcm_locked
#                bit3 = clock_wiz_locked
#
# DRAM map used by the current FPGA top-level:
#   0x80000000 : external DDR window through the MIG AXI bridge

    li x1, 0x00000100
    li x2, 0x00000104
    li x5, 0x0000010c
    li x6, 0x00000108
    slui x7, 0x02FAF
    saddi x7, x7, 128
    slui x8, 0x05F5E
    saddi x8, x8, 256
    slui x10, 0x80000

wait_dram_ready_init:
    sld x24, 0(x6)
wait_dram_ready:
    sld x3, 0(x5)
    sandi x3, x3, 0x00000007
    li x4, 7
    sbeq x3, x4, dram_ready
    nop
    nop
    sld x25, 0(x6)
    ssub x26, x25, x24
    sbltu x26, x8, wait_dram_ready
    nop
    nop
    sjal x0, timeout_error_init

dram_ready:
    li x4, 72
    sst x4, 0(x10)
    li x4, 101
    sst x4, 4(x10)
    li x4, 108
    sst x4, 8(x10)
    li x4, 108
    sst x4, 12(x10)
    li x4, 111
    sst x4, 16(x10)
    li x4, 32
    sst x4, 20(x10)
    li x4, 87
    sst x4, 24(x10)
    li x4, 111
    sst x4, 28(x10)
    li x4, 114
    sst x4, 32(x10)
    li x4, 108
    sst x4, 36(x10)
    li x4, 100
    sst x4, 40(x10)
    li x4, 13
    sst x4, 44(x10)
    li x4, 10
    sst x4, 48(x10)

    sld x11, 0(x10)
    sld x12, 4(x10)
    sld x13, 8(x10)
    sld x14, 12(x10)
    sld x15, 16(x10)
    sld x16, 20(x10)
    sld x17, 24(x10)
    sld x18, 28(x10)
    sld x19, 32(x10)
    sld x20, 36(x10)
    sld x21, 40(x10)
    sld x22, 44(x10)
    sld x23, 48(x10)

wait_00_init:
    sld x24, 0(x6)
wait_00:
    sld x3, 0(x1)
    sbeq x3, x0, wait_00_retry
    nop
    nop
    sst x11, 0(x2)

wait_01_init:
    sld x24, 0(x6)
wait_01:
    sld x3, 0(x1)
    sbeq x3, x0, wait_01_retry
    nop
    nop
    sst x12, 0(x2)

wait_02_init:
    sld x24, 0(x6)
wait_02:
    sld x3, 0(x1)
    sbeq x3, x0, wait_02_retry
    nop
    nop
    sst x13, 0(x2)

wait_03_init:
    sld x24, 0(x6)
wait_03:
    sld x3, 0(x1)
    sbeq x3, x0, wait_03_retry
    nop
    nop
    sst x14, 0(x2)

wait_04_init:
    sld x24, 0(x6)
wait_04:
    sld x3, 0(x1)
    sbeq x3, x0, wait_04_retry
    nop
    nop
    sst x15, 0(x2)

wait_05_init:
    sld x24, 0(x6)
wait_05:
    sld x3, 0(x1)
    sbeq x3, x0, wait_05_retry
    nop
    nop
    sst x16, 0(x2)

wait_06_init:
    sld x24, 0(x6)
wait_06:
    sld x3, 0(x1)
    sbeq x3, x0, wait_06_retry
    nop
    nop
    sst x17, 0(x2)

wait_07_init:
    sld x24, 0(x6)
wait_07:
    sld x3, 0(x1)
    sbeq x3, x0, wait_07_retry
    nop
    nop
    sst x18, 0(x2)

wait_08_init:
    sld x24, 0(x6)
wait_08:
    sld x3, 0(x1)
    sbeq x3, x0, wait_08_retry
    nop
    nop
    sst x19, 0(x2)

wait_09_init:
    sld x24, 0(x6)
wait_09:
    sld x3, 0(x1)
    sbeq x3, x0, wait_09_retry
    nop
    nop
    sst x20, 0(x2)

wait_10_init:
    sld x24, 0(x6)
wait_10:
    sld x3, 0(x1)
    sbeq x3, x0, wait_10_retry
    nop
    nop
    sst x21, 0(x2)

wait_11_init:
    sld x24, 0(x6)
wait_11:
    sld x3, 0(x1)
    sbeq x3, x0, wait_11_retry
    nop
    nop
    sst x22, 0(x2)

wait_12_init:
    sld x24, 0(x6)
wait_12:
    sld x3, 0(x1)
    sbeq x3, x0, wait_12_retry
    nop
    nop
    sst x23, 0(x2)

    sebreak

wait_00_retry:
    sld x25, 0(x6)
    ssub x26, x25, x24
    sbltu x26, x7, wait_00
    nop
    nop
    sjal x0, timeout_halt

wait_01_retry:
    sld x25, 0(x6)
    ssub x26, x25, x24
    sbltu x26, x7, wait_01
    nop
    nop
    sjal x0, timeout_halt

wait_02_retry:
    sld x25, 0(x6)
    ssub x26, x25, x24
    sbltu x26, x7, wait_02
    nop
    nop
    sjal x0, timeout_halt

wait_03_retry:
    sld x25, 0(x6)
    ssub x26, x25, x24
    sbltu x26, x7, wait_03
    nop
    nop
    sjal x0, timeout_halt

wait_04_retry:
    sld x25, 0(x6)
    ssub x26, x25, x24
    sbltu x26, x7, wait_04
    nop
    nop
    sjal x0, timeout_halt

wait_05_retry:
    sld x25, 0(x6)
    ssub x26, x25, x24
    sbltu x26, x7, wait_05
    nop
    nop
    sjal x0, timeout_halt

wait_06_retry:
    sld x25, 0(x6)
    ssub x26, x25, x24
    sbltu x26, x7, wait_06
    nop
    nop
    sjal x0, timeout_halt

wait_07_retry:
    sld x25, 0(x6)
    ssub x26, x25, x24
    sbltu x26, x7, wait_07
    nop
    nop
    sjal x0, timeout_halt

wait_08_retry:
    sld x25, 0(x6)
    ssub x26, x25, x24
    sbltu x26, x7, wait_08
    nop
    nop
    sjal x0, timeout_halt

wait_09_retry:
    sld x25, 0(x6)
    ssub x26, x25, x24
    sbltu x26, x7, wait_09
    nop
    nop
    sjal x0, timeout_halt

wait_10_retry:
    sld x25, 0(x6)
    ssub x26, x25, x24
    sbltu x26, x7, wait_10
    nop
    nop
    sjal x0, timeout_halt

wait_11_retry:
    sld x25, 0(x6)
    ssub x26, x25, x24
    sbltu x26, x7, wait_11
    nop
    nop
    sjal x0, timeout_halt

wait_12_retry:
    sld x25, 0(x6)
    ssub x26, x25, x24
    sbltu x26, x7, wait_12
    nop
    nop
    sjal x0, timeout_halt

timeout_error_init:
    li x4, 33
    sld x24, 0(x6)
timeout_error_wait_0:
    sld x3, 0(x1)
    sbeq x3, x0, timeout_error_retry_0
    nop
    nop
    sst x4, 0(x2)

    li x4, 13
    sld x24, 0(x6)
timeout_error_wait_1:
    sld x3, 0(x1)
    sbeq x3, x0, timeout_error_retry_1
    nop
    nop
    sst x4, 0(x2)

    li x4, 10
    sld x24, 0(x6)
timeout_error_wait_2:
    sld x3, 0(x1)
    sbeq x3, x0, timeout_error_retry_2
    nop
    nop
    sst x4, 0(x2)

timeout_halt:
    sebreak

timeout_error_retry_0:
    sld x25, 0(x6)
    ssub x26, x25, x24
    sbltu x26, x7, timeout_error_wait_0
    nop
    nop
    sjal x0, timeout_halt

timeout_error_retry_1:
    sld x25, 0(x6)
    ssub x26, x25, x24
    sbltu x26, x7, timeout_error_wait_1
    nop
    nop
    sjal x0, timeout_halt

timeout_error_retry_2:
    sld x25, 0(x6)
    ssub x26, x25, x24
    sbltu x26, x7, timeout_error_wait_2
    nop
    nop
    sjal x0, timeout_halt
