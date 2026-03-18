# Flattened from upstream riscv-tests rv32ui/sw.S
#
# Penguin test-result convention for imported riscv-tests programs:
#   x10 = 0 on pass, 1 on fail
#   x3  = failing upstream test number on fail, 0 on pass
#   secall terminates the test and returns control to the host harness

test_2:
    saddi x3, x0, 2
    li x2, VMEM_BASE + 0x100
    slui x1, 0x00aa0
    saddi x1, x1, 170
    sw x1, 0(x2)
    lw x14, 0(x2)
    slui x7, 0x00aa0
    saddi x7, x7, 170
    sbne x14, x7, fail
    nop
    nop
test_3:
    saddi x3, x0, 3
    li x2, VMEM_BASE + 0x100
    slui x1, 0xaa00b
    saddi x1, x1, -1536
    sw x1, 4(x2)
    lw x14, 4(x2)
    slui x7, 0xaa00b
    saddi x7, x7, -1536
    sbne x14, x7, fail
    nop
    nop
test_4:
    saddi x3, x0, 4
    li x2, VMEM_BASE + 0x100
    slui x1, 0x0aa01
    saddi x1, x1, -1376
    sw x1, 8(x2)
    lw x14, 8(x2)
    slui x7, 0x0aa01
    saddi x7, x7, -1376
    sbne x14, x7, fail
    nop
    nop
test_5:
    saddi x3, x0, 5
    li x2, VMEM_BASE + 0x100
    slui x1, 0xa00aa
    saddi x1, x1, 10
    sw x1, 12(x2)
    lw x14, 12(x2)
    slui x7, 0xa00aa
    saddi x7, x7, 10
    sbne x14, x7, fail
    nop
    nop
test_6:
    saddi x3, x0, 6
    li x2, VMEM_BASE + 0x11c
    slui x1, 0x00aa0
    saddi x1, x1, 170
    sw x1, -12(x2)
    lw x14, -12(x2)
    slui x7, 0x00aa0
    saddi x7, x7, 170
    sbne x14, x7, fail
    nop
    nop
test_7:
    saddi x3, x0, 7
    li x2, VMEM_BASE + 0x11c
    slui x1, 0xaa00b
    saddi x1, x1, -1536
    sw x1, -8(x2)
    lw x14, -8(x2)
    slui x7, 0xaa00b
    saddi x7, x7, -1536
    sbne x14, x7, fail
    nop
    nop
test_8:
    saddi x3, x0, 8
    li x2, VMEM_BASE + 0x11c
    slui x1, 0x0aa01
    saddi x1, x1, -1376
    sw x1, -4(x2)
    lw x14, -4(x2)
    slui x7, 0x0aa01
    saddi x7, x7, -1376
    sbne x14, x7, fail
    nop
    nop
test_9:
    saddi x3, x0, 9
    li x2, VMEM_BASE + 0x11c
    slui x1, 0xa00aa
    saddi x1, x1, 10
    sw x1, 0(x2)
    lw x14, 0(x2)
    slui x7, 0xa00aa
    saddi x7, x7, 10
    sbne x14, x7, fail
    nop
    nop
test_12:
    saddi x3, x0, 12
    saddi x4, x0, 0
sw_test_12_loop_1:
    slui x13, 0xaabbd
    saddi x13, x13, -803
    li x12, VMEM_BASE + 0x100
    sw x13, 0(x12)
    lw x14, 0(x12)
    slui x7, 0xaabbd
    saddi x7, x7, -803
    sbne x14, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sw_test_12_loop_1
    nop
    nop
test_13:
    saddi x3, x0, 13
    saddi x4, x0, 0
sw_test_13_loop_2:
    slui x13, 0xdaabc
    saddi x13, x13, -819
    li x12, VMEM_BASE + 0x100
    nop
    sw x13, 4(x12)
    lw x14, 4(x12)
    slui x7, 0xdaabc
    saddi x7, x7, -819
    sbne x14, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sw_test_13_loop_2
    nop
    nop
test_14:
    saddi x3, x0, 14
    saddi x4, x0, 0
sw_test_14_loop_3:
    slui x13, 0xddaac
    saddi x13, x13, -1076
    li x12, VMEM_BASE + 0x100
    nop
    nop
    sw x13, 8(x12)
    lw x14, 8(x12)
    slui x7, 0xddaac
    saddi x7, x7, -1076
    sbne x14, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sw_test_14_loop_3
    nop
    nop
test_15:
    saddi x3, x0, 15
    saddi x4, x0, 0
sw_test_15_loop_4:
    slui x13, 0xcddab
    saddi x13, x13, -1092
    nop
    li x12, VMEM_BASE + 0x100
    sw x13, 12(x12)
    lw x14, 12(x12)
    slui x7, 0xcddab
    saddi x7, x7, -1092
    sbne x14, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sw_test_15_loop_4
    nop
    nop
test_16:
    saddi x3, x0, 16
    saddi x4, x0, 0
sw_test_16_loop_5:
    slui x13, 0xccddb
    saddi x13, x13, -1349
    nop
    li x12, VMEM_BASE + 0x100
    nop
    sw x13, 16(x12)
    lw x14, 16(x12)
    slui x7, 0xccddb
    saddi x7, x7, -1349
    sbne x14, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sw_test_16_loop_5
    nop
    nop
test_17:
    saddi x3, x0, 17
    saddi x4, x0, 0
sw_test_17_loop_6:
    slui x13, 0xbccde
    saddi x13, x13, -1365
    nop
    nop
    li x12, VMEM_BASE + 0x100
    sw x13, 20(x12)
    lw x14, 20(x12)
    slui x7, 0xbccde
    saddi x7, x7, -1365
    sbne x14, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sw_test_17_loop_6
    nop
    nop
test_18:
    saddi x3, x0, 18
    saddi x4, x0, 0
sw_test_18_loop_7:
    li x2, VMEM_BASE + 0x100
    slui x1, 0x00112
    saddi x1, x1, 563
    sw x1, 0(x2)
    lw x14, 0(x2)
    slui x7, 0x00112
    saddi x7, x7, 563
    sbne x14, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sw_test_18_loop_7
    nop
    nop
test_19:
    saddi x3, x0, 19
    saddi x4, x0, 0
sw_test_19_loop_8:
    li x2, VMEM_BASE + 0x100
    slui x1, 0x30011
    saddi x1, x1, 547
    nop
    sw x1, 4(x2)
    lw x14, 4(x2)
    slui x7, 0x30011
    saddi x7, x7, 547
    sbne x14, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sw_test_19_loop_8
    nop
    nop
test_20:
    saddi x3, x0, 20
    saddi x4, x0, 0
sw_test_20_loop_9:
    li x2, VMEM_BASE + 0x100
    slui x1, 0x33001
    saddi x1, x1, 290
    nop
    nop
    sw x1, 8(x2)
    lw x14, 8(x2)
    slui x7, 0x33001
    saddi x7, x7, 290
    sbne x14, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sw_test_20_loop_9
    nop
    nop
test_21:
    saddi x3, x0, 21
    saddi x4, x0, 0
sw_test_21_loop_10:
    li x2, VMEM_BASE + 0x100
    nop
    slui x1, 0x23300
    saddi x1, x1, 274
    sw x1, 12(x2)
    lw x14, 12(x2)
    slui x7, 0x23300
    saddi x7, x7, 274
    sbne x14, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sw_test_21_loop_10
    nop
    nop
test_22:
    saddi x3, x0, 22
    saddi x4, x0, 0
sw_test_22_loop_11:
    li x2, VMEM_BASE + 0x100
    nop
    slui x1, 0x22330
    saddi x1, x1, 17
    nop
    sw x1, 16(x2)
    lw x14, 16(x2)
    slui x7, 0x22330
    saddi x7, x7, 17
    sbne x14, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sw_test_22_loop_11
    nop
    nop
test_23:
    saddi x3, x0, 23
    saddi x4, x0, 0
sw_test_23_loop_12:
    li x2, VMEM_BASE + 0x100
    nop
    nop
    slui x1, 0x12233
    saddi x1, x1, 1
    sw x1, 20(x2)
    lw x14, 20(x2)
    slui x7, 0x12233
    saddi x7, x7, 1
    sbne x14, x7, fail
    nop
    nop
    saddi x4, x4, 1
    saddi x5, x0, 2
    sbne x4, x5, sw_test_23_loop_12
    nop
    nop

test_pass:
    saddi x3, x0, 0
    saddi x10, x0, 0
    secall

fail:
    saddi x10, x0, 1
    secall
