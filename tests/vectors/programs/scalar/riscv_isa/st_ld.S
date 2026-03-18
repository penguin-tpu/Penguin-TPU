# Flattened from upstream riscv-tests rv32ui/st_ld.S
#
# Penguin test-result convention for imported riscv-tests programs:
#   x10 = 0 on pass, 1 on fail
#   x3  = failing upstream test number on fail, 0 on pass
#   secall terminates the test and returns control to the host harness

test_2:
    saddi x3, x0, 2
    li x2, VMEM_BASE + 0x100
    slui x1, 0x100000
    saddi x1, x1, -35
    sb x1, 0(x2)
    lb x14, 0(x2)
    slui x7, 0x100000
    saddi x7, x7, -35
    sbne x14, x7, fail
    nop
    nop
test_3:
    saddi x3, x0, 3
    li x2, VMEM_BASE + 0x100
    slui x1, 0x100000
    saddi x1, x1, -51
    sb x1, 1(x2)
    lb x14, 1(x2)
    slui x7, 0x100000
    saddi x7, x7, -51
    sbne x14, x7, fail
    nop
    nop
test_4:
    saddi x3, x0, 4
    li x2, VMEM_BASE + 0x100
    slui x1, 0x100000
    saddi x1, x1, -52
    sb x1, 2(x2)
    lb x14, 2(x2)
    slui x7, 0x100000
    saddi x7, x7, -52
    sbne x14, x7, fail
    nop
    nop
test_5:
    saddi x3, x0, 5
    li x2, VMEM_BASE + 0x100
    slui x1, 0x100000
    saddi x1, x1, -68
    sb x1, 3(x2)
    lb x14, 3(x2)
    slui x7, 0x100000
    saddi x7, x7, -68
    sbne x14, x7, fail
    nop
    nop
test_6:
    saddi x3, x0, 6
    li x2, VMEM_BASE + 0x100
    slui x1, 0x100000
    saddi x1, x1, -69
    sb x1, 4(x2)
    lb x14, 4(x2)
    slui x7, 0x100000
    saddi x7, x7, -69
    sbne x14, x7, fail
    nop
    nop
test_7:
    saddi x3, x0, 7
    li x2, VMEM_BASE + 0x100
    slui x1, 0x100000
    saddi x1, x1, -85
    sb x1, 5(x2)
    lb x14, 5(x2)
    slui x7, 0x100000
    saddi x7, x7, -85
    sbne x14, x7, fail
    nop
    nop
test_8:
    saddi x3, x0, 8
    li x2, VMEM_BASE + 0x100
    saddi x1, x0, 51
    sb x1, 0(x2)
    lb x14, 0(x2)
    saddi x7, x0, 51
    sbne x14, x7, fail
    nop
    nop
test_9:
    saddi x3, x0, 9
    li x2, VMEM_BASE + 0x100
    saddi x1, x0, 35
    sb x1, 1(x2)
    lb x14, 1(x2)
    saddi x7, x0, 35
    sbne x14, x7, fail
    nop
    nop
test_10:
    saddi x3, x0, 10
    li x2, VMEM_BASE + 0x100
    saddi x1, x0, 34
    sb x1, 2(x2)
    lb x14, 2(x2)
    saddi x7, x0, 34
    sbne x14, x7, fail
    nop
    nop
test_11:
    saddi x3, x0, 11
    li x2, VMEM_BASE + 0x100
    saddi x1, x0, 18
    sb x1, 3(x2)
    lb x14, 3(x2)
    saddi x7, x0, 18
    sbne x14, x7, fail
    nop
    nop
test_12:
    saddi x3, x0, 12
    li x2, VMEM_BASE + 0x100
    saddi x1, x0, 17
    sb x1, 4(x2)
    lb x14, 4(x2)
    saddi x7, x0, 17
    sbne x14, x7, fail
    nop
    nop
test_13:
    saddi x3, x0, 13
    li x2, VMEM_BASE + 0x100
    saddi x1, x0, 1
    sb x1, 5(x2)
    lb x14, 5(x2)
    saddi x7, x0, 1
    sbne x14, x7, fail
    nop
    nop
test_14:
    saddi x3, x0, 14
    li x2, VMEM_BASE + 0x100
    saddi x1, x0, 51
    sb x1, 0(x2)
    lbu x14, 0(x2)
    saddi x7, x0, 51
    sbne x14, x7, fail
    nop
    nop
test_15:
    saddi x3, x0, 15
    li x2, VMEM_BASE + 0x100
    saddi x1, x0, 35
    sb x1, 1(x2)
    lbu x14, 1(x2)
    saddi x7, x0, 35
    sbne x14, x7, fail
    nop
    nop
test_16:
    saddi x3, x0, 16
    li x2, VMEM_BASE + 0x100
    saddi x1, x0, 34
    sb x1, 2(x2)
    lbu x14, 2(x2)
    saddi x7, x0, 34
    sbne x14, x7, fail
    nop
    nop
test_17:
    saddi x3, x0, 17
    li x2, VMEM_BASE + 0x100
    saddi x1, x0, 18
    sb x1, 3(x2)
    lbu x14, 3(x2)
    saddi x7, x0, 18
    sbne x14, x7, fail
    nop
    nop
test_18:
    saddi x3, x0, 18
    li x2, VMEM_BASE + 0x100
    saddi x1, x0, 17
    sb x1, 4(x2)
    lbu x14, 4(x2)
    saddi x7, x0, 17
    sbne x14, x7, fail
    nop
    nop
test_19:
    saddi x3, x0, 19
    li x2, VMEM_BASE + 0x100
    saddi x1, x0, 1
    sb x1, 5(x2)
    lbu x14, 5(x2)
    saddi x7, x0, 1
    sbne x14, x7, fail
    nop
    nop
test_20:
    saddi x3, x0, 20
    li x2, VMEM_BASE + 0x100
    slui x1, 0xaabbd
    saddi x1, x1, -803
    sw x1, 0(x2)
    lw x14, 0(x2)
    slui x7, 0xaabbd
    saddi x7, x7, -803
    sbne x14, x7, fail
    nop
    nop
test_21:
    saddi x3, x0, 21
    li x2, VMEM_BASE + 0x100
    slui x1, 0xdaabc
    saddi x1, x1, -819
    sw x1, 4(x2)
    lw x14, 4(x2)
    slui x7, 0xdaabc
    saddi x7, x7, -819
    sbne x14, x7, fail
    nop
    nop
test_22:
    saddi x3, x0, 22
    li x2, VMEM_BASE + 0x100
    slui x1, 0xddaac
    saddi x1, x1, -1076
    sw x1, 8(x2)
    lw x14, 8(x2)
    slui x7, 0xddaac
    saddi x7, x7, -1076
    sbne x14, x7, fail
    nop
    nop
test_23:
    saddi x3, x0, 23
    li x2, VMEM_BASE + 0x100
    slui x1, 0xcddab
    saddi x1, x1, -1092
    sw x1, 12(x2)
    lw x14, 12(x2)
    slui x7, 0xcddab
    saddi x7, x7, -1092
    sbne x14, x7, fail
    nop
    nop
test_24:
    saddi x3, x0, 24
    li x2, VMEM_BASE + 0x100
    slui x1, 0xccddb
    saddi x1, x1, -1349
    sw x1, 16(x2)
    lw x14, 16(x2)
    slui x7, 0xccddb
    saddi x7, x7, -1349
    sbne x14, x7, fail
    nop
    nop
test_25:
    saddi x3, x0, 25
    li x2, VMEM_BASE + 0x100
    slui x1, 0xbccde
    saddi x1, x1, -1365
    sw x1, 20(x2)
    lw x14, 20(x2)
    slui x7, 0xbccde
    saddi x7, x7, -1365
    sbne x14, x7, fail
    nop
    nop
test_26:
    saddi x3, x0, 26
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
test_27:
    saddi x3, x0, 27
    li x2, VMEM_BASE + 0x100
    slui x1, 0x30011
    saddi x1, x1, 547
    sw x1, 4(x2)
    lw x14, 4(x2)
    slui x7, 0x30011
    saddi x7, x7, 547
    sbne x14, x7, fail
    nop
    nop
test_28:
    saddi x3, x0, 28
    li x2, VMEM_BASE + 0x100
    slui x1, 0x33001
    saddi x1, x1, 290
    sw x1, 8(x2)
    lw x14, 8(x2)
    slui x7, 0x33001
    saddi x7, x7, 290
    sbne x14, x7, fail
    nop
    nop
test_29:
    saddi x3, x0, 29
    li x2, VMEM_BASE + 0x100
    slui x1, 0x23300
    saddi x1, x1, 274
    sw x1, 12(x2)
    lw x14, 12(x2)
    slui x7, 0x23300
    saddi x7, x7, 274
    sbne x14, x7, fail
    nop
    nop
test_30:
    saddi x3, x0, 30
    li x2, VMEM_BASE + 0x100
    slui x1, 0x22330
    saddi x1, x1, 17
    sw x1, 16(x2)
    lw x14, 16(x2)
    slui x7, 0x22330
    saddi x7, x7, 17
    sbne x14, x7, fail
    nop
    nop
test_31:
    saddi x3, x0, 31
    li x2, VMEM_BASE + 0x100
    slui x1, 0x12233
    saddi x1, x1, 1
    sw x1, 20(x2)
    lw x14, 20(x2)
    slui x7, 0x12233
    saddi x7, x7, 1
    sbne x14, x7, fail
    nop
    nop
test_32:
    saddi x3, x0, 32
    li x2, VMEM_BASE + 0x100
    slui x1, 0xffffd
    saddi x1, x1, -803
    sh x1, 0(x2)
    lh x14, 0(x2)
    slui x7, 0xffffd
    saddi x7, x7, -803
    sbne x14, x7, fail
    nop
    nop
test_33:
    saddi x3, x0, 33
    li x2, VMEM_BASE + 0x100
    slui x1, 0xffffc
    saddi x1, x1, -819
    sh x1, 2(x2)
    lh x14, 2(x2)
    slui x7, 0xffffc
    saddi x7, x7, -819
    sbne x14, x7, fail
    nop
    nop
test_34:
    saddi x3, x0, 34
    li x2, VMEM_BASE + 0x100
    slui x1, 0xffffc
    saddi x1, x1, -1076
    sh x1, 4(x2)
    lh x14, 4(x2)
    slui x7, 0xffffc
    saddi x7, x7, -1076
    sbne x14, x7, fail
    nop
    nop
test_35:
    saddi x3, x0, 35
    li x2, VMEM_BASE + 0x100
    slui x1, 0xffffb
    saddi x1, x1, -1092
    sh x1, 6(x2)
    lh x14, 6(x2)
    slui x7, 0xffffb
    saddi x7, x7, -1092
    sbne x14, x7, fail
    nop
    nop
test_36:
    saddi x3, x0, 36
    li x2, VMEM_BASE + 0x100
    slui x1, 0xffffb
    saddi x1, x1, -1349
    sh x1, 8(x2)
    lh x14, 8(x2)
    slui x7, 0xffffb
    saddi x7, x7, -1349
    sbne x14, x7, fail
    nop
    nop
test_37:
    saddi x3, x0, 37
    li x2, VMEM_BASE + 0x100
    slui x1, 0xffffe
    saddi x1, x1, -1365
    sh x1, 10(x2)
    lh x14, 10(x2)
    slui x7, 0xffffe
    saddi x7, x7, -1365
    sbne x14, x7, fail
    nop
    nop
test_38:
    saddi x3, x0, 38
    li x2, VMEM_BASE + 0x100
    slui x1, 0x00002
    saddi x1, x1, 563
    sh x1, 0(x2)
    lh x14, 0(x2)
    slui x7, 0x00002
    saddi x7, x7, 563
    sbne x14, x7, fail
    nop
    nop
test_39:
    saddi x3, x0, 39
    li x2, VMEM_BASE + 0x100
    slui x1, 0x00001
    saddi x1, x1, 547
    sh x1, 2(x2)
    lh x14, 2(x2)
    slui x7, 0x00001
    saddi x7, x7, 547
    sbne x14, x7, fail
    nop
    nop
test_40:
    saddi x3, x0, 40
    li x2, VMEM_BASE + 0x100
    slui x1, 0x00001
    saddi x1, x1, 290
    sh x1, 4(x2)
    lh x14, 4(x2)
    slui x7, 0x00001
    saddi x7, x7, 290
    sbne x14, x7, fail
    nop
    nop
test_41:
    saddi x3, x0, 41
    li x2, VMEM_BASE + 0x100
    saddi x1, x0, 274
    sh x1, 6(x2)
    lh x14, 6(x2)
    saddi x7, x0, 274
    sbne x14, x7, fail
    nop
    nop
test_42:
    saddi x3, x0, 42
    li x2, VMEM_BASE + 0x100
    saddi x1, x0, 17
    sh x1, 8(x2)
    lh x14, 8(x2)
    saddi x7, x0, 17
    sbne x14, x7, fail
    nop
    nop
test_43:
    saddi x3, x0, 43
    li x2, VMEM_BASE + 0x100
    slui x1, 0x00003
    saddi x1, x1, 1
    sh x1, 10(x2)
    lh x14, 10(x2)
    slui x7, 0x00003
    saddi x7, x7, 1
    sbne x14, x7, fail
    nop
    nop
test_44:
    saddi x3, x0, 44
    li x2, VMEM_BASE + 0x100
    slui x1, 0x00002
    saddi x1, x1, 563
    sh x1, 0(x2)
    lhu x14, 0(x2)
    slui x7, 0x00002
    saddi x7, x7, 563
    sbne x14, x7, fail
    nop
    nop
test_45:
    saddi x3, x0, 45
    li x2, VMEM_BASE + 0x100
    slui x1, 0x00001
    saddi x1, x1, 547
    sh x1, 2(x2)
    lhu x14, 2(x2)
    slui x7, 0x00001
    saddi x7, x7, 547
    sbne x14, x7, fail
    nop
    nop
test_46:
    saddi x3, x0, 46
    li x2, VMEM_BASE + 0x100
    slui x1, 0x00001
    saddi x1, x1, 290
    sh x1, 4(x2)
    lhu x14, 4(x2)
    slui x7, 0x00001
    saddi x7, x7, 290
    sbne x14, x7, fail
    nop
    nop
test_47:
    saddi x3, x0, 47
    li x2, VMEM_BASE + 0x100
    saddi x1, x0, 274
    sh x1, 6(x2)
    lhu x14, 6(x2)
    saddi x7, x0, 274
    sbne x14, x7, fail
    nop
    nop
test_48:
    saddi x3, x0, 48
    li x2, VMEM_BASE + 0x100
    saddi x1, x0, 17
    sh x1, 8(x2)
    lhu x14, 8(x2)
    saddi x7, x0, 17
    sbne x14, x7, fail
    nop
    nop
test_49:
    saddi x3, x0, 49
    li x2, VMEM_BASE + 0x100
    slui x1, 0x00003
    saddi x1, x1, 1
    sh x1, 10(x2)
    lhu x14, 10(x2)
    slui x7, 0x00003
    saddi x7, x7, 1
    sbne x14, x7, fail
    nop
    nop
    saddi x10, x0, 239
    li x11, VMEM_BASE + 0x100
    sb x10, 3(x11)
    lb x12, 3(x11)

test_pass:
    saddi x3, x0, 0
    saddi x10, x0, 0
    secall

fail:
    saddi x10, x0, 1
    secall
