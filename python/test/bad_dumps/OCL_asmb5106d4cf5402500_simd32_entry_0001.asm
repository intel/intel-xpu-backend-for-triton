//.kernel kernel
//.platform PVCXT
//.thread_config numGRF=128, numAcc=4, numSWSB=16
//.options_string "-emitCrossThreadOffR0Reloc "
//.full_options "-emitLocation -enableCoalesceScalarMoves -hasRNEandDenorm -noStitchExternFunc -emitCrossThreadOffR0Reloc -linker 63 -preserver0 -abortOnSpill 4 -enableBundleCR 3 -freqBasedSpillCost 8 -freqBasedSpillCostFunc 1 -boundsChecking -presched-ctrl 6 -presched-rp 100 -nodpsendreorder -SBIDDepLoc -PVCSendWARWA -output -binary -dumpcommonisa -dumpcombinedcisa -dumpvisa -printHexFloatInAsm -noverifyCISA -enableHalfLSC -partialInt64 -noSendSrcDstOverlap -activeThreadsOnlyBarrier "
//.instCount 67
//.RA type	LOCAL_ROUND_ROBIN_RA
//.git-hash 6cf36b7b97350cc07149a285f84685325287b49e

//.declare BuiltInR0 (0)  rf=r size=64 type=ud align=32 words (r0.0) IsBuiltin
//.declare  (1)  rf=r size=64 type=ud alias=BuiltInR0+0 align=32 words (r0.0) IsBuiltin
//.declare BuiltinA0 (2)  rf=a size=4 type=ud align=1 words (a0.0) IsBuiltin
//.declare BuiltinA0Dot2 (3)  rf=a size=4 type=ud align=1 words (a0.2) IsBuiltin
//.declare BuiltinSR0Dot1 (5)  rf=r size=4 type=ud align=2 words IsBuiltin
//.declare %null (10)  rf=r size=4 type=ud align=2 words
//.declare %local_id_x (13)  rf=r size=4 type=ud align=2 words (r1.10)
//.declare %local_id_y (14)  rf=r size=4 type=ud align=2 words (r1.11)
//.declare %local_size_x (15)  rf=r size=4 type=ud align=2 words (r1.6)
//.declare %local_size_y (16)  rf=r size=4 type=ud align=2 words (r1.7)
//.declare %group_id_x (17)  rf=r size=4 type=ud align=2 words (r0.1)
//.declare %group_id_y (18)  rf=r size=4 type=ud align=2 words (r0.6)
//.declare %group_id_z (19)  rf=r size=4 type=ud align=2 words (r0.7)
//.declare %group_count_x (20)  rf=r size=4 type=ud align=2 words (r1.8)
//.declare %group_count_y (21)  rf=r size=4 type=ud align=2 words (r1.9)
//.declare %tsc (22)  rf=r size=20 type=ud align=2 words
//.declare %arg (23)  rf=r size=0 type=ud align=32 words (r26.0)
//.declare %retval (24)  rf=r size=0 type=ud align=32 words (r26.0) Output
//.declare %sp (25)  rf=r size=8 type=uq align=4 words (r127.3)
//.declare %fp (26)  rf=r size=8 type=uq align=4 words (r127.2)
//.declare %sr0 (27)  rf=r size=16 type=ud align=2 words
//.declare %cr0 (28)  rf=r size=12 type=ud align=2 words
//.declare %ce0 (29)  rf=r size=4 type=ud align=2 words
//.declare %dbg0 (30)  rf=r size=8 type=ud align=2 words
//.declare implBufPtr (32)  rf=r size=8 type=uq align=4 words (r126.0)
//.declare localIdBufPtr (33)  rf=r size=8 type=uq align=4 words (r126.3)
//.declare %msg0 (34)  rf=r size=12 type=ud align=2 words
//.declare %null (35)  rf=r size=4 type=ud align=2 words
//.declare V0033 (43)  rf=r size=64 type=d alias=+0 align=32 words (r0.0)
//.declare V0034 (44)  rf=r size=8 type=uq align=4 words (r4.4)
//.declare V0035 (45)  rf=r size=8 type=uq align=4 words (r4.5)
//.declare V0036 (46)  rf=r size=8 type=uq align=4 words (r4.6)
//.declare V0037 (47)  rf=r size=8 type=uq align=4 words (r4.7)
//.declare V0039 (49)  rf=r size=32 type=d alias=+0 align=32 words (r0.0)
//.declare V0040 (50)  rf=r size=32 type=d align=16 words (r4.0)
//.declare V0041 (51)  rf=r size=64 type=w align=32 words (r1.0)
//.declare V0042 (52)  rf=r size=64 type=w align=32 words (r2.0)
//.declare V0043 (53)  rf=r size=64 type=w align=32 words (r3.0)
//.declare V0044 (54)  rf=r size=8 type=uq align=4 words (r5.2)
//.declare V0051 (61)  rf=r size=64 type=w align=32 words (r2.0)
//.declare V0052 (62)  rf=r size=8 type=q alias=V0034+0 align=4 words (r4.4)
//.declare V0054 (64)  rf=r size=64 type=uw alias=V0051+0 align=32 words (r2.0)
//.declare V0056 (66)  rf=r size=256 type=q align=32 words (r5.0)
//.declare V0057 (67)  rf=r size=256 type=uq alias=V0056+0 align=32 words (r5.0)
//.declare V0058 (68)  rf=r size=128 type=f align=32 words (r9.0)
//.declare V0059 (69)  rf=r size=8 type=q alias=V0035+0 align=4 words (r4.5)
//.declare V0060 (70)  rf=r size=256 type=q align=32 words (r11.0)
//.declare V0061 (71)  rf=r size=256 type=uq alias=V0060+0 align=32 words (r11.0)
//.declare V0062 (72)  rf=r size=128 type=f align=32 words (r15.0)
//.declare V0063 (73)  rf=r size=128 type=d align=32 words (r17.0)
//.declare V0064 (74)  rf=r size=128 type=d alias=V0062+0 align=32 words (r15.0)
//.declare P01 (75)  rf=f32  size=4 type=uw align=2 words (f2.0)
//.declare V0065 (76)  rf=r size=128 type=f align=32 words (r19.0)
//.declare P02 (77)  rf=f32  size=4 type=uw align=2 words (f0.0)
//.declare V0066 (78)  rf=r size=128 type=ud alias=V0063+0 align=32 words (r17.0)
//.declare V0067 (79)  rf=r size=128 type=f align=32 words (r21.0)
//.declare V0068 (80)  rf=r size=128 type=f align=32 words (r23.0)
//.declare V0069 (81)  rf=r size=128 type=f align=32 words (r25.0)
//.declare V0070 (82)  rf=r size=128 type=f align=32 words (r27.0)
//.declare V0071 (83)  rf=r size=128 type=f align=32 words (r29.0)
//.declare P03 (84)  rf=f32  size=4 type=uw align=2 words (f3.0)
//.declare P04 (86)  rf=f32  size=4 type=uw align=2 words (f1.0)
//.declare P05 (87)  rf=f32  size=4 type=uw align=2 words (f1.0)
//.declare P06 (88)  rf=f32  size=4 type=uw align=2 words (f0.0)
//.declare V0073 (89)  rf=r size=128 type=f align=32 words (r31.0)
//.declare V0074 (90)  rf=r size=8 type=q alias=V0036+0 align=4 words (r4.6)
//.declare V0075 (91)  rf=r size=256 type=q align=32 words (r33.0)
//.declare V0076 (92)  rf=r size=256 type=uq alias=V0075+0 align=32 words (r33.0)
//.declare V0077 (93)  rf=r size=8 type=q alias=V0037+0 align=4 words (r4.7)
//.declare V0078 (94)  rf=r size=256 type=q align=32 words (r37.0)
//.declare V0079 (95)  rf=r size=256 type=uq alias=V0078+0 align=32 words (r37.0)
//.declare V0080 (96)  rf=r size=8 type=uq align=4 words (r5.0)
//.declare V0081 (97)  rf=r size=8 type=uq align=4 words (r5.1)
//.declare  (98)  rf=r size=64 type=ud align=32 words (r127.0)
//.declare  (99)  rf=r size=64 type=uw align=32 words (r3.0)
//.declare  (100)  rf=r size=64 type=uw align=32 words (r41.0)
//.declare  (101)  rf=r size=4 type=f align=2 words (r4.0)
//.declare  (102)  rf=r size=2 type=uw align=1 words (r4.2)
//.declare  (104)  rf=r size=128 type=uw align=32 words (r42.0)
//.declare  (105)  rf=r size=128 type=uw align=32 words (r44.0)
//.declare  (108)  rf=r size=128 type=q align=32 words (r46.0)
//.declare  (109)  rf=r size=128 type=q align=32 words (r48.0)
//.declare r0 (110)  rf=r size=64 type=ud align=32 words (r0.0)
//.declare rtmp (111)  rf=r size=64 type=ud align=32 words (r127.0)
//.declare  (112)  rf=r size=128 type=ud align=32 words (r1.0)
//.declare  (113)  rf=r size=4 type=ud align=2 words (r126.0)
//.declare  (114)  rf=r size=64 type=ud align=32 words (r3.0)
//.declare  (115)  rf=r size=64 type=ud align=32 words (r4.0)
//.declare  (116)  rf=r size=4 type=ud align=2 words (r126.0)
//.declare  (117)  rf=r size=32 type=ud align=2 words (r5.0)
//.declare  (118)  rf=r size=4 type=ud align=2 words (r127.0)
//.declare  (119)  rf=r size=4 type=ud align=2 words (r126.0)
//.declare  (120)  rf=r size=4 type=ud align=2 words (r127.0)
//.declare  (121)  rf=r size=4 type=ud align=2 words (r126.0)

// .inputs
// +----------+----------+--------+----------+------------------+
// | id       | type     |  bytes | at       | from             |
// +----------+----------+--------+----------+------------------+
// | V0041    | :w x 32  |   0x40 | r1       | pti[tid]+0x0     |
// | V0042    | :w x 32  |   0x40 | r2       | pti[tid]+0x40    |
// | V0043    | :w x 32  |   0x40 | r3       | pti[tid]+0x80    |
// | V0040    | :d x 8   |   0x20 | r4       | cti+0x0          |
// | V0034    | :uq      |    0x8 | r4+0x20  | cti+0x20         |
// | V0035    | :uq      |    0x8 | r4+0x28  | cti+0x28         |
// | V0036    | :uq      |    0x8 | r4+0x30  | cti+0x30         |
// | V0037    | :uq      |    0x8 | r4+0x38  | cti+0x38         |
// | V0080    | :uq      |    0x8 | r5       | cti+0x40         |
// | V0081    | :uq      |    0x8 | r5+0x8   | cti+0x48         |
// | V0044    | :uq      |    0x8 | r5+0x10  | cti+0x50         |
// +----------+----------+--------+----------+------------------+


// B000: Preds:{},  Succs:{B001}
per_thread_prolog:
(W)     mov (16|M0)              r127.0<1>:ud  0x0:ud                                                //  ALU pipe: int; 
(W)     and (1|M0)               r127.2<1>:ud  r0.0<0;1,0>:ud    0xFFFFFFC0:ud                       //  ALU pipe: int; 
(W)     and (1|M0)               r127.0<1>:uw  r0.4<0;1,0>:uw    0xFF:uw                             //  ALU pipe: int; 
(W)     add (1|M0)               r127.2<1>:ud  r127.2<0;1,0>:ud  0x60:ud              {I@2}          //  ALU pipe: int; 
(W)     add (1|M0)               r127.2<1>:ud  r127.2<0;1,0>:ud  0x0:ud              {I@1}           //  R_SYM_ADDR_32: __INTEL_PATCH_CROSS_THREAD_OFFSET_OFF_R0; ALU pipe: int; 
(W)     mad (1|M0)               r127.0<1>:ud  r127.2<0;0>:ud    r127.0<0;0>:uw    0xC0:uw              {I@1} //  ALU pipe: int; 
(W)     load.ugm.d32x32t.a32.ca.ca (1|M0)  r1:2 bti[255][r127:1]   {A@1,$0} // ex_desc:0xFF000000; desc:0x6228E500 // 
(W)     add (1|M0)               r126.0<1>:ud  r127.0<0;1,0>:ud  0x80:uw                             //  ALU pipe: int; 
(W)     load.ugm.d32x16t.a32.ca.ca (1|M0)  r3:1 bti[255][r126:1]   {A@1,$1} // ex_desc:0xFF000000; desc:0x6218D500 // 
(W)     mov (1|M0)               null<1>:ud    0x30433:ud                              {$1.src}      // 
(W)     mov (1|M0)               null<1>:ud    r127.0<0;1,0>:ud                                      // 
(W)     mov (1|M0)               null<1>:ud    r126.0<0;1,0>:ud                                      // 
        sync.nop                             null                             {I@1}                  // 
        sync.allrd                           ($0)                                                    // 
        nop                                                                                          // 
        nop                                                                                          // 
// B001: Preds:{B000},  Succs:{B002}
// cross_thread_prolog:
(W)     and (1|M0)               r127.0<1>:ud  r0.0<0;1,0>:ud    0xFFFFFFC0:ud                       //  ALU pipe: int; 
(W)     add (1|M0)               r127.0<1>:ud  r127.0<0;1,0>:ud  0x0:ud              {I@1}           //  R_SYM_ADDR_32: __INTEL_PATCH_CROSS_THREAD_OFFSET_OFF_R0; ALU pipe: int; 
(W)     load.ugm.d32x16t.a32.ca.ca (1|M0)  r4:1 bti[255][r127:1]   {I@1,$2} // ex_desc:0xFF000000; desc:0x6218D500 // 
(W)     add (1|M0)               r126.0<1>:ud  r127.0<0;1,0>:ud  0x40:uw                             //  ALU pipe: int; 
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r5:1  bti[255][r126:1]   {I@1,$3} // ex_desc:0xFF000000; desc:0x6218C500 // 
(W)     mov (1|M0)               null<1>:ud    0x30433:ud                              {$3.src}      // 
(W)     mov (1|M0)               null<1>:ud    r127.0<0;1,0>:ud                 {Compacted}          // 
(W)     mov (1|M0)               null<1>:ud    r126.0<0;1,0>:ud                 {Compacted}          // 
        sync.nop                             null                             {Compacted,I@1}        // 
        sync.allrd                           ($2)                 {Compacted}                        // 
// B002: Preds:{B001},  Succs:{}
// _main:
(W)     or (1|M0)                cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x4C0:uw              {Compacted,A@1} // $1
        and (32|M0)              r2.0<1>:w     r1.0<1;1,0>:w     127:w               {A@1,$0.dst}    //  ALU pipe: int; $2
        mov (16|M0)              r42.0<4>:uw   r2.0<1;1,0>:uw                   {I@1}                //  ALU pipe: int; $4
        mov (16|M16)             r44.0<4>:uw   r2.16<1;1,0>:uw                                       //  ALU pipe: int; $4
        shl (16|M0)              r46.0<1>:q    r42.0<4;1,0>:uw   2:w               {I@2}             //  ALU pipe: int; $4
        shl (16|M16)             r48.0<1>:q    r44.0<4;1,0>:uw   2:w               {I@2}             //  ALU pipe: int; $4
        sync.nop                             null                             {Compacted,I@2}        // $7
        add (16|M0)              r11.0<1>:q    r46.0<1;1,0>:q    r4.5<0;1,0>:q    {Compacted,$2.dst} //  ALU pipe: int; $7
        add (16|M16)             r13.0<1>:q    r48.0<1;1,0>:q    r4.5<0;1,0>:q    {Compacted,I@2}    //  ALU pipe: int; $7
        load.ugm.d32.a64 (32|M0)  r15:2         [r11:4]            {A@1,$4} // ex_desc:0x0; desc:0x8200580 // $8
        add (16|M0)              r5.0<1>:q     r46.0<1;1,0>:q    r4.4<0;1,0>:q    {Compacted,$3.dst} //  ALU pipe: int; $5
        add (16|M16)             r7.0<1>:q     r48.0<1;1,0>:q    r4.4<0;1,0>:q    {Compacted}        //  ALU pipe: int; $5
        load.ugm.d32.a64 (32|M0)  r9:2          [r5:4]             {A@1,$5} // ex_desc:0x0; desc:0x8200580 // $6
(W)     mov (1|M0)               r4.0<1>:f     0x4F800000:f                               {Compacted} //  ALU pipe: float; $11
(W)     mov (1|M0)               r4.2<1>:hf    0x1:hf                                                //  ALU pipe: float; $21
        add (16|M0)              r33.0<1>:q    r46.0<1;1,0>:q    r4.6<0;1,0>:q    {Compacted}        //  ALU pipe: int; $26
        add (16|M16)             r35.0<1>:q    r48.0<1;1,0>:q    r4.6<0;1,0>:q    {Compacted}        //  ALU pipe: int; $26
        add (16|M0)              r37.0<1>:q    r46.0<1;1,0>:q    r4.7<0;1,0>:q    {Compacted}        //  ALU pipe: int; $28
        add (16|M16)             r39.0<1>:q    r48.0<1;1,0>:q    r4.7<0;1,0>:q    {Compacted}        //  ALU pipe: int; $28
(W)     mov (16|M0)              r127.0<1>:f   r0.0<1;1,0>:f                    {Compacted}          //  ALU pipe: float; $30
        and (32|M0)   (eq)f2.0   r17.0<1>:d    r15.0<1;1,0>:d    2139095040:d               {$4.dst} //  ALU pipe: int; $9
        cmp (32|M0)   (ge)f0.0   null<1>:ud    r17.0<1;1,0>:ud   0x64000000:ud              {I@1}    //  ALU pipe: int; $12
(f2.0)  sel (32|M0)              acc0.0<1>:f   r4.0<0;1,0>:f     0x3F800000:f               {F@3}    //  ALU pipe: float; $11
        and (32|M0)   (eq)f3.0   null<1>:d     r15.0<1;1,0>:d    8388607:d                           //  ALU pipe: int; $18
        cmp (32|M0)   (eq)f1.0   null<1>:d     r17.0<1;1,0>:d    0:w                                 //  ALU pipe: int; $20
(~f0.0) sel (32|M0)              acc0.0<1>:f   acc0.0<1;1,0>:f   0x2F800000:f                        //  ALU pipe: float; $13
        mul (32|M0)              r23.0<1>:f    r15.0<1;1,0>:f    acc0.0<1;1,0>:f                     //  ALU pipe: float; $14
(f3.0)  sel (32|M0)              r41.0<1>:uw   r4.2<0;1,0>:uw    0x0:uw              {F@5}           //  ALU pipe: int; $21
(f1.0)  sel (32|M0)              r3.0<1>:uw    r4.2<0;1,0>:uw    0x0:uw              {$1.dst}        //  ALU pipe: int; $21
        math.inv (32|M0)         r25.0<1>:f    r23.0<1;1,0>:f                   {F@1}                //  ALU pipe: math; $15
        or (32|M0)    (ne)f1.0   null<2>:uw    r3.0<1;1,0>:uw    r41.0<1;1,0>:uw  {I@1}              //  ALU pipe: int; $21
(W)     not (1|M0)               f0.0<1>:ud    f1.0<0;1,0>:ud                                        //  ALU pipe: int; $22
        sync.nop                             null                             {Compacted,M@1}        // $16
        mul (32|M0)              r27.0<1>:f    r25.0<1;1,0>:f    r9.0<1;1,0>:f    {Compacted,$5.dst} //  ALU pipe: float; $16
(f0.0)  cmp (32|M0)   (eq)f0.0   null<1>:f     r9.0<1;1,0>:f     r15.0<1;1,0>:f   {I@1}              //  ALU pipe: float; $23
        mul (32|M0)              acc0.0<1>:f   r27.0<1;1,0>:f    acc0.0<1;1,0>:f  {F@2}              //  ALU pipe: float; $17
(~f0.0) sel (32|M0)              r31.0<1>:f    acc0.0<1;1,0>:f   0x3F800000:f                        //  ALU pipe: float; $25
        store.ugm.d32.a64 (32|M0)  [r33:4]      r31:2              {F@1,$6} // ex_desc:0x0; desc:0x8000584 // $27
        store.ugm.d32.a64 (32|M0)  [r37:4]      r31:2              {$7} // ex_desc:0x0; desc:0x8000584 // $29
(W)     send.gtwy (1|M0)         null     r127    null:0  0x0            0x02000010           {EOT,$8} // wr:1+0, rd:0; end of thread // $30
L912:
        nop                                                                                          // $30


//.BankConflicts: 0
//.ByteRMWs: 0
//


//.numALUInst: 56
//.accSubDef: 3
//.accSubUse: 4
//.accSubCandidateDef: 4
//.accSubCandidateUse: 5
//
//
//.singlePipeAtOneDistNum: 12
//.allAtOneDistNum: 6
//.syncInstCount: 2
//.tokenReuseCount: 0
//.AfterWriteTokenDepCount: 6
//.AfterReadTokenDepCount: 0
