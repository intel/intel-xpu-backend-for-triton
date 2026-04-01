//.kernel matmul_kernel_with_tensor_descriptors
//.platform XE2
//.thread_config numGRF=128, numAcc=4, numSWSB=16
//.options_string "-emitCrossThreadOffR0Reloc -hashmovs 743001711 4124613444 -hashmovs1 0 1 "
//.full_options "-emitLocation -enableCoalesceScalarMoves -supportLSCImmScale 0 -samplerHeaderWA -enablePreemptionR0Only -hasRNEandDenorm -noStitchExternFunc -useInlineData -emitCrossThreadOffR0Reloc -TotalGRFNum 128 -abortOnSpill 4 -enableBCR -enableBundleCR 3 -freqBasedSpillCost 8 -freqBasedSpillCostFunc 1 -boundsChecking -presched-rp 100 -nodpsendreorder -SBIDDepLoc -PVCSendWARWA -output -binary -dumpcommonisa -dumpcombinedcisa -dumpvisa -printHexFloatInAsm -noverifyCISA -enableHalfLSC -partialInt64 -activeThreadsOnlyBarrier -generateDebugInfo -hashmovs 743001711 4124613444 -hashmovs1 0 1 "
//.instCount 628
//.RA type	LOCAL_ROUND_ROBIN_BC_RA
//.git-hash 8a9a27bbde4417e20b54507ca89a46693ac9f225

//.declare BuiltInR0 (0)  rf=r size=64 type=ud align=32 words (r0.0) IsBuiltin
//.declare  (1)  rf=r size=64 type=ud align=32 words (r89.0) IsBuiltin
//.declare BuiltinA0 (2)  rf=a size=4 type=ud align=1 words (a0.0) IsBuiltin
//.declare BuiltinA0Dot2 (3)  rf=a size=4 type=ud align=1 words (a0.2) IsBuiltin
//.declare BuiltinSR0Dot1 (5)  rf=r size=4 type=ud align=2 words IsBuiltin
//.declare %null (10)  rf=r size=4 type=ud align=2 words
//.declare %local_id_x (13)  rf=r size=4 type=ud align=2 words (r1.8)
//.declare %local_id_y (14)  rf=r size=4 type=ud align=2 words (r1.9)
//.declare %local_size_x (15)  rf=r size=4 type=ud align=2 words (r1.4)
//.declare %local_size_y (16)  rf=r size=4 type=ud align=2 words (r1.5)
//.declare %group_id_x (17)  rf=r size=4 type=ud align=2 words (r0.1)
//.declare %group_id_y (18)  rf=r size=4 type=ud align=2 words (r0.6)
//.declare %group_id_z (19)  rf=r size=4 type=ud align=2 words (r0.7)
//.declare %group_count_x (20)  rf=r size=4 type=ud align=2 words (r1.6)
//.declare %group_count_y (21)  rf=r size=4 type=ud align=2 words (r1.7)
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
//.declare %scratchloc (35)  rf=r size=8 type=uq align=4 words (s0.7)
//.declare V0033 (43)  rf=r size=64 type=d alias=+0 align=32 words (r89.0)
//.declare V0034 (44)  rf=r size=8 type=uq align=4 words (r4.0)
//.declare V0035 (45)  rf=r size=8 type=uq align=4 words (r4.1)
//.declare V0036 (46)  rf=r size=8 type=uq align=4 words (r4.2)
//.declare V0038 (48)  rf=r size=32 type=d alias=+0 align=32 words (r89.0)
//.declare V0040 (50)  rf=r size=32 type=w align=16 words (r1.0)
//.declare V0041 (51)  rf=r size=32 type=w align=16 words (r2.0)
//.declare V0042 (52)  rf=r size=32 type=w align=16 words (r3.0)
//.declare V0043 (53)  rf=r size=8 type=uq align=4 words (r5.1)
//.declare V0054 (64)  rf=r size=512 type=w align=32 words (r5.0)
//.declare V0055 (65)  rf=r size=1024 type=d align=32 words (r13.0)
//.declare V0056 (66)  rf=r size=512 type=w align=32 words (r29.0)
//.declare V0057 (67)  rf=r size=1024 type=d align=32 words (r37.0)
//.declare V0058 (68)  rf=r size=512 type=w align=32 words (r55.0)
//.declare V0059 (69)  rf=r size=1024 type=d align=32 words (r63.0)
//.declare V0060 (70)  rf=r size=512 type=w align=32 words (r79.0)
//.declare V0061 (71)  rf=r size=1024 type=d align=32 words (r5.0)
//.declare V0062 (72)  rf=r size=512 type=w align=32 words (r23.0)
//.declare V0063 (73)  rf=r size=1024 type=d align=32 words (r31.0)
//.declare V0064 (74)  rf=r size=512 type=w align=32 words (r47.0)
//.declare V0065 (75)  rf=r size=1024 type=d align=32 words (r55.0)
//.declare V0066 (76)  rf=r size=512 type=w align=32 words (r73.0)
//.declare V0067 (77)  rf=r size=1024 type=d align=32 words (r5.0)
//.declare V0068 (78)  rf=r size=512 type=w align=32 words (r21.0)
//.declare V0069 (79)  rf=r size=1024 type=d align=32 words (r29.0)
//.declare V0070 (80)  rf=r size=512 type=w align=32 words (r47.0)
//.declare V0071 (81)  rf=r size=1024 type=d align=32 words (r55.0)
//.declare V0072 (82)  rf=r size=512 type=w align=32 words (r71.0)
//.declare V0073 (83)  rf=r size=1024 type=d align=32 words (r14.0)
//.declare V0074 (84)  rf=r size=512 type=w align=32 words (r33.0)
//.declare V0075 (85)  rf=r size=1024 type=d align=32 words (r41.0)
//.declare V0076 (86)  rf=r size=512 type=w align=32 words (r57.0)
//.declare V0077 (87)  rf=r size=1024 type=d align=32 words (r65.0)
//.declare V0078 (88)  rf=r size=512 type=w align=32 words (r5.0)
//.declare V0079 (89)  rf=r size=1024 type=d align=32 words (r13.0)
//.declare V0080 (90)  rf=r size=512 type=w align=32 words (r29.0)
//.declare V0081 (91)  rf=r size=1024 type=d align=32 words (r37.0)
//.declare V0082 (92)  rf=r size=512 type=w align=32 words (r55.0)
//.declare V0083 (93)  rf=r size=1024 type=d align=32 words (r63.0)
//.declare V0084 (94)  rf=r size=512 type=w align=32 words (r79.0)
//.declare V0085 (95)  rf=r size=1024 type=d align=32 words (r5.0)
//.declare V0086 (96)  rf=r size=512 type=w align=32 words (r23.0)
//.declare V0087 (97)  rf=r size=1024 type=d align=32 words (r31.0)
//.declare V0088 (98)  rf=r size=512 type=w align=32 words (r47.0)
//.declare V0089 (99)  rf=r size=1024 type=d align=32 words (r55.0)
//.declare V0090 (100)  rf=r size=512 type=w align=32 words (r73.0)
//.declare V0091 (101)  rf=r size=1024 type=d align=32 words (r5.0)
//.declare V0092 (102)  rf=r size=512 type=w align=32 words (r21.0)
//.declare V0093 (103)  rf=r size=1024 type=d align=32 words (r29.0)
//.declare V0094 (104)  rf=r size=512 type=w align=32 words (r47.0)
//.declare V0095 (105)  rf=r size=1024 type=d align=32 words (r55.0)
//.declare V0096 (106)  rf=r size=512 type=w align=32 words (r71.0)
//.declare V0097 (107)  rf=r size=1024 type=d align=32 words (r14.0)
//.declare V0098 (108)  rf=r size=512 type=w align=32 words (r33.0)
//.declare V0099 (109)  rf=r size=1024 type=d align=32 words (r41.0)
//.declare V0100 (110)  rf=r size=512 type=w align=32 words (r57.0)
//.declare V0101 (111)  rf=r size=1024 type=d align=32 words (r65.0)
//.declare V0102 (112)  rf=r size=512 type=w align=32 words (r5.0)
//.declare V0103 (113)  rf=r size=1024 type=d align=32 words (r13.0)
//.declare V0104 (114)  rf=r size=512 type=w align=32 words (r29.0)
//.declare V0105 (115)  rf=r size=1024 type=d align=32 words (r37.0)
//.declare V0106 (116)  rf=r size=512 type=w align=32 words (r57.0)
//.declare V0107 (117)  rf=r size=1024 type=d align=32 words (r65.0)
//.declare V0108 (118)  rf=r size=512 type=w align=32 words (r81.0)
//.declare V0109 (119)  rf=r size=1024 type=d align=32 words (r5.0)
//.declare V0110 (120)  rf=r size=512 type=w align=32 words (r25.0)
//.declare V0111 (121)  rf=r size=1024 type=d align=32 words (r33.0)
//.declare V0112 (122)  rf=r size=512 type=w align=32 words (r49.0)
//.declare V0113 (123)  rf=r size=1024 type=d align=32 words (r57.0)
//.declare V0114 (124)  rf=r size=512 type=w align=32 words (r75.0)
//.declare V0115 (125)  rf=r size=1024 type=d align=32 words (r5.0)
//.declare V0116 (126)  rf=r size=512 type=w align=32 words (r23.0)
//.declare V0117 (127)  rf=r size=1024 type=d align=32 words (r31.0)
//.declare V0118 (128)  rf=r size=4 type=d align=2 words (r2.0)
//.declare V0119 (129)  rf=r size=4 type=d align=32 words (r3.0)
//.declare V0120 (130)  rf=r size=4 type=ud alias=V0118+0 align=2 words (r2.0)
//.declare V0121 (131)  rf=r size=4 type=ud alias=V0119+0 align=2 words (r3.0)
//.declare V0122 (132)  rf=r size=4 type=d align=2 words (r125.0)
//.declare V0123 (133)  rf=r size=4 type=ud alias=V0122+0 align=2 words (r125.0)
//.declare V0124 (134)  rf=r size=4 type=d alias=+0 align=2 words (r123.4)
//.declare V0125 (135)  rf=r size=4 type=d alias=+4 align=2 words (r123.5)
//.declare P01 (136)  rf=f16  size=2 type=uw align=1 words (f0.1)
//.declare V0126 (137)  rf=r size=4 type=d align=2 words (r90.0)
//.declare V0127 (138)  rf=r size=4 type=d alias=+0 align=2 words (r2.0)
//.declare V0128 (139)  rf=r size=4 type=d alias=+4 align=2 words (r2.1)
//.declare V0129 (140)  rf=r size=4 type=d align=2 words (r3.0)
//.declare V0130 (141)  rf=r size=4 type=d align=2 words (r4.6)
//.declare V0131 (142)  rf=r size=4 type=d align=2 words (r5.0)
//.declare V0132 (143)  rf=r size=4 type=d align=2 words (r6.0)
//.declare V0133 (144)  rf=r size=4 type=f align=2 words (r8.0)
//.declare V0134 (145)  rf=r size=4 type=ud alias=V0130+0 align=2 words (r4.6)
//.declare V0135 (146)  rf=r size=4 type=d align=2 words (r9.0)
//.declare V0136 (147)  rf=r size=4 type=ud alias=V0135+0 align=2 words (r9.0)
//.declare V0137 (148)  rf=r size=4 type=d alias=+0 align=2 words (r10.0)
//.declare V0138 (149)  rf=r size=4 type=f align=2 words (r11.0)
//.declare V0139 (150)  rf=r size=4 type=ud alias=V0132+0 align=2 words (r6.0)
//.declare V0140 (151)  rf=r size=4 type=f align=2 words (r12.0)
//.declare V0141 (152)  rf=r size=4 type=f align=2 words (r14.0)
//.declare V0142 (153)  rf=r size=4 type=f align=2 words (r15.0)
//.declare V0143 (154)  rf=r size=4 type=d align=2 words (r16.0)
//.declare V0144 (155)  rf=r size=4 type=ud alias=V0143+0 align=2 words (r16.0)
//.declare V0145 (156)  rf=r size=4 type=d alias=+4 align=2 words (r10.1)
//.declare V0146 (157)  rf=r size=4 type=d align=2 words (r17.0)
//.declare V0147 (158)  rf=r size=4 type=ud alias=V0146+0 align=2 words (r17.0)
//.declare V0148 (159)  rf=r size=4 type=f alias=+0 align=2 words (r18.0)
//.declare V0149 (160)  rf=r size=4 type=ud alias=V0137+0 align=2 words (r10.0)
//.declare V0150 (161)  rf=r size=4 type=f alias=+4 align=2 words (r18.1)
//.declare V0151 (162)  rf=r size=4 type=ud alias=V0145+0 align=2 words (r10.1)
//.declare V0152 (163)  rf=r size=4 type=f align=2 words (r19.0)
//.declare V0154 (165)  rf=r size=4 type=f align=2 words (r20.0)
//.declare V0156 (167)  rf=r size=4 type=f align=2 words (r21.0)
//.declare V0157 (168)  rf=r size=4 type=f align=2 words (r22.0)
//.declare V0158 (169)  rf=r size=4 type=f align=2 words (r23.0)
//.declare V0159 (170)  rf=r size=4 type=d align=2 words (r24.0)
//.declare V0160 (171)  rf=r size=4 type=ud alias=V0159+0 align=2 words (r24.0)
//.declare V0161 (172)  rf=r size=4 type=d align=2 words (r25.0)
//.declare V0162 (173)  rf=r size=4 type=d align=2 words (r27.0)
//.declare V0163 (174)  rf=r size=4 type=d align=32 words (r28.0)
//.declare V0164 (175)  rf=r size=4 type=d align=2 words (r29.0)
//.declare V0165 (176)  rf=r size=4 type=d align=2 words (r30.0)
//.declare V0166 (177)  rf=r size=4 type=ud alias=V0164+0 align=2 words (r29.0)
//.declare V0167 (178)  rf=r size=4 type=ud alias=V0165+0 align=2 words (r30.0)
//.declare  (179)  rf=f16  size=2 type=uw align=1 words (f1.1)
//.declare V0168 (180)  rf=r size=4 type=d align=2 words (r31.0)
//.declare V0169 (181)  rf=r size=4 type=d align=32 words (r2.0)
//.declare V0170 (182)  rf=r size=4 type=d align=2 words (r3.0)
//.declare V0171 (183)  rf=r size=4 type=d align=2 words (r126.0)
//.declare V0172 (184)  rf=r size=4 type=d align=2 words (r125.1)
//.declare V0173 (185)  rf=r size=32 type=b alias=V0038+0 align=16 words (r89.0)
//.declare V0174 (186)  rf=r size=1 type=b align=1 words (r4.24)
//.declare V0175 (187)  rf=r size=2 type=w align=1 words (r5.0)
//.declare V0176 (188)  rf=r size=2 type=w align=1 words (r6.0)
//.declare V0177 (189)  rf=r size=4 type=d align=2 words (r7.0)
//.declare V0178 (190)  rf=r size=1 type=ub alias=V0174+0 align=1 words (r4.24)
//.declare V0179 (191)  rf=r size=4 type=d align=2 words (r8.0)
//.declare V0180 (192)  rf=r size=4 type=d align=2 words (r10.0)
//.declare V0181 (193)  rf=r size=8 type=q alias=V0035+0 align=4 words (r4.1)
//.declare V0184 (196)  rf=r size=8 type=d alias=V0035+0 align=4 words (r4.2)
//.declare V0187 (199)  rf=r size=8 type=q align=4 words (r90.1)
//.declare V0188 (200)  rf=r size=8 type=d alias=V0187+0 align=4 words (r90.2)
//.declare V0190 (202)  rf=r size=4 type=d align=2 words (r11.0)
//.declare V0191 (203)  rf=r size=4 type=d align=2 words (r13.0)
//.declare V0192 (204)  rf=r size=4 type=ud alias=V0190+0 align=2 words (r11.0)
//.declare V0193 (205)  rf=r size=4 type=ud alias=V0191+0 align=2 words (r13.0)
//.declare V0194 (206)  rf=r size=4 type=d alias=+0 align=2 words (r90.4)
//.declare V0195 (207)  rf=r size=4 type=d align=2 words (r126.1)
//.declare V0196 (208)  rf=r size=2 type=b align=1 words (r14.0)
//.declare V0197 (209)  rf=r size=4 type=d alias=+4 align=2 words (r90.5)
//.declare V0198 (210)  rf=r size=2 type=ub alias=V0196+0 align=1 words (r14.0)
//.declare  (211)  rf=r size=64 type=ud align=32 words (r15.0)
//.declare  (212)  rf=r size=64 type=uq alias=+0 align=32 words (r15.0)
//.declare V0199 (213)  rf=r size=8 type=q alias=V0034+0 align=4 words (r4.0)
//.declare V0202 (216)  rf=r size=8 type=d alias=V0034+0 align=4 words (r4.0)
//.declare V0205 (219)  rf=r size=8 type=q align=4 words (r16.0)
//.declare V0206 (220)  rf=r size=8 type=d alias=V0205+0 align=4 words (r16.0)
//.declare V0208 (222)  rf=r size=4 type=d align=2 words (r17.0)
//.declare V0209 (223)  rf=r size=4 type=d align=2 words (r90.1)
//.declare V0210 (224)  rf=r size=4 type=ud alias=V0208+0 align=2 words (r17.0)
//.declare V0211 (225)  rf=r size=4 type=ud alias=V0209+0 align=2 words (r90.1)
//.declare V0212 (226)  rf=r size=4 type=d align=2 words (r18.0)
//.declare V0213 (227)  rf=r size=4 type=d align=2 words (r19.0)
//.declare V0214 (228)  rf=r size=4 type=d align=2 words (r126.2)
//.declare V0216 (230)  rf=r size=4 type=d alias=+0 align=2 words (r90.8)
//.declare V0217 (231)  rf=r size=32 type=d align=32 words (r91.0)
//.declare V0218 (232)  rf=r size=32 type=q alias=V0217+0 align=32 words (r91.0)
//.declare V0219 (233)  rf=r size=32 type=d align=32 words (r124.0)
//.declare V0220 (234)  rf=r size=32 type=q alias=V0219+0 align=32 words (r124.0)
//.declare V0221 (235)  rf=r size=4 type=d align=2 words (r125.2)
//.declare V0222 (236)  rf=r size=4 type=d align=2 words (r125.3)
//.declare V0223 (237)  rf=r size=4 type=d align=2 words (r125.4)
//.declare V0224 (238)  rf=r size=4 type=d align=2 words (r125.5)
//.declare V0225 (239)  rf=r size=4 type=d align=2 words (r125.6)
//.declare V0226 (240)  rf=r size=4 type=d align=2 words (r125.7)
//.declare V0227 (241)  rf=r size=4 type=d align=2 words (r125.8)
//.declare V0228 (242)  rf=r size=4 type=d align=2 words (r125.9)
//.declare V0229 (243)  rf=r size=4 type=d align=2 words (r125.10)
//.declare V0230 (244)  rf=r size=4 type=d align=2 words (r125.11)
//.declare V0231 (245)  rf=r size=4 type=d align=2 words (r125.12)
//.declare V0232 (246)  rf=r size=4 type=d align=2 words (r125.13)
//.declare V0233 (247)  rf=r size=4 type=d align=2 words (r125.14)
//.declare V0234 (248)  rf=r size=4 type=d align=2 words (r125.15)
//.declare V0235 (249)  rf=r size=4 type=d align=2 words (r123.0)
//.declare V0236 (250)  rf=r size=4 type=d alias=+4 align=2 words (r90.9)
//.declare V0237 (251)  rf=r size=512 type=f align=32 words (r92.0)
//.declare V0238 (252)  rf=r size=4 type=d align=2 words (r1.8)
//.declare  (254)  rf=r size=64 type=ud align=32 words (r2.0)
//.declare  (255)  rf=r size=64 type=uq alias=+0 align=32 words (r2.0)
//.declare V0243 (259)  rf=r size=512 type=ud alias=V0054+0 align=32 words (r5.0)
//.declare V0244 (260)  rf=r size=512 type=ud alias=V0056+0 align=32 words (r29.0)
//.declare V0245 (261)  rf=r size=4 type=d align=2 words (r53.0)
//.declare  (263)  rf=r size=64 type=ud align=32 words (r54.0)
//.declare  (264)  rf=r size=64 type=uq alias=+0 align=32 words (r54.0)
//.declare V0250 (268)  rf=r size=512 type=ud alias=V0058+0 align=32 words (r55.0)
//.declare V0251 (269)  rf=r size=512 type=ud alias=V0060+0 align=32 words (r79.0)
//.declare V0252 (270)  rf=r size=4 type=d align=2 words (r21.0)
//.declare  (272)  rf=r size=64 type=ud align=32 words (r22.0)
//.declare  (273)  rf=r size=64 type=uq alias=+0 align=32 words (r22.0)
//.declare V0257 (277)  rf=r size=512 type=ud alias=V0062+0 align=32 words (r23.0)
//.declare V0258 (278)  rf=r size=512 type=ud alias=V0064+0 align=32 words (r47.0)
//.declare V0259 (279)  rf=r size=4 type=d align=2 words (r71.0)
//.declare  (281)  rf=r size=64 type=ud align=32 words (r72.0)
//.declare  (282)  rf=r size=64 type=uq alias=+0 align=32 words (r72.0)
//.declare V0264 (286)  rf=r size=512 type=ud alias=V0066+0 align=32 words (r73.0)
//.declare V0265 (287)  rf=r size=512 type=ud alias=V0068+0 align=32 words (r21.0)
//.declare V0266 (288)  rf=r size=4 type=d align=2 words (r45.0)
//.declare  (290)  rf=r size=64 type=ud align=32 words (r46.0)
//.declare  (291)  rf=r size=64 type=uq alias=+0 align=32 words (r46.0)
//.declare V0271 (295)  rf=r size=512 type=ud alias=V0070+0 align=32 words (r47.0)
//.declare V0272 (296)  rf=r size=512 type=ud alias=V0072+0 align=32 words (r71.0)
//.declare V0273 (297)  rf=r size=4 type=d align=2 words (r30.0)
//.declare  (299)  rf=r size=64 type=ud align=32 words (r31.0)
//.declare  (300)  rf=r size=64 type=uq alias=+0 align=32 words (r31.0)
//.declare V0278 (304)  rf=r size=512 type=ud alias=V0074+0 align=32 words (r33.0)
//.declare V0279 (305)  rf=r size=512 type=ud alias=V0076+0 align=32 words (r57.0)
//.declare V0280 (306)  rf=r size=4 type=d align=2 words (r81.0)
//.declare  (308)  rf=r size=64 type=ud align=32 words (r82.0)
//.declare  (309)  rf=r size=64 type=uq alias=+0 align=32 words (r82.0)
//.declare V0285 (313)  rf=r size=512 type=ud alias=V0078+0 align=32 words (r5.0)
//.declare V0286 (314)  rf=r size=512 type=ud alias=V0080+0 align=32 words (r29.0)
//.declare V0287 (315)  rf=r size=4 type=d align=2 words (r53.0)
//.declare  (317)  rf=r size=64 type=ud align=32 words (r54.0)
//.declare  (318)  rf=r size=64 type=uq alias=+0 align=32 words (r54.0)
//.declare V0292 (322)  rf=r size=512 type=ud alias=V0082+0 align=32 words (r55.0)
//.declare V0293 (323)  rf=r size=512 type=ud alias=V0084+0 align=32 words (r79.0)
//.declare V0294 (324)  rf=r size=4 type=d align=2 words (r21.0)
//.declare  (326)  rf=r size=64 type=ud align=32 words (r22.0)
//.declare  (327)  rf=r size=64 type=uq alias=+0 align=32 words (r22.0)
//.declare V0299 (331)  rf=r size=512 type=ud alias=V0086+0 align=32 words (r23.0)
//.declare V0300 (332)  rf=r size=512 type=ud alias=V0088+0 align=32 words (r47.0)
//.declare V0301 (333)  rf=r size=4 type=d align=2 words (r71.0)
//.declare  (335)  rf=r size=64 type=ud align=32 words (r72.0)
//.declare  (336)  rf=r size=64 type=uq alias=+0 align=32 words (r72.0)
//.declare V0306 (340)  rf=r size=512 type=ud alias=V0090+0 align=32 words (r73.0)
//.declare V0307 (341)  rf=r size=512 type=ud alias=V0092+0 align=32 words (r21.0)
//.declare V0308 (342)  rf=r size=4 type=d align=2 words (r45.0)
//.declare  (344)  rf=r size=64 type=ud align=32 words (r46.0)
//.declare  (345)  rf=r size=64 type=uq alias=+0 align=32 words (r46.0)
//.declare V0313 (349)  rf=r size=512 type=ud alias=V0094+0 align=32 words (r47.0)
//.declare V0314 (350)  rf=r size=512 type=ud alias=V0096+0 align=32 words (r71.0)
//.declare V0315 (351)  rf=r size=4 type=d align=2 words (r30.0)
//.declare  (353)  rf=r size=64 type=ud align=32 words (r31.0)
//.declare  (354)  rf=r size=64 type=uq alias=+0 align=32 words (r31.0)
//.declare V0320 (358)  rf=r size=512 type=ud alias=V0098+0 align=32 words (r33.0)
//.declare V0321 (359)  rf=r size=512 type=ud alias=V0100+0 align=32 words (r57.0)
//.declare V0322 (360)  rf=r size=4 type=d align=2 words (r81.0)
//.declare  (362)  rf=r size=64 type=ud align=32 words (r82.0)
//.declare  (363)  rf=r size=64 type=uq alias=+0 align=32 words (r82.0)
//.declare V0327 (367)  rf=r size=512 type=ud alias=V0102+0 align=32 words (r5.0)
//.declare V0328 (368)  rf=r size=512 type=ud alias=V0104+0 align=32 words (r29.0)
//.declare V0329 (369)  rf=r size=4 type=d align=2 words (r54.0)
//.declare  (371)  rf=r size=64 type=ud align=32 words (r55.0)
//.declare  (372)  rf=r size=64 type=uq alias=+0 align=32 words (r55.0)
//.declare V0334 (376)  rf=r size=512 type=ud alias=V0106+0 align=32 words (r57.0)
//.declare V0335 (377)  rf=r size=512 type=ud alias=V0108+0 align=32 words (r81.0)
//.declare V0336 (378)  rf=r size=4 type=d align=2 words (r22.0)
//.declare  (380)  rf=r size=64 type=ud align=32 words (r23.0)
//.declare  (381)  rf=r size=64 type=uq alias=+0 align=32 words (r23.0)
//.declare V0341 (385)  rf=r size=512 type=ud alias=V0110+0 align=32 words (r25.0)
//.declare V0342 (386)  rf=r size=512 type=ud alias=V0112+0 align=32 words (r49.0)
//.declare V0343 (387)  rf=r size=4 type=d align=2 words (r126.3)
//.declare  (389)  rf=r size=64 type=ud align=32 words (r73.0)
//.declare  (390)  rf=r size=64 type=uq alias=+0 align=32 words (r73.0)
//.declare V0348 (394)  rf=r size=512 type=ud alias=V0114+0 align=32 words (r75.0)
//.declare V0349 (395)  rf=r size=512 type=ud alias=V0116+0 align=32 words (r23.0)
//.declare P02 (396)  rf=f16  size=2 type=uw align=1 words (f1.0)
//.declare V0350 (397)  rf=r size=4 type=ud alias=V0336+0 align=2 words (r22.0)
//.declare P03 (398)  rf=f16  size=2 type=uw align=1 words (f0.0)
//.declare V0352 (400)  rf=r size=64 type=d align=32 words (r2.0)
//.declare V0353 (401)  rf=r size=512 type=d alias=V0237+0 align=32 words (r92.0)
//.declare V0354 (402)  rf=r size=8 type=q alias=V0036+0 align=4 words (r4.2)
//.declare V0357 (405)  rf=r size=8 type=d alias=V0036+0 align=4 words (r4.4)
//.declare V0360 (408)  rf=r size=8 type=q align=4 words (r3.0)
//.declare V0361 (409)  rf=r size=8 type=d alias=V0360+0 align=4 words (r3.0)
//.declare V0363 (411)  rf=r size=4 type=d align=2 words (r4.6)
//.declare V0364 (412)  rf=r size=4 type=d align=2 words (r6.0)
//.declare V0365 (413)  rf=r size=4 type=ud alias=V0363+0 align=2 words (r4.6)
//.declare V0366 (414)  rf=r size=4 type=ud alias=V0364+0 align=2 words (r6.0)
//.declare  (417)  rf=r size=64 type=ud align=32 words (r7.0)
//.declare  (418)  rf=r size=64 type=uq alias=+0 align=32 words (r7.0)
//.declare V0369 (419)  rf=r size=8 type=uq align=4 words (r4.3)
//.declare V0370 (420)  rf=r size=8 type=uq align=4 words (r5.0)
//.declare  (421)  rf=r size=64 type=ud align=32 words (r127.0)
//.declare  (422)  rf=r size=4 type=d alias=V0181+0 align=2 words (r4.2)
//.declare  (423)  rf=r size=4 type=d alias=V0199+0 align=2 words (r4.0)
//.declare  (424)  rf=r size=4 type=d alias=V0354+0 align=2 words (r4.4)
//.declare  (425)  rf=r size=8 type=d align=8 words (r2.0)
//.declare  (426)  rf=r size=8 type=d align=8 words (r123.4)
//.declare  (427)  rf=r size=8 type=f align=8 words (r18.0)
//.declare  (428)  rf=r size=8 type=ud align=8 words (r10.0)
//.declare  (429)  rf=r size=8 type=d align=8 words (r90.4)
//.declare  (430)  rf=r size=8 type=d align=8 words (r90.8)
//.declare  (431)  rf=r size=4 type=f align=2 words (r13.0)
//.declare r0 (432)  rf=r size=64 type=ud align=32 words (r0.0)
//.declare rtmp (433)  rf=r size=64 type=ud align=32 words (r127.0)
//.declare inlineRegFromTDL (434)  rf=r size=32 type=ud align=2 words (r1.0)
//.declare inlineRegExpectedLocation (435)  rf=r size=32 type=ud align=2 words (r4.0)
//.declare  (436)  rf=r size=128 type=ud align=32 words (r1.0)
//.declare  (437)  rf=r size=64 type=ud align=32 words (r3.0)
//.declare  (438)  rf=r size=32 type=ud align=2 words (r5.0)

// .inputs
// +----------+----------+--------+----------+------------------+
// | id       | type     |  bytes | at       | from             |
// +----------+----------+--------+----------+------------------+
// | V0040    | :w x 16  |   0x20 | r1       | pti[tid]+0x0     |
// | V0041    | :w x 16  |   0x20 | r2       | pti[tid]+0x40    |
// | V0042    | :w x 16  |   0x20 | r3       | pti[tid]+0x80    |
// | V0034    | :uq      |    0x8 | r4       | inline+0x0       |
// | V0035    | :uq      |    0x8 | r4+0x8   | inline+0x8       |
// | V0036    | :uq      |    0x8 | r4+0x10  | inline+0x10      |
// | V0369    | :uq      |    0x8 | r4+0x18  | inline+0x18      |
// | V0370    | :uq      |    0x8 | r5       | cti+0x20         |
// | V0043    | :uq      |    0x8 | r5+0x8   | cti+0x28         |
// +----------+----------+--------+----------+------------------+


// B000: Preds:{},  Succs:{B001}
per_thread_prolog:
(W)     mov (16|M0)              r127.0<1>:ud  0x0:ud                                                //  ALU pipe: int; 
(W)     and (1|M0)               r127.2<1>:ud  r0.0<0;1,0>:ud    0xFFFFFFC0:ud                       //  ALU pipe: int; 
(W)     and (1|M0)               r127.0<1>:uw  r0.4<0;1,0>:uw    0xFF:uw                             //  ALU pipe: int; 
(W)     add (1|M0)               r127.2<1>:ud  r127.2<0;1,0>:ud  0x20:ud              {I@2}          //  ALU pipe: int; 
(W)     add (1|M0)               r127.2<1>:ud  r127.2<0;1,0>:ud  0x0:ud              {I@1}           //  R_SYM_ADDR_32: __INTEL_PATCH_CROSS_THREAD_OFFSET_OFF_R0; ALU pipe: int; 
(W)     mad (1|M0)               r127.0<1>:ud  r127.2<0;0>:ud    r127.0<0;0>:uw    0xC0:uw              {I@1} //  ALU pipe: int; 
(W)     mov (8|M0)               r4.0<1>:ud    r1.0<1;1,0>:ud                                        //  ALU pipe: int; 

// File: /home/gta/workspace/intel-xpu-backend-for-triton/benchmarks/triton_kernels_benchmark/gemm_benchmark.py

// Line 38:  def matmul_kernel_with_tensor_descriptors(
(W)     load.ugm.d32x32t.a32.ca.cc (1|M0)  r1:2 bti[255][r127:1]   {A@1,$0} // ex_desc:0xFF000000; desc:0x6229E500 // 
(W)     load.ugm.d32x16t.a32.ca.cc (1|M0)  r3:1 bti[255][r127:1+0x80]  {$1} // ex_desc:0xFF080000; desc:0x6219D500 // 
        nop                                                                                          // 
        nop                                                                                          // 
        nop                                                                                          // 
// B001: Preds:{B000},  Succs:{B002}
// cross_thread_prolog:
        sync.nop                             null                             {Compacted,$1.src}     // 
(W)     and (1|M0)               r127.0<1>:ud  r0.0<0;1,0>:ud    0xFFFFFFC0:ud              {$0.src} //  ALU pipe: int; 
(W)     add (1|M0)               r127.0<1>:ud  r127.0<0;1,0>:ud  0x0:ud              {I@1}           //  R_SYM_ADDR_32: __INTEL_PATCH_CROSS_THREAD_OFFSET_OFF_R0; ALU pipe: int; 
(W)     load.ugm.d32x8t.a32.ca.cc (1|M0)  r5:1  bti[255][r127:1]   {I@1,$2} // ex_desc:0xFF000000; desc:0x6219C500 // 
// B002: Preds:{B001},  Succs:{B003, B004}
// _main:
(W)     mov (16|M0)              r89.0<1>:ud   r0.0<1;1,0>:ud                   {Compacted}          //  ALU pipe: int; $1
(W)     or (1|M0)                cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x400004C0:ud              {A@1}    // $1
(W)     mov (1|M0)               r2.0<1>:d     r89.1<0;1,0>:d                   {Compacted,A@1,$0.dst} //  ALU pipe: int; $2

// Line 54:  group_id = pid // num_pid_in_group
(W)     mul (1|M0)               acc0.0<1>:ud  r2.0<0;1,0>:ud    0xAAAB:uw              {I@1}        //  ALU pipe: int; $5
(W)     mach (1|M0)              r3.0<1>:ud    r2.0<0;1,0>:ud    0xAAAAAAAB:ud              {$1.dst} //  ALU pipe: int; 
(W)     shr (1|M0)               r125.0<1>:ud  r3.0<0;1,0>:ud    4:w               {I@1}             //  ALU pipe: int; $6

// Line 58:  pid_n = (pid % num_pid_in_group) // group_size_m
(W)     cmp (16|M0)   (eq)f0.1   null<1>:d     r125.0<0;1,0>:d   1:w               {I@1}             //  ALU pipe: int; $12

// Line 56:  group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
(W)     add (1|M0)               r123.4<1>:d   -r125.0<0;1,0>:d  1:w               {Compacted}       //  ALU pipe: int; $8

// Line 57:  pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
(W)     mad (1|M0)               r123.5<1>:d   r2.0<0;0>:d       r125.0<0;0>:d     -24:w               //  ALU pipe: int; $10

// Line 58:  pid_n = (pid % num_pid_in_group) // group_size_m
(W&~f0.1) jmpi                               _0_007                                                  //  ALU pipe: int; $13
// B003: Preds:{B002},  Succs:{B005}
_0_008:
(W)     mov (1|M0)               r90.0<1>:d    -1:w                               {Compacted}        //  ALU pipe: int; $15
(W)     jmpi                                 _0_009                                                  // $16
// B004: Preds:{B002},  Succs:{B005}
_0_007:
(W)     asr (2|M0)               r2.0<1>:d     r123.4<1;1,0>:d   31:w               {Compacted,I@4}  //  ALU pipe: int; $18
(W)     add3 (1|M0)              r3.0<1>:d     r2.0<0;0>:d       -r125.0<0;0>:d    1:w               {I@1} //  ALU pipe: int; $20
(W)     add (1|M0)               r5.0<1>:d     r2.1<0;1,0>:d     r123.5<0;1,0>:d  {Compacted,$2.dst} //  ALU pipe: int; $22
(W)     xor (1|M0)               r4.6<1>:d     r3.0<0;1,0>:d     r2.0<0;1,0>:d    {I@2}              //  ALU pipe: int; $21
(W)     xor (1|M0)               r6.0<1>:d     r5.0<0;1,0>:d     r2.1<0;1,0>:d    {Compacted,I@2}    //  ALU pipe: int; $23
(W)     xor (1|M0)               cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x30:uw              {A@1}          // $24
(W)     mov (1|M0)               r8.0<1>:f     r4.6<0;1,0>:ud                   {A@1}                //  ALU pipe: float; $25
(W)     mov (1|M0)               r13.0<1>:f    0xB4C00000:f                               {Compacted} //  ALU pipe: float; $30
(W)     math.inv (1|M0)          r12.0<1>:f    r8.0<0;1,0>:f                    {F@2}                //  ALU pipe: math; $29
(W)     mov (1|M0)               r11.0<1>:f    r6.0<0;1,0>:ud                   {I@2}                //  ALU pipe: float; $28
(W)     mad (1|M0)               r14.0<1>:f    r12.0<0;0>:f      r13.0<0;0>:f      r12.0<0>:f       {Compacted,A@1} //  ALU pipe: float; $30
(W)     mov (1|M0)               r9.0<1>:ud    r8.0<0;1,0>:f                                         //  ALU pipe: int; $26
(W)     mov (1|M0)               r16.0<1>:ud   r11.0<0;1,0>:f                   {F@2}                //  ALU pipe: int; $32
(W)     mul (1|M0)               r15.0<1>:f    r11.0<0;1,0>:f    r14.0<0;1,0>:f   {Compacted,F@1}    //  ALU pipe: float; $31
(W)     add (1|M0)               r10.0<1>:d    r4.6<0;1,0>:d     -r9.0<0;1,0>:d   {I@2}              //  ALU pipe: int; $27
(W)     add (1|M0)               r10.1<1>:d    r6.0<0;1,0>:d     -r16.0<0;1,0>:d  {I@2}              //  ALU pipe: int; $33
(W)     mov (1|M0)               r17.0<1>:ud   r15.0<0;1,0>:f                   {F@1}                //  ALU pipe: int; $34
(W)     mov (1|M0)               r18.0<1>:f    r10.0<0;1,0>:ud                  {I@3}                //  ALU pipe: float; $35
(W)     mov (1|M0)               r18.1<1>:f    r10.1<0;1,0>:ud                  {I@2}                //  ALU pipe: float; $35
(W)     mov (1|M0)               r19.0<1>:f    r17.0<0;1,0>:ud                  {I@1}                //  ALU pipe: float; $37
(W)     mad (1|M0)               r20.0<1>:f    r11.0<0;0>:f      r19.0<0;0>:f      -r8.0<0>:f       {F@1} //  ALU pipe: float; $39
(W)     mad (1|M0)               r21.0<1>:f    r18.1<0;0>:f      r19.0<0;0>:f      -r18.0<0>:f       //  ALU pipe: float; $41
(W)     add (1|M0)               r22.0<1>:f    r20.0<0;1,0>:f    r21.0<0;1,0>:f   {Compacted,F@1}    //  ALU pipe: float; $42
(W)     mul (1|M0)               r23.0<1>:f    r14.0<0;1,0>:f    r22.0<0;1,0>:f   {Compacted,F@1}    //  ALU pipe: float; $43
(W)     xor (1|M0)               cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x30:uw              {A@1}          // $44
(W)     mov (1|M0)               r24.0<1>:ud   r23.0<0;1,0>:f                   {A@1}                //  ALU pipe: int; $45
(W)     xor (1|M0)               r27.0<1>:d    r2.0<0;1,0>:d     r2.1<0;1,0>:d    {Compacted}        //  ALU pipe: int; $47
(W)     add (1|M0)               r25.0<1>:d    r24.0<0;1,0>:d    r17.0<0;1,0>:d   {Compacted,I@2}    //  ALU pipe: int; $46
(W)     mul (1|M0)               acc0.0<1>:d   r25.0<0;1,0>:d    r4.12<0;1,0>:uw  {I@1}              //  ALU pipe: int; $48
(W)     macl (1|M0)              r28.0<1>:d    r25.0<0;1,0>:d    r4.6<0;1,0>:d    {Compacted}        //  ALU pipe: int; $49
(W)     add (1|M0)               r29.0<1>:d    r6.0<0;1,0>:d     -r28.0<0;1,0>:d  {I@1}              //  ALU pipe: int; $49
(W)     cmp (1|M0)    (ge)f1.1   r30.0<1>:ud   r29.0<0;1,0>:ud   r4.6<0;1,0>:ud   {I@1}              //  ALU pipe: int; $50
(W)     add3 (1|M0)              r31.0<1>:d    r25.0<0;0>:d      r27.0<0;0>:d      -r30.0<0>:d      {I@1} //  ALU pipe: int; $51
(W)     bfn.(s0^s1^s2) (1|M0)    r90.0<1>:ud   r31.0<0;0>:ud     r2.0<0;0>:ud      r2.1<0>:ud       {I@1} //  ALU pipe: int; $52
// B005: Preds:{B004, B003},  Succs:{B006}
_0_009:

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r4.24<2>:b    r89.8<0;1,0>:b                                        //  ALU pipe: int; $61
(W)     and (1|M0)               r11.0<1>:d    r4.2<0;1,0>:d     63:w               {Compacted}      //  ALU pipe: int; $75
(W)     mov (1|M0)               r5.0<1>:w     r4.24<0;1,0>:b                   {@2,$2.dst}          //  ALU pipe: int; $62
(W)     mov (1|M0)               r7.0<1>:d     r4.24<0;1,0>:ub                                       //  ALU pipe: int; $64
(W)     and (1|M0)               r6.0<1>:w     r5.0<0;1,0>:w     48:w               {I@2}            //  ALU pipe: int; $63
(W)     shl (1|M0)               r8.0<1>:d     r7.0<0;1,0>:d     5:w               {Compacted,I@2}   //  ALU pipe: int; $65
(W)     shl (1|M0)               r125.1<1>:d   r90.0<0;1,0>:d    9:w               {Compacted}       //  ALU pipe: int; $60
(W)     shr (1|M0)               r13.0<1>:ud   r11.0<0;1,0>:ud   1:w                                 //  ALU pipe: int; $76
(W)     mov (1|M0)               r14.0<2>:b    r6.0<0;1,0>:w                    {I@4}                //  ALU pipe: int; $79
(W)     and (1|M0)               r10.0<1>:d    r8.0<0;1,0>:d     480:w               {Compacted,I@4} //  ALU pipe: int; $66

// Line 57:  pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
(W)     mul (1|M0)               acc0.0<1>:d   r90.0<0;1,0>:d    r123.8<0;1,0>:uw                    //  ALU pipe: int; $55

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r90.3<1>:f    r4.3<0;1,0>:f                                         //  ALU pipe: float; $68
(W)     and (1|M0)               r90.2<1>:d    r4.2<0;1,0>:d     -64:w                               //  ALU pipe: int; $69
(W)     add (1|M0)               r126.1<1>:d   r11.0<0;1,0>:d    24575:w                             //  ALU pipe: int; $78
(W)     shl (1|M0)               r19.0<1>:d    r7.0<0;1,0>:d     4:w               {Compacted}       //  ALU pipe: int; $93
(W)     mov (1|M0)               r90.5<1>:d    r14.0<0;1,0>:ub                  {I@6}                //  ALU pipe: int; $80

// Line 57:  pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
(W)     macl (1|M0)              r2.0<1>:d     r90.0<0;1,0>:d    r123.4<0;1,0>:d  {Compacted}        //  ALU pipe: int; $56

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     bfn.(s0|s1|s2) (1|M0)    r90.4<1>:ud   r13.0<0;0>:ud     r10.0<0;0>:ud     r125.1<0>:ud     {I@7} //  ALU pipe: int; $77 R{} IR{}{O:6,E:5,O:6,},  {BC=1}
(W)     mov (1|M0)               r15.3<1>:ud   4095:w                                                //  blk2d.heightM1; ALU pipe: int; $81
(W)     mov (1|M0)               r15.4<1>:ud   24575:w                                               //  blk2d.pitchM1; ALU pipe: int; $81
(W)     mov (1|M0)               r15.7<1>:ud   0xF1F:uw                                              //  bkl2d.shape = 1x32x16; ALU pipe: int; $81
(W)     mov (1|M0)               r16.1<1>:f    r4.1<0;1,0>:f                                         //  ALU pipe: float; $83
(W)     and (1|M0)               r16.0<1>:d    r4.0<0;1,0>:d     -64:w               {Compacted}     //  ALU pipe: int; $84
(W)     and (1|M0)               r17.0<1>:d    r4.0<0;1,0>:d     63:w               {Compacted}      //  ALU pipe: int; $90
(W)     or (1|M0)                r18.0<1>:d    r13.0<0;1,0>:d    r125.1<0;1,0>:d  {Compacted}        //  ALU pipe: int; $92 R{} IR{}{O:6,O:6,},  {BC=1}
(W)     mov (1|M0)               r15.0<1>:uq   r90.1<0;1,0>:q                   {F@2}                //  ALU pipe: int; $81
(W)     mov (1|M0)               r15.2<1>:ud   r126.1<0;1,0>:d                                       //  blk2d.widthM1; ALU pipe: int; $81
(W)     and (1|M0)               r126.2<1>:d   r19.0<0;1,0>:d    496:w               {Compacted}     //  ALU pipe: int; $94

// Line 57:  pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
(W)     add3 (1|M0)              r3.0<1>:d     r125.0<0;0>:d     r123.5<0;0>:d     -r2.0<0>:d       {I@7} //  ALU pipe: int; $56

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (2|M0)               r15.5<1>:ud   r90.4<1;1,0>:d                   {I@7}                //  blk2d.X; ALU pipe: int; $81

// Line 75:  for _ in range(0, K, BLOCK_SIZE_K):
        mov (16|M0)              r92.0<1>:f    0x0:f                               {Compacted}       //  ALU pipe: float; $128
        mov (16|M0)              r93.0<1>:f    0x0:f                               {Compacted}       //  ALU pipe: float; $129
        mov (16|M0)              r94.0<1>:f    0x0:f                               {Compacted}       //  ALU pipe: float; $130
        mov (16|M0)              r95.0<1>:f    0x0:f                               {Compacted}       //  ALU pipe: float; $131
        mov (16|M0)              r96.0<1>:f    0x0:f                               {Compacted}       //  ALU pipe: float; $132
        mov (16|M0)              r97.0<1>:f    0x0:f                               {Compacted}       //  ALU pipe: float; $133
        mov (16|M0)              r98.0<1>:f    0x0:f                               {Compacted}       //  ALU pipe: float; $134
        mov (16|M0)              r99.0<1>:f    0x0:f                               {Compacted}       //  ALU pipe: float; $135

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        load_block2d.ugm.d16.a64.ca.ca (1|M0)  null:0 [r15:1]      {I@1,$3} // ex_desc:0x0; desc:0x2080203 // $81
(W)     mov (1|M0)               r91.3<1>:d    3:w                                                   //  ALU pipe: int; $99
(W)     mov (1|M0)               r91.4<1>:d    8191:w                                                //  ALU pipe: int; $100
(W)     mov (1|M0)               r91.5<1>:d    0:w                                                   //  ALU pipe: int; $101
(W)     mov (1|M0)               r91.6<1>:d    0:w                                                   //  ALU pipe: int; $102
(W)     mov (1|M0)               r91.7<1>:f    0x1070F:f                                             //  (0x0001070f:f); ALU pipe: float; $103
(W)     mov (1|M0)               r124.3<1>:d   4095:w                                                //  ALU pipe: int; $106
(W)     mov (1|M0)               r124.4<1>:d   24575:w                                               //  ALU pipe: int; $107
(W)     mov (1|M0)               r124.5<1>:d   0:w                                                   //  ALU pipe: int; $108
(W)     mov (1|M0)               r124.6<1>:d   0:w                                                   //  ALU pipe: int; $109
(W)     mov (1|M0)               r124.7<1>:d   7951:w                                                //  ALU pipe: int; $110

// Line 75:  for _ in range(0, K, BLOCK_SIZE_K):
(W)     mov (1|M0)               r125.2<1>:d   32:w                               {Compacted}        //  ALU pipe: int; $112
(W)     mov (1|M0)               r125.3<1>:d   896:w                                                 //  ALU pipe: int; $113
(W)     mov (1|M0)               r125.4<1>:d   832:w                               {Compacted}       //  ALU pipe: int; $114
(W)     mov (1|M0)               r125.5<1>:d   768:w                                                 //  ALU pipe: int; $115
(W)     mov (1|M0)               r125.6<1>:d   704:w                                                 //  ALU pipe: int; $116
(W)     mov (1|M0)               r125.7<1>:d   640:w                                                 //  ALU pipe: int; $117
(W)     mov (1|M0)               r125.8<1>:d   576:w                                                 //  ALU pipe: int; $118
(W)     mov (1|M0)               r125.9<1>:d   512:w                                                 //  ALU pipe: int; $119
(W)     mov (1|M0)               r125.10<1>:d  448:w                                                 //  ALU pipe: int; $120
(W)     mov (1|M0)               r125.11<1>:d  384:w                                                 //  ALU pipe: int; $121
(W)     mov (1|M0)               r125.12<1>:d  320:w                                                 //  ALU pipe: int; $122
(W)     mov (1|M0)               r125.13<1>:d  256:w                                                 //  ALU pipe: int; $123
(W)     mov (1|M0)               r125.14<1>:d  192:w                                                 //  ALU pipe: int; $124
(W)     mov (1|M0)               r125.15<1>:d  128:w                                                 //  ALU pipe: int; $125
(W)     mov (1|M0)               r123.0<1>:d   64:w                               {Compacted}        //  ALU pipe: int; $126
(W)     mov (1|M0)               r90.9<1>:d    0:w                                                   //  ALU pipe: int; $127

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r124.0<1>:q   r90.1<0;1,0>:q                                        //  ALU pipe: int; $104
(W)     mov (1|M0)               r124.2<1>:d   r126.1<0;1,0>:d                                       //  ALU pipe: int; $105
(W)     mov (1|M0)               r91.0<1>:q    r16.0<0;1,0>:q                   {F@7}                //  ALU pipe: int; $97
(W)     shr (1|M0)               r90.1<1>:ud   r17.0<0;1,0>:ud   1:w                                 //  ALU pipe: int; $91
(W)     add (1|M0)               r91.2<1>:d    r17.0<0;1,0>:d    8191:w                              //  ALU pipe: int; $95
(W)     add (1|M0)               r90.8<1>:d    r18.0<0;1,0>:d    r126.2<0;1,0>:d                     //  ALU pipe: int; $96

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     shl (1|M0)               r126.0<1>:d   r3.0<0;1,0>:d     3:w               {Compacted}       //  ALU pipe: int; $58
// B006: Preds:{B007, B005},  Succs:{B007, B008}
_0_010:
(W)     or (1|M0)                r91.5<1>:d    r90.9<0;1,0>:d    r90.1<0;1,0>:d   {I@4}              //  ALU pipe: int; $143

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r91.6<1>:d    r126.0<0;1,0>:d                  {I@2}                //  ALU pipe: int; $148

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     mov (2|M0)               r124.5<1>:d   r90.8<1;1,0>:d                                        //  ALU pipe: int; $151

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        load_block2d.ugm.d16.a64 (1|M0)  r5:8   [r91:1]            {A@1,$7} // ex_desc:0x0; desc:0x2800203 // $149

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        sync.nop                             null                             {Compacted,$6.src}     // $153
        load_block2d.ugm.d16v.a64 (1|M0)  r13:16 [r124:1]           {I@1,$8} // ex_desc:0x0; desc:0x3000283 // $153
(W)     bfn.(s0|s1|s2) (1|M0)    r91.5<1>:ud   r90.9<0;0>:ud     r90.1<0;0>:ud     r125.2<0>:ud     {$7.src} //  ALU pipe: int; $144

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r91.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $158
(W)     or (1|M0)                r124.6<1>:d   r90.9<0;1,0>:d    32:w               {$8.src}         //  ALU pipe: int; $146

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     mov (1|M0)               r124.5<1>:d   r90.8<0;1,0>:d                                        //  ALU pipe: int; $161

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        load_block2d.ugm.d16.a64 (1|M0)  r29:8  [r91:1]            {I@3,$9} // ex_desc:0x0; desc:0x2800203 // $159

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        load_block2d.ugm.d16v.a64 (1|M0)  r37:16 [r124:1]           {I@1,$10} // ex_desc:0x0; desc:0x3000283 // $163

// Line 85:  off_k += BLOCK_SIZE_K
(W)     or (1|M0)                r1.8<1>:d     r90.9<0;1,0>:d    64:w                                //  ALU pipe: int; $138

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     bfn.(s0|s1|s2) (1|M0)    r91.5<1>:ud   r90.9<0;0>:ud     r123.0<0;0>:ud    r90.1<0>:ud      {$9.src} //  ALU pipe: int; $177

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r91.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $182

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     mov (1|M0)               r124.5<1>:d   r90.8<0;1,0>:d                   {$10.src}            //  ALU pipe: int; $185
(W)     mov (1|M0)               r124.6<1>:d   r1.8<0;1,0>:d                    {I@4}                //  ALU pipe: int; $186

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        load_block2d.ugm.d16.a64 (1|M0)  r55:8  [r91:1]            {I@3,$11} // ex_desc:0x0; desc:0x2800203 // $183

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        load_block2d.ugm.d16v.a64 (1|M0)  r63:16 [r124:1]           {I@1,$12} // ex_desc:0x0; desc:0x3000283 // $187
(W)     bfn.(s0|s1|s2) (1|M0)    r91.5<1>:ud   r1.8<0;0>:ud      r90.1<0;0>:ud     r125.2<0>:ud     {$11.src} //  ALU pipe: int; $178

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r91.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $192
(W)     or (1|M0)                r124.6<1>:d   r90.9<0;1,0>:d    96:w               {$12.src}        //  ALU pipe: int; $180

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     mov (1|M0)               r124.5<1>:d   r90.8<0;1,0>:d                                        //  ALU pipe: int; $195

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        load_block2d.ugm.d16.a64 (1|M0)  r79:8  [r91:1]            {I@3,$13} // ex_desc:0x0; desc:0x2800203 // $193

// Line 85:  off_k += BLOCK_SIZE_K
(W)     or (1|M0)                r53.0<1>:d    r90.9<0;1,0>:d    128:w               {Compacted}     //  ALU pipe: int; $172

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     bfn.(s0|s1|s2) (1|M0)    r91.5<1>:ud   r90.9<0;0>:ud     r125.15<0;0>:ud   r90.1<0>:ud      {$13.src} //  ALU pipe: int; $211

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r91.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $216
(W)     bfn.(s0|s1|s2) (1|M0)    r54.6<1>:ud   r90.5<0;0>:ud     r90.9<0;0>:ud     r125.15<0>:ud     //  ALU pipe: int; $174
(W)     mov (1|M0)               r54.0<1>:uq   r90.1<0;1,0>:q                                        //  ALU pipe: int; $175
(W)     mov (1|M0)               r54.2<1>:ud   r126.1<0;1,0>:d                                       //  blk2d.widthM1; ALU pipe: int; $175
(W)     mov (1|M0)               r54.3<1>:ud   4095:w                                                //  blk2d.heightM1; ALU pipe: int; $175
(W)     mov (1|M0)               r54.4<1>:ud   24575:w                                               //  blk2d.pitchM1; ALU pipe: int; $175
(W)     mov (1|M0)               r54.5<1>:ud   r90.4<0;1,0>:d                                        //  blk2d.X; ALU pipe: int; $175
(W)     mov (1|M0)               r54.7<1>:ud   0xF1F:uw                                              //  bkl2d.shape = 1x32x16; ALU pipe: int; $175

// Line 85:  off_k += BLOCK_SIZE_K
(W)     add (1|M0)               r126.3<1>:d   r90.9<0;1,0>:d    1024:w                              //  ALU pipe: int; $648

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        load_block2d.ugm.d16.a64.ca.ca (1|M0)  null:0 [r54:1]      {I@2,$14} // ex_desc:0x0; desc:0x2080203 // $175
(W)     bfn.(s0|s1|s2) (1|M0)    r2.6<1>:ud    r90.5<0;0>:ud     r90.9<0;0>:ud     r123.0<0>:ud     {$5.src} //  ALU pipe: int; $140
(W)     mov (1|M0)               r2.0<1>:uq    r90.1<0;1,0>:q                                        //  ALU pipe: int; $141
(W)     mov (1|M0)               r2.2<1>:ud    r126.1<0;1,0>:d                                       //  blk2d.widthM1; ALU pipe: int; $141
(W)     mov (1|M0)               r2.3<1>:ud    4095:w                                                //  blk2d.heightM1; ALU pipe: int; $141
(W)     mov (1|M0)               r2.4<1>:ud    24575:w                                               //  blk2d.pitchM1; ALU pipe: int; $141
(W)     mov (1|M0)               r2.5<1>:ud    r90.4<0;1,0>:d                                        //  blk2d.X; ALU pipe: int; $141
(W)     mov (1|M0)               r2.7<1>:ud    0xF1F:uw                                              //  bkl2d.shape = 1x32x16; ALU pipe: int; $141
        load_block2d.ugm.d16.a64.ca.ca (1|M0)  null:0 [r2:1]       {I@1,$5} // ex_desc:0x0; desc:0x2080203 // $141
        sync.allwr                           ($6,$8)                                                 // $167
        dpas.8x8 (16|M0)         r92:f         r92:f             r13:bf            r5.0:bf          {Compacted,$7} // $167
        dpas.8x8 (16|M0)         r92:f         r92:f             r21:bf            r9.0:bf          {Compacted,$7} // $168

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        sync.nop                             null                             {Compacted,$7.src}     // $197
        load_block2d.ugm.d16v.a64 (1|M0)  r5:16  [r124:1]           {$15} // ex_desc:0x0; desc:0x3000283 // $197
(W)     mov (1|M0)               r124.5<1>:d   r90.8<0;1,0>:d                   {$15.src}            //  ALU pipe: int; $219
(W)     mov (1|M0)               r124.6<1>:d   r53.0<0;1,0>:d                                        //  ALU pipe: int; $220

// Line 85:  off_k += BLOCK_SIZE_K
(W)     or (1|M0)                r21.0<1>:d    r90.9<0;1,0>:d    192:w               {Compacted}     //  ALU pipe: int; $206

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     bfn.(s0|s1|s2) (1|M0)    r22.6<1>:ud   r90.5<0;0>:ud     r90.9<0;0>:ud     r125.14<0>:ud     //  ALU pipe: int; $208
(W)     mov (1|M0)               r22.0<1>:uq   r90.1<0;1,0>:q                                        //  ALU pipe: int; $209
        sync.allwr                           ($7,$10)                                                // $169
        dpas.8x8 (16|M0)         r92:f         r92:f             r37:bf            r29.0:bf         {Compacted,$9} // $169
        sync.nop                             null                             {Compacted,$9.src}     // $217
        load_block2d.ugm.d16.a64 (1|M0)  r23:8  [r91:1]            {$0} // ex_desc:0x0; desc:0x2800203 // $217

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     bfn.(s0|s1|s2) (1|M0)    r91.5<1>:ud   r53.0<0;0>:ud     r90.1<0;0>:ud     r125.2<0>:ud     {$0.src} //  ALU pipe: int; $212

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r91.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $226
(W)     mov (1|M0)               r22.2<1>:ud   r126.1<0;1,0>:d                                       //  blk2d.widthM1; ALU pipe: int; $209
(W)     mov (1|M0)               r22.3<1>:ud   4095:w                                                //  blk2d.heightM1; ALU pipe: int; $209
(W)     mov (1|M0)               r22.4<1>:ud   24575:w                                               //  blk2d.pitchM1; ALU pipe: int; $209
        dpas.8x8 (16|M0)         r92:f         r92:f             r45:bf            r33.0:bf         {Compacted,$9} // $170

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        sync.nop                             null                             {Compacted,$9.src}     // $221
        load_block2d.ugm.d16v.a64 (1|M0)  r31:16 [r124:1]           {I@7,$1} // ex_desc:0x0; desc:0x3000283 // $221

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     or (1|M0)                r124.6<1>:d   r90.9<0;1,0>:d    160:w               {$1.src}        //  ALU pipe: int; $214

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     mov (1|M0)               r124.5<1>:d   r90.8<0;1,0>:d                                        //  ALU pipe: int; $229

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        load_block2d.ugm.d16.a64 (1|M0)  r47:8  [r91:1]            {I@6,$2} // ex_desc:0x0; desc:0x2800203 // $227

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     bfn.(s0|s1|s2) (1|M0)    r91.5<1>:ud   r90.9<0;0>:ud     r125.14<0;0>:ud   r90.1<0>:ud      {$2.src} //  ALU pipe: int; $245

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r91.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $250
        sync.allwr                           ($9,$12)                                                // $201
        dpas.8x8 (16|M0)         r92:f         r92:f             r63:bf            r55.0:bf         {Compacted,$11} // $201
(W)     mov (1|M0)               r22.5<1>:ud   r90.4<0;1,0>:d                                        //  blk2d.X; ALU pipe: int; $209
(W)     mov (1|M0)               r22.7<1>:ud   0xF1F:uw                                              //  bkl2d.shape = 1x32x16; ALU pipe: int; $209
        load_block2d.ugm.d16.a64.ca.ca (1|M0)  null:0 [r22:1]      {I@1,$3} // ex_desc:0x0; desc:0x2080203 // $209
        dpas.8x8 (16|M0)         r92:f         r92:f             r71:bf            r59.0:bf         {Compacted,$11} // $202

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        sync.nop                             null                             {Compacted,$11.src}    // $231
        load_block2d.ugm.d16v.a64 (1|M0)  r55:16 [r124:1]           {$4} // ex_desc:0x0; desc:0x3000283 // $231
(W)     mov (1|M0)               r124.5<1>:d   r90.8<0;1,0>:d                   {$4.src}             //  ALU pipe: int; $253
(W)     mov (1|M0)               r124.6<1>:d   r21.0<0;1,0>:d                                        //  ALU pipe: int; $254

// Line 85:  off_k += BLOCK_SIZE_K
(W)     or (1|M0)                r71.0<1>:d    r90.9<0;1,0>:d    256:w               {Compacted}     //  ALU pipe: int; $240

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     bfn.(s0|s1|s2) (1|M0)    r72.6<1>:ud   r90.5<0;0>:ud     r90.9<0;0>:ud     r125.13<0>:ud     //  ALU pipe: int; $242
(W)     mov (1|M0)               r72.0<1>:uq   r90.1<0;1,0>:q                                        //  ALU pipe: int; $243
(W)     mov (1|M0)               r72.2<1>:ud   r126.1<0;1,0>:d                                       //  blk2d.widthM1; ALU pipe: int; $243
(W)     mov (1|M0)               r72.3<1>:ud   4095:w                                                //  blk2d.heightM1; ALU pipe: int; $243
(W)     mov (1|M0)               r72.4<1>:ud   24575:w                                               //  blk2d.pitchM1; ALU pipe: int; $243
(W)     mov (1|M0)               r72.5<1>:ud   r90.4<0;1,0>:d                                        //  blk2d.X; ALU pipe: int; $243
(W)     mov (1|M0)               r72.7<1>:ud   0xF1F:uw                                              //  bkl2d.shape = 1x32x16; ALU pipe: int; $243
        load_block2d.ugm.d16.a64.ca.ca (1|M0)  null:0 [r72:1]      {I@1,$6} // ex_desc:0x0; desc:0x2080203 // $243
        sync.allwr                           ($11,$13)                                               // $203
        dpas.8x8 (16|M0)         r92:f         r92:f             r5:bf             r79.0:bf         {Compacted,$15} // $203
        sync.nop                             null                             {Compacted,$15.src}    // $251
        load_block2d.ugm.d16.a64 (1|M0)  r73:8  [r91:1]            {$8} // ex_desc:0x0; desc:0x2800203 // $251

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     bfn.(s0|s1|s2) (1|M0)    r91.5<1>:ud   r21.0<0;0>:ud     r90.1<0;0>:ud     r125.2<0>:ud     {$8.src} //  ALU pipe: int; $246

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r91.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $260

// Line 85:  off_k += BLOCK_SIZE_K
(W)     or (1|M0)                r81.0<1>:d    r90.9<0;1,0>:d    448:w               {Compacted}     //  ALU pipe: int; $342

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     bfn.(s0|s1|s2) (1|M0)    r82.6<1>:ud   r90.5<0;0>:ud     r90.9<0;0>:ud     r125.10<0>:ud     //  ALU pipe: int; $344
(W)     mov (1|M0)               r82.0<1>:uq   r90.1<0;1,0>:q                                        //  ALU pipe: int; $345
        dpas.8x8 (16|M0)         r92:f         r92:f             r13:bf            r83.0:bf         {Compacted,$15} // $204

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        sync.nop                             null                             {Compacted,$15.src}    // $255
        load_block2d.ugm.d16v.a64 (1|M0)  r5:16  [r124:1]           {$14} // ex_desc:0x0; desc:0x3000283 // $255

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     or (1|M0)                r124.6<1>:d   r90.9<0;1,0>:d    224:w               {$14.src}       //  ALU pipe: int; $248

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     mov (1|M0)               r124.5<1>:d   r90.8<0;1,0>:d                                        //  ALU pipe: int; $263

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r82.2<1>:ud   r126.1<0;1,0>:d                                       //  blk2d.widthM1; ALU pipe: int; $345
(W)     mov (1|M0)               r82.3<1>:ud   4095:w                                                //  blk2d.heightM1; ALU pipe: int; $345
(W)     mov (1|M0)               r82.4<1>:ud   24575:w                                               //  blk2d.pitchM1; ALU pipe: int; $345
        sync.allwr                           ($1,$15)                                                // $235
        dpas.8x8 (16|M0)         r92:f         r92:f             r31:bf            r23.0:bf         {Compacted,$0} // $235
(W)     mov (1|M0)               r82.5<1>:ud   r90.4<0;1,0>:d                                        //  blk2d.X; ALU pipe: int; $345
(W)     mov (1|M0)               r82.7<1>:ud   0xF1F:uw                                              //  bkl2d.shape = 1x32x16; ALU pipe: int; $345
        load_block2d.ugm.d16.a64.ca.ca (1|M0)  null:0 [r82:1]      {I@1,$7} // ex_desc:0x0; desc:0x2080203 // $345
        dpas.8x8 (16|M0)         r92:f         r92:f             r39:bf            r27.0:bf         {Compacted,$0} // $236
        sync.nop                             null                             {Compacted,$0.src}     // $261
        load_block2d.ugm.d16.a64 (1|M0)  r21:8  [r91:1]            {$9} // ex_desc:0x0; desc:0x2800203 // $261

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        load_block2d.ugm.d16v.a64 (1|M0)  r29:16 [r124:1]           {$10} // ex_desc:0x0; desc:0x3000283 // $265
(W)     bfn.(s0|s1|s2) (1|M0)    r91.5<1>:ud   r90.9<0;0>:ud     r125.13<0;0>:ud   r90.1<0>:ud      {$9.src} //  ALU pipe: int; $279

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r91.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $284

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     mov (1|M0)               r124.5<1>:d   r90.8<0;1,0>:d                   {$10.src}            //  ALU pipe: int; $287
(W)     mov (1|M0)               r124.6<1>:d   r71.0<0;1,0>:d                                        //  ALU pipe: int; $288

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        sync.allwr                           ($0,$4)                                                 // $237
        dpas.8x8 (16|M0)         r92:f         r92:f             r55:bf            r47.0:bf         {Compacted,$2} // $237

// Line 85:  off_k += BLOCK_SIZE_K
(W)     or (1|M0)                r45.0<1>:d    r90.9<0;1,0>:d    320:w               {Compacted}     //  ALU pipe: int; $274

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     bfn.(s0|s1|s2) (1|M0)    r46.6<1>:ud   r90.5<0;0>:ud     r90.9<0;0>:ud     r125.12<0>:ud     //  ALU pipe: int; $276
(W)     mov (1|M0)               r46.0<1>:uq   r90.1<0;1,0>:q                                        //  ALU pipe: int; $277
(W)     mov (1|M0)               r46.2<1>:ud   r126.1<0;1,0>:d                                       //  blk2d.widthM1; ALU pipe: int; $277
(W)     mov (1|M0)               r46.3<1>:ud   4095:w                                                //  blk2d.heightM1; ALU pipe: int; $277
(W)     mov (1|M0)               r46.4<1>:ud   24575:w                                               //  blk2d.pitchM1; ALU pipe: int; $277
        dpas.8x8 (16|M0)         r92:f         r92:f             r63:bf            r51.0:bf         {Compacted,$2} // $238
        sync.nop                             null                             {Compacted,$2.src}     // $285
        load_block2d.ugm.d16.a64 (1|M0)  r47:8  [r91:1]            {I@7,$11} // ex_desc:0x0; desc:0x2800203 // $285

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        load_block2d.ugm.d16v.a64 (1|M0)  r55:16 [r124:1]           {I@7,$12} // ex_desc:0x0; desc:0x3000283 // $289
(W)     bfn.(s0|s1|s2) (1|M0)    r91.5<1>:ud   r71.0<0;0>:ud     r90.1<0;0>:ud     r125.2<0>:ud     {$11.src} //  ALU pipe: int; $280

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r91.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $294
(W)     or (1|M0)                r124.6<1>:d   r90.9<0;1,0>:d    288:w               {$12.src}       //  ALU pipe: int; $282

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     mov (1|M0)               r124.5<1>:d   r90.8<0;1,0>:d                                        //  ALU pipe: int; $297

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r46.5<1>:ud   r90.4<0;1,0>:d                                        //  blk2d.X; ALU pipe: int; $277
(W)     mov (1|M0)               r46.7<1>:ud   0xF1F:uw                                              //  bkl2d.shape = 1x32x16; ALU pipe: int; $277
        load_block2d.ugm.d16.a64.ca.ca (1|M0)  null:0 [r46:1]      {I@1,$13} // ex_desc:0x0; desc:0x2080203 // $277
        sync.allwr                           ($2,$14)                                                // $269
        dpas.8x8 (16|M0)         r92:f         r92:f             r5:bf             r73.0:bf         {Compacted,$8} // $269
        dpas.8x8 (16|M0)         r92:f         r92:f             r13:bf            r77.0:bf         {Compacted,$8} // $270 R{} IR{}{E:6,O:6,O:6,},  R{} IR{}{O:14,E:7,E:7,},  {BC=2}
        sync.nop                             null                             {Compacted,$8.src}     // $295
        load_block2d.ugm.d16.a64 (1|M0)  r71:8  [r91:1]            {$15} // ex_desc:0x0; desc:0x2800203 // $295

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     bfn.(s0|s1|s2) (1|M0)    r91.5<1>:ud   r90.9<0;0>:ud     r125.12<0;0>:ud   r90.1<0>:ud      {$15.src} //  ALU pipe: int; $313

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r91.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $318
        sync.allwr                           ($8,$10)                                                // $271
        dpas.8x8 (16|M0)         r92:f         r92:f             r29:bf            r21.0:bf         {Compacted,$9} // $271

// Line 85:  off_k += BLOCK_SIZE_K
(W)     or (1|M0)                r30.0<1>:d    r90.9<0;1,0>:d    384:w               {Compacted,$9.src} //  ALU pipe: int; $308

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     bfn.(s0|s1|s2) (1|M0)    r31.6<1>:ud   r90.5<0;0>:ud     r90.9<0;0>:ud     r125.11<0>:ud     //  ALU pipe: int; $310
(W)     mov (1|M0)               r31.0<1>:uq   r90.1<0;1,0>:q                                        //  ALU pipe: int; $311
(W)     mov (1|M0)               r31.2<1>:ud   r126.1<0;1,0>:d                                       //  blk2d.widthM1; ALU pipe: int; $311
(W)     mov (1|M0)               r31.3<1>:ud   4095:w                                                //  blk2d.heightM1; ALU pipe: int; $311
(W)     mov (1|M0)               r31.4<1>:ud   24575:w                                               //  blk2d.pitchM1; ALU pipe: int; $311
        dpas.8x8 (16|M0)         r92:f         r92:f             r37:bf            r25.0:bf         {Compacted,$9} // $272

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        sync.nop                             null                             {Compacted,$9.src}     // $299
        load_block2d.ugm.d16v.a64 (1|M0)  r14:16 [r124:1]           {$1} // ex_desc:0x0; desc:0x3000283 // $299
(W)     mov (1|M0)               r124.5<1>:d   r90.8<0;1,0>:d                   {$1.src}             //  ALU pipe: int; $321
(W)     mov (1|M0)               r124.6<1>:d   r45.0<0;1,0>:d                                        //  ALU pipe: int; $322

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        load_block2d.ugm.d16.a64 (1|M0)  r33:8  [r91:1]            {I@7,$5} // ex_desc:0x0; desc:0x2800203 // $319

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     bfn.(s0|s1|s2) (1|M0)    r91.5<1>:ud   r45.0<0;0>:ud     r90.1<0;0>:ud     r125.2<0>:ud     {$5.src} //  ALU pipe: int; $314 R{} IR{}{O:6,E:5,O:6,},  {BC=1}

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r91.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $328
        sync.allwr                           ($9,$12)                                                // $303
        dpas.8x8 (16|M0)         r92:f         r92:f             r55:bf            r47.0:bf         {Compacted,$11} // $303
(W)     mov (1|M0)               r31.5<1>:ud   r90.4<0;1,0>:d                                        //  blk2d.X; ALU pipe: int; $311
(W)     mov (1|M0)               r31.7<1>:ud   0xF1F:uw                                              //  bkl2d.shape = 1x32x16; ALU pipe: int; $311
        load_block2d.ugm.d16.a64.ca.ca (1|M0)  null:0 [r31:1]      {I@1,$3} // ex_desc:0x0; desc:0x2080203 // $311
        dpas.8x8 (16|M0)         r92:f         r92:f             r63:bf            r51.0:bf         {Compacted,$11} // $304

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        sync.nop                             null                             {Compacted,$11.src}    // $323
        load_block2d.ugm.d16v.a64 (1|M0)  r41:16 [r124:1]           {$4} // ex_desc:0x0; desc:0x3000283 // $323

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     or (1|M0)                r124.6<1>:d   r90.9<0;1,0>:d    352:w               {$4.src}        //  ALU pipe: int; $316

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     mov (1|M0)               r124.5<1>:d   r90.8<0;1,0>:d                                        //  ALU pipe: int; $331

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        load_block2d.ugm.d16.a64 (1|M0)  r57:8  [r91:1]            {$7} // ex_desc:0x0; desc:0x2800203 // $329

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     bfn.(s0|s1|s2) (1|M0)    r91.5<1>:ud   r90.9<0;0>:ud     r125.11<0;0>:ud   r90.1<0>:ud      {$7.src} //  ALU pipe: int; $347

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r91.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $352
        load_block2d.ugm.d16.a64 (1|M0)  r5:8   [r91:1]            {I@1,$0} // ex_desc:0x0; desc:0x2800203 // $353

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     bfn.(s0|s1|s2) (1|M0)    r91.5<1>:ud   r30.0<0;0>:ud     r90.1<0;0>:ud     r125.2<0>:ud     {$0.src} //  ALU pipe: int; $348

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r91.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $362
        sync.allwr                           ($1,$11)                                                // $305
        dpas.8x8 (16|M0)         r92:f         r92:f             r14:bf            r71.0:bf         {Compacted,$15} // $305
        dpas.8x8 (16|M0)         r92:f         r92:f             r22:bf            r75.0:bf         {Compacted,$15} // $306

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        sync.nop                             null                             {Compacted,$15.src}    // $333
        load_block2d.ugm.d16v.a64 (1|M0)  r65:16 [r124:1]           {$6} // ex_desc:0x0; desc:0x3000283 // $333
(W)     mov (1|M0)               r124.5<1>:d   r90.8<0;1,0>:d                   {$6.src}             //  ALU pipe: int; $355
(W)     mov (1|M0)               r124.6<1>:d   r30.0<0;1,0>:d                                        //  ALU pipe: int; $356
        load_block2d.ugm.d16v.a64 (1|M0)  r13:16 [r124:1]           {I@1,$14} // ex_desc:0x0; desc:0x3000283 // $357

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     or (1|M0)                r124.6<1>:d   r90.9<0;1,0>:d    416:w               {$14.src}       //  ALU pipe: int; $350
        sync.allwr                           ($4,$15)                                                // $337
        dpas.8x8 (16|M0)         r92:f         r92:f             r41:bf            r33.0:bf         {Compacted,$5} // $337

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     mov (1|M0)               r124.5<1>:d   r90.8<0;1,0>:d                                        //  ALU pipe: int; $365

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        sync.nop                             null                             {Compacted,$5.src}     // $363
        load_block2d.ugm.d16.a64 (1|M0)  r29:8  [r91:1]            {$10} // ex_desc:0x0; desc:0x2800203 // $363

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     bfn.(s0|s1|s2) (1|M0)    r91.5<1>:ud   r90.9<0;0>:ud     r125.10<0;0>:ud   r90.1<0>:ud      {$10.src} //  ALU pipe: int; $381

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r91.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $386
        dpas.8x8 (16|M0)         r92:f         r92:f             r49:bf            r37.0:bf         {Compacted,$5} // $338

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        sync.nop                             null                             {Compacted,$5.src}     // $367
        load_block2d.ugm.d16v.a64 (1|M0)  r37:16 [r124:1]           {I@3,$2} // ex_desc:0x0; desc:0x3000283 // $367
(W)     mov (1|M0)               r124.5<1>:d   r90.8<0;1,0>:d                   {$2.src}             //  ALU pipe: int; $389
(W)     mov (1|M0)               r124.6<1>:d   r81.0<0;1,0>:d                                        //  ALU pipe: int; $390

// Line 85:  off_k += BLOCK_SIZE_K
(W)     or (1|M0)                r53.0<1>:d    r90.9<0;1,0>:d    512:w               {Compacted}     //  ALU pipe: int; $376

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     bfn.(s0|s1|s2) (1|M0)    r54.6<1>:ud   r90.5<0;0>:ud     r90.9<0;0>:ud     r125.9<0>:ud      //  ALU pipe: int; $378
(W)     mov (1|M0)               r54.0<1>:uq   r90.1<0;1,0>:q                                        //  ALU pipe: int; $379
(W)     mov (1|M0)               r54.2<1>:ud   r126.1<0;1,0>:d                                       //  blk2d.widthM1; ALU pipe: int; $379
(W)     mov (1|M0)               r54.3<1>:ud   4095:w                                                //  blk2d.heightM1; ALU pipe: int; $379
(W)     mov (1|M0)               r54.4<1>:ud   24575:w                                               //  blk2d.pitchM1; ALU pipe: int; $379
(W)     mov (1|M0)               r54.5<1>:ud   r90.4<0;1,0>:d                                        //  blk2d.X; ALU pipe: int; $379
(W)     mov (1|M0)               r54.7<1>:ud   0xF1F:uw                                              //  bkl2d.shape = 1x32x16; ALU pipe: int; $379
        load_block2d.ugm.d16.a64.ca.ca (1|M0)  null:0 [r54:1]      {I@1,$13} // ex_desc:0x0; desc:0x2080203 // $379
        sync.allwr                           ($5,$6)                                                 // $339
        dpas.8x8 (16|M0)         r92:f         r92:f             r65:bf            r57.0:bf         {Compacted,$7} // $339
        dpas.8x8 (16|M0)         r92:f         r92:f             r73:bf            r61.0:bf         {Compacted,$7} // $340
        sync.nop                             null                             {Compacted,$7.src}     // $387
        load_block2d.ugm.d16.a64 (1|M0)  r55:8  [r91:1]            {$15} // ex_desc:0x0; desc:0x2800203 // $387

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        load_block2d.ugm.d16v.a64 (1|M0)  r63:16 [r124:1]           {$1} // ex_desc:0x0; desc:0x3000283 // $391
(W)     bfn.(s0|s1|s2) (1|M0)    r91.5<1>:ud   r81.0<0;0>:ud     r90.1<0;0>:ud     r125.2<0>:ud     {$15.src} //  ALU pipe: int; $382

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r91.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $396
(W)     or (1|M0)                r124.6<1>:d   r90.9<0;1,0>:d    480:w               {$1.src}        //  ALU pipe: int; $384

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     mov (1|M0)               r124.5<1>:d   r90.8<0;1,0>:d                                        //  ALU pipe: int; $399

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        sync.allwr                           ($7,$14)                                                // $371
        dpas.8x8 (16|M0)         r92:f         r92:f             r13:bf            r5.0:bf          {Compacted,$0} // $371
        load_block2d.ugm.d16.a64 (1|M0)  r79:8  [r91:1]            {I@3,$8} // ex_desc:0x0; desc:0x2800203 // $397

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     bfn.(s0|s1|s2) (1|M0)    r91.5<1>:ud   r90.9<0;0>:ud     r125.9<0;0>:ud    r90.1<0>:ud      {$8.src} //  ALU pipe: int; $415

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r91.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $420
        dpas.8x8 (16|M0)         r92:f         r92:f             r21:bf            r9.0:bf          {Compacted,$0} // $372

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        sync.nop                             null                             {Compacted,$0.src}     // $401
        load_block2d.ugm.d16v.a64 (1|M0)  r5:16  [r124:1]           {I@3,$9} // ex_desc:0x0; desc:0x3000283 // $401
(W)     mov (1|M0)               r124.5<1>:d   r90.8<0;1,0>:d                   {$9.src}             //  ALU pipe: int; $423
(W)     mov (1|M0)               r124.6<1>:d   r53.0<0;1,0>:d                                        //  ALU pipe: int; $424

// Line 85:  off_k += BLOCK_SIZE_K
(W)     or (1|M0)                r21.0<1>:d    r90.9<0;1,0>:d    576:w               {Compacted}     //  ALU pipe: int; $410

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     bfn.(s0|s1|s2) (1|M0)    r22.6<1>:ud   r90.5<0;0>:ud     r90.9<0;0>:ud     r125.8<0>:ud      //  ALU pipe: int; $412
(W)     mov (1|M0)               r22.0<1>:uq   r90.1<0;1,0>:q                                        //  ALU pipe: int; $413
        sync.allwr                           ($0,$2)                                                 // $373
        dpas.8x8 (16|M0)         r92:f         r92:f             r37:bf            r29.0:bf         {Compacted,$10} // $373
        sync.nop                             null                             {Compacted,$10.src}    // $421
        load_block2d.ugm.d16.a64 (1|M0)  r23:8  [r91:1]            {I@6,$11} // ex_desc:0x0; desc:0x2800203 // $421

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     bfn.(s0|s1|s2) (1|M0)    r91.5<1>:ud   r53.0<0;0>:ud     r90.1<0;0>:ud     r125.2<0>:ud     {$11.src} //  ALU pipe: int; $416

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r91.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $430
(W)     mov (1|M0)               r22.2<1>:ud   r126.1<0;1,0>:d                                       //  blk2d.widthM1; ALU pipe: int; $413
(W)     mov (1|M0)               r22.3<1>:ud   4095:w                                                //  blk2d.heightM1; ALU pipe: int; $413
(W)     mov (1|M0)               r22.4<1>:ud   24575:w                                               //  blk2d.pitchM1; ALU pipe: int; $413
        dpas.8x8 (16|M0)         r92:f         r92:f             r45:bf            r33.0:bf         {Compacted,$10} // $374

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        sync.nop                             null                             {Compacted,$10.src}    // $425
        load_block2d.ugm.d16v.a64 (1|M0)  r31:16 [r124:1]           {I@7,$12} // ex_desc:0x0; desc:0x3000283 // $425

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     or (1|M0)                r124.6<1>:d   r90.9<0;1,0>:d    544:w               {$12.src}       //  ALU pipe: int; $418

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     mov (1|M0)               r124.5<1>:d   r90.8<0;1,0>:d                                        //  ALU pipe: int; $433

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        load_block2d.ugm.d16.a64 (1|M0)  r47:8  [r91:1]            {I@6,$3} // ex_desc:0x0; desc:0x2800203 // $431

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     bfn.(s0|s1|s2) (1|M0)    r91.5<1>:ud   r90.9<0;0>:ud     r125.8<0;0>:ud    r90.1<0>:ud      {$3.src} //  ALU pipe: int; $449

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r91.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $454
(W)     mov (1|M0)               r22.5<1>:ud   r90.4<0;1,0>:d                                        //  blk2d.X; ALU pipe: int; $413
(W)     mov (1|M0)               r22.7<1>:ud   0xF1F:uw                                              //  bkl2d.shape = 1x32x16; ALU pipe: int; $413
        load_block2d.ugm.d16.a64.ca.ca (1|M0)  null:0 [r22:1]      {I@1,$4} // ex_desc:0x0; desc:0x2080203 // $413
        sync.allwr                           ($1,$10)                                                // $405
        dpas.8x8 (16|M0)         r92:f         r92:f             r63:bf            r55.0:bf         {Compacted,$15} // $405
        dpas.8x8 (16|M0)         r92:f         r92:f             r71:bf            r59.0:bf         {Compacted,$15} // $406

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        sync.nop                             null                             {Compacted,$15.src}    // $435
        load_block2d.ugm.d16v.a64 (1|M0)  r55:16 [r124:1]           {$6} // ex_desc:0x0; desc:0x3000283 // $435
(W)     mov (1|M0)               r124.5<1>:d   r90.8<0;1,0>:d                   {$6.src}             //  ALU pipe: int; $457
(W)     mov (1|M0)               r124.6<1>:d   r21.0<0;1,0>:d                                        //  ALU pipe: int; $458

// Line 85:  off_k += BLOCK_SIZE_K
(W)     or (1|M0)                r71.0<1>:d    r90.9<0;1,0>:d    640:w               {Compacted}     //  ALU pipe: int; $444

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     bfn.(s0|s1|s2) (1|M0)    r72.6<1>:ud   r90.5<0;0>:ud     r90.9<0;0>:ud     r125.7<0>:ud      //  ALU pipe: int; $446
(W)     mov (1|M0)               r72.0<1>:uq   r90.1<0;1,0>:q                                        //  ALU pipe: int; $447
        sync.allwr                           ($9,$15)                                                // $407
        dpas.8x8 (16|M0)         r92:f         r92:f             r5:bf             r79.0:bf         {Compacted,$8} // $407
        sync.nop                             null                             {Compacted,$8.src}     // $455
        load_block2d.ugm.d16.a64 (1|M0)  r73:8  [r91:1]            {$14} // ex_desc:0x0; desc:0x2800203 // $455

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     bfn.(s0|s1|s2) (1|M0)    r91.5<1>:ud   r21.0<0;0>:ud     r90.1<0;0>:ud     r125.2<0>:ud     {$14.src} //  ALU pipe: int; $450

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r91.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $464
(W)     mov (1|M0)               r72.2<1>:ud   r126.1<0;1,0>:d                                       //  blk2d.widthM1; ALU pipe: int; $447
(W)     mov (1|M0)               r72.3<1>:ud   4095:w                                                //  blk2d.heightM1; ALU pipe: int; $447
(W)     mov (1|M0)               r72.4<1>:ud   24575:w                                               //  blk2d.pitchM1; ALU pipe: int; $447
        dpas.8x8 (16|M0)         r92:f         r92:f             r13:bf            r83.0:bf         {Compacted,$8} // $408

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        sync.nop                             null                             {Compacted,$8.src}     // $459
        load_block2d.ugm.d16v.a64 (1|M0)  r5:16  [r124:1]           {I@7,$13} // ex_desc:0x0; desc:0x3000283 // $459

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     or (1|M0)                r124.6<1>:d   r90.9<0;1,0>:d    608:w               {$13.src}       //  ALU pipe: int; $452

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     mov (1|M0)               r124.5<1>:d   r90.8<0;1,0>:d                                        //  ALU pipe: int; $467

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r72.5<1>:ud   r90.4<0;1,0>:d                                        //  blk2d.X; ALU pipe: int; $447
(W)     mov (1|M0)               r72.7<1>:ud   0xF1F:uw                                              //  bkl2d.shape = 1x32x16; ALU pipe: int; $447

// Line 85:  off_k += BLOCK_SIZE_K
(W)     or (1|M0)                r81.0<1>:d    r90.9<0;1,0>:d    832:w               {Compacted}     //  ALU pipe: int; $546

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        sync.allwr                           ($8,$12)                                                // $439
        dpas.8x8 (16|M0)         r92:f         r92:f             r31:bf            r23.0:bf         {Compacted,$11} // $439
        load_block2d.ugm.d16.a64.ca.ca (1|M0)  null:0 [r72:1]      {I@2,$15} // ex_desc:0x0; desc:0x2080203 // $447
(W)     bfn.(s0|s1|s2) (1|M0)    r82.6<1>:ud   r90.5<0;0>:ud     r90.9<0;0>:ud     r125.4<0>:ud      //  ALU pipe: int; $548
(W)     mov (1|M0)               r82.0<1>:uq   r90.1<0;1,0>:q                                        //  ALU pipe: int; $549
(W)     mov (1|M0)               r82.2<1>:ud   r126.1<0;1,0>:d                                       //  blk2d.widthM1; ALU pipe: int; $549
(W)     mov (1|M0)               r82.3<1>:ud   4095:w                                                //  blk2d.heightM1; ALU pipe: int; $549
(W)     mov (1|M0)               r82.4<1>:ud   24575:w                                               //  blk2d.pitchM1; ALU pipe: int; $549
        dpas.8x8 (16|M0)         r92:f         r92:f             r39:bf            r27.0:bf         {Compacted,$11} // $440
        sync.nop                             null                             {Compacted,$11.src}    // $465
        load_block2d.ugm.d16.a64 (1|M0)  r21:8  [r91:1]            {$0} // ex_desc:0x0; desc:0x2800203 // $465

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        load_block2d.ugm.d16v.a64 (1|M0)  r29:16 [r124:1]           {$1} // ex_desc:0x0; desc:0x3000283 // $469
(W)     bfn.(s0|s1|s2) (1|M0)    r91.5<1>:ud   r90.9<0;0>:ud     r125.7<0;0>:ud    r90.1<0>:ud      {$0.src} //  ALU pipe: int; $483

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r91.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $488

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     mov (1|M0)               r124.5<1>:d   r90.8<0;1,0>:d                   {$1.src}             //  ALU pipe: int; $491
(W)     mov (1|M0)               r124.6<1>:d   r71.0<0;1,0>:d                                        //  ALU pipe: int; $492

// Line 85:  off_k += BLOCK_SIZE_K
(W)     or (1|M0)                r45.0<1>:d    r90.9<0;1,0>:d    704:w               {Compacted}     //  ALU pipe: int; $478

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     bfn.(s0|s1|s2) (1|M0)    r46.6<1>:ud   r90.5<0;0>:ud     r90.9<0;0>:ud     r125.6<0>:ud      //  ALU pipe: int; $480
(W)     mov (1|M0)               r46.0<1>:uq   r90.1<0;1,0>:q                                        //  ALU pipe: int; $481
(W)     mov (1|M0)               r46.2<1>:ud   r126.1<0;1,0>:d                                       //  blk2d.widthM1; ALU pipe: int; $481
(W)     mov (1|M0)               r46.3<1>:ud   4095:w                                                //  blk2d.heightM1; ALU pipe: int; $481
(W)     mov (1|M0)               r46.4<1>:ud   24575:w                                               //  blk2d.pitchM1; ALU pipe: int; $481
(W)     mov (1|M0)               r46.5<1>:ud   r90.4<0;1,0>:d                                        //  blk2d.X; ALU pipe: int; $481
(W)     mov (1|M0)               r46.7<1>:ud   0xF1F:uw                                              //  bkl2d.shape = 1x32x16; ALU pipe: int; $481
(W)     mov (1|M0)               r82.5<1>:ud   r90.4<0;1,0>:d                                        //  blk2d.X; ALU pipe: int; $549
        load_block2d.ugm.d16.a64.ca.ca (1|M0)  null:0 [r46:1]      {I@2,$2} // ex_desc:0x0; desc:0x2080203 // $481
(W)     mov (1|M0)               r82.7<1>:ud   0xF1F:uw                                              //  bkl2d.shape = 1x32x16; ALU pipe: int; $549
        sync.allwr                           ($6,$11)                                                // $441
        dpas.8x8 (16|M0)         r92:f         r92:f             r55:bf            r47.0:bf         {Compacted,$3} // $441
        load_block2d.ugm.d16.a64.ca.ca (1|M0)  null:0 [r82:1]      {I@1,$7} // ex_desc:0x0; desc:0x2080203 // $549
        dpas.8x8 (16|M0)         r92:f         r92:f             r63:bf            r51.0:bf         {Compacted,$3} // $442
        sync.nop                             null                             {Compacted,$3.src}     // $489
        load_block2d.ugm.d16.a64 (1|M0)  r47:8  [r91:1]            {$8} // ex_desc:0x0; desc:0x2800203 // $489

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        load_block2d.ugm.d16v.a64 (1|M0)  r55:16 [r124:1]           {$9} // ex_desc:0x0; desc:0x3000283 // $493
(W)     bfn.(s0|s1|s2) (1|M0)    r91.5<1>:ud   r71.0<0;0>:ud     r90.1<0;0>:ud     r125.2<0>:ud     {$8.src} //  ALU pipe: int; $484

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r91.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $498
(W)     or (1|M0)                r124.6<1>:d   r90.9<0;1,0>:d    672:w               {$9.src}        //  ALU pipe: int; $486

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     mov (1|M0)               r124.5<1>:d   r90.8<0;1,0>:d                                        //  ALU pipe: int; $501

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        sync.allwr                           ($3,$13)                                                // $473
        dpas.8x8 (16|M0)         r92:f         r92:f             r5:bf             r73.0:bf         {Compacted,$14} // $473
        dpas.8x8 (16|M0)         r92:f         r92:f             r13:bf            r77.0:bf         {Compacted,$14} // $474 R{} IR{}{E:6,O:6,O:6,},  R{} IR{}{O:14,E:7,E:7,},  {BC=2}
        sync.nop                             null                             {Compacted,$14.src}    // $499
        load_block2d.ugm.d16.a64 (1|M0)  r71:8  [r91:1]            {I@3,$10} // ex_desc:0x0; desc:0x2800203 // $499

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     bfn.(s0|s1|s2) (1|M0)    r91.5<1>:ud   r90.9<0;0>:ud     r125.6<0;0>:ud    r90.1<0>:ud      {$10.src} //  ALU pipe: int; $517

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r91.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $522
        sync.allwr                           ($1,$14)                                                // $475
        dpas.8x8 (16|M0)         r92:f         r92:f             r29:bf            r21.0:bf         {Compacted,$0} // $475

// Line 85:  off_k += BLOCK_SIZE_K
(W)     or (1|M0)                r30.0<1>:d    r90.9<0;1,0>:d    768:w               {Compacted,$0.src} //  ALU pipe: int; $512

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     bfn.(s0|s1|s2) (1|M0)    r31.6<1>:ud   r90.5<0;0>:ud     r90.9<0;0>:ud     r125.5<0>:ud      //  ALU pipe: int; $514
(W)     mov (1|M0)               r31.0<1>:uq   r90.1<0;1,0>:q                                        //  ALU pipe: int; $515
(W)     mov (1|M0)               r31.2<1>:ud   r126.1<0;1,0>:d                                       //  blk2d.widthM1; ALU pipe: int; $515
(W)     mov (1|M0)               r31.3<1>:ud   4095:w                                                //  blk2d.heightM1; ALU pipe: int; $515
(W)     mov (1|M0)               r31.4<1>:ud   24575:w                                               //  blk2d.pitchM1; ALU pipe: int; $515
        dpas.8x8 (16|M0)         r92:f         r92:f             r37:bf            r25.0:bf         {Compacted,$0} // $476

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        sync.nop                             null                             {Compacted,$0.src}     // $503
        load_block2d.ugm.d16v.a64 (1|M0)  r14:16 [r124:1]           {I@7,$12} // ex_desc:0x0; desc:0x3000283 // $503
(W)     mov (1|M0)               r124.5<1>:d   r90.8<0;1,0>:d                   {$12.src}            //  ALU pipe: int; $525
(W)     mov (1|M0)               r124.6<1>:d   r45.0<0;1,0>:d                                        //  ALU pipe: int; $526

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        load_block2d.ugm.d16.a64 (1|M0)  r33:8  [r91:1]            {I@7,$5} // ex_desc:0x0; desc:0x2800203 // $523

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     bfn.(s0|s1|s2) (1|M0)    r91.5<1>:ud   r45.0<0;0>:ud     r90.1<0;0>:ud     r125.2<0>:ud     {$5.src} //  ALU pipe: int; $518 R{} IR{}{O:6,E:5,O:6,},  {BC=1}

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r91.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $532
(W)     mov (1|M0)               r31.5<1>:ud   r90.4<0;1,0>:d                                        //  blk2d.X; ALU pipe: int; $515
(W)     mov (1|M0)               r31.7<1>:ud   0xF1F:uw                                              //  bkl2d.shape = 1x32x16; ALU pipe: int; $515
        load_block2d.ugm.d16.a64.ca.ca (1|M0)  null:0 [r31:1]      {I@1,$15} // ex_desc:0x0; desc:0x2080203 // $515
        sync.allwr                           ($0,$9)                                                 // $507
        dpas.8x8 (16|M0)         r92:f         r92:f             r55:bf            r47.0:bf         {Compacted,$8} // $507
        dpas.8x8 (16|M0)         r92:f         r92:f             r63:bf            r51.0:bf         {Compacted,$8} // $508

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        sync.nop                             null                             {Compacted,$8.src}     // $527
        load_block2d.ugm.d16v.a64 (1|M0)  r41:16 [r124:1]           {$4} // ex_desc:0x0; desc:0x3000283 // $527

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     or (1|M0)                r124.6<1>:d   r90.9<0;1,0>:d    736:w               {$4.src}        //  ALU pipe: int; $520

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     mov (1|M0)               r124.5<1>:d   r90.8<0;1,0>:d                                        //  ALU pipe: int; $535

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        load_block2d.ugm.d16.a64 (1|M0)  r57:8  [r91:1]            {$6} // ex_desc:0x0; desc:0x2800203 // $533

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     bfn.(s0|s1|s2) (1|M0)    r91.5<1>:ud   r90.9<0;0>:ud     r125.5<0;0>:ud    r90.1<0>:ud      {$6.src} //  ALU pipe: int; $551

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r91.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $556
        load_block2d.ugm.d16.a64 (1|M0)  r5:8   [r91:1]            {I@1,$13} // ex_desc:0x0; desc:0x2800203 // $557

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     bfn.(s0|s1|s2) (1|M0)    r91.5<1>:ud   r30.0<0;0>:ud     r90.1<0;0>:ud     r125.2<0>:ud     {$13.src} //  ALU pipe: int; $552

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r91.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $566
        sync.allwr                           ($8,$12)                                                // $509
        dpas.8x8 (16|M0)         r92:f         r92:f             r14:bf            r71.0:bf         {Compacted,$10} // $509
        dpas.8x8 (16|M0)         r92:f         r92:f             r22:bf            r75.0:bf         {Compacted,$10} // $510

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        sync.nop                             null                             {Compacted,$10.src}    // $537
        load_block2d.ugm.d16v.a64 (1|M0)  r65:16 [r124:1]           {$11} // ex_desc:0x0; desc:0x3000283 // $537
(W)     mov (1|M0)               r124.5<1>:d   r90.8<0;1,0>:d                   {$11.src}            //  ALU pipe: int; $559
(W)     mov (1|M0)               r124.6<1>:d   r30.0<0;1,0>:d                                        //  ALU pipe: int; $560
        load_block2d.ugm.d16v.a64 (1|M0)  r13:16 [r124:1]           {I@1,$1} // ex_desc:0x0; desc:0x3000283 // $561

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     or (1|M0)                r124.6<1>:d   r90.9<0;1,0>:d    800:w               {$1.src}        //  ALU pipe: int; $554

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     mov (1|M0)               r124.5<1>:d   r90.8<0;1,0>:d                                        //  ALU pipe: int; $569

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        sync.allwr                           ($4,$10)                                                // $541
        dpas.8x8 (16|M0)         r92:f         r92:f             r41:bf            r33.0:bf         {Compacted,$5} // $541
        sync.nop                             null                             {Compacted,$5.src}     // $567
        load_block2d.ugm.d16.a64 (1|M0)  r29:8  [r91:1]            {$2} // ex_desc:0x0; desc:0x2800203 // $567

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     bfn.(s0|s1|s2) (1|M0)    r91.5<1>:ud   r90.9<0;0>:ud     r125.4<0;0>:ud    r90.1<0>:ud      {$2.src} //  ALU pipe: int; $585

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r91.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $590
        dpas.8x8 (16|M0)         r92:f         r92:f             r49:bf            r37.0:bf         {Compacted,$5} // $542

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        sync.nop                             null                             {Compacted,$5.src}     // $571
        load_block2d.ugm.d16v.a64 (1|M0)  r37:16 [r124:1]           {I@3,$7} // ex_desc:0x0; desc:0x3000283 // $571
(W)     mov (1|M0)               r124.5<1>:d   r90.8<0;1,0>:d                   {$7.src}             //  ALU pipe: int; $593
(W)     mov (1|M0)               r124.6<1>:d   r81.0<0;1,0>:d                                        //  ALU pipe: int; $594

// Line 85:  off_k += BLOCK_SIZE_K
(W)     or (1|M0)                r54.0<1>:d    r90.9<0;1,0>:d    896:w               {Compacted}     //  ALU pipe: int; $580

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     bfn.(s0|s1|s2) (1|M0)    r55.6<1>:ud   r90.5<0;0>:ud     r90.9<0;0>:ud     r125.3<0>:ud      //  ALU pipe: int; $582
(W)     mov (1|M0)               r55.0<1>:uq   r90.1<0;1,0>:q                                        //  ALU pipe: int; $583
(W)     mov (1|M0)               r55.2<1>:ud   r126.1<0;1,0>:d                                       //  blk2d.widthM1; ALU pipe: int; $583
(W)     mov (1|M0)               r55.3<1>:ud   4095:w                                                //  blk2d.heightM1; ALU pipe: int; $583
(W)     mov (1|M0)               r55.4<1>:ud   24575:w                                               //  blk2d.pitchM1; ALU pipe: int; $583
(W)     mov (1|M0)               r55.5<1>:ud   r90.4<0;1,0>:d                                        //  blk2d.X; ALU pipe: int; $583
(W)     mov (1|M0)               r55.7<1>:ud   0xF1F:uw                                              //  bkl2d.shape = 1x32x16; ALU pipe: int; $583
        load_block2d.ugm.d16.a64.ca.ca (1|M0)  null:0 [r55:1]      {I@1,$14} // ex_desc:0x0; desc:0x2080203 // $583
        sync.allwr                           ($5,$11)                                                // $543
        dpas.8x8 (16|M0)         r92:f         r92:f             r65:bf            r57.0:bf         {Compacted,$6} // $543
        dpas.8x8 (16|M0)         r92:f         r92:f             r73:bf            r61.0:bf         {Compacted,$6} // $544
        sync.nop                             null                             {Compacted,$6.src}     // $591
        load_block2d.ugm.d16.a64 (1|M0)  r57:8  [r91:1]            {$3} // ex_desc:0x0; desc:0x2800203 // $591

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        load_block2d.ugm.d16v.a64 (1|M0)  r65:16 [r124:1]           {$9} // ex_desc:0x0; desc:0x3000283 // $595
(W)     bfn.(s0|s1|s2) (1|M0)    r91.5<1>:ud   r81.0<0;0>:ud     r90.1<0;0>:ud     r125.2<0>:ud     {$3.src} //  ALU pipe: int; $586

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r91.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $600
(W)     or (1|M0)                r124.6<1>:d   r90.9<0;1,0>:d    864:w               {$9.src}        //  ALU pipe: int; $588

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     mov (1|M0)               r124.5<1>:d   r90.8<0;1,0>:d                                        //  ALU pipe: int; $603

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        sync.allwr                           ($1,$6)                                                 // $575
        dpas.8x8 (16|M0)         r92:f         r92:f             r13:bf            r5.0:bf          {Compacted,$13} // $575
        load_block2d.ugm.d16.a64 (1|M0)  r81:8  [r91:1]            {I@3,$0} // ex_desc:0x0; desc:0x2800203 // $601

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     bfn.(s0|s1|s2) (1|M0)    r91.5<1>:ud   r90.9<0;0>:ud     r125.3<0;0>:ud    r90.1<0>:ud      {$0.src} //  ALU pipe: int; $619

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r91.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $624
        dpas.8x8 (16|M0)         r92:f         r92:f             r21:bf            r9.0:bf          {Compacted,$13} // $576

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        sync.nop                             null                             {Compacted,$13.src}    // $605
        load_block2d.ugm.d16v.a64 (1|M0)  r5:16  [r124:1]           {I@3,$15} // ex_desc:0x0; desc:0x3000283 // $605
(W)     mov (1|M0)               r124.5<1>:d   r90.8<0;1,0>:d                   {$15.src}            //  ALU pipe: int; $627
(W)     mov (1|M0)               r124.6<1>:d   r54.0<0;1,0>:d                                        //  ALU pipe: int; $628

// Line 85:  off_k += BLOCK_SIZE_K
(W)     or (1|M0)                r22.0<1>:d    r90.9<0;1,0>:d    960:w               {Compacted}     //  ALU pipe: int; $614

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r23.0<1>:uq   r90.1<0;1,0>:q                                        //  ALU pipe: int; $617
(W)     mov (1|M0)               r23.2<1>:ud   r126.1<0;1,0>:d                                       //  blk2d.widthM1; ALU pipe: int; $617
        sync.allwr                           ($7,$13)                                                // $577
        dpas.8x8 (16|M0)         r92:f         r92:f             r37:bf            r29.0:bf         {Compacted,$2} // $577
        sync.nop                             null                             {Compacted,$2.src}     // $625
        load_block2d.ugm.d16.a64 (1|M0)  r25:8  [r91:1]            {I@6,$8} // ex_desc:0x0; desc:0x2800203 // $625

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     bfn.(s0|s1|s2) (1|M0)    r91.5<1>:ud   r54.0<0;0>:ud     r90.1<0;0>:ud     r125.2<0>:ud     {$8.src} //  ALU pipe: int; $620

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r91.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $634
(W)     mov (1|M0)               r23.3<1>:ud   4095:w                                                //  blk2d.heightM1; ALU pipe: int; $617
(W)     mov (1|M0)               r23.4<1>:ud   24575:w                                               //  blk2d.pitchM1; ALU pipe: int; $617
(W)     mov (1|M0)               r23.5<1>:ud   r90.4<0;1,0>:d                                        //  blk2d.X; ALU pipe: int; $617
        dpas.8x8 (16|M0)         r92:f         r92:f             r45:bf            r33.0:bf         {Compacted,$2} // $578

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        sync.nop                             null                             {Compacted,$2.src}     // $629
        load_block2d.ugm.d16v.a64 (1|M0)  r33:16 [r124:1]           {I@7,$10} // ex_desc:0x0; desc:0x3000283 // $629

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     or (1|M0)                r124.6<1>:d   r90.9<0;1,0>:d    928:w               {$10.src}       //  ALU pipe: int; $622

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     mov (1|M0)               r124.5<1>:d   r90.8<0;1,0>:d                                        //  ALU pipe: int; $637

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        load_block2d.ugm.d16.a64 (1|M0)  r49:8  [r91:1]            {I@6,$12} // ex_desc:0x0; desc:0x2800203 // $635

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     or (1|M0)                r91.5<1>:d    r22.0<0;1,0>:d    r90.1<0;1,0>:d   {$12.src}          //  ALU pipe: int; $653

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r91.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $658
(W)     mov (1|M0)               r23.7<1>:ud   0xF1F:uw                                              //  bkl2d.shape = 1x32x16; ALU pipe: int; $617
(W)     or (1|M0)                r23.6<1>:ud   r90.5<0;1,0>:d    r22.0<0;1,0>:d                      //  ALU pipe: int; $616

// Line 75:  for _ in range(0, K, BLOCK_SIZE_K):
(W)     cmp (16|M0)   (lt)f1.0   null<1>:d     r22.0<0;1,0>:ud   0xFC0:uw                            //  ALU pipe: int; $682

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        load_block2d.ugm.d16.a64.ca.ca (1|M0)  null:0 [r23:1]      {I@2,$11} // ex_desc:0x0; desc:0x2080203 // $617
        sync.allwr                           ($2,$9)                                                 // $609
        dpas.8x8 (16|M0)         r92:f         r92:f             r65:bf            r57.0:bf         {Compacted,$3} // $609
        dpas.8x8 (16|M0)         r92:f         r92:f             r73:bf            r61.0:bf         {Compacted,$3} // $610

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        sync.nop                             null                             {Compacted,$3.src}     // $639
        load_block2d.ugm.d16v.a64 (1|M0)  r57:16 [r124:1]           {$1} // ex_desc:0x0; desc:0x3000283 // $639
(W)     mov (1|M0)               r124.5<1>:d   r90.8<0;1,0>:d                   {$1.src}             //  ALU pipe: int; $661
(W)     mov (1|M0)               r124.6<1>:d   r22.0<0;1,0>:d                                        //  ALU pipe: int; $662

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     or (1|M0)                r73.6<1>:ud   r90.5<0;1,0>:d    r126.3<0;1,0>:d                     //  ALU pipe: int; $650
(W)     mov (1|M0)               r73.0<1>:uq   r90.1<0;1,0>:q                                        //  ALU pipe: int; $651
(W)     mov (1|M0)               r73.2<1>:ud   r126.1<0;1,0>:d                                       //  blk2d.widthM1; ALU pipe: int; $651
        sync.allwr                           ($3,$15)                                                // $611
        dpas.8x8 (16|M0)         r92:f         r92:f             r5:bf             r81.0:bf         {Compacted,$0} // $611
        sync.nop                             null                             {Compacted,$0.src}     // $659
        load_block2d.ugm.d16.a64 (1|M0)  r75:8  [r91:1]            {$5} // ex_desc:0x0; desc:0x2800203 // $659

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     bfn.(s0|s1|s2) (1|M0)    r91.5<1>:ud   r22.0<0;0>:ud     r90.1<0;0>:ud     r125.2<0>:ud     {$5.src} //  ALU pipe: int; $654

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r91.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $668
(W)     mov (1|M0)               r73.3<1>:ud   4095:w                                                //  blk2d.heightM1; ALU pipe: int; $651
(W)     mov (1|M0)               r73.4<1>:ud   24575:w                                               //  blk2d.pitchM1; ALU pipe: int; $651
(W)     mov (1|M0)               r73.5<1>:ud   r90.4<0;1,0>:d                                        //  blk2d.X; ALU pipe: int; $651
        dpas.8x8 (16|M0)         r92:f         r92:f             r13:bf            r85.0:bf         {Compacted,$0} // $612

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        sync.nop                             null                             {Compacted,$0.src}     // $663
        load_block2d.ugm.d16v.a64 (1|M0)  r5:16  [r124:1]           {I@7,$14} // ex_desc:0x0; desc:0x3000283 // $663

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     or (1|M0)                r124.6<1>:d   r90.9<0;1,0>:d    992:w               {$14.src}       //  ALU pipe: int; $656

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     mov (1|M0)               r124.5<1>:d   r90.8<0;1,0>:d                                        //  ALU pipe: int; $671

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r73.7<1>:ud   0xF1F:uw                                              //  bkl2d.shape = 1x32x16; ALU pipe: int; $651
        load_block2d.ugm.d16.a64.ca.ca (1|M0)  null:0 [r73:1]      {I@1,$4} // ex_desc:0x0; desc:0x2080203 // $651
        sync.allwr                           ($0,$10)                                                // $643
        dpas.8x8 (16|M0)         r92:f         r92:f             r33:bf            r25.0:bf         {Compacted,$8} // $643
        dpas.8x8 (16|M0)         r92:f         r92:f             r41:bf            r29.0:bf         {Compacted,$8} // $644
        sync.nop                             null                             {Compacted,$8.src}     // $669
        load_block2d.ugm.d16.a64 (1|M0)  r23:8  [r91:1]            {$7} // ex_desc:0x0; desc:0x2800203 // $669

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        load_block2d.ugm.d16v.a64 (1|M0)  r31:16 [r124:1]           {$13} // ex_desc:0x0; desc:0x3000283 // $673

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        sync.allwr                           ($1,$8)                                                 // $645
        dpas.8x8 (16|M0)         r92:f         r92:f             r57:bf            r49.0:bf         {Compacted,$12} // $645
        dpas.8x8 (16|M0)         r92:f         r92:f             r65:bf            r53.0:bf         {Compacted,$12} // $646
        sync.allwr                           ($12,$14)                                               // $677
        dpas.8x8 (16|M0)         r92:f         r92:f             r5:bf             r75.0:bf         {Compacted,$5} // $677
        dpas.8x8 (16|M0)         r92:f         r92:f             r13:bf            r79.0:bf         {Compacted,$5} // $678
        sync.allwr                           ($5,$13)                                                // $679
        dpas.8x8 (16|M0)         r92:f         r92:f             r31:bf            r23.0:bf         {Compacted,$7} // $679
        sync.nop                             null                             {Compacted,$7.dst}     // $680
        dpas.8x8 (16|M0)         r92:f         r92:f             r39:bf            r27.0:bf         {Compacted,$6} // $680

// Line 75:  for _ in range(0, K, BLOCK_SIZE_K):
(W&~f1.0) jmpi                               _0_011                                                  //  ALU pipe: int; $683
// B007: Preds:{B006},  Succs:{B006}
_0_012:
(W)     mov (1|M0)               r90.9<1>:d    r126.3<0;1,0>:d                                       //  ALU pipe: int; $685
(W)     jmpi                                 _0_010                                                  // $686
// B008: Preds:{B006},  Succs:{}
_0_011:

// Line 90:  c_desc.store([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], c)
        and (16|M0)   (eq)f0.0   null<2>:w     r1.0<1;1,0>:w     512:w                               //  ALU pipe: int; $689
(W)     and (1|M0)               r4.6<1>:d     r4.4<0;1,0>:d     63:w                                //  ALU pipe: int; $700
(W)     mov (1|M0)               r3.1<1>:d     r4.5<0;1,0>:d                                         //  ALU pipe: int; $693
(W)     and (1|M0)               r3.0<1>:d     r4.4<0;1,0>:d     -64:w               {Compacted}     //  ALU pipe: int; $694
(f0.0)  sel (16|M0)              r2.0<1>:d     r126.0<0;1,0>:d   4:w               {Compacted,$5.src} //  ALU pipe: int; $691
(W)     shr (1|M0)               r6.0<1>:ud    r4.6<0;1,0>:ud    2:w               {I@4}             //  ALU pipe: int; $701
(W)     mov (1|M0)               r7.3<1>:ud    3:w                                                   //  blk2d.heightM1; ALU pipe: int; $704
(W)     mov (1|M0)               r7.4<1>:ud    0xBFFF:uw                                             //  blk2d.pitchM1; ALU pipe: int; $704
(W)     mov (1|M0)               r7.7<1>:ud    0x70F:uw                                              //  bkl2d.shape = 1x16x8; ALU pipe: int; $704

// Line 38:  def matmul_kernel_with_tensor_descriptors(
(W)     mov (16|M0)              r127.0<1>:f   r89.0<1;1,0>:f                   {Compacted}          //  ALU pipe: float; $706

// Line 90:  c_desc.store([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], c)
(W)     add (1|M0)               r7.2<1>:ud    r4.6<0;1,0>:d     0xBFFF:uw                           //  ALU pipe: int; $703
(W)     mov (1|M0)               r7.0<1>:uq    r3.0<0;1,0>:q                    {I@7}                //  ALU pipe: int; $704
(W)     mov (1|M0)               r7.6<1>:ud    r2.0<0;1,0>:d                    {I@7}                //  blk2d.Y; ALU pipe: int; $704
(W)     bfn.(s0|s1|s2) (1|M0)    r7.5<1>:ud    r6.0<0;0>:ud      r126.2<0;0>:ud    r125.1<0>:ud     {I@7} //  ALU pipe: int; $702
        store_block2d.ugm.d32.a64 (1|M0)  [r7:1] r92:8             {I@1,$6} // ex_desc:0x0; desc:0x2000407 // $704

// Line 38:  def matmul_kernel_with_tensor_descriptors(
(W)     send.gtwy (1|M0)         null     r127  null:0  0x0            0x02000010           {EOT,F@1,$11} // wr:1+0, rd:0; end of thread // $706
L8720:
(W)     mov (16|M0)              null<1>:ud    0x2C494E6F:ud                                         // 
(W)     mov (16|M0)              null<1>:ud    0xF5D89B44:ud                                         // 
(W)     mov (16|M0)              null<1>:ud    0x0:ud                                                // 
(W)     mov (16|M0)              null<1>:ud    0x1:ud                                                // 


//.BankConflicts: 8
//.ByteRMWs: 2
//


//.numALUInst: 471
//.accSubDef: 0
//.accSubUse: 0
//.accSubCandidateDef: 0
//.accSubCandidateUse: 0
//
//
//.singlePipeAtOneDistNum: 41
//.allAtOneDistNum: 9
//.syncInstCount: 40
//.tokenReuseCount: 27
//.AfterWriteTokenDepCount: 69
//.AfterReadTokenDepCount: 106
