//.kernel matmul_kernel_with_tensor_descriptors
//.platform XE2
//.thread_config numGRF=128, numAcc=4, numSWSB=16
//.options_string "-emitCrossThreadOffR0Reloc -hashmovs 1615620884 1561956450 -hashmovs1 0 1 "
//.full_options "-emitLocation -enableCoalesceScalarMoves -supportLSCImmScale 0 -samplerHeaderWA -enablePreemptionR0Only -hasRNEandDenorm -noStitchExternFunc -useInlineData -emitCrossThreadOffR0Reloc -TotalGRFNum 128 -abortOnSpill 4 -enableBCR -enableBundleCR 3 -freqBasedSpillCost 8 -freqBasedSpillCostFunc 1 -boundsChecking -presched-rp 100 -nodpsendreorder -SBIDDepLoc -PVCSendWARWA -output -binary -dumpcommonisa -dumpcombinedcisa -dumpvisa -printHexFloatInAsm -noverifyCISA -enableHalfLSC -partialInt64 -activeThreadsOnlyBarrier -generateDebugInfo -hashmovs 1615620884 1561956450 -hashmovs1 0 1 "
//.instCount 623
//.RA type	LOCAL_FIRST_FIT_BC_RA
//.git-hash 8a9a27bbde4417e20b54507ca89a46693ac9f225

//.declare BuiltInR0 (0)  rf=r size=64 type=ud align=32 words (r0.0) IsBuiltin
//.declare  (1)  rf=r size=64 type=ud align=32 words (r53.0) IsBuiltin
//.declare BuiltinA0 (2)  rf=a size=4 type=ud align=1 words (a0.0) IsBuiltin
//.declare BuiltinA0Dot2 (3)  rf=a size=4 type=ud align=1 words (a0.2) IsBuiltin
//.declare BuiltinSR0Dot1 (5)  rf=r size=4 type=ud align=2 words IsBuiltin
//.declare %null (10)  rf=r size=4 type=ud align=32 words
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
//.declare %sp (25)  rf=r size=8 type=uq align=32 words (r127.3)
//.declare %fp (26)  rf=r size=8 type=uq align=32 words (r127.2)
//.declare %sr0 (27)  rf=r size=16 type=ud align=2 words
//.declare %cr0 (28)  rf=r size=12 type=ud align=2 words
//.declare %ce0 (29)  rf=r size=4 type=ud align=2 words
//.declare %dbg0 (30)  rf=r size=8 type=ud align=2 words
//.declare implBufPtr (32)  rf=r size=8 type=uq align=32 words (r126.0)
//.declare localIdBufPtr (33)  rf=r size=8 type=uq align=32 words (r126.3)
//.declare %msg0 (34)  rf=r size=12 type=ud align=2 words
//.declare %scratchloc (35)  rf=r size=8 type=uq align=4 words (s0.7)
//.declare V0033 (43)  rf=r size=64 type=d alias=+0 align=32 words (r53.0)
//.declare V0034 (44)  rf=r size=8 type=uq align=4 words (r4.0)
//.declare V0035 (45)  rf=r size=8 type=uq align=4 words (r4.1)
//.declare V0036 (46)  rf=r size=8 type=uq align=4 words (r4.2)
//.declare V0038 (48)  rf=r size=32 type=d alias=+0 align=32 words (r53.0)
//.declare V0040 (50)  rf=r size=32 type=w align=16 words (r1.0)
//.declare V0041 (51)  rf=r size=32 type=w align=16 words (r2.0)
//.declare V0042 (52)  rf=r size=32 type=w align=16 words (r3.0)
//.declare V0043 (53)  rf=r size=8 type=uq align=4 words (r5.1)
//.declare V0054 (64)  rf=r size=512 type=w align=32 words (r5.0)
//.declare V0055 (65)  rf=r size=1024 type=d align=32 words (r13.0)
//.declare V0056 (66)  rf=r size=512 type=w align=32 words (r29.0)
//.declare V0057 (67)  rf=r size=1024 type=d align=32 words (r37.0)
//.declare V0058 (68)  rf=r size=512 type=w align=32 words (r5.0)
//.declare V0059 (69)  rf=r size=1024 type=d align=32 words (r13.0)
//.declare V0060 (70)  rf=r size=512 type=w align=32 words (r29.0)
//.declare V0061 (71)  rf=r size=1024 type=d align=32 words (r37.0)
//.declare V0062 (72)  rf=r size=512 type=w align=32 words (r5.0)
//.declare V0063 (73)  rf=r size=1024 type=d align=32 words (r13.0)
//.declare V0064 (74)  rf=r size=512 type=w align=32 words (r29.0)
//.declare V0065 (75)  rf=r size=1024 type=d align=32 words (r37.0)
//.declare V0066 (76)  rf=r size=512 type=w align=32 words (r5.0)
//.declare V0067 (77)  rf=r size=1024 type=d align=32 words (r13.0)
//.declare V0068 (78)  rf=r size=512 type=w align=32 words (r29.0)
//.declare V0069 (79)  rf=r size=1024 type=d align=32 words (r37.0)
//.declare V0070 (80)  rf=r size=512 type=w align=32 words (r5.0)
//.declare V0071 (81)  rf=r size=1024 type=d align=32 words (r13.0)
//.declare V0072 (82)  rf=r size=512 type=w align=32 words (r29.0)
//.declare V0073 (83)  rf=r size=1024 type=d align=32 words (r37.0)
//.declare V0074 (84)  rf=r size=512 type=w align=32 words (r5.0)
//.declare V0075 (85)  rf=r size=1024 type=d align=32 words (r13.0)
//.declare V0076 (86)  rf=r size=512 type=w align=32 words (r29.0)
//.declare V0077 (87)  rf=r size=1024 type=d align=32 words (r37.0)
//.declare V0078 (88)  rf=r size=512 type=w align=32 words (r5.0)
//.declare V0079 (89)  rf=r size=1024 type=d align=32 words (r13.0)
//.declare V0080 (90)  rf=r size=512 type=w align=32 words (r29.0)
//.declare V0081 (91)  rf=r size=1024 type=d align=32 words (r37.0)
//.declare V0082 (92)  rf=r size=512 type=w align=32 words (r5.0)
//.declare V0083 (93)  rf=r size=1024 type=d align=32 words (r13.0)
//.declare V0084 (94)  rf=r size=512 type=w align=32 words (r29.0)
//.declare V0085 (95)  rf=r size=1024 type=d align=32 words (r37.0)
//.declare V0086 (96)  rf=r size=512 type=w align=32 words (r5.0)
//.declare V0087 (97)  rf=r size=1024 type=d align=32 words (r13.0)
//.declare V0088 (98)  rf=r size=512 type=w align=32 words (r29.0)
//.declare V0089 (99)  rf=r size=1024 type=d align=32 words (r37.0)
//.declare V0090 (100)  rf=r size=512 type=w align=32 words (r5.0)
//.declare V0091 (101)  rf=r size=1024 type=d align=32 words (r13.0)
//.declare V0092 (102)  rf=r size=512 type=w align=32 words (r29.0)
//.declare V0093 (103)  rf=r size=1024 type=d align=32 words (r37.0)
//.declare V0094 (104)  rf=r size=512 type=w align=32 words (r5.0)
//.declare V0095 (105)  rf=r size=1024 type=d align=32 words (r13.0)
//.declare V0096 (106)  rf=r size=512 type=w align=32 words (r29.0)
//.declare V0097 (107)  rf=r size=1024 type=d align=32 words (r37.0)
//.declare V0098 (108)  rf=r size=512 type=w align=32 words (r5.0)
//.declare V0099 (109)  rf=r size=1024 type=d align=32 words (r13.0)
//.declare V0100 (110)  rf=r size=512 type=w align=32 words (r29.0)
//.declare V0101 (111)  rf=r size=1024 type=d align=32 words (r37.0)
//.declare V0102 (112)  rf=r size=512 type=w align=32 words (r5.0)
//.declare V0103 (113)  rf=r size=1024 type=d align=32 words (r13.0)
//.declare V0104 (114)  rf=r size=512 type=w align=32 words (r29.0)
//.declare V0105 (115)  rf=r size=1024 type=d align=32 words (r37.0)
//.declare V0106 (116)  rf=r size=512 type=w align=32 words (r5.0)
//.declare V0107 (117)  rf=r size=1024 type=d align=32 words (r13.0)
//.declare V0108 (118)  rf=r size=512 type=w align=32 words (r29.0)
//.declare V0109 (119)  rf=r size=1024 type=d align=32 words (r37.0)
//.declare V0110 (120)  rf=r size=512 type=w align=32 words (r5.0)
//.declare V0111 (121)  rf=r size=1024 type=d align=32 words (r13.0)
//.declare V0112 (122)  rf=r size=512 type=w align=32 words (r29.0)
//.declare V0113 (123)  rf=r size=1024 type=d align=32 words (r37.0)
//.declare V0114 (124)  rf=r size=512 type=w align=32 words (r5.0)
//.declare V0115 (125)  rf=r size=1024 type=d align=32 words (r13.0)
//.declare V0116 (126)  rf=r size=512 type=w align=32 words (r29.0)
//.declare V0117 (127)  rf=r size=1024 type=d align=32 words (r37.0)
//.declare V0118 (128)  rf=r size=4 type=d align=2 words (r2.0)
//.declare V0119 (129)  rf=r size=4 type=d align=32 words (r3.0)
//.declare V0120 (130)  rf=r size=4 type=ud alias=V0118+0 align=2 words (r2.0)
//.declare V0121 (131)  rf=r size=4 type=ud alias=V0119+0 align=2 words (r3.0)
//.declare V0122 (132)  rf=r size=4 type=d align=2 words (r125.0)
//.declare V0123 (133)  rf=r size=4 type=ud alias=V0122+0 align=2 words (r125.0)
//.declare V0124 (134)  rf=r size=4 type=d alias=+0 align=2 words (r121.0)
//.declare V0125 (135)  rf=r size=4 type=d alias=+4 align=2 words (r121.1)
//.declare P01 (136)  rf=f16  size=2 type=uw align=1 words (f0.1)
//.declare V0126 (137)  rf=r size=4 type=d align=2 words (r54.0)
//.declare V0127 (138)  rf=r size=4 type=d alias=+0 align=2 words (r2.0)
//.declare V0128 (139)  rf=r size=4 type=d alias=+4 align=2 words (r2.1)
//.declare V0129 (140)  rf=r size=4 type=d align=2 words (r1.8)
//.declare V0130 (141)  rf=r size=4 type=d align=2 words (r2.2)
//.declare V0131 (142)  rf=r size=4 type=d align=2 words (r2.3)
//.declare V0132 (143)  rf=r size=4 type=d align=2 words (r3.0)
//.declare V0133 (144)  rf=r size=4 type=f align=2 words (r4.6)
//.declare V0134 (145)  rf=r size=4 type=ud alias=V0130+0 align=2 words (r2.2)
//.declare V0135 (146)  rf=r size=4 type=d align=2 words (r3.1)
//.declare V0136 (147)  rf=r size=4 type=ud alias=V0135+0 align=2 words (r3.1)
//.declare V0137 (148)  rf=r size=4 type=d alias=+0 align=2 words (r4.8)
//.declare V0138 (149)  rf=r size=4 type=f align=2 words (r4.7)
//.declare V0139 (150)  rf=r size=4 type=ud alias=V0132+0 align=2 words (r3.0)
//.declare V0140 (151)  rf=r size=4 type=f align=2 words (r4.10)
//.declare V0141 (152)  rf=r size=4 type=f align=2 words (r6.0)
//.declare V0142 (153)  rf=r size=4 type=f align=2 words (r6.1)
//.declare V0143 (154)  rf=r size=4 type=d align=2 words (r1.8)
//.declare V0144 (155)  rf=r size=4 type=ud alias=V0143+0 align=2 words (r1.8)
//.declare V0145 (156)  rf=r size=4 type=d alias=+4 align=2 words (r4.9)
//.declare V0146 (157)  rf=r size=4 type=d align=2 words (r2.3)
//.declare V0147 (158)  rf=r size=4 type=ud alias=V0146+0 align=2 words (r2.3)
//.declare V0148 (159)  rf=r size=4 type=f alias=+0 align=2 words (r2.4)
//.declare V0149 (160)  rf=r size=4 type=ud alias=V0137+0 align=2 words (r4.8)
//.declare V0150 (161)  rf=r size=4 type=f alias=+4 align=2 words (r2.5)
//.declare V0151 (162)  rf=r size=4 type=ud alias=V0145+0 align=2 words (r4.9)
//.declare V0152 (163)  rf=r size=4 type=f align=2 words (r7.0)
//.declare V0154 (165)  rf=r size=4 type=f align=2 words (r2.6)
//.declare V0156 (167)  rf=r size=4 type=f align=2 words (r3.1)
//.declare V0157 (168)  rf=r size=4 type=f align=2 words (r8.0)
//.declare V0158 (169)  rf=r size=4 type=f align=2 words (r9.0)
//.declare V0159 (170)  rf=r size=4 type=d align=2 words (r5.0)
//.declare V0160 (171)  rf=r size=4 type=ud alias=V0159+0 align=2 words (r5.0)
//.declare V0161 (172)  rf=r size=4 type=d align=2 words (r10.0)
//.declare V0162 (173)  rf=r size=4 type=d align=2 words (r1.8)
//.declare V0163 (174)  rf=r size=4 type=d align=32 words (r11.0)
//.declare V0164 (175)  rf=r size=4 type=d align=2 words (r1.9)
//.declare V0165 (176)  rf=r size=4 type=d align=2 words (r10.1)
//.declare V0166 (177)  rf=r size=4 type=ud alias=V0164+0 align=2 words (r1.9)
//.declare V0167 (178)  rf=r size=4 type=ud alias=V0165+0 align=2 words (r10.1)
//.declare  (179)  rf=f16  size=2 type=uw align=1 words (f1.1)
//.declare V0168 (180)  rf=r size=4 type=d align=2 words (r13.0)
//.declare V0169 (181)  rf=r size=4 type=d align=32 words (r2.0)
//.declare V0170 (182)  rf=r size=4 type=d align=2 words (r1.8)
//.declare V0171 (183)  rf=r size=4 type=d align=2 words (r126.0)
//.declare V0172 (184)  rf=r size=4 type=d align=2 words (r125.1)
//.declare V0173 (185)  rf=r size=32 type=b alias=V0038+0 align=32 words (r53.0)
//.declare V0174 (186)  rf=r size=1 type=b align=1 words (r3.0)
//.declare V0175 (187)  rf=r size=2 type=w align=1 words (r3.1)
//.declare V0176 (188)  rf=r size=2 type=w align=1 words (r4.12)
//.declare V0177 (189)  rf=r size=4 type=d align=2 words (r4.7)
//.declare V0178 (190)  rf=r size=1 type=ub alias=V0174+0 align=1 words (r3.0)
//.declare V0179 (191)  rf=r size=4 type=d align=2 words (r4.8)
//.declare V0180 (192)  rf=r size=4 type=d align=2 words (r6.0)
//.declare V0181 (193)  rf=r size=8 type=q alias=V0035+0 align=32 words (r4.1)
//.declare V0184 (196)  rf=r size=8 type=d alias=V0035+0 align=32 words (r4.2)
//.declare V0187 (199)  rf=r size=8 type=q align=4 words (r54.1)
//.declare V0188 (200)  rf=r size=8 type=d alias=V0187+0 align=4 words (r54.2)
//.declare V0190 (202)  rf=r size=4 type=d align=2 words (r5.0)
//.declare V0191 (203)  rf=r size=4 type=d align=2 words (r5.1)
//.declare V0192 (204)  rf=r size=4 type=ud alias=V0190+0 align=2 words (r5.0)
//.declare V0193 (205)  rf=r size=4 type=ud alias=V0191+0 align=2 words (r5.1)
//.declare V0194 (206)  rf=r size=4 type=d alias=+0 align=2 words (r54.4)
//.declare V0195 (207)  rf=r size=4 type=d align=2 words (r126.1)
//.declare V0196 (208)  rf=r size=2 type=b align=1 words (r1.32)
//.declare V0197 (209)  rf=r size=4 type=d alias=+4 align=2 words (r54.5)
//.declare V0198 (210)  rf=r size=2 type=ub alias=V0196+0 align=1 words (r1.32)
//.declare  (211)  rf=r size=64 type=ud align=32 words (r2.0)
//.declare  (212)  rf=r size=64 type=uq alias=+0 align=32 words (r2.0)
//.declare V0199 (213)  rf=r size=8 type=q alias=V0034+0 align=32 words (r4.0)
//.declare V0202 (216)  rf=r size=8 type=d alias=V0034+0 align=32 words (r4.0)
//.declare V0205 (219)  rf=r size=8 type=q align=4 words (r3.0)
//.declare V0206 (220)  rf=r size=8 type=d alias=V0205+0 align=4 words (r3.0)
//.declare V0208 (222)  rf=r size=4 type=d align=2 words (r3.2)
//.declare V0209 (223)  rf=r size=4 type=d align=2 words (r54.1)
//.declare V0210 (224)  rf=r size=4 type=ud alias=V0208+0 align=2 words (r3.2)
//.declare V0211 (225)  rf=r size=4 type=ud alias=V0209+0 align=2 words (r54.1)
//.declare V0212 (226)  rf=r size=4 type=d align=2 words (r1.8)
//.declare V0213 (227)  rf=r size=4 type=d align=2 words (r1.9)
//.declare V0214 (228)  rf=r size=4 type=d align=2 words (r126.2)
//.declare V0216 (230)  rf=r size=4 type=d alias=+0 align=2 words (r54.8)
//.declare V0217 (231)  rf=r size=32 type=d align=32 words (r55.0)
//.declare V0218 (232)  rf=r size=32 type=q alias=V0217+0 align=32 words (r55.0)
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
//.declare V0236 (250)  rf=r size=4 type=d align=2 words (r123.1)
//.declare V0237 (251)  rf=r size=4 type=d align=2 words (r123.2)
//.declare V0238 (252)  rf=r size=4 type=d align=2 words (r123.3)
//.declare V0239 (253)  rf=r size=4 type=d align=2 words (r123.4)
//.declare V0240 (254)  rf=r size=4 type=d align=2 words (r123.5)
//.declare V0241 (255)  rf=r size=4 type=d align=2 words (r123.6)
//.declare V0242 (256)  rf=r size=4 type=d align=2 words (r123.7)
//.declare V0243 (257)  rf=r size=4 type=d align=2 words (r123.8)
//.declare V0244 (258)  rf=r size=4 type=d align=2 words (r123.9)
//.declare V0245 (259)  rf=r size=4 type=d align=2 words (r123.10)
//.declare V0246 (260)  rf=r size=4 type=d align=2 words (r123.11)
//.declare V0247 (261)  rf=r size=4 type=d align=2 words (r123.12)
//.declare V0248 (262)  rf=r size=4 type=d align=2 words (r123.13)
//.declare V0249 (263)  rf=r size=4 type=d align=2 words (r123.14)
//.declare V0250 (264)  rf=r size=4 type=d align=2 words (r123.15)
//.declare V0251 (265)  rf=r size=4 type=d alias=+4 align=2 words (r54.9)
//.declare V0252 (266)  rf=r size=512 type=f align=32 words (r56.0)
//.declare  (269)  rf=r size=64 type=ud align=32 words (r2.0)
//.declare  (270)  rf=r size=64 type=uq alias=+0 align=32 words (r2.0)
//.declare V0258 (274)  rf=r size=512 type=ud alias=V0054+0 align=32 words (r5.0)
//.declare V0259 (275)  rf=r size=512 type=ud alias=V0056+0 align=32 words (r29.0)
//.declare  (278)  rf=r size=64 type=ud align=32 words (r2.0)
//.declare  (279)  rf=r size=64 type=uq alias=+0 align=32 words (r2.0)
//.declare V0265 (283)  rf=r size=512 type=ud alias=V0058+0 align=32 words (r5.0)
//.declare V0266 (284)  rf=r size=512 type=ud alias=V0060+0 align=32 words (r29.0)
//.declare  (287)  rf=r size=64 type=ud align=32 words (r2.0)
//.declare  (288)  rf=r size=64 type=uq alias=+0 align=32 words (r2.0)
//.declare V0272 (292)  rf=r size=512 type=ud alias=V0062+0 align=32 words (r5.0)
//.declare V0273 (293)  rf=r size=512 type=ud alias=V0064+0 align=32 words (r29.0)
//.declare  (296)  rf=r size=64 type=ud align=32 words (r2.0)
//.declare  (297)  rf=r size=64 type=uq alias=+0 align=32 words (r2.0)
//.declare V0279 (301)  rf=r size=512 type=ud alias=V0066+0 align=32 words (r5.0)
//.declare V0280 (302)  rf=r size=512 type=ud alias=V0068+0 align=32 words (r29.0)
//.declare  (305)  rf=r size=64 type=ud align=32 words (r2.0)
//.declare  (306)  rf=r size=64 type=uq alias=+0 align=32 words (r2.0)
//.declare V0286 (310)  rf=r size=512 type=ud alias=V0070+0 align=32 words (r5.0)
//.declare V0287 (311)  rf=r size=512 type=ud alias=V0072+0 align=32 words (r29.0)
//.declare  (314)  rf=r size=64 type=ud align=32 words (r2.0)
//.declare  (315)  rf=r size=64 type=uq alias=+0 align=32 words (r2.0)
//.declare V0293 (319)  rf=r size=512 type=ud alias=V0074+0 align=32 words (r5.0)
//.declare V0294 (320)  rf=r size=512 type=ud alias=V0076+0 align=32 words (r29.0)
//.declare  (323)  rf=r size=64 type=ud align=32 words (r2.0)
//.declare  (324)  rf=r size=64 type=uq alias=+0 align=32 words (r2.0)
//.declare V0300 (328)  rf=r size=512 type=ud alias=V0078+0 align=32 words (r5.0)
//.declare V0301 (329)  rf=r size=512 type=ud alias=V0080+0 align=32 words (r29.0)
//.declare  (332)  rf=r size=64 type=ud align=32 words (r2.0)
//.declare  (333)  rf=r size=64 type=uq alias=+0 align=32 words (r2.0)
//.declare V0307 (337)  rf=r size=512 type=ud alias=V0082+0 align=32 words (r5.0)
//.declare V0308 (338)  rf=r size=512 type=ud alias=V0084+0 align=32 words (r29.0)
//.declare  (341)  rf=r size=64 type=ud align=32 words (r2.0)
//.declare  (342)  rf=r size=64 type=uq alias=+0 align=32 words (r2.0)
//.declare V0314 (346)  rf=r size=512 type=ud alias=V0086+0 align=32 words (r5.0)
//.declare V0315 (347)  rf=r size=512 type=ud alias=V0088+0 align=32 words (r29.0)
//.declare  (350)  rf=r size=64 type=ud align=32 words (r2.0)
//.declare  (351)  rf=r size=64 type=uq alias=+0 align=32 words (r2.0)
//.declare V0321 (355)  rf=r size=512 type=ud alias=V0090+0 align=32 words (r5.0)
//.declare V0322 (356)  rf=r size=512 type=ud alias=V0092+0 align=32 words (r29.0)
//.declare  (359)  rf=r size=64 type=ud align=32 words (r2.0)
//.declare  (360)  rf=r size=64 type=uq alias=+0 align=32 words (r2.0)
//.declare V0328 (364)  rf=r size=512 type=ud alias=V0094+0 align=32 words (r5.0)
//.declare V0329 (365)  rf=r size=512 type=ud alias=V0096+0 align=32 words (r29.0)
//.declare  (368)  rf=r size=64 type=ud align=32 words (r2.0)
//.declare  (369)  rf=r size=64 type=uq alias=+0 align=32 words (r2.0)
//.declare V0335 (373)  rf=r size=512 type=ud alias=V0098+0 align=32 words (r5.0)
//.declare V0336 (374)  rf=r size=512 type=ud alias=V0100+0 align=32 words (r29.0)
//.declare  (377)  rf=r size=64 type=ud align=32 words (r2.0)
//.declare  (378)  rf=r size=64 type=uq alias=+0 align=32 words (r2.0)
//.declare V0342 (382)  rf=r size=512 type=ud alias=V0102+0 align=32 words (r5.0)
//.declare V0343 (383)  rf=r size=512 type=ud alias=V0104+0 align=32 words (r29.0)
//.declare  (386)  rf=r size=64 type=ud align=32 words (r2.0)
//.declare  (387)  rf=r size=64 type=uq alias=+0 align=32 words (r2.0)
//.declare V0349 (391)  rf=r size=512 type=ud alias=V0106+0 align=32 words (r5.0)
//.declare V0350 (392)  rf=r size=512 type=ud alias=V0108+0 align=32 words (r29.0)
//.declare V0351 (393)  rf=r size=4 type=d align=2 words (r1.8)
//.declare  (395)  rf=r size=64 type=ud align=32 words (r2.0)
//.declare  (396)  rf=r size=64 type=uq alias=+0 align=32 words (r2.0)
//.declare V0356 (400)  rf=r size=512 type=ud alias=V0110+0 align=32 words (r5.0)
//.declare V0357 (401)  rf=r size=512 type=ud alias=V0112+0 align=32 words (r29.0)
//.declare V0358 (402)  rf=r size=4 type=d align=2 words (r126.3)
//.declare  (404)  rf=r size=64 type=ud align=32 words (r2.0)
//.declare  (405)  rf=r size=64 type=uq alias=+0 align=32 words (r2.0)
//.declare V0363 (409)  rf=r size=512 type=ud alias=V0114+0 align=32 words (r5.0)
//.declare V0364 (410)  rf=r size=512 type=ud alias=V0116+0 align=32 words (r29.0)
//.declare P02 (411)  rf=f16  size=2 type=uw align=1 words (f1.0)
//.declare V0365 (412)  rf=r size=4 type=ud alias=V0351+0 align=2 words (r1.8)
//.declare P03 (413)  rf=f16  size=2 type=uw align=1 words (f0.0)
//.declare V0367 (415)  rf=r size=64 type=d align=32 words (r2.0)
//.declare V0368 (416)  rf=r size=512 type=d alias=V0252+0 align=32 words (r56.0)
//.declare V0369 (417)  rf=r size=8 type=q alias=V0036+0 align=32 words (r4.2)
//.declare V0372 (420)  rf=r size=8 type=d alias=V0036+0 align=32 words (r4.4)
//.declare V0375 (423)  rf=r size=8 type=q align=4 words (r1.4)
//.declare V0376 (424)  rf=r size=8 type=d alias=V0375+0 align=4 words (r1.8)
//.declare V0378 (426)  rf=r size=4 type=d align=2 words (r1.10)
//.declare V0379 (427)  rf=r size=4 type=d align=2 words (r4.6)
//.declare V0380 (428)  rf=r size=4 type=ud alias=V0378+0 align=2 words (r1.10)
//.declare V0381 (429)  rf=r size=4 type=ud alias=V0379+0 align=2 words (r4.6)
//.declare  (432)  rf=r size=64 type=ud align=32 words (r3.0)
//.declare  (433)  rf=r size=64 type=uq alias=+0 align=32 words (r3.0)
//.declare V0384 (434)  rf=r size=8 type=uq align=4 words (r4.3)
//.declare V0385 (435)  rf=r size=8 type=uq align=4 words (r5.0)
//.declare  (436)  rf=r size=64 type=ud align=32 words (r127.0)
//.declare  (437)  rf=r size=4 type=d alias=V0181+0 align=32 words (r4.2)
//.declare  (438)  rf=r size=4 type=d alias=V0199+0 align=32 words (r4.0)
//.declare  (439)  rf=r size=4 type=d alias=V0369+0 align=32 words (r4.4)
//.declare  (440)  rf=r size=8 type=d align=8 words (r2.0)
//.declare  (441)  rf=r size=8 type=d align=8 words (r121.0)
//.declare  (442)  rf=r size=8 type=f align=8 words (r2.4)
//.declare  (443)  rf=r size=8 type=ud align=8 words (r4.8)
//.declare  (444)  rf=r size=8 type=d align=8 words (r54.4)
//.declare  (445)  rf=r size=8 type=d align=8 words (r54.8)
//.declare  (446)  rf=r size=4 type=f align=2 words (r5.0)
//.declare r0 (447)  rf=r size=64 type=ud align=32 words (r0.0)
//.declare rtmp (448)  rf=r size=64 type=ud align=32 words (r127.0)
//.declare inlineRegFromTDL (449)  rf=r size=32 type=ud align=2 words (r1.0)
//.declare inlineRegExpectedLocation (450)  rf=r size=32 type=ud align=2 words (r4.0)
//.declare  (451)  rf=r size=128 type=ud align=32 words (r1.0)
//.declare  (452)  rf=r size=64 type=ud align=32 words (r3.0)
//.declare  (453)  rf=r size=32 type=ud align=2 words (r5.0)

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
// | V0384    | :uq      |    0x8 | r4+0x18  | inline+0x18      |
// | V0385    | :uq      |    0x8 | r5       | cti+0x20         |
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
(W)     mov (16|M0)              r53.0<1>:ud   r0.0<1;1,0>:ud                   {Compacted}          //  ALU pipe: int; $1
(W)     or (1|M0)                cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x400004C0:ud              {A@1}    // $1
(W)     mov (1|M0)               r2.0<1>:d     r53.1<0;1,0>:d                   {Compacted,A@1,$0.dst} //  ALU pipe: int; $2

// Line 54:  group_id = pid // num_pid_in_group
(W)     mul (1|M0)               acc0.0<1>:ud  r2.0<0;1,0>:ud    0xAAAB:uw              {I@1}        //  ALU pipe: int; $5
(W)     mach (1|M0)              r3.0<1>:ud    r2.0<0;1,0>:ud    0xAAAAAAAB:ud              {$1.dst} //  ALU pipe: int; 
(W)     shr (1|M0)               r125.0<1>:ud  r3.0<0;1,0>:ud    4:w               {I@1}             //  ALU pipe: int; $6

// Line 58:  pid_n = (pid % num_pid_in_group) // group_size_m
(W)     cmp (16|M0)   (eq)f0.1   null<1>:d     r125.0<0;1,0>:d   1:w               {I@1}             //  ALU pipe: int; $12

// Line 56:  group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
(W)     add (1|M0)               r121.0<1>:d   -r125.0<0;1,0>:d  1:w               {Compacted}       //  ALU pipe: int; $8

// Line 57:  pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
(W)     mad (1|M0)               r121.1<1>:d   r2.0<0;0>:d       r125.0<0;0>:d     -24:w               //  ALU pipe: int; $10

// Line 58:  pid_n = (pid % num_pid_in_group) // group_size_m
(W&~f0.1) jmpi                               _0_007                                                  //  ALU pipe: int; $13
// B003: Preds:{B002},  Succs:{B005}
_0_008:
(W)     mov (1|M0)               r54.0<1>:d    -1:w                               {Compacted}        //  ALU pipe: int; $15
(W)     jmpi                                 _0_009                                                  // $16
// B004: Preds:{B002},  Succs:{B005}
_0_007:
(W)     asr (2|M0)               r2.0<1>:d     r121.0<1;1,0>:d   31:w               {Compacted,I@4}  //  ALU pipe: int; $18
(W)     add3 (1|M0)              r1.8<1>:d     r2.0<0;0>:d       -r125.0<0;0>:d    1:w               {I@1} //  ALU pipe: int; $20
(W)     add (1|M0)               r2.3<1>:d     r2.1<0;1,0>:d     r121.1<0;1,0>:d                     //  ALU pipe: int; $22
(W)     xor (1|M0)               r2.2<1>:d     r1.8<0;1,0>:d     r2.0<0;1,0>:d    {I@2}              //  ALU pipe: int; $21
(W)     xor (1|M0)               r3.0<1>:d     r2.3<0;1,0>:d     r2.1<0;1,0>:d    {Compacted,I@2}    //  ALU pipe: int; $23
(W)     xor (1|M0)               cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x30:uw              {A@1}          // $24
(W)     mov (1|M0)               r4.6<1>:f     r2.2<0;1,0>:ud                   {A@1}                //  ALU pipe: float; $25
(W)     mov (1|M0)               r5.0<1>:f     0xB4C00000:f                               {Compacted,$2.dst} //  ALU pipe: float; $30
(W)     math.inv (1|M0)          r4.10<1>:f    r4.6<0;1,0>:f                    {F@2}                //  ALU pipe: math; $29
(W)     mov (1|M0)               r4.7<1>:f     r3.0<0;1,0>:ud                   {I@2}                //  ALU pipe: float; $28
(W)     mad (1|M0)               r6.0<1>:f     r4.10<0;0>:f      r5.0<0;0>:f       r4.10<0>:f       {A@1} //  ALU pipe: float; $30
(W)     mov (1|M0)               r3.1<1>:ud    r4.6<0;1,0>:f                                         //  ALU pipe: int; $26
(W)     mov (1|M0)               r1.8<1>:ud    r4.7<0;1,0>:f                    {F@2}                //  ALU pipe: int; $32
(W)     mul (1|M0)               r6.1<1>:f     r4.7<0;1,0>:f     r6.0<0;1,0>:f    {F@1}              //  ALU pipe: float; $31
(W)     add (1|M0)               r4.8<1>:d     r2.2<0;1,0>:d     -r3.1<0;1,0>:d   {I@2}              //  ALU pipe: int; $27
(W)     add (1|M0)               r4.9<1>:d     r3.0<0;1,0>:d     -r1.8<0;1,0>:d   {I@2}              //  ALU pipe: int; $33
(W)     mov (1|M0)               r2.3<1>:ud    r6.1<0;1,0>:f                    {F@1}                //  ALU pipe: int; $34
(W)     mov (1|M0)               r2.4<1>:f     r4.8<0;1,0>:ud                   {I@3}                //  ALU pipe: float; $35
(W)     mov (1|M0)               r2.5<1>:f     r4.9<0;1,0>:ud                   {I@2}                //  ALU pipe: float; $35
(W)     mov (1|M0)               r7.0<1>:f     r2.3<0;1,0>:ud                   {I@1}                //  ALU pipe: float; $37
(W)     mad (1|M0)               r2.6<1>:f     r4.7<0;0>:f       r7.0<0;0>:f       -r4.6<0>:f       {F@1} //  ALU pipe: float; $39
(W)     mad (1|M0)               r3.1<1>:f     r2.5<0;0>:f       r7.0<0;0>:f       -r2.4<0>:f        //  ALU pipe: float; $41
(W)     add (1|M0)               r8.0<1>:f     r2.6<0;1,0>:f     r3.1<0;1,0>:f    {Compacted,F@1}    //  ALU pipe: float; $42
(W)     mul (1|M0)               r9.0<1>:f     r6.0<0;1,0>:f     r8.0<0;1,0>:f    {Compacted,F@1}    //  ALU pipe: float; $43
(W)     xor (1|M0)               cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x30:uw              {A@1}          // $44
(W)     mov (1|M0)               r5.0<1>:ud    r9.0<0;1,0>:f                    {A@1}                //  ALU pipe: int; $45
(W)     xor (1|M0)               r1.8<1>:d     r2.0<0;1,0>:d     r2.1<0;1,0>:d                       //  ALU pipe: int; $47
(W)     add (1|M0)               r10.0<1>:d    r5.0<0;1,0>:d     r2.3<0;1,0>:d    {Compacted,I@2}    //  ALU pipe: int; $46
(W)     mul (1|M0)               acc0.0<1>:d   r10.0<0;1,0>:d    r2.4<0;1,0>:uw   {I@1}              //  ALU pipe: int; $48
(W)     macl (1|M0)              r11.0<1>:d    r10.0<0;1,0>:d    r2.2<0;1,0>:d    {Compacted}        //  ALU pipe: int; $49
(W)     add (1|M0)               r1.9<1>:d     r3.0<0;1,0>:d     -r11.0<0;1,0>:d  {I@1}              //  ALU pipe: int; $49
(W)     cmp (1|M0)    (ge)f1.1   r10.1<1>:ud   r1.9<0;1,0>:ud    r2.2<0;1,0>:ud   {I@1}              //  ALU pipe: int; $50
(W)     add3 (1|M0)              r13.0<1>:d    r10.0<0;0>:d      r1.8<0;0>:d       -r10.1<0>:d      {I@1} //  ALU pipe: int; $51
(W)     bfn.(s0^s1^s2) (1|M0)    r54.0<1>:ud   r13.0<0;0>:ud     r2.0<0;0>:ud      r2.1<0>:ud       {I@1} //  ALU pipe: int; $52
// B005: Preds:{B004, B003},  Succs:{B006}
_0_009:

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r3.0<2>:b     r53.8<0;1,0>:b                                        //  ALU pipe: int; $61

// Line 57:  pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
(W)     mul (1|M0)               acc0.0<1>:d   r54.0<0;1,0>:d    r121.0<0;1,0>:uw {I@2}              //  ALU pipe: int; $55
(W)     macl (1|M0)              r2.0<1>:d     r54.0<0;1,0>:d    r121.0<0;1,0>:d  {Compacted}        //  ALU pipe: int; $56

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r3.1<1>:w     r3.0<0;1,0>:b                    {I@3}                //  ALU pipe: int; $62
(W)     mov (1|M0)               r4.7<1>:d     r3.0<0;1,0>:ub                                        //  ALU pipe: int; $64
(W)     and (1|M0)               r5.0<1>:d     r4.2<0;1,0>:d     63:w               {Compacted,$2.dst} //  ALU pipe: int; $75

// Line 57:  pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
(W)     add3 (1|M0)              r1.8<1>:d     r125.0<0;0>:d     r121.1<0;0>:d     -r2.0<0>:d       {I@4} //  ALU pipe: int; $56

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     and (1|M0)               r4.12<1>:w    r3.1<0;1,0>:w     48:w               {I@4}            //  ALU pipe: int; $63
(W)     shl (1|M0)               r4.8<1>:d     r4.7<0;1,0>:d     5:w               {I@4}             //  ALU pipe: int; $65
(W)     shl (1|M0)               r125.1<1>:d   r54.0<0;1,0>:d    9:w               {Compacted}       //  ALU pipe: int; $60
(W)     shr (1|M0)               r5.1<1>:ud    r5.0<0;1,0>:ud    1:w               {I@5}             //  ALU pipe: int; $76

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     shl (1|M0)               r126.0<1>:d   r1.8<0;1,0>:d     3:w               {Compacted,I@5}   //  ALU pipe: int; $58

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     and (1|M0)               r6.0<1>:d     r4.8<0;1,0>:d     480:w               {Compacted,I@4} //  ALU pipe: int; $66
(W)     mov (1|M0)               r1.32<2>:b    r4.12<0;1,0>:w                                        //  ALU pipe: int; $79
(W)     mov (1|M0)               r54.3<1>:f    r4.3<0;1,0>:f                                         //  ALU pipe: float; $68
(W)     and (1|M0)               r54.2<1>:d    r4.2<0;1,0>:d     -64:w                               //  ALU pipe: int; $69
(W)     shl (1|M0)               r1.9<1>:d     r4.7<0;1,0>:d     4:w                                 //  ALU pipe: int; $93
(W)     add (1|M0)               r126.1<1>:d   r5.0<0;1,0>:d     24575:w                             //  ALU pipe: int; $78
(W)     bfn.(s0|s1|s2) (1|M0)    r54.4<1>:ud   r5.1<0;0>:ud      r6.0<0;0>:ud      r125.1<0>:ud     {I@5} //  ALU pipe: int; $77
(W)     mov (1|M0)               r54.5<1>:d    r1.32<0;1,0>:ub                  {I@5}                //  ALU pipe: int; $80
(W)     mov (1|M0)               r2.3<1>:ud    4095:w                                                //  blk2d.heightM1; ALU pipe: int; $81
(W)     mov (1|M0)               r2.4<1>:ud    24575:w                                               //  blk2d.pitchM1; ALU pipe: int; $81
(W)     mov (1|M0)               r2.7<1>:ud    0xF1F:uw                                              //  bkl2d.shape = 1x32x16; ALU pipe: int; $81
(W)     mov (1|M0)               r3.1<1>:f     r4.1<0;1,0>:f                                         //  ALU pipe: float; $83
(W)     and (1|M0)               r3.2<1>:d     r4.0<0;1,0>:d     63:w               {Compacted}      //  ALU pipe: int; $90
(W)     and (1|M0)               r3.0<1>:d     r4.0<0;1,0>:d     -64:w               {Compacted}     //  ALU pipe: int; $84
(W)     mov (1|M0)               r2.0<1>:uq    r54.1<0;1,0>:q                   {A@2}                //  ALU pipe: int; $81
(W)     and (1|M0)               r126.2<1>:d   r1.9<0;1,0>:d     496:w               {I@7}           //  ALU pipe: int; $94
(W)     mov (1|M0)               r2.2<1>:ud    r126.1<0;1,0>:d                  {I@7}                //  blk2d.widthM1; ALU pipe: int; $81
(W)     or (1|M0)                r1.8<1>:d     r5.1<0;1,0>:d     r125.1<0;1,0>:d                     //  ALU pipe: int; $92
(W)     mov (2|M0)               r2.5<1>:ud    r54.4<1;1,0>:d                   {I@7}                //  blk2d.X; ALU pipe: int; $81

// Line 75:  for _ in range(0, K, BLOCK_SIZE_K):
        mov (16|M0)              r56.0<1>:f    0x0:f                               {Compacted}       //  ALU pipe: float; $143
        mov (16|M0)              r57.0<1>:f    0x0:f                               {Compacted}       //  ALU pipe: float; $144
        mov (16|M0)              r58.0<1>:f    0x0:f                               {Compacted}       //  ALU pipe: float; $145
        mov (16|M0)              r59.0<1>:f    0x0:f                               {Compacted}       //  ALU pipe: float; $146
        mov (16|M0)              r60.0<1>:f    0x0:f                               {Compacted}       //  ALU pipe: float; $147
        mov (16|M0)              r61.0<1>:f    0x0:f                               {Compacted}       //  ALU pipe: float; $148
        mov (16|M0)              r62.0<1>:f    0x0:f                               {Compacted}       //  ALU pipe: float; $149
        mov (16|M0)              r63.0<1>:f    0x0:f                               {Compacted}       //  ALU pipe: float; $150

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        load_block2d.ugm.d16.a64.ca.ca (1|M0)  null:0 [r2:1]       {I@1,$3} // ex_desc:0x0; desc:0x2080203 // $81
(W)     mov (1|M0)               r55.3<1>:d    3:w                                                   //  ALU pipe: int; $99
(W)     mov (1|M0)               r55.4<1>:d    8191:w                                                //  ALU pipe: int; $100
(W)     mov (1|M0)               r55.5<1>:d    0:w                                                   //  ALU pipe: int; $101
(W)     mov (1|M0)               r55.6<1>:d    0:w                                                   //  ALU pipe: int; $102
(W)     mov (1|M0)               r55.7<1>:f    0x1070F:f                                             //  (0x0001070f:f); ALU pipe: float; $103
(W)     mov (1|M0)               r124.3<1>:d   4095:w                                                //  ALU pipe: int; $106
(W)     mov (1|M0)               r124.4<1>:d   24575:w                                               //  ALU pipe: int; $107
(W)     mov (1|M0)               r124.5<1>:d   0:w                                                   //  ALU pipe: int; $108
(W)     mov (1|M0)               r124.6<1>:d   0:w                                                   //  ALU pipe: int; $109
(W)     mov (1|M0)               r124.7<1>:d   7951:w                                                //  ALU pipe: int; $110

// Line 75:  for _ in range(0, K, BLOCK_SIZE_K):
(W)     mov (1|M0)               r125.2<1>:d   992:w                               {Compacted}       //  ALU pipe: int; $112
(W)     mov (1|M0)               r125.3<1>:d   928:w                                                 //  ALU pipe: int; $113
(W)     mov (1|M0)               r125.4<1>:d   896:w                               {Compacted}       //  ALU pipe: int; $114
(W)     mov (1|M0)               r125.5<1>:d   864:w                                                 //  ALU pipe: int; $115
(W)     mov (1|M0)               r125.6<1>:d   832:w                                                 //  ALU pipe: int; $116
(W)     mov (1|M0)               r125.7<1>:d   800:w                                                 //  ALU pipe: int; $117
(W)     mov (1|M0)               r125.8<1>:d   768:w                                                 //  ALU pipe: int; $118
(W)     mov (1|M0)               r125.9<1>:d   736:w                                                 //  ALU pipe: int; $119
(W)     mov (1|M0)               r125.10<1>:d  704:w                                                 //  ALU pipe: int; $120
(W)     mov (1|M0)               r125.11<1>:d  672:w                                                 //  ALU pipe: int; $121
(W)     mov (1|M0)               r125.12<1>:d  640:w                                                 //  ALU pipe: int; $122
(W)     mov (1|M0)               r125.13<1>:d  608:w                                                 //  ALU pipe: int; $123
(W)     mov (1|M0)               r125.14<1>:d  576:w                                                 //  ALU pipe: int; $124
(W)     mov (1|M0)               r125.15<1>:d  544:w                                                 //  ALU pipe: int; $125
(W)     mov (1|M0)               r123.0<1>:d   512:w                               {Compacted}       //  ALU pipe: int; $126
(W)     mov (1|M0)               r123.1<1>:d   480:w                               {Compacted}       //  ALU pipe: int; $127
(W)     mov (1|M0)               r123.2<1>:d   448:w                               {Compacted}       //  ALU pipe: int; $128
(W)     mov (1|M0)               r123.3<1>:d   416:w                                                 //  ALU pipe: int; $129
(W)     mov (1|M0)               r123.4<1>:d   384:w                               {Compacted}       //  ALU pipe: int; $130
(W)     mov (1|M0)               r123.5<1>:d   352:w                                                 //  ALU pipe: int; $131
(W)     mov (1|M0)               r123.6<1>:d   320:w                                                 //  ALU pipe: int; $132
(W)     mov (1|M0)               r123.7<1>:d   288:w                                                 //  ALU pipe: int; $133
(W)     mov (1|M0)               r123.8<1>:d   256:w                                                 //  ALU pipe: int; $134
(W)     mov (1|M0)               r123.9<1>:d   224:w                                                 //  ALU pipe: int; $135
(W)     mov (1|M0)               r123.10<1>:d  192:w                                                 //  ALU pipe: int; $136
(W)     mov (1|M0)               r123.11<1>:d  160:w                                                 //  ALU pipe: int; $137
(W)     mov (1|M0)               r123.12<1>:d  128:w                                                 //  ALU pipe: int; $138
(W)     mov (1|M0)               r123.13<1>:d  96:w                                                  //  ALU pipe: int; $139
(W)     mov (1|M0)               r123.14<1>:d  64:w                                                  //  ALU pipe: int; $140
(W)     mov (1|M0)               r123.15<1>:d  32:w                                                  //  ALU pipe: int; $141
(W)     mov (1|M0)               r54.9<1>:d    0:w                                                   //  ALU pipe: int; $142

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r124.0<1>:q   r54.1<0;1,0>:q                                        //  ALU pipe: int; $104
(W)     mov (1|M0)               r124.2<1>:d   r126.1<0;1,0>:d                                       //  ALU pipe: int; $105
(W)     shr (1|M0)               r54.1<1>:ud   r3.2<0;1,0>:ud    1:w                                 //  ALU pipe: int; $91
(W)     add (1|M0)               r55.2<1>:d    r3.2<0;1,0>:d     8191:w                              //  ALU pipe: int; $95
(W)     mov (1|M0)               r55.0<1>:q    r3.0<0;1,0>:q                    {F@7}                //  ALU pipe: int; $97
(W)     add (1|M0)               r54.8<1>:d    r1.8<0;1,0>:d     r126.2<0;1,0>:d                     //  ALU pipe: int; $96
// B006: Preds:{B007, B005},  Succs:{B007, B008}
_0_010:

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     or (1|M0)                r55.5<1>:d    r54.9<0;1,0>:d    r54.1<0;1,0>:d   {I@4}              //  ALU pipe: int; $158
(W)     mov (1|M0)               r55.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $162
(W)     mov (2|M0)               r124.5<1>:d   r54.8<1;1,0>:d                   {I@3}                //  ALU pipe: int; $164
        load_block2d.ugm.d16.a64 (1|M0)  r5:8   [r55:1]            {A@1,$6} // ex_desc:0x0; desc:0x2800203 // $163
        load_block2d.ugm.d16v.a64 (1|M0)  r13:16 [r124:1]           {I@1,$7} // ex_desc:0x0; desc:0x3000283 // $166
(W)     bfn.(s0|s1|s2) (1|M0)    r55.5<1>:ud   r54.9<0;0>:ud     r123.15<0;0>:ud   r54.1<0>:ud      {$6.src} //  ALU pipe: int; $160

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r55.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $171

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     or (1|M0)                r124.6<1>:d   r54.9<0;1,0>:d    32:w               {$7.src}         //  ALU pipe: int; $159
(W)     mov (1|M0)               r124.5<1>:d   r54.8<0;1,0>:d                                        //  ALU pipe: int; $174

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        sync.nop                             null                             {Compacted,$5.src}     // $172
        load_block2d.ugm.d16.a64 (1|M0)  r29:8  [r55:1]            {I@3,$8} // ex_desc:0x0; desc:0x2800203 // $172

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        load_block2d.ugm.d16v.a64 (1|M0)  r37:16 [r124:1]           {I@1,$9} // ex_desc:0x0; desc:0x3000283 // $176
(W)     bfn.(s0|s1|s2) (1|M0)    r55.5<1>:ud   r54.9<0;0>:ud     r123.14<0;0>:ud   r54.1<0>:ud      {$8.src} //  ALU pipe: int; $190
(W)     mov (1|M0)               r55.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $194

// Line 85:  off_k += BLOCK_SIZE_K
(W)     or (1|M0)                r124.6<1>:d   r54.9<0;1,0>:d    64:w               {$9.src}         //  ALU pipe: int; $153

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     mov (1|M0)               r124.5<1>:d   r54.8<0;1,0>:d                                        //  ALU pipe: int; $196

// Line 85:  off_k += BLOCK_SIZE_K
(W)     or (1|M0)                r1.8<1>:d     r54.9<0;1,0>:d    960:w                               //  ALU pipe: int; $601

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        sync.nop                             null                             {Compacted,$4.src}     // $155
(W)     bfn.(s0|s1|s2) (1|M0)    r2.6<1>:ud    r54.5<0;0>:ud     r54.9<0;0>:ud     r123.14<0>:ud    {$3.src} //  ALU pipe: int; $155
(W)     mov (1|M0)               r2.0<1>:uq    r54.1<0;1,0>:q                                        //  ALU pipe: int; $156
(W)     mov (1|M0)               r2.2<1>:ud    r126.1<0;1,0>:d                                       //  blk2d.widthM1; ALU pipe: int; $156
(W)     mov (1|M0)               r2.3<1>:ud    4095:w                                                //  blk2d.heightM1; ALU pipe: int; $156
(W)     mov (1|M0)               r2.4<1>:ud    24575:w                                               //  blk2d.pitchM1; ALU pipe: int; $156
(W)     mov (1|M0)               r2.5<1>:ud    r54.4<0;1,0>:d                                        //  blk2d.X; ALU pipe: int; $156
(W)     mov (1|M0)               r2.7<1>:ud    0xF1F:uw                                              //  bkl2d.shape = 1x32x16; ALU pipe: int; $156

// Line 85:  off_k += BLOCK_SIZE_K
(W)     add (1|M0)               r126.3<1>:d   r54.9<0;1,0>:d    1024:w                              //  ALU pipe: int; $633

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        load_block2d.ugm.d16.a64.ca.ca (1|M0)  null:0 [r2:1]       {I@2,$10} // ex_desc:0x0; desc:0x2080203 // $156
(W)     bfn.(s0|s1|s2) (1|M0)    r2.6<1>:ud    r54.5<0;0>:ud     r54.9<0;0>:ud     r123.12<0>:ud    {$10.src} //  ALU pipe: int; $187
(W)     mov (1|M0)               r2.0<1>:uq    r54.1<0;1,0>:q                                        //  ALU pipe: int; $188
(W)     mov (1|M0)               r2.2<1>:ud    r126.1<0;1,0>:d                                       //  blk2d.widthM1; ALU pipe: int; $188
(W)     mov (1|M0)               r2.3<1>:ud    4095:w                                                //  blk2d.heightM1; ALU pipe: int; $188
(W)     mov (1|M0)               r2.4<1>:ud    24575:w                                               //  blk2d.pitchM1; ALU pipe: int; $188
(W)     mov (1|M0)               r2.5<1>:ud    r54.4<0;1,0>:d                                        //  blk2d.X; ALU pipe: int; $188
(W)     mov (1|M0)               r2.7<1>:ud    0xF1F:uw                                              //  bkl2d.shape = 1x32x16; ALU pipe: int; $188

// Line 75:  for _ in range(0, K, BLOCK_SIZE_K):
(W)     cmp (16|M0)   (lt)f1.0   null<1>:d     r1.8<0;1,0>:ud    0xFC0:uw                            //  ALU pipe: int; $665

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        load_block2d.ugm.d16.a64.ca.ca (1|M0)  null:0 [r2:1]       {I@2,$11} // ex_desc:0x0; desc:0x2080203 // $188
(W)     bfn.(s0|s1|s2) (1|M0)    r2.6<1>:ud    r54.5<0;0>:ud     r54.9<0;0>:ud     r123.10<0>:ud    {$11.src} //  ALU pipe: int; $219
(W)     mov (1|M0)               r2.0<1>:uq    r54.1<0;1,0>:q                                        //  ALU pipe: int; $220
(W)     mov (1|M0)               r2.2<1>:ud    r126.1<0;1,0>:d                                       //  blk2d.widthM1; ALU pipe: int; $220
(W)     mov (1|M0)               r2.3<1>:ud    4095:w                                                //  blk2d.heightM1; ALU pipe: int; $220
(W)     mov (1|M0)               r2.4<1>:ud    24575:w                                               //  blk2d.pitchM1; ALU pipe: int; $220
(W)     mov (1|M0)               r2.5<1>:ud    r54.4<0;1,0>:d                                        //  blk2d.X; ALU pipe: int; $220
(W)     mov (1|M0)               r2.7<1>:ud    0xF1F:uw                                              //  bkl2d.shape = 1x32x16; ALU pipe: int; $220
        load_block2d.ugm.d16.a64.ca.ca (1|M0)  null:0 [r2:1]       {I@1,$12} // ex_desc:0x0; desc:0x2080203 // $220
(W)     bfn.(s0|s1|s2) (1|M0)    r2.6<1>:ud    r54.5<0;0>:ud     r54.9<0;0>:ud     r123.8<0>:ud     {$12.src} //  ALU pipe: int; $251
(W)     mov (1|M0)               r2.0<1>:uq    r54.1<0;1,0>:q                                        //  ALU pipe: int; $252
(W)     mov (1|M0)               r2.2<1>:ud    r126.1<0;1,0>:d                                       //  blk2d.widthM1; ALU pipe: int; $252
(W)     mov (1|M0)               r2.3<1>:ud    4095:w                                                //  blk2d.heightM1; ALU pipe: int; $252
(W)     mov (1|M0)               r2.4<1>:ud    24575:w                                               //  blk2d.pitchM1; ALU pipe: int; $252
(W)     mov (1|M0)               r2.5<1>:ud    r54.4<0;1,0>:d                                        //  blk2d.X; ALU pipe: int; $252
(W)     mov (1|M0)               r2.7<1>:ud    0xF1F:uw                                              //  bkl2d.shape = 1x32x16; ALU pipe: int; $252
        load_block2d.ugm.d16.a64.ca.ca (1|M0)  null:0 [r2:1]       {I@1,$13} // ex_desc:0x0; desc:0x2080203 // $252
(W)     bfn.(s0|s1|s2) (1|M0)    r2.6<1>:ud    r54.5<0;0>:ud     r54.9<0;0>:ud     r123.6<0>:ud     {$13.src} //  ALU pipe: int; $283
        sync.allwr                           ($5,$7)                                                 // $180
        dpas.8x8 (16|M0)         r56:f         r56:f             r13:bf            r5.0:bf          {Compacted,$6} // $180
(W)     mov (1|M0)               r2.0<1>:uq    r54.1<0;1,0>:q                                        //  ALU pipe: int; $284
(W)     mov (1|M0)               r2.2<1>:ud    r126.1<0;1,0>:d                                       //  blk2d.widthM1; ALU pipe: int; $284
(W)     mov (1|M0)               r2.3<1>:ud    4095:w                                                //  blk2d.heightM1; ALU pipe: int; $284
(W)     mov (1|M0)               r2.4<1>:ud    24575:w                                               //  blk2d.pitchM1; ALU pipe: int; $284
(W)     mov (1|M0)               r2.5<1>:ud    r54.4<0;1,0>:d                                        //  blk2d.X; ALU pipe: int; $284
(W)     mov (1|M0)               r2.7<1>:ud    0xF1F:uw                                              //  bkl2d.shape = 1x32x16; ALU pipe: int; $284
        dpas.8x8 (16|M0)         r56:f         r56:f             r21:bf            r9.0:bf          {Compacted,$6} // $181

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        sync.nop                             null                             {Compacted,$6.src}     // $195
        load_block2d.ugm.d16.a64 (1|M0)  r5:8   [r55:1]            {$14} // ex_desc:0x0; desc:0x2800203 // $195
        load_block2d.ugm.d16v.a64 (1|M0)  r13:16 [r124:1]           {$15} // ex_desc:0x0; desc:0x3000283 // $198
(W)     bfn.(s0|s1|s2) (1|M0)    r55.5<1>:ud   r54.9<0;0>:ud     r123.13<0;0>:ud   r54.1<0>:ud      {$14.src} //  ALU pipe: int; $192

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r55.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $203

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     or (1|M0)                r124.6<1>:d   r54.9<0;1,0>:d    96:w               {$15.src}        //  ALU pipe: int; $191
(W)     mov (1|M0)               r124.5<1>:d   r54.8<0;1,0>:d                                        //  ALU pipe: int; $206

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        sync.allwr                           ($6,$9)                                                 // $182
        dpas.8x8 (16|M0)         r56:f         r56:f             r37:bf            r29.0:bf         {Compacted,$8} // $182
        load_block2d.ugm.d16.a64.ca.ca (1|M0)  null:0 [r2:1]       {I@5,$0} // ex_desc:0x0; desc:0x2080203 // $284
(W)     bfn.(s0|s1|s2) (1|M0)    r2.6<1>:ud    r54.5<0;0>:ud     r54.9<0;0>:ud     r123.4<0>:ud     {$0.src} //  ALU pipe: int; $315
(W)     mov (1|M0)               r2.0<1>:uq    r54.1<0;1,0>:q                                        //  ALU pipe: int; $316
(W)     mov (1|M0)               r2.2<1>:ud    r126.1<0;1,0>:d                                       //  blk2d.widthM1; ALU pipe: int; $316
(W)     mov (1|M0)               r2.3<1>:ud    4095:w                                                //  blk2d.heightM1; ALU pipe: int; $316
(W)     mov (1|M0)               r2.4<1>:ud    24575:w                                               //  blk2d.pitchM1; ALU pipe: int; $316
        dpas.8x8 (16|M0)         r56:f         r56:f             r45:bf            r33.0:bf         {Compacted,$8} // $183
        sync.nop                             null                             {Compacted,$8.src}     // $204
        load_block2d.ugm.d16.a64 (1|M0)  r29:8  [r55:1]            {I@7,$1} // ex_desc:0x0; desc:0x2800203 // $204

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        load_block2d.ugm.d16v.a64 (1|M0)  r37:16 [r124:1]           {I@6,$2} // ex_desc:0x0; desc:0x3000283 // $208
(W)     bfn.(s0|s1|s2) (1|M0)    r55.5<1>:ud   r54.9<0;0>:ud     r123.12<0;0>:ud   r54.1<0>:ud      {$1.src} //  ALU pipe: int; $222
(W)     mov (1|M0)               r55.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $226

// Line 85:  off_k += BLOCK_SIZE_K
(W)     or (1|M0)                r124.6<1>:d   r54.9<0;1,0>:d    128:w               {$2.src}        //  ALU pipe: int; $185

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     mov (1|M0)               r124.5<1>:d   r54.8<0;1,0>:d                                        //  ALU pipe: int; $228

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r2.5<1>:ud    r54.4<0;1,0>:d                                        //  blk2d.X; ALU pipe: int; $316
(W)     mov (1|M0)               r2.7<1>:ud    0xF1F:uw                                              //  bkl2d.shape = 1x32x16; ALU pipe: int; $316
        load_block2d.ugm.d16.a64.ca.ca (1|M0)  null:0 [r2:1]       {I@1,$3} // ex_desc:0x0; desc:0x2080203 // $316
(W)     bfn.(s0|s1|s2) (1|M0)    r2.6<1>:ud    r54.5<0;0>:ud     r54.9<0;0>:ud     r123.2<0>:ud     {$3.src} //  ALU pipe: int; $347
(W)     mov (1|M0)               r2.0<1>:uq    r54.1<0;1,0>:q                                        //  ALU pipe: int; $348
(W)     mov (1|M0)               r2.2<1>:ud    r126.1<0;1,0>:d                                       //  blk2d.widthM1; ALU pipe: int; $348
(W)     mov (1|M0)               r2.3<1>:ud    4095:w                                                //  blk2d.heightM1; ALU pipe: int; $348
(W)     mov (1|M0)               r2.4<1>:ud    24575:w                                               //  blk2d.pitchM1; ALU pipe: int; $348
(W)     mov (1|M0)               r2.5<1>:ud    r54.4<0;1,0>:d                                        //  blk2d.X; ALU pipe: int; $348
(W)     mov (1|M0)               r2.7<1>:ud    0xF1F:uw                                              //  bkl2d.shape = 1x32x16; ALU pipe: int; $348
        load_block2d.ugm.d16.a64.ca.ca (1|M0)  null:0 [r2:1]       {I@1,$10} // ex_desc:0x0; desc:0x2080203 // $348
(W)     bfn.(s0|s1|s2) (1|M0)    r2.6<1>:ud    r54.5<0;0>:ud     r54.9<0;0>:ud     r123.0<0>:ud     {$10.src} //  ALU pipe: int; $379
(W)     mov (1|M0)               r2.0<1>:uq    r54.1<0;1,0>:q                                        //  ALU pipe: int; $380
(W)     mov (1|M0)               r2.2<1>:ud    r126.1<0;1,0>:d                                       //  blk2d.widthM1; ALU pipe: int; $380
(W)     mov (1|M0)               r2.3<1>:ud    4095:w                                                //  blk2d.heightM1; ALU pipe: int; $380
(W)     mov (1|M0)               r2.4<1>:ud    24575:w                                               //  blk2d.pitchM1; ALU pipe: int; $380
(W)     mov (1|M0)               r2.5<1>:ud    r54.4<0;1,0>:d                                        //  blk2d.X; ALU pipe: int; $380
(W)     mov (1|M0)               r2.7<1>:ud    0xF1F:uw                                              //  bkl2d.shape = 1x32x16; ALU pipe: int; $380
        load_block2d.ugm.d16.a64.ca.ca (1|M0)  null:0 [r2:1]       {I@1,$11} // ex_desc:0x0; desc:0x2080203 // $380
(W)     bfn.(s0|s1|s2) (1|M0)    r2.6<1>:ud    r54.5<0;0>:ud     r54.9<0;0>:ud     r125.14<0>:ud    {$11.src} //  ALU pipe: int; $411
(W)     mov (1|M0)               r2.0<1>:uq    r54.1<0;1,0>:q                                        //  ALU pipe: int; $412
(W)     mov (1|M0)               r2.2<1>:ud    r126.1<0;1,0>:d                                       //  blk2d.widthM1; ALU pipe: int; $412
(W)     mov (1|M0)               r2.3<1>:ud    4095:w                                                //  blk2d.heightM1; ALU pipe: int; $412
(W)     mov (1|M0)               r2.4<1>:ud    24575:w                                               //  blk2d.pitchM1; ALU pipe: int; $412
(W)     mov (1|M0)               r2.5<1>:ud    r54.4<0;1,0>:d                                        //  blk2d.X; ALU pipe: int; $412
(W)     mov (1|M0)               r2.7<1>:ud    0xF1F:uw                                              //  bkl2d.shape = 1x32x16; ALU pipe: int; $412
        sync.allwr                           ($8,$15)                                                // $212
        dpas.8x8 (16|M0)         r56:f         r56:f             r13:bf            r5.0:bf          {Compacted,$14} // $212
        load_block2d.ugm.d16.a64.ca.ca (1|M0)  null:0 [r2:1]       {I@1,$12} // ex_desc:0x0; desc:0x2080203 // $412
(W)     bfn.(s0|s1|s2) (1|M0)    r2.6<1>:ud    r54.5<0;0>:ud     r54.9<0;0>:ud     r125.12<0>:ud    {$12.src} //  ALU pipe: int; $443
(W)     mov (1|M0)               r2.0<1>:uq    r54.1<0;1,0>:q                                        //  ALU pipe: int; $444
(W)     mov (1|M0)               r2.2<1>:ud    r126.1<0;1,0>:d                                       //  blk2d.widthM1; ALU pipe: int; $444
(W)     mov (1|M0)               r2.3<1>:ud    4095:w                                                //  blk2d.heightM1; ALU pipe: int; $444
(W)     mov (1|M0)               r2.4<1>:ud    24575:w                                               //  blk2d.pitchM1; ALU pipe: int; $444
        dpas.8x8 (16|M0)         r56:f         r56:f             r21:bf            r9.0:bf          {Compacted,$14} // $213

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        sync.nop                             null                             {Compacted,$14.src}    // $227
        load_block2d.ugm.d16.a64 (1|M0)  r5:8   [r55:1]            {$13} // ex_desc:0x0; desc:0x2800203 // $227
        load_block2d.ugm.d16v.a64 (1|M0)  r13:16 [r124:1]           {$6} // ex_desc:0x0; desc:0x3000283 // $230
(W)     bfn.(s0|s1|s2) (1|M0)    r55.5<1>:ud   r54.9<0;0>:ud     r123.11<0;0>:ud   r54.1<0>:ud      {$13.src} //  ALU pipe: int; $224

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r55.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $235

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     or (1|M0)                r124.6<1>:d   r54.9<0;1,0>:d    160:w               {$6.src}        //  ALU pipe: int; $223
(W)     mov (1|M0)               r124.5<1>:d   r54.8<0;1,0>:d                                        //  ALU pipe: int; $238

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        sync.allwr                           ($2,$14)                                                // $214
        dpas.8x8 (16|M0)         r56:f         r56:f             r37:bf            r29.0:bf         {Compacted,$1} // $214
(W)     mov (1|M0)               r2.5<1>:ud    r54.4<0;1,0>:d                                        //  blk2d.X; ALU pipe: int; $444
(W)     mov (1|M0)               r2.7<1>:ud    0xF1F:uw                                              //  bkl2d.shape = 1x32x16; ALU pipe: int; $444
        load_block2d.ugm.d16.a64.ca.ca (1|M0)  null:0 [r2:1]       {I@1,$7} // ex_desc:0x0; desc:0x2080203 // $444
(W)     bfn.(s0|s1|s2) (1|M0)    r2.6<1>:ud    r54.5<0;0>:ud     r54.9<0;0>:ud     r125.10<0>:ud    {$7.src} //  ALU pipe: int; $475
(W)     mov (1|M0)               r2.0<1>:uq    r54.1<0;1,0>:q                                        //  ALU pipe: int; $476
        dpas.8x8 (16|M0)         r56:f         r56:f             r45:bf            r33.0:bf         {Compacted,$1} // $215
        sync.nop                             null                             {Compacted,$1.src}     // $236
        load_block2d.ugm.d16.a64 (1|M0)  r29:8  [r55:1]            {$9} // ex_desc:0x0; desc:0x2800203 // $236

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        load_block2d.ugm.d16v.a64 (1|M0)  r37:16 [r124:1]           {$0} // ex_desc:0x0; desc:0x3000283 // $240
(W)     bfn.(s0|s1|s2) (1|M0)    r55.5<1>:ud   r54.9<0;0>:ud     r123.10<0;0>:ud   r54.1<0>:ud      {$9.src} //  ALU pipe: int; $254
(W)     mov (1|M0)               r55.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $258

// Line 85:  off_k += BLOCK_SIZE_K
(W)     or (1|M0)                r124.6<1>:d   r54.9<0;1,0>:d    192:w               {$0.src}        //  ALU pipe: int; $217

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     mov (1|M0)               r124.5<1>:d   r54.8<0;1,0>:d                                        //  ALU pipe: int; $260

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r2.2<1>:ud    r126.1<0;1,0>:d                                       //  blk2d.widthM1; ALU pipe: int; $476
(W)     mov (1|M0)               r2.3<1>:ud    4095:w                                                //  blk2d.heightM1; ALU pipe: int; $476
(W)     mov (1|M0)               r2.4<1>:ud    24575:w                                               //  blk2d.pitchM1; ALU pipe: int; $476
(W)     mov (1|M0)               r2.5<1>:ud    r54.4<0;1,0>:d                                        //  blk2d.X; ALU pipe: int; $476
(W)     mov (1|M0)               r2.7<1>:ud    0xF1F:uw                                              //  bkl2d.shape = 1x32x16; ALU pipe: int; $476
        load_block2d.ugm.d16.a64.ca.ca (1|M0)  null:0 [r2:1]       {I@1,$3} // ex_desc:0x0; desc:0x2080203 // $476
(W)     bfn.(s0|s1|s2) (1|M0)    r2.6<1>:ud    r54.5<0;0>:ud     r54.9<0;0>:ud     r125.8<0>:ud     {$3.src} //  ALU pipe: int; $507
(W)     mov (1|M0)               r2.0<1>:uq    r54.1<0;1,0>:q                                        //  ALU pipe: int; $508
(W)     mov (1|M0)               r2.2<1>:ud    r126.1<0;1,0>:d                                       //  blk2d.widthM1; ALU pipe: int; $508
(W)     mov (1|M0)               r2.3<1>:ud    4095:w                                                //  blk2d.heightM1; ALU pipe: int; $508
(W)     mov (1|M0)               r2.4<1>:ud    24575:w                                               //  blk2d.pitchM1; ALU pipe: int; $508
(W)     mov (1|M0)               r2.5<1>:ud    r54.4<0;1,0>:d                                        //  blk2d.X; ALU pipe: int; $508
(W)     mov (1|M0)               r2.7<1>:ud    0xF1F:uw                                              //  bkl2d.shape = 1x32x16; ALU pipe: int; $508
        load_block2d.ugm.d16.a64.ca.ca (1|M0)  null:0 [r2:1]       {I@1,$10} // ex_desc:0x0; desc:0x2080203 // $508
(W)     bfn.(s0|s1|s2) (1|M0)    r2.6<1>:ud    r54.5<0;0>:ud     r54.9<0;0>:ud     r125.6<0>:ud     {$10.src} //  ALU pipe: int; $539
(W)     mov (1|M0)               r2.0<1>:uq    r54.1<0;1,0>:q                                        //  ALU pipe: int; $540
(W)     mov (1|M0)               r2.2<1>:ud    r126.1<0;1,0>:d                                       //  blk2d.widthM1; ALU pipe: int; $540
(W)     mov (1|M0)               r2.3<1>:ud    4095:w                                                //  blk2d.heightM1; ALU pipe: int; $540
(W)     mov (1|M0)               r2.4<1>:ud    24575:w                                               //  blk2d.pitchM1; ALU pipe: int; $540
(W)     mov (1|M0)               r2.5<1>:ud    r54.4<0;1,0>:d                                        //  blk2d.X; ALU pipe: int; $540
(W)     mov (1|M0)               r2.7<1>:ud    0xF1F:uw                                              //  bkl2d.shape = 1x32x16; ALU pipe: int; $540
        load_block2d.ugm.d16.a64.ca.ca (1|M0)  null:0 [r2:1]       {I@1,$11} // ex_desc:0x0; desc:0x2080203 // $540
(W)     bfn.(s0|s1|s2) (1|M0)    r2.6<1>:ud    r54.5<0;0>:ud     r54.9<0;0>:ud     r125.4<0>:ud     {$11.src} //  ALU pipe: int; $571
(W)     mov (1|M0)               r2.0<1>:uq    r54.1<0;1,0>:q                                        //  ALU pipe: int; $572
(W)     mov (1|M0)               r2.2<1>:ud    r126.1<0;1,0>:d                                       //  blk2d.widthM1; ALU pipe: int; $572
(W)     mov (1|M0)               r2.3<1>:ud    4095:w                                                //  blk2d.heightM1; ALU pipe: int; $572
        sync.allwr                           ($1,$6)                                                 // $244
        dpas.8x8 (16|M0)         r56:f         r56:f             r13:bf            r5.0:bf          {Compacted,$13} // $244
(W)     mov (1|M0)               r2.4<1>:ud    24575:w                                               //  blk2d.pitchM1; ALU pipe: int; $572
(W)     mov (1|M0)               r2.5<1>:ud    r54.4<0;1,0>:d                                        //  blk2d.X; ALU pipe: int; $572
(W)     mov (1|M0)               r2.7<1>:ud    0xF1F:uw                                              //  bkl2d.shape = 1x32x16; ALU pipe: int; $572
        load_block2d.ugm.d16.a64.ca.ca (1|M0)  null:0 [r2:1]       {I@1,$15} // ex_desc:0x0; desc:0x2080203 // $572
(W)     or (1|M0)                r2.6<1>:ud    r54.5<0;1,0>:d    r1.8<0;1,0>:d    {$15.src}          //  ALU pipe: int; $603
        dpas.8x8 (16|M0)         r56:f         r56:f             r21:bf            r9.0:bf          {Compacted,$13} // $245

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        sync.nop                             null                             {Compacted,$13.src}    // $259
        load_block2d.ugm.d16.a64 (1|M0)  r5:8   [r55:1]            {$2} // ex_desc:0x0; desc:0x2800203 // $259
        load_block2d.ugm.d16v.a64 (1|M0)  r13:16 [r124:1]           {$8} // ex_desc:0x0; desc:0x3000283 // $262
(W)     bfn.(s0|s1|s2) (1|M0)    r55.5<1>:ud   r54.9<0;0>:ud     r123.9<0;0>:ud    r54.1<0>:ud      {$2.src} //  ALU pipe: int; $256

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r55.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $267

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     or (1|M0)                r124.6<1>:d   r54.9<0;1,0>:d    224:w               {$8.src}        //  ALU pipe: int; $255
(W)     mov (1|M0)               r124.5<1>:d   r54.8<0;1,0>:d                                        //  ALU pipe: int; $270

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        sync.allwr                           ($0,$13)                                                // $246
        dpas.8x8 (16|M0)         r56:f         r56:f             r37:bf            r29.0:bf         {Compacted,$9} // $246
(W)     mov (1|M0)               r2.0<1>:uq    r54.1<0;1,0>:q                                        //  ALU pipe: int; $604
(W)     mov (1|M0)               r2.2<1>:ud    r126.1<0;1,0>:d                                       //  blk2d.widthM1; ALU pipe: int; $604
(W)     mov (1|M0)               r2.3<1>:ud    4095:w                                                //  blk2d.heightM1; ALU pipe: int; $604
(W)     mov (1|M0)               r2.4<1>:ud    24575:w                                               //  blk2d.pitchM1; ALU pipe: int; $604
(W)     mov (1|M0)               r2.5<1>:ud    r54.4<0;1,0>:d                                        //  blk2d.X; ALU pipe: int; $604
(W)     mov (1|M0)               r2.7<1>:ud    0xF1F:uw                                              //  bkl2d.shape = 1x32x16; ALU pipe: int; $604
        dpas.8x8 (16|M0)         r56:f         r56:f             r45:bf            r33.0:bf         {Compacted,$9} // $247
        sync.nop                             null                             {Compacted,$9.src}     // $268
        load_block2d.ugm.d16.a64 (1|M0)  r29:8  [r55:1]            {I@7,$12} // ex_desc:0x0; desc:0x2800203 // $268

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        load_block2d.ugm.d16v.a64 (1|M0)  r37:16 [r124:1]           {I@7,$14} // ex_desc:0x0; desc:0x3000283 // $272
(W)     bfn.(s0|s1|s2) (1|M0)    r55.5<1>:ud   r54.9<0;0>:ud     r123.8<0;0>:ud    r54.1<0>:ud      {$12.src} //  ALU pipe: int; $286
(W)     mov (1|M0)               r55.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $290

// Line 85:  off_k += BLOCK_SIZE_K
(W)     or (1|M0)                r124.6<1>:d   r54.9<0;1,0>:d    256:w               {$14.src}       //  ALU pipe: int; $249

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     mov (1|M0)               r124.5<1>:d   r54.8<0;1,0>:d                                        //  ALU pipe: int; $292

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        load_block2d.ugm.d16.a64.ca.ca (1|M0)  null:0 [r2:1]       {I@5,$7} // ex_desc:0x0; desc:0x2080203 // $604
(W)     or (1|M0)                r2.6<1>:ud    r54.5<0;1,0>:d    r126.3<0;1,0>:d  {$7.src}           //  ALU pipe: int; $635
(W)     mov (1|M0)               r2.0<1>:uq    r54.1<0;1,0>:q                                        //  ALU pipe: int; $636
(W)     mov (1|M0)               r2.2<1>:ud    r126.1<0;1,0>:d                                       //  blk2d.widthM1; ALU pipe: int; $636
(W)     mov (1|M0)               r2.3<1>:ud    4095:w                                                //  blk2d.heightM1; ALU pipe: int; $636
(W)     mov (1|M0)               r2.4<1>:ud    24575:w                                               //  blk2d.pitchM1; ALU pipe: int; $636
(W)     mov (1|M0)               r2.5<1>:ud    r54.4<0;1,0>:d                                        //  blk2d.X; ALU pipe: int; $636
(W)     mov (1|M0)               r2.7<1>:ud    0xF1F:uw                                              //  bkl2d.shape = 1x32x16; ALU pipe: int; $636
        load_block2d.ugm.d16.a64.ca.ca (1|M0)  null:0 [r2:1]       {I@1,$4} // ex_desc:0x0; desc:0x2080203 // $636
        sync.allwr                           ($8,$9)                                                 // $276
        dpas.8x8 (16|M0)         r56:f         r56:f             r13:bf            r5.0:bf          {Compacted,$2} // $276
        dpas.8x8 (16|M0)         r56:f         r56:f             r21:bf            r9.0:bf          {Compacted,$2} // $277

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        sync.nop                             null                             {Compacted,$2.src}     // $291
        load_block2d.ugm.d16.a64 (1|M0)  r5:8   [r55:1]            {$3} // ex_desc:0x0; desc:0x2800203 // $291
        load_block2d.ugm.d16v.a64 (1|M0)  r13:16 [r124:1]           {$10} // ex_desc:0x0; desc:0x3000283 // $294
(W)     bfn.(s0|s1|s2) (1|M0)    r55.5<1>:ud   r54.9<0;0>:ud     r123.7<0;0>:ud    r54.1<0>:ud      {$3.src} //  ALU pipe: int; $288

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r55.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $299

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     or (1|M0)                r124.6<1>:d   r54.9<0;1,0>:d    288:w               {$10.src}       //  ALU pipe: int; $287
(W)     mov (1|M0)               r124.5<1>:d   r54.8<0;1,0>:d                                        //  ALU pipe: int; $302

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        sync.allwr                           ($2,$14)                                                // $278
        dpas.8x8 (16|M0)         r56:f         r56:f             r37:bf            r29.0:bf         {Compacted,$12} // $278
        dpas.8x8 (16|M0)         r56:f         r56:f             r45:bf            r33.0:bf         {Compacted,$12} // $279
        sync.nop                             null                             {Compacted,$12.src}    // $300
        load_block2d.ugm.d16.a64 (1|M0)  r29:8  [r55:1]            {I@3,$11} // ex_desc:0x0; desc:0x2800203 // $300

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        load_block2d.ugm.d16v.a64 (1|M0)  r37:16 [r124:1]           {I@1,$1} // ex_desc:0x0; desc:0x3000283 // $304
(W)     bfn.(s0|s1|s2) (1|M0)    r55.5<1>:ud   r54.9<0;0>:ud     r123.6<0;0>:ud    r54.1<0>:ud      {$11.src} //  ALU pipe: int; $318
(W)     mov (1|M0)               r55.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $322

// Line 85:  off_k += BLOCK_SIZE_K
(W)     or (1|M0)                r124.6<1>:d   r54.9<0;1,0>:d    320:w               {$1.src}        //  ALU pipe: int; $281

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     mov (1|M0)               r124.5<1>:d   r54.8<0;1,0>:d                                        //  ALU pipe: int; $324

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        sync.allwr                           ($10,$12)                                               // $308
        dpas.8x8 (16|M0)         r56:f         r56:f             r13:bf            r5.0:bf          {Compacted,$3} // $308
        dpas.8x8 (16|M0)         r56:f         r56:f             r21:bf            r9.0:bf          {Compacted,$3} // $309

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        sync.nop                             null                             {Compacted,$3.src}     // $323
        load_block2d.ugm.d16.a64 (1|M0)  r5:8   [r55:1]            {I@3,$6} // ex_desc:0x0; desc:0x2800203 // $323
        load_block2d.ugm.d16v.a64 (1|M0)  r13:16 [r124:1]           {I@1,$13} // ex_desc:0x0; desc:0x3000283 // $326
(W)     bfn.(s0|s1|s2) (1|M0)    r55.5<1>:ud   r54.9<0;0>:ud     r123.5<0;0>:ud    r54.1<0>:ud      {$6.src} //  ALU pipe: int; $320

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r55.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $331

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     or (1|M0)                r124.6<1>:d   r54.9<0;1,0>:d    352:w               {$13.src}       //  ALU pipe: int; $319
(W)     mov (1|M0)               r124.5<1>:d   r54.8<0;1,0>:d                                        //  ALU pipe: int; $334

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        sync.allwr                           ($1,$3)                                                 // $310
        dpas.8x8 (16|M0)         r56:f         r56:f             r37:bf            r29.0:bf         {Compacted,$11} // $310
        dpas.8x8 (16|M0)         r56:f         r56:f             r45:bf            r33.0:bf         {Compacted,$11} // $311
        sync.nop                             null                             {Compacted,$11.src}    // $332
        load_block2d.ugm.d16.a64 (1|M0)  r29:8  [r55:1]            {I@3,$15} // ex_desc:0x0; desc:0x2800203 // $332

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        load_block2d.ugm.d16v.a64 (1|M0)  r37:16 [r124:1]           {I@1,$0} // ex_desc:0x0; desc:0x3000283 // $336
(W)     bfn.(s0|s1|s2) (1|M0)    r55.5<1>:ud   r54.9<0;0>:ud     r123.4<0;0>:ud    r54.1<0>:ud      {$15.src} //  ALU pipe: int; $350
(W)     mov (1|M0)               r55.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $354

// Line 85:  off_k += BLOCK_SIZE_K
(W)     or (1|M0)                r124.6<1>:d   r54.9<0;1,0>:d    384:w               {$0.src}        //  ALU pipe: int; $313

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     mov (1|M0)               r124.5<1>:d   r54.8<0;1,0>:d                                        //  ALU pipe: int; $356

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        sync.allwr                           ($11,$13)                                               // $340
        dpas.8x8 (16|M0)         r56:f         r56:f             r13:bf            r5.0:bf          {Compacted,$6} // $340
        dpas.8x8 (16|M0)         r56:f         r56:f             r21:bf            r9.0:bf          {Compacted,$6} // $341

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        sync.nop                             null                             {Compacted,$6.src}     // $355
        load_block2d.ugm.d16.a64 (1|M0)  r5:8   [r55:1]            {I@3,$5} // ex_desc:0x0; desc:0x2800203 // $355
        load_block2d.ugm.d16v.a64 (1|M0)  r13:16 [r124:1]           {I@1,$8} // ex_desc:0x0; desc:0x3000283 // $358
(W)     bfn.(s0|s1|s2) (1|M0)    r55.5<1>:ud   r54.9<0;0>:ud     r123.3<0;0>:ud    r54.1<0>:ud      {$5.src} //  ALU pipe: int; $352

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r55.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $363

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     or (1|M0)                r124.6<1>:d   r54.9<0;1,0>:d    416:w               {$8.src}        //  ALU pipe: int; $351
(W)     mov (1|M0)               r124.5<1>:d   r54.8<0;1,0>:d                                        //  ALU pipe: int; $366

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        sync.allwr                           ($0,$6)                                                 // $342
        dpas.8x8 (16|M0)         r56:f         r56:f             r37:bf            r29.0:bf         {Compacted,$15} // $342
        dpas.8x8 (16|M0)         r56:f         r56:f             r45:bf            r33.0:bf         {Compacted,$15} // $343
        sync.nop                             null                             {Compacted,$15.src}    // $364
        load_block2d.ugm.d16.a64 (1|M0)  r29:8  [r55:1]            {I@3,$7} // ex_desc:0x0; desc:0x2800203 // $364

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        load_block2d.ugm.d16v.a64 (1|M0)  r37:16 [r124:1]           {I@1,$9} // ex_desc:0x0; desc:0x3000283 // $368
(W)     bfn.(s0|s1|s2) (1|M0)    r55.5<1>:ud   r54.9<0;0>:ud     r123.2<0;0>:ud    r54.1<0>:ud      {$7.src} //  ALU pipe: int; $382
(W)     mov (1|M0)               r55.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $386

// Line 85:  off_k += BLOCK_SIZE_K
(W)     or (1|M0)                r124.6<1>:d   r54.9<0;1,0>:d    448:w               {$9.src}        //  ALU pipe: int; $345

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     mov (1|M0)               r124.5<1>:d   r54.8<0;1,0>:d                                        //  ALU pipe: int; $388

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        sync.allwr                           ($8,$15)                                                // $372
        dpas.8x8 (16|M0)         r56:f         r56:f             r13:bf            r5.0:bf          {Compacted,$5} // $372
        dpas.8x8 (16|M0)         r56:f         r56:f             r21:bf            r9.0:bf          {Compacted,$5} // $373

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        sync.nop                             null                             {Compacted,$5.src}     // $387
        load_block2d.ugm.d16.a64 (1|M0)  r5:8   [r55:1]            {I@3,$10} // ex_desc:0x0; desc:0x2800203 // $387
        load_block2d.ugm.d16v.a64 (1|M0)  r13:16 [r124:1]           {I@1,$11} // ex_desc:0x0; desc:0x3000283 // $390
(W)     bfn.(s0|s1|s2) (1|M0)    r55.5<1>:ud   r54.9<0;0>:ud     r123.1<0;0>:ud    r54.1<0>:ud      {$10.src} //  ALU pipe: int; $384

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r55.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $395

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     or (1|M0)                r124.6<1>:d   r54.9<0;1,0>:d    480:w               {$11.src}       //  ALU pipe: int; $383
(W)     mov (1|M0)               r124.5<1>:d   r54.8<0;1,0>:d                                        //  ALU pipe: int; $398

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        sync.allwr                           ($5,$9)                                                 // $374
        dpas.8x8 (16|M0)         r56:f         r56:f             r37:bf            r29.0:bf         {Compacted,$7} // $374
        dpas.8x8 (16|M0)         r56:f         r56:f             r45:bf            r33.0:bf         {Compacted,$7} // $375
        sync.nop                             null                             {Compacted,$7.src}     // $396
        load_block2d.ugm.d16.a64 (1|M0)  r29:8  [r55:1]            {I@3,$12} // ex_desc:0x0; desc:0x2800203 // $396

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        load_block2d.ugm.d16v.a64 (1|M0)  r37:16 [r124:1]           {I@1,$13} // ex_desc:0x0; desc:0x3000283 // $400
(W)     bfn.(s0|s1|s2) (1|M0)    r55.5<1>:ud   r54.9<0;0>:ud     r123.0<0;0>:ud    r54.1<0>:ud      {$12.src} //  ALU pipe: int; $414
(W)     mov (1|M0)               r55.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $418

// Line 85:  off_k += BLOCK_SIZE_K
(W)     or (1|M0)                r124.6<1>:d   r54.9<0;1,0>:d    512:w               {$13.src}       //  ALU pipe: int; $377

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     mov (1|M0)               r124.5<1>:d   r54.8<0;1,0>:d                                        //  ALU pipe: int; $420

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        sync.allwr                           ($7,$11)                                                // $404
        dpas.8x8 (16|M0)         r56:f         r56:f             r13:bf            r5.0:bf          {Compacted,$10} // $404
        dpas.8x8 (16|M0)         r56:f         r56:f             r21:bf            r9.0:bf          {Compacted,$10} // $405

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        sync.nop                             null                             {Compacted,$10.src}    // $419
        load_block2d.ugm.d16.a64 (1|M0)  r5:8   [r55:1]            {I@3,$14} // ex_desc:0x0; desc:0x2800203 // $419
        load_block2d.ugm.d16v.a64 (1|M0)  r13:16 [r124:1]           {I@1,$15} // ex_desc:0x0; desc:0x3000283 // $422
(W)     bfn.(s0|s1|s2) (1|M0)    r55.5<1>:ud   r54.9<0;0>:ud     r125.15<0;0>:ud   r54.1<0>:ud      {$14.src} //  ALU pipe: int; $416

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r55.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $427

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     or (1|M0)                r124.6<1>:d   r54.9<0;1,0>:d    544:w               {$15.src}       //  ALU pipe: int; $415
(W)     mov (1|M0)               r124.5<1>:d   r54.8<0;1,0>:d                                        //  ALU pipe: int; $430

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        sync.allwr                           ($10,$13)                                               // $406
        dpas.8x8 (16|M0)         r56:f         r56:f             r37:bf            r29.0:bf         {Compacted,$12} // $406
        dpas.8x8 (16|M0)         r56:f         r56:f             r45:bf            r33.0:bf         {Compacted,$12} // $407
        sync.nop                             null                             {Compacted,$12.src}    // $428
        load_block2d.ugm.d16.a64 (1|M0)  r29:8  [r55:1]            {I@3,$0} // ex_desc:0x0; desc:0x2800203 // $428

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        load_block2d.ugm.d16v.a64 (1|M0)  r37:16 [r124:1]           {I@1,$1} // ex_desc:0x0; desc:0x3000283 // $432
(W)     bfn.(s0|s1|s2) (1|M0)    r55.5<1>:ud   r54.9<0;0>:ud     r125.14<0;0>:ud   r54.1<0>:ud      {$0.src} //  ALU pipe: int; $446
(W)     mov (1|M0)               r55.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $450

// Line 85:  off_k += BLOCK_SIZE_K
(W)     or (1|M0)                r124.6<1>:d   r54.9<0;1,0>:d    576:w               {$1.src}        //  ALU pipe: int; $409

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     mov (1|M0)               r124.5<1>:d   r54.8<0;1,0>:d                                        //  ALU pipe: int; $452

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        sync.allwr                           ($12,$15)                                               // $436
        dpas.8x8 (16|M0)         r56:f         r56:f             r13:bf            r5.0:bf          {Compacted,$14} // $436
        dpas.8x8 (16|M0)         r56:f         r56:f             r21:bf            r9.0:bf          {Compacted,$14} // $437

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        sync.nop                             null                             {Compacted,$14.src}    // $451
        load_block2d.ugm.d16.a64 (1|M0)  r5:8   [r55:1]            {I@3,$2} // ex_desc:0x0; desc:0x2800203 // $451
        load_block2d.ugm.d16v.a64 (1|M0)  r13:16 [r124:1]           {I@1,$3} // ex_desc:0x0; desc:0x3000283 // $454
(W)     bfn.(s0|s1|s2) (1|M0)    r55.5<1>:ud   r54.9<0;0>:ud     r125.13<0;0>:ud   r54.1<0>:ud      {$2.src} //  ALU pipe: int; $448

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r55.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $459

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     or (1|M0)                r124.6<1>:d   r54.9<0;1,0>:d    608:w               {$3.src}        //  ALU pipe: int; $447
(W)     mov (1|M0)               r124.5<1>:d   r54.8<0;1,0>:d                                        //  ALU pipe: int; $462

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        sync.allwr                           ($1,$14)                                                // $438
        dpas.8x8 (16|M0)         r56:f         r56:f             r37:bf            r29.0:bf         {Compacted,$0} // $438
        dpas.8x8 (16|M0)         r56:f         r56:f             r45:bf            r33.0:bf         {Compacted,$0} // $439
        sync.nop                             null                             {Compacted,$0.src}     // $460
        load_block2d.ugm.d16.a64 (1|M0)  r29:8  [r55:1]            {I@3,$6} // ex_desc:0x0; desc:0x2800203 // $460

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        load_block2d.ugm.d16v.a64 (1|M0)  r37:16 [r124:1]           {I@1,$7} // ex_desc:0x0; desc:0x3000283 // $464
(W)     bfn.(s0|s1|s2) (1|M0)    r55.5<1>:ud   r54.9<0;0>:ud     r125.12<0;0>:ud   r54.1<0>:ud      {$6.src} //  ALU pipe: int; $478
(W)     mov (1|M0)               r55.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $482

// Line 85:  off_k += BLOCK_SIZE_K
(W)     or (1|M0)                r124.6<1>:d   r54.9<0;1,0>:d    640:w               {$7.src}        //  ALU pipe: int; $441

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     mov (1|M0)               r124.5<1>:d   r54.8<0;1,0>:d                                        //  ALU pipe: int; $484

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        sync.allwr                           ($0,$3)                                                 // $468
        dpas.8x8 (16|M0)         r56:f         r56:f             r13:bf            r5.0:bf          {Compacted,$2} // $468
        dpas.8x8 (16|M0)         r56:f         r56:f             r21:bf            r9.0:bf          {Compacted,$2} // $469

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        sync.nop                             null                             {Compacted,$2.src}     // $483
        load_block2d.ugm.d16.a64 (1|M0)  r5:8   [r55:1]            {I@3,$8} // ex_desc:0x0; desc:0x2800203 // $483
        load_block2d.ugm.d16v.a64 (1|M0)  r13:16 [r124:1]           {I@1,$9} // ex_desc:0x0; desc:0x3000283 // $486
(W)     bfn.(s0|s1|s2) (1|M0)    r55.5<1>:ud   r54.9<0;0>:ud     r125.11<0;0>:ud   r54.1<0>:ud      {$8.src} //  ALU pipe: int; $480

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r55.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $491

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     or (1|M0)                r124.6<1>:d   r54.9<0;1,0>:d    672:w               {$9.src}        //  ALU pipe: int; $479
(W)     mov (1|M0)               r124.5<1>:d   r54.8<0;1,0>:d                                        //  ALU pipe: int; $494

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        sync.allwr                           ($2,$7)                                                 // $470
        dpas.8x8 (16|M0)         r56:f         r56:f             r37:bf            r29.0:bf         {Compacted,$6} // $470
        dpas.8x8 (16|M0)         r56:f         r56:f             r45:bf            r33.0:bf         {Compacted,$6} // $471
        sync.nop                             null                             {Compacted,$6.src}     // $492
        load_block2d.ugm.d16.a64 (1|M0)  r29:8  [r55:1]            {I@3,$10} // ex_desc:0x0; desc:0x2800203 // $492

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        load_block2d.ugm.d16v.a64 (1|M0)  r37:16 [r124:1]           {I@1,$11} // ex_desc:0x0; desc:0x3000283 // $496
(W)     bfn.(s0|s1|s2) (1|M0)    r55.5<1>:ud   r54.9<0;0>:ud     r125.10<0;0>:ud   r54.1<0>:ud      {$10.src} //  ALU pipe: int; $510
(W)     mov (1|M0)               r55.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $514

// Line 85:  off_k += BLOCK_SIZE_K
(W)     or (1|M0)                r124.6<1>:d   r54.9<0;1,0>:d    704:w               {$11.src}       //  ALU pipe: int; $473

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     mov (1|M0)               r124.5<1>:d   r54.8<0;1,0>:d                                        //  ALU pipe: int; $516

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        sync.allwr                           ($6,$9)                                                 // $500
        dpas.8x8 (16|M0)         r56:f         r56:f             r13:bf            r5.0:bf          {Compacted,$8} // $500
        dpas.8x8 (16|M0)         r56:f         r56:f             r21:bf            r9.0:bf          {Compacted,$8} // $501

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        sync.nop                             null                             {Compacted,$8.src}     // $515
        load_block2d.ugm.d16.a64 (1|M0)  r5:8   [r55:1]            {I@3,$12} // ex_desc:0x0; desc:0x2800203 // $515
        load_block2d.ugm.d16v.a64 (1|M0)  r13:16 [r124:1]           {I@1,$13} // ex_desc:0x0; desc:0x3000283 // $518
(W)     bfn.(s0|s1|s2) (1|M0)    r55.5<1>:ud   r54.9<0;0>:ud     r125.9<0;0>:ud    r54.1<0>:ud      {$12.src} //  ALU pipe: int; $512

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r55.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $523

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     or (1|M0)                r124.6<1>:d   r54.9<0;1,0>:d    736:w               {$13.src}       //  ALU pipe: int; $511
(W)     mov (1|M0)               r124.5<1>:d   r54.8<0;1,0>:d                                        //  ALU pipe: int; $526

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        sync.allwr                           ($8,$11)                                                // $502
        dpas.8x8 (16|M0)         r56:f         r56:f             r37:bf            r29.0:bf         {Compacted,$10} // $502
        dpas.8x8 (16|M0)         r56:f         r56:f             r45:bf            r33.0:bf         {Compacted,$10} // $503
        sync.nop                             null                             {Compacted,$10.src}    // $524
        load_block2d.ugm.d16.a64 (1|M0)  r29:8  [r55:1]            {I@3,$14} // ex_desc:0x0; desc:0x2800203 // $524

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        load_block2d.ugm.d16v.a64 (1|M0)  r37:16 [r124:1]           {I@1,$15} // ex_desc:0x0; desc:0x3000283 // $528
(W)     bfn.(s0|s1|s2) (1|M0)    r55.5<1>:ud   r54.9<0;0>:ud     r125.8<0;0>:ud    r54.1<0>:ud      {$14.src} //  ALU pipe: int; $542
(W)     mov (1|M0)               r55.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $546

// Line 85:  off_k += BLOCK_SIZE_K
(W)     or (1|M0)                r124.6<1>:d   r54.9<0;1,0>:d    768:w               {$15.src}       //  ALU pipe: int; $505

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     mov (1|M0)               r124.5<1>:d   r54.8<0;1,0>:d                                        //  ALU pipe: int; $548

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        sync.allwr                           ($10,$13)                                               // $532
        dpas.8x8 (16|M0)         r56:f         r56:f             r13:bf            r5.0:bf          {Compacted,$12} // $532
        dpas.8x8 (16|M0)         r56:f         r56:f             r21:bf            r9.0:bf          {Compacted,$12} // $533

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        sync.nop                             null                             {Compacted,$12.src}    // $547
        load_block2d.ugm.d16.a64 (1|M0)  r5:8   [r55:1]            {I@3,$0} // ex_desc:0x0; desc:0x2800203 // $547
        load_block2d.ugm.d16v.a64 (1|M0)  r13:16 [r124:1]           {I@1,$1} // ex_desc:0x0; desc:0x3000283 // $550
(W)     bfn.(s0|s1|s2) (1|M0)    r55.5<1>:ud   r54.9<0;0>:ud     r125.7<0;0>:ud    r54.1<0>:ud      {$0.src} //  ALU pipe: int; $544

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r55.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $555

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     or (1|M0)                r124.6<1>:d   r54.9<0;1,0>:d    800:w               {$1.src}        //  ALU pipe: int; $543
(W)     mov (1|M0)               r124.5<1>:d   r54.8<0;1,0>:d                                        //  ALU pipe: int; $558

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        sync.allwr                           ($12,$15)                                               // $534
        dpas.8x8 (16|M0)         r56:f         r56:f             r37:bf            r29.0:bf         {Compacted,$14} // $534
        dpas.8x8 (16|M0)         r56:f         r56:f             r45:bf            r33.0:bf         {Compacted,$14} // $535
        sync.nop                             null                             {Compacted,$14.src}    // $556
        load_block2d.ugm.d16.a64 (1|M0)  r29:8  [r55:1]            {I@3,$2} // ex_desc:0x0; desc:0x2800203 // $556

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        load_block2d.ugm.d16v.a64 (1|M0)  r37:16 [r124:1]           {I@1,$3} // ex_desc:0x0; desc:0x3000283 // $560
(W)     bfn.(s0|s1|s2) (1|M0)    r55.5<1>:ud   r54.9<0;0>:ud     r125.6<0;0>:ud    r54.1<0>:ud      {$2.src} //  ALU pipe: int; $574
(W)     mov (1|M0)               r55.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $578

// Line 85:  off_k += BLOCK_SIZE_K
(W)     or (1|M0)                r124.6<1>:d   r54.9<0;1,0>:d    832:w               {$3.src}        //  ALU pipe: int; $537

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     mov (1|M0)               r124.5<1>:d   r54.8<0;1,0>:d                                        //  ALU pipe: int; $580

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        sync.allwr                           ($1,$14)                                                // $564
        dpas.8x8 (16|M0)         r56:f         r56:f             r13:bf            r5.0:bf          {Compacted,$0} // $564
        dpas.8x8 (16|M0)         r56:f         r56:f             r21:bf            r9.0:bf          {Compacted,$0} // $565

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        sync.nop                             null                             {Compacted,$0.src}     // $579
        load_block2d.ugm.d16.a64 (1|M0)  r5:8   [r55:1]            {I@3,$6} // ex_desc:0x0; desc:0x2800203 // $579
        load_block2d.ugm.d16v.a64 (1|M0)  r13:16 [r124:1]           {I@1,$7} // ex_desc:0x0; desc:0x3000283 // $582
(W)     bfn.(s0|s1|s2) (1|M0)    r55.5<1>:ud   r54.9<0;0>:ud     r125.5<0;0>:ud    r54.1<0>:ud      {$6.src} //  ALU pipe: int; $576

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r55.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $587

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     or (1|M0)                r124.6<1>:d   r54.9<0;1,0>:d    864:w               {$7.src}        //  ALU pipe: int; $575
(W)     mov (1|M0)               r124.5<1>:d   r54.8<0;1,0>:d                                        //  ALU pipe: int; $590

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        sync.allwr                           ($0,$3)                                                 // $566
        dpas.8x8 (16|M0)         r56:f         r56:f             r37:bf            r29.0:bf         {Compacted,$2} // $566
        dpas.8x8 (16|M0)         r56:f         r56:f             r45:bf            r33.0:bf         {Compacted,$2} // $567
        sync.nop                             null                             {Compacted,$2.src}     // $588
        load_block2d.ugm.d16.a64 (1|M0)  r29:8  [r55:1]            {I@3,$8} // ex_desc:0x0; desc:0x2800203 // $588

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        load_block2d.ugm.d16v.a64 (1|M0)  r37:16 [r124:1]           {I@1,$9} // ex_desc:0x0; desc:0x3000283 // $592
(W)     bfn.(s0|s1|s2) (1|M0)    r55.5<1>:ud   r54.9<0;0>:ud     r125.4<0;0>:ud    r54.1<0>:ud      {$8.src} //  ALU pipe: int; $606
(W)     mov (1|M0)               r55.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $610

// Line 85:  off_k += BLOCK_SIZE_K
(W)     or (1|M0)                r124.6<1>:d   r54.9<0;1,0>:d    896:w               {$9.src}        //  ALU pipe: int; $569

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     mov (1|M0)               r124.5<1>:d   r54.8<0;1,0>:d                                        //  ALU pipe: int; $612

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        sync.allwr                           ($2,$7)                                                 // $596
        dpas.8x8 (16|M0)         r56:f         r56:f             r13:bf            r5.0:bf          {Compacted,$6} // $596
        dpas.8x8 (16|M0)         r56:f         r56:f             r21:bf            r9.0:bf          {Compacted,$6} // $597

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        sync.nop                             null                             {Compacted,$6.src}     // $611
        load_block2d.ugm.d16.a64 (1|M0)  r5:8   [r55:1]            {I@3,$10} // ex_desc:0x0; desc:0x2800203 // $611
        load_block2d.ugm.d16v.a64 (1|M0)  r13:16 [r124:1]           {I@1,$11} // ex_desc:0x0; desc:0x3000283 // $614
(W)     bfn.(s0|s1|s2) (1|M0)    r55.5<1>:ud   r54.9<0;0>:ud     r125.3<0;0>:ud    r54.1<0>:ud      {$10.src} //  ALU pipe: int; $608

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r55.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $619

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     or (1|M0)                r124.6<1>:d   r54.9<0;1,0>:d    928:w               {$11.src}       //  ALU pipe: int; $607
(W)     mov (1|M0)               r124.5<1>:d   r54.8<0;1,0>:d                                        //  ALU pipe: int; $622

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        sync.allwr                           ($6,$9)                                                 // $598
        dpas.8x8 (16|M0)         r56:f         r56:f             r37:bf            r29.0:bf         {Compacted,$8} // $598
        dpas.8x8 (16|M0)         r56:f         r56:f             r45:bf            r33.0:bf         {Compacted,$8} // $599
        sync.nop                             null                             {Compacted,$8.src}     // $620
        load_block2d.ugm.d16.a64 (1|M0)  r29:8  [r55:1]            {I@3,$12} // ex_desc:0x0; desc:0x2800203 // $620

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        load_block2d.ugm.d16v.a64 (1|M0)  r37:16 [r124:1]           {I@1,$13} // ex_desc:0x0; desc:0x3000283 // $624
(W)     or (1|M0)                r55.5<1>:d    r1.8<0;1,0>:d     r54.1<0;1,0>:d   {$12.src}          //  ALU pipe: int; $638
(W)     mov (1|M0)               r55.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $642
(W)     mov (1|M0)               r124.5<1>:d   r54.8<0;1,0>:d                   {$13.src}            //  ALU pipe: int; $644
(W)     mov (1|M0)               r124.6<1>:d   r1.8<0;1,0>:d                                         //  ALU pipe: int; $645

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        sync.allwr                           ($8,$11)                                                // $628
        dpas.8x8 (16|M0)         r56:f         r56:f             r13:bf            r5.0:bf          {Compacted,$10} // $628
        dpas.8x8 (16|M0)         r56:f         r56:f             r21:bf            r9.0:bf          {Compacted,$10} // $629

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        sync.nop                             null                             {Compacted,$10.src}    // $643
        load_block2d.ugm.d16.a64 (1|M0)  r5:8   [r55:1]            {I@3,$14} // ex_desc:0x0; desc:0x2800203 // $643
        load_block2d.ugm.d16v.a64 (1|M0)  r13:16 [r124:1]           {I@1,$15} // ex_desc:0x0; desc:0x3000283 // $646
(W)     bfn.(s0|s1|s2) (1|M0)    r55.5<1>:ud   r54.9<0;0>:ud     r125.2<0;0>:ud    r54.1<0>:ud      {$14.src} //  ALU pipe: int; $640

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
(W)     mov (1|M0)               r55.6<1>:d    r126.0<0;1,0>:d                                       //  ALU pipe: int; $651

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
(W)     or (1|M0)                r124.6<1>:d   r54.9<0;1,0>:d    992:w               {$15.src}       //  ALU pipe: int; $639
(W)     mov (1|M0)               r124.5<1>:d   r54.8<0;1,0>:d                                        //  ALU pipe: int; $654

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        sync.allwr                           ($10,$13)                                               // $630
        dpas.8x8 (16|M0)         r56:f         r56:f             r37:bf            r29.0:bf         {Compacted,$12} // $630
        dpas.8x8 (16|M0)         r56:f         r56:f             r45:bf            r33.0:bf         {Compacted,$12} // $631
        sync.nop                             null                             {Compacted,$12.src}    // $652
        load_block2d.ugm.d16.a64 (1|M0)  r29:8  [r55:1]            {I@3,$0} // ex_desc:0x0; desc:0x2800203 // $652

// Line 79:  a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        load_block2d.ugm.d16v.a64 (1|M0)  r37:16 [r124:1]           {I@1,$1} // ex_desc:0x0; desc:0x3000283 // $656

// Line 83:  b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        sync.allwr                           ($12,$15)                                               // $660
        dpas.8x8 (16|M0)         r56:f         r56:f             r13:bf            r5.0:bf          {Compacted,$14} // $660
        dpas.8x8 (16|M0)         r56:f         r56:f             r21:bf            r9.0:bf          {Compacted,$14} // $661
        sync.allwr                           ($1,$14)                                                // $662
        dpas.8x8 (16|M0)         r56:f         r56:f             r37:bf            r29.0:bf         {Compacted,$0} // $662
        sync.nop                             null                             {Compacted,$0.dst}     // $663
        dpas.8x8 (16|M0)         r56:f         r56:f             r45:bf            r33.0:bf         {Compacted,$5} // $663

// Line 75:  for _ in range(0, K, BLOCK_SIZE_K):
(W&~f1.0) jmpi                               _0_011                                                  //  ALU pipe: int; $666
// B007: Preds:{B006},  Succs:{B006}
_0_012:
(W)     mov (1|M0)               r54.9<1>:d    r126.3<0;1,0>:d                                       //  ALU pipe: int; $668
(W)     jmpi                                 _0_010                                                  // $669
// B008: Preds:{B006},  Succs:{}
_0_011:

// Line 90:  c_desc.store([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], c)
        and (16|M0)   (eq)f0.0   null<2>:w     r1.0<1;1,0>:w     512:w                               //  ALU pipe: int; $672
(W)     and (1|M0)               r1.10<1>:d    r4.4<0;1,0>:d     63:w                                //  ALU pipe: int; $683
(W)     mov (1|M0)               r1.9<1>:d     r4.5<0;1,0>:d                                         //  ALU pipe: int; $676
(W)     and (1|M0)               r1.8<1>:d     r4.4<0;1,0>:d     -64:w                               //  ALU pipe: int; $677
(f0.0)  sel (16|M0)              r2.0<1>:d     r126.0<0;1,0>:d   4:w               {Compacted,$4.src} //  ALU pipe: int; $674
(W)     shr (1|M0)               r4.6<1>:ud    r1.10<0;1,0>:ud   2:w               {I@4}             //  ALU pipe: int; $684
(W)     mov (1|M0)               r3.3<1>:ud    3:w                                                   //  blk2d.heightM1; ALU pipe: int; $687
(W)     mov (1|M0)               r3.4<1>:ud    0xBFFF:uw                                             //  blk2d.pitchM1; ALU pipe: int; $687
(W)     mov (1|M0)               r3.7<1>:ud    0x70F:uw                                              //  bkl2d.shape = 1x16x8; ALU pipe: int; $687

// Line 38:  def matmul_kernel_with_tensor_descriptors(
(W)     mov (16|M0)              r127.0<1>:f   r53.0<1;1,0>:f                   {Compacted}          //  ALU pipe: float; $689

// Line 90:  c_desc.store([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], c)
(W)     add (1|M0)               r3.2<1>:ud    r1.10<0;1,0>:d    0xBFFF:uw                           //  ALU pipe: int; $686
(W)     mov (1|M0)               r3.0<1>:uq    r1.4<0;1,0>:q                    {I@7}                //  ALU pipe: int; $687
(W)     mov (1|M0)               r3.6<1>:ud    r2.0<0;1,0>:d                    {I@7}                //  blk2d.Y; ALU pipe: int; $687
(W)     bfn.(s0|s1|s2) (1|M0)    r3.5<1>:ud    r4.6<0;0>:ud      r126.2<0;0>:ud    r125.1<0>:ud     {I@7} //  ALU pipe: int; $685
        store_block2d.ugm.d32.a64 (1|M0)  [r3:1] r56:8             {I@1,$5} // ex_desc:0x0; desc:0x2000407 // $687

// Line 38:  def matmul_kernel_with_tensor_descriptors(
(W)     send.gtwy (1|M0)         null     r127  null:0  0x0            0x02000010           {EOT,F@1,$2} // wr:1+0, rd:0; end of thread // $689
L8848:
(W)     mov (16|M0)              null<1>:ud    0x604C6B14:ud                                         // 
(W)     mov (16|M0)              null<1>:ud    0x5D199062:ud                                         // 
(W)     mov (16|M0)              null<1>:ud    0x0:ud                                                // 
(W)     mov (16|M0)              null<1>:ud    0x1:ud                                                // 


//.BankConflicts: 0
//.ByteRMWs: 2
//


//.numALUInst: 466
//.accSubDef: 0
//.accSubUse: 0
//.accSubCandidateDef: 0
//.accSubCandidateUse: 0
//
//
//.singlePipeAtOneDistNum: 59
//.allAtOneDistNum: 9
//.syncInstCount: 34
//.tokenReuseCount: 2
//.AfterWriteTokenDepCount: 69
//.AfterReadTokenDepCount: 113
