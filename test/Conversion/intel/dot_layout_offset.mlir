// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s

#dpas = #ttig.dpas<{repeatCount=8, systolicDepth=8, executionSize = 8, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA=[1, 1], repCluster=[2, 2]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#dpas, kWidth=1}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @dot_layout_emit_offset()
  tt.func public @dot_layout_emit_offset() {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #dot_operand_a>
    // CHECK-COUNT-64:  {{.*}} = llvm.extractvalue {{.*}}

    // COM: Base index of the dot layout.
    // CHECK:           %[[THREAD_ID_I64:.*]] = llvm.call spir_funccc @_Z12get_local_idj
    // CHECK:           %[[THREAD_ID_I32:.*]] = llvm.trunc %[[THREAD_ID_I64]] : i64 to i32
    // CHECK:           %[[VAL_145:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[LANE_ID:.*]] = llvm.urem %[[THREAD_ID_I32]], %[[VAL_145]]  : i32
    // CHECK:           %[[WARP_ID:.*]] = llvm.udiv %[[THREAD_ID_I32]], %[[VAL_145]]  : i32
    // CHECK-COUNT-3:   %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_149:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[VAL_150:.*]] = llvm.and %[[LANE_ID]], %[[VAL_149]]  : i32
    // CHECK:           %[[VAL_151:.*]] = llvm.icmp "eq" %[[VAL_150]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_152:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[VAL_153:.*]] = llvm.select %[[VAL_151]], %[[CST_0]], %[[VAL_152]] : i1, i32
    // CHECK:           %[[VAL_154:.*]] = llvm.xor %[[CST_0]], %[[VAL_153]]  : i32
    // CHECK:           %[[VAL_155:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK:           %[[VAL_156:.*]] = llvm.and %[[LANE_ID]], %[[VAL_155]]  : i32
    // CHECK:           %[[VAL_157:.*]] = llvm.icmp "eq" %[[VAL_156]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_158:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK:           %[[VAL_159:.*]] = llvm.select %[[VAL_157]], %[[CST_0]], %[[VAL_158]] : i1, i32
    // CHECK:           %[[VAL_160:.*]] = llvm.xor %[[VAL_154]], %[[VAL_159]]  : i32
    // CHECK:           %[[VAL_161:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK:           %[[VAL_162:.*]] = llvm.and %[[LANE_ID]], %[[VAL_161]]  : i32
    // CHECK:           %[[VAL_163:.*]] = llvm.icmp "eq" %[[VAL_162]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_164:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK:           %[[VAL_165:.*]] = llvm.select %[[VAL_163]], %[[CST_0]], %[[VAL_164]] : i1, i32
    // CHECK:           %[[VAL_182:.*]] = llvm.xor %[[VAL_160]], %[[VAL_165]]  : i32
    // CHECK:           %[[VAL_167:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK:           %[[VAL_168:.*]] = llvm.and %[[LANE_ID]], %[[VAL_167]]  : i32
    // CHECK:           %[[VAL_169:.*]] = llvm.icmp "eq" %[[VAL_168]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_170:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK:           %[[VAL_171:.*]] = llvm.select %[[VAL_169]], %[[CST_0]], %[[VAL_170]] : i1, i32
    // CHECK:           %[[VAL_181:.*]] = llvm.xor %[[VAL_182]], %[[VAL_171]]  : i32

    // COM: There are total [4, 2] repetitions of tensor shape [32, 32] per warp.
    // COM: The repetitions are clustered as [2, 1] for A operand. The repetitions orders are [0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [3, 0], [2, 1], [3, 1]
    // COM: Offsets of rep [0, 0].
    // CHECK:           %[[VAL_183:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_184:.*]] = llvm.xor %[[CST_0]], %[[VAL_183]] : i32
    // CHECK:           %[[VAL_185:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_186:.*]] = llvm.xor %[[VAL_181]], %[[VAL_185]] : i32
    // CHECK:           %[[VAL_187:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[VAL_188:.*]] = llvm.xor %[[CST_0]], %[[VAL_187]] : i32
    // CHECK:           %[[VAL_189:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_190:.*]] = llvm.xor %[[VAL_181]], %[[VAL_189]] : i32
    // CHECK:           %[[VAL_191:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK:           %[[VAL_192:.*]] = llvm.xor %[[CST_0]], %[[VAL_191]] : i32
    // CHECK:           %[[VAL_193:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_194:.*]] = llvm.xor %[[VAL_181]], %[[VAL_193]] : i32
    // CHECK:           %[[VAL_195:.*]] = llvm.mlir.constant(3 : i32) : i32
    // CHECK:           %[[VAL_196:.*]] = llvm.xor %[[CST_0]], %[[VAL_195]] : i32
    // CHECK:           %[[VAL_197:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_198:.*]] = llvm.xor %[[VAL_181]], %[[VAL_197]] : i32
    // CHECK:           %[[VAL_199:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK:           %[[VAL_200:.*]] = llvm.xor %[[CST_0]], %[[VAL_199]] : i32
    // CHECK:           %[[VAL_201:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_202:.*]] = llvm.xor %[[VAL_181]], %[[VAL_201]] : i32
    // CHECK:           %[[VAL_203:.*]] = llvm.mlir.constant(5 : i32) : i32
    // CHECK:           %[[VAL_204:.*]] = llvm.xor %[[CST_0]], %[[VAL_203]] : i32
    // CHECK:           %[[VAL_205:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_206:.*]] = llvm.xor %[[VAL_181]], %[[VAL_205]] : i32
    // CHECK:           %[[VAL_207:.*]] = llvm.mlir.constant(6 : i32) : i32
    // CHECK:           %[[VAL_208:.*]] = llvm.xor %[[CST_0]], %[[VAL_207]] : i32
    // CHECK:           %[[VAL_209:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_210:.*]] = llvm.xor %[[VAL_181]], %[[VAL_209]] : i32
    // CHECK:           %[[VAL_211:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[VAL_212:.*]] = llvm.xor %[[CST_0]], %[[VAL_211]] : i32
    // CHECK:           %[[VAL_213:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_214:.*]] = llvm.xor %[[VAL_181]], %[[VAL_213]] : i32

    // COM: Offsets of rep [1, 0].
    // CHECK:           %[[VAL_215:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK:           %[[VAL_216:.*]] = llvm.xor %[[CST_0]], %[[VAL_215]] : i32
    // CHECK:           %[[VAL_217:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_218:.*]] = llvm.xor %[[VAL_181]], %[[VAL_217]] : i32
    // CHECK:           %[[VAL_219:.*]] = llvm.mlir.constant(9 : i32) : i32
    // CHECK:           %[[VAL_220:.*]] = llvm.xor %[[CST_0]], %[[VAL_219]] : i32
    // CHECK:           %[[VAL_221:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_222:.*]] = llvm.xor %[[VAL_181]], %[[VAL_221]] : i32
    // CHECK:           %[[VAL_223:.*]] = llvm.mlir.constant(10 : i32) : i32
    // CHECK:           %[[VAL_224:.*]] = llvm.xor %[[CST_0]], %[[VAL_223]] : i32
    // CHECK:           %[[VAL_225:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_226:.*]] = llvm.xor %[[VAL_181]], %[[VAL_225]] : i32
    // CHECK:           %[[VAL_227:.*]] = llvm.mlir.constant(11 : i32) : i32
    // CHECK:           %[[VAL_228:.*]] = llvm.xor %[[CST_0]], %[[VAL_227]] : i32
    // CHECK:           %[[VAL_229:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_230:.*]] = llvm.xor %[[VAL_181]], %[[VAL_229]] : i32
    // CHECK:           %[[VAL_231:.*]] = llvm.mlir.constant(12 : i32) : i32
    // CHECK:           %[[VAL_232:.*]] = llvm.xor %[[CST_0]], %[[VAL_231]] : i32
    // CHECK:           %[[VAL_233:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_234:.*]] = llvm.xor %[[VAL_181]], %[[VAL_233]] : i32
    // CHECK:           %[[VAL_235:.*]] = llvm.mlir.constant(13 : i32) : i32
    // CHECK:           %[[VAL_236:.*]] = llvm.xor %[[CST_0]], %[[VAL_235]] : i32
    // CHECK:           %[[VAL_237:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_238:.*]] = llvm.xor %[[VAL_181]], %[[VAL_237]] : i32
    // CHECK:           %[[VAL_239:.*]] = llvm.mlir.constant(14 : i32) : i32
    // CHECK:           %[[VAL_240:.*]] = llvm.xor %[[CST_0]], %[[VAL_239]] : i32
    // CHECK:           %[[VAL_241:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_242:.*]] = llvm.xor %[[VAL_181]], %[[VAL_241]] : i32
    // CHECK:           %[[VAL_243:.*]] = llvm.mlir.constant(15 : i32) : i32
    // CHECK:           %[[VAL_244:.*]] = llvm.xor %[[CST_0]], %[[VAL_243]] : i32
    // CHECK:           %[[VAL_245:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_246:.*]] = llvm.xor %[[VAL_181]], %[[VAL_245]] : i32

    // COM: Offsets of rep [0, 1].
    // CHECK:           %[[VAL_247:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_248:.*]] = llvm.xor %[[CST_0]], %[[VAL_247]] : i32
    // CHECK:           %[[VAL_249:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_250:.*]] = llvm.xor %[[VAL_181]], %[[VAL_249]] : i32
    // CHECK:           %[[VAL_251:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[VAL_252:.*]] = llvm.xor %[[CST_0]], %[[VAL_251]] : i32
    // CHECK:           %[[VAL_253:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_254:.*]] = llvm.xor %[[VAL_181]], %[[VAL_253]] : i32
    // CHECK:           %[[VAL_255:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK:           %[[VAL_256:.*]] = llvm.xor %[[CST_0]], %[[VAL_255]] : i32
    // CHECK:           %[[VAL_257:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_258:.*]] = llvm.xor %[[VAL_181]], %[[VAL_257]] : i32
    // CHECK:           %[[VAL_259:.*]] = llvm.mlir.constant(3 : i32) : i32
    // CHECK:           %[[VAL_260:.*]] = llvm.xor %[[CST_0]], %[[VAL_259]] : i32
    // CHECK:           %[[VAL_261:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_262:.*]] = llvm.xor %[[VAL_181]], %[[VAL_261]] : i32
    // CHECK:           %[[VAL_263:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK:           %[[VAL_264:.*]] = llvm.xor %[[CST_0]], %[[VAL_263]] : i32
    // CHECK:           %[[VAL_265:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_266:.*]] = llvm.xor %[[VAL_181]], %[[VAL_265]] : i32
    // CHECK:           %[[VAL_267:.*]] = llvm.mlir.constant(5 : i32) : i32
    // CHECK:           %[[VAL_268:.*]] = llvm.xor %[[CST_0]], %[[VAL_267]] : i32
    // CHECK:           %[[VAL_269:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_270:.*]] = llvm.xor %[[VAL_181]], %[[VAL_269]] : i32
    // CHECK:           %[[VAL_271:.*]] = llvm.mlir.constant(6 : i32) : i32
    // CHECK:           %[[VAL_272:.*]] = llvm.xor %[[CST_0]], %[[VAL_271]] : i32
    // CHECK:           %[[VAL_273:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_274:.*]] = llvm.xor %[[VAL_181]], %[[VAL_273]] : i32
    // CHECK:           %[[VAL_275:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:           %[[VAL_276:.*]] = llvm.xor %[[CST_0]], %[[VAL_275]] : i32
    // CHECK:           %[[VAL_277:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_278:.*]] = llvm.xor %[[VAL_181]], %[[VAL_277]] : i32

    // COM: Offsets of rep [1, 1].
    // CHECK:           %[[VAL_279:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK:           %[[VAL_280:.*]] = llvm.xor %[[CST_0]], %[[VAL_279]] : i32
    // CHECK:           %[[VAL_281:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_282:.*]] = llvm.xor %[[VAL_181]], %[[VAL_281]] : i32
    // CHECK:           %[[VAL_283:.*]] = llvm.mlir.constant(9 : i32) : i32
    // CHECK:           %[[VAL_284:.*]] = llvm.xor %[[CST_0]], %[[VAL_283]] : i32
    // CHECK:           %[[VAL_285:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_286:.*]] = llvm.xor %[[VAL_181]], %[[VAL_285]] : i32
    // CHECK:           %[[VAL_287:.*]] = llvm.mlir.constant(10 : i32) : i32
    // CHECK:           %[[VAL_288:.*]] = llvm.xor %[[CST_0]], %[[VAL_287]] : i32
    // CHECK:           %[[VAL_289:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_290:.*]] = llvm.xor %[[VAL_181]], %[[VAL_289]] : i32
    // CHECK:           %[[VAL_291:.*]] = llvm.mlir.constant(11 : i32) : i32
    // CHECK:           %[[VAL_292:.*]] = llvm.xor %[[CST_0]], %[[VAL_291]] : i32
    // CHECK:           %[[VAL_293:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_294:.*]] = llvm.xor %[[VAL_181]], %[[VAL_293]] : i32
    // CHECK:           %[[VAL_295:.*]] = llvm.mlir.constant(12 : i32) : i32
    // CHECK:           %[[VAL_296:.*]] = llvm.xor %[[CST_0]], %[[VAL_295]] : i32
    // CHECK:           %[[VAL_297:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_298:.*]] = llvm.xor %[[VAL_181]], %[[VAL_297]] : i32
    // CHECK:           %[[VAL_299:.*]] = llvm.mlir.constant(13 : i32) : i32
    // CHECK:           %[[VAL_300:.*]] = llvm.xor %[[CST_0]], %[[VAL_299]] : i32
    // CHECK:           %[[VAL_301:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_302:.*]] = llvm.xor %[[VAL_181]], %[[VAL_301]] : i32
    // CHECK:           %[[VAL_303:.*]] = llvm.mlir.constant(14 : i32) : i32
    // CHECK:           %[[VAL_304:.*]] = llvm.xor %[[CST_0]], %[[VAL_303]] : i32
    // CHECK:           %[[VAL_305:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_306:.*]] = llvm.xor %[[VAL_181]], %[[VAL_305]] : i32
    // CHECK:           %[[VAL_307:.*]] = llvm.mlir.constant(15 : i32) : i32
    // CHECK:           %[[VAL_308:.*]] = llvm.xor %[[CST_0]], %[[VAL_307]] : i32
    // CHECK:           %[[VAL_309:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_310:.*]] = llvm.xor %[[VAL_181]], %[[VAL_309]] : i32

    // COM: Offsets of rep [2, 0].
    // CHECK:           %[[VAL_311:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_312:.*]] = llvm.xor %[[CST_0]], %[[VAL_311]] : i32
    // CHECK:           %[[VAL_313:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_314:.*]] = llvm.xor %[[VAL_181]], %[[VAL_313]] : i32
    // CHECK:           %[[VAL_315:.*]] = llvm.mlir.constant(17 : i32) : i32
    // CHECK:           %[[VAL_316:.*]] = llvm.xor %[[CST_0]], %[[VAL_315]] : i32
    // CHECK:           %[[VAL_317:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_318:.*]] = llvm.xor %[[VAL_181]], %[[VAL_317]] : i32
    // CHECK:           %[[VAL_319:.*]] = llvm.mlir.constant(18 : i32) : i32
    // CHECK:           %[[VAL_320:.*]] = llvm.xor %[[CST_0]], %[[VAL_319]] : i32
    // CHECK:           %[[VAL_321:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_322:.*]] = llvm.xor %[[VAL_181]], %[[VAL_321]] : i32
    // CHECK:           %[[VAL_323:.*]] = llvm.mlir.constant(19 : i32) : i32
    // CHECK:           %[[VAL_324:.*]] = llvm.xor %[[CST_0]], %[[VAL_323]] : i32
    // CHECK:           %[[VAL_325:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_326:.*]] = llvm.xor %[[VAL_181]], %[[VAL_325]] : i32
    // CHECK:           %[[VAL_327:.*]] = llvm.mlir.constant(20 : i32) : i32
    // CHECK:           %[[VAL_328:.*]] = llvm.xor %[[CST_0]], %[[VAL_327]] : i32
    // CHECK:           %[[VAL_329:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_330:.*]] = llvm.xor %[[VAL_181]], %[[VAL_329]] : i32
    // CHECK:           %[[VAL_331:.*]] = llvm.mlir.constant(21 : i32) : i32
    // CHECK:           %[[VAL_332:.*]] = llvm.xor %[[CST_0]], %[[VAL_331]] : i32
    // CHECK:           %[[VAL_333:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_334:.*]] = llvm.xor %[[VAL_181]], %[[VAL_333]] : i32
    // CHECK:           %[[VAL_335:.*]] = llvm.mlir.constant(22 : i32) : i32
    // CHECK:           %[[VAL_336:.*]] = llvm.xor %[[CST_0]], %[[VAL_335]] : i32
    // CHECK:           %[[VAL_337:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_338:.*]] = llvm.xor %[[VAL_181]], %[[VAL_337]] : i32
    // CHECK:           %[[VAL_339:.*]] = llvm.mlir.constant(23 : i32) : i32
    // CHECK:           %[[VAL_340:.*]] = llvm.xor %[[CST_0]], %[[VAL_339]] : i32
    // CHECK:           %[[VAL_341:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_342:.*]] = llvm.xor %[[VAL_181]], %[[VAL_341]] : i32

    // COM: Offsets of rep [3, 0].
    // CHECK:           %[[VAL_343:.*]] = llvm.mlir.constant(24 : i32) : i32
    // CHECK:           %[[VAL_344:.*]] = llvm.xor %[[CST_0]], %[[VAL_343]] : i32
    // CHECK:           %[[VAL_345:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_346:.*]] = llvm.xor %[[VAL_181]], %[[VAL_345]] : i32
    // CHECK:           %[[VAL_347:.*]] = llvm.mlir.constant(25 : i32) : i32
    // CHECK:           %[[VAL_348:.*]] = llvm.xor %[[CST_0]], %[[VAL_347]] : i32
    // CHECK:           %[[VAL_349:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_350:.*]] = llvm.xor %[[VAL_181]], %[[VAL_349]] : i32
    // CHECK:           %[[VAL_351:.*]] = llvm.mlir.constant(26 : i32) : i32
    // CHECK:           %[[VAL_352:.*]] = llvm.xor %[[CST_0]], %[[VAL_351]] : i32
    // CHECK:           %[[VAL_353:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_354:.*]] = llvm.xor %[[VAL_181]], %[[VAL_353]] : i32
    // CHECK:           %[[VAL_355:.*]] = llvm.mlir.constant(27 : i32) : i32
    // CHECK:           %[[VAL_356:.*]] = llvm.xor %[[CST_0]], %[[VAL_355]] : i32
    // CHECK:           %[[VAL_357:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_358:.*]] = llvm.xor %[[VAL_181]], %[[VAL_357]] : i32
    // CHECK:           %[[VAL_359:.*]] = llvm.mlir.constant(28 : i32) : i32
    // CHECK:           %[[VAL_360:.*]] = llvm.xor %[[CST_0]], %[[VAL_359]] : i32
    // CHECK:           %[[VAL_361:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_362:.*]] = llvm.xor %[[VAL_181]], %[[VAL_361]] : i32
    // CHECK:           %[[VAL_363:.*]] = llvm.mlir.constant(29 : i32) : i32
    // CHECK:           %[[VAL_364:.*]] = llvm.xor %[[CST_0]], %[[VAL_363]] : i32
    // CHECK:           %[[VAL_365:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_366:.*]] = llvm.xor %[[VAL_181]], %[[VAL_365]] : i32
    // CHECK:           %[[VAL_367:.*]] = llvm.mlir.constant(30 : i32) : i32
    // CHECK:           %[[VAL_368:.*]] = llvm.xor %[[CST_0]], %[[VAL_367]] : i32
    // CHECK:           %[[VAL_369:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_370:.*]] = llvm.xor %[[VAL_181]], %[[VAL_369]] : i32
    // CHECK:           %[[VAL_371:.*]] = llvm.mlir.constant(31 : i32) : i32
    // CHECK:           %[[VAL_372:.*]] = llvm.xor %[[CST_0]], %[[VAL_371]] : i32
    // CHECK:           %[[VAL_373:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_374:.*]] = llvm.xor %[[VAL_181]], %[[VAL_373]] : i32

    // COM: Offsets of rep [2, 1].
    // CHECK:           %[[VAL_375:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_376:.*]] = llvm.xor %[[CST_0]], %[[VAL_375]] : i32
    // CHECK:           %[[VAL_377:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_378:.*]] = llvm.xor %[[VAL_181]], %[[VAL_377]] : i32
    // CHECK:           %[[VAL_379:.*]] = llvm.mlir.constant(17 : i32) : i32
    // CHECK:           %[[VAL_380:.*]] = llvm.xor %[[CST_0]], %[[VAL_379]] : i32
    // CHECK:           %[[VAL_381:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_382:.*]] = llvm.xor %[[VAL_181]], %[[VAL_381]] : i32
    // CHECK:           %[[VAL_383:.*]] = llvm.mlir.constant(18 : i32) : i32
    // CHECK:           %[[VAL_384:.*]] = llvm.xor %[[CST_0]], %[[VAL_383]] : i32
    // CHECK:           %[[VAL_385:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_386:.*]] = llvm.xor %[[VAL_181]], %[[VAL_385]] : i32
    // CHECK:           %[[VAL_387:.*]] = llvm.mlir.constant(19 : i32) : i32
    // CHECK:           %[[VAL_388:.*]] = llvm.xor %[[CST_0]], %[[VAL_387]] : i32
    // CHECK:           %[[VAL_389:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_390:.*]] = llvm.xor %[[VAL_181]], %[[VAL_389]] : i32
    // CHECK:           %[[VAL_391:.*]] = llvm.mlir.constant(20 : i32) : i32
    // CHECK:           %[[VAL_392:.*]] = llvm.xor %[[CST_0]], %[[VAL_391]] : i32
    // CHECK:           %[[VAL_393:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_394:.*]] = llvm.xor %[[VAL_181]], %[[VAL_393]] : i32
    // CHECK:           %[[VAL_395:.*]] = llvm.mlir.constant(21 : i32) : i32
    // CHECK:           %[[VAL_396:.*]] = llvm.xor %[[CST_0]], %[[VAL_395]] : i32
    // CHECK:           %[[VAL_397:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_398:.*]] = llvm.xor %[[VAL_181]], %[[VAL_397]] : i32
    // CHECK:           %[[VAL_399:.*]] = llvm.mlir.constant(22 : i32) : i32
    // CHECK:           %[[VAL_400:.*]] = llvm.xor %[[CST_0]], %[[VAL_399]] : i32
    // CHECK:           %[[VAL_401:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_402:.*]] = llvm.xor %[[VAL_181]], %[[VAL_401]] : i32
    // CHECK:           %[[VAL_403:.*]] = llvm.mlir.constant(23 : i32) : i32
    // CHECK:           %[[VAL_404:.*]] = llvm.xor %[[CST_0]], %[[VAL_403]] : i32
    // CHECK:           %[[VAL_405:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_406:.*]] = llvm.xor %[[VAL_181]], %[[VAL_405]] : i32

    // COM: Offsets of rep [2, 2].
    // CHECK:           %[[VAL_407:.*]] = llvm.mlir.constant(24 : i32) : i32
    // CHECK:           %[[VAL_408:.*]] = llvm.xor %[[CST_0]], %[[VAL_407]] : i32
    // CHECK:           %[[VAL_409:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_410:.*]] = llvm.xor %[[VAL_181]], %[[VAL_409]] : i32
    // CHECK:           %[[VAL_411:.*]] = llvm.mlir.constant(25 : i32) : i32
    // CHECK:           %[[VAL_412:.*]] = llvm.xor %[[CST_0]], %[[VAL_411]] : i32
    // CHECK:           %[[VAL_413:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_414:.*]] = llvm.xor %[[VAL_181]], %[[VAL_413]] : i32
    // CHECK:           %[[VAL_415:.*]] = llvm.mlir.constant(26 : i32) : i32
    // CHECK:           %[[VAL_416:.*]] = llvm.xor %[[CST_0]], %[[VAL_415]] : i32
    // CHECK:           %[[VAL_417:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_418:.*]] = llvm.xor %[[VAL_181]], %[[VAL_417]] : i32
    // CHECK:           %[[VAL_419:.*]] = llvm.mlir.constant(27 : i32) : i32
    // CHECK:           %[[VAL_420:.*]] = llvm.xor %[[CST_0]], %[[VAL_419]] : i32
    // CHECK:           %[[VAL_421:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_422:.*]] = llvm.xor %[[VAL_181]], %[[VAL_421]] : i32
    // CHECK:           %[[VAL_423:.*]] = llvm.mlir.constant(28 : i32) : i32
    // CHECK:           %[[VAL_424:.*]] = llvm.xor %[[CST_0]], %[[VAL_423]] : i32
    // CHECK:           %[[VAL_425:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_426:.*]] = llvm.xor %[[VAL_181]], %[[VAL_425]] : i32
    // CHECK:           %[[VAL_427:.*]] = llvm.mlir.constant(29 : i32) : i32
    // CHECK:           %[[VAL_428:.*]] = llvm.xor %[[CST_0]], %[[VAL_427]] : i32
    // CHECK:           %[[VAL_429:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_430:.*]] = llvm.xor %[[VAL_181]], %[[VAL_429]] : i32
    // CHECK:           %[[VAL_431:.*]] = llvm.mlir.constant(30 : i32) : i32
    // CHECK:           %[[VAL_432:.*]] = llvm.xor %[[CST_0]], %[[VAL_431]] : i32
    // CHECK:           %[[VAL_433:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_434:.*]] = llvm.xor %[[VAL_181]], %[[VAL_433]] : i32
    // CHECK:           %[[VAL_435:.*]] = llvm.mlir.constant(31 : i32) : i32
    // CHECK:           %[[VAL_436:.*]] = llvm.xor %[[CST_0]], %[[VAL_435]] : i32
    // CHECK:           %[[VAL_437:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_438:.*]] = llvm.xor %[[VAL_181]], %[[VAL_437]] : i32
    tt.print " x: " {hex = false, isSigned = array<i32: 0>} : %cst : tensor<32x32xf16, #dot_operand_a>
    tt.return
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount=8, systolicDepth=8, executionSize = 8, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA=[1, 1], repCluster=[2, 2]}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#dpas, kWidth=2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {

  // CHECK-LABEL:   llvm.func spir_kernelcc @dot_layout_emit_offset()
  tt.func public @dot_layout_emit_offset() {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #dot_operand_b>
    // CHECK-COUNT-64:           {{.*}} = llvm.extractvalue {{.*}}
    // CHECK:           %[[VAL_142:.*]] = llvm.mlir.constant(0 : i32) : i32

    // COM: Base index of the dot layout.
    // CHECK:           %[[THREAD_ID_I64:.*]] = llvm.call spir_funccc @_Z12get_local_idj(%[[VAL_142]])
    // CHECK:           %[[THREAD_ID_I32:.*]] = llvm.trunc %[[THREAD_ID_I64]] : i64 to i32
    // CHECK:           %[[VAL_145:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[LANE_ID:.*]] = llvm.urem %[[THREAD_ID_I32]], %[[VAL_145]]  : i32
    // CHECK:           %[[WARP_ID:.*]] = llvm.udiv %[[THREAD_ID_I32]], %[[VAL_145]]  : i32
    // CHECK-COUNT-3:   %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_149:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[VAL_150:.*]] = llvm.and %[[LANE_ID]], %[[VAL_149]]  : i32
    // CHECK:           %[[VAL_151:.*]] = llvm.icmp "eq" %[[VAL_150]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_152:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[VAL_153:.*]] = llvm.select %[[VAL_151]], %[[CST_0]], %[[VAL_152]] : i1, i32
    // CHECK:           %[[VAL_154:.*]] = llvm.xor %[[CST_0]], %[[VAL_153]]  : i32
    // CHECK:           %[[VAL_155:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK:           %[[VAL_156:.*]] = llvm.and %[[LANE_ID]], %[[VAL_155]]  : i32
    // CHECK:           %[[VAL_157:.*]] = llvm.icmp "eq" %[[VAL_156]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_158:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK:           %[[VAL_159:.*]] = llvm.select %[[VAL_157]], %[[CST_0]], %[[VAL_158]] : i1, i32
    // CHECK:           %[[VAL_160:.*]] = llvm.xor %[[VAL_154]], %[[VAL_159]]  : i32
    // CHECK:           %[[VAL_161:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK:           %[[VAL_162:.*]] = llvm.and %[[LANE_ID]], %[[VAL_161]]  : i32
    // CHECK:           %[[VAL_163:.*]] = llvm.icmp "eq" %[[VAL_162]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_164:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK:           %[[VAL_165:.*]] = llvm.select %[[VAL_163]], %[[CST_0]], %[[VAL_164]] : i1, i32
    // CHECK:           %[[VAL_182:.*]] = llvm.xor %[[VAL_160]], %[[VAL_165]]  : i32
    // CHECK:           %[[VAL_167:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK:           %[[VAL_168:.*]] = llvm.and %[[LANE_ID]], %[[VAL_167]]  : i32
    // CHECK:           %[[VAL_169:.*]] = llvm.icmp "eq" %[[VAL_168]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_170:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK:           %[[VAL_171:.*]] = llvm.select %[[VAL_169]], %[[CST_0]], %[[VAL_170]] : i1, i32
    // CHECK:           %[[VAL_181:.*]] = llvm.xor %[[CST_0]], %[[VAL_171]]  : i32

    // COM: There are total [2, 4] repetitions of tensor shape [32, 32] per warp of B.
    // COM: The repetitions are clustered as [1, 2] for B operand. The repetitions orders are [0, 0], [0, 1], [1, 0], [1, 1], [0, 2], [0, 3], [1, 2], [1, 3]
    // COM: Offsets of rep [0, 0].
    // CHECK:           %[[VAL_183:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_184:.*]] = llvm.xor %[[VAL_181]], %[[VAL_183]] : i32
    // CHECK:           %[[VAL_185:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_186:.*]] = llvm.xor %[[VAL_182]], %[[VAL_185]] : i32
    // CHECK:           %[[VAL_187:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[VAL_188:.*]] = llvm.xor %[[VAL_181]], %[[VAL_187]] : i32
    // CHECK:           %[[VAL_189:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_190:.*]] = llvm.xor %[[VAL_182]], %[[VAL_189]] : i32
    // CHECK:           %[[VAL_191:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK:           %[[VAL_192:.*]] = llvm.xor %[[VAL_181]], %[[VAL_191]] : i32
    // CHECK:           %[[VAL_193:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_194:.*]] = llvm.xor %[[VAL_182]], %[[VAL_193]] : i32
    // CHECK:           %[[VAL_195:.*]] = llvm.mlir.constant(5 : i32) : i32
    // CHECK:           %[[VAL_196:.*]] = llvm.xor %[[VAL_181]], %[[VAL_195]] : i32
    // CHECK:           %[[VAL_197:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_198:.*]] = llvm.xor %[[VAL_182]], %[[VAL_197]] : i32
    // CHECK:           %[[VAL_199:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK:           %[[VAL_200:.*]] = llvm.xor %[[VAL_181]], %[[VAL_199]] : i32
    // CHECK:           %[[VAL_201:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_202:.*]] = llvm.xor %[[VAL_182]], %[[VAL_201]] : i32
    // CHECK:           %[[VAL_203:.*]] = llvm.mlir.constant(9 : i32) : i32
    // CHECK:           %[[VAL_204:.*]] = llvm.xor %[[VAL_181]], %[[VAL_203]] : i32
    // CHECK:           %[[VAL_205:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_206:.*]] = llvm.xor %[[VAL_182]], %[[VAL_205]] : i32
    // CHECK:           %[[VAL_207:.*]] = llvm.mlir.constant(12 : i32) : i32
    // CHECK:           %[[VAL_208:.*]] = llvm.xor %[[VAL_181]], %[[VAL_207]] : i32
    // CHECK:           %[[VAL_209:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_210:.*]] = llvm.xor %[[VAL_182]], %[[VAL_209]] : i32
    // CHECK:           %[[VAL_211:.*]] = llvm.mlir.constant(13 : i32) : i32
    // CHECK:           %[[VAL_212:.*]] = llvm.xor %[[VAL_181]], %[[VAL_211]] : i32
    // CHECK:           %[[VAL_213:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_214:.*]] = llvm.xor %[[VAL_182]], %[[VAL_213]] : i32

    // COM: Offsets of rep [0, 1].
    // CHECK:           %[[VAL_215:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_216:.*]] = llvm.xor %[[VAL_181]], %[[VAL_215]] : i32
    // CHECK:           %[[VAL_217:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK:           %[[VAL_218:.*]] = llvm.xor %[[VAL_182]], %[[VAL_217]] : i32
    // CHECK:           %[[VAL_219:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[VAL_220:.*]] = llvm.xor %[[VAL_181]], %[[VAL_219]] : i32
    // CHECK:           %[[VAL_221:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK:           %[[VAL_222:.*]] = llvm.xor %[[VAL_182]], %[[VAL_221]] : i32
    // CHECK:           %[[VAL_223:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK:           %[[VAL_224:.*]] = llvm.xor %[[VAL_181]], %[[VAL_223]] : i32
    // CHECK:           %[[VAL_225:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK:           %[[VAL_226:.*]] = llvm.xor %[[VAL_182]], %[[VAL_225]] : i32
    // CHECK:           %[[VAL_227:.*]] = llvm.mlir.constant(5 : i32) : i32
    // CHECK:           %[[VAL_228:.*]] = llvm.xor %[[VAL_181]], %[[VAL_227]] : i32
    // CHECK:           %[[VAL_229:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK:           %[[VAL_230:.*]] = llvm.xor %[[VAL_182]], %[[VAL_229]] : i32
    // CHECK:           %[[VAL_231:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK:           %[[VAL_232:.*]] = llvm.xor %[[VAL_181]], %[[VAL_231]] : i32
    // CHECK:           %[[VAL_233:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK:           %[[VAL_234:.*]] = llvm.xor %[[VAL_182]], %[[VAL_233]] : i32
    // CHECK:           %[[VAL_235:.*]] = llvm.mlir.constant(9 : i32) : i32
    // CHECK:           %[[VAL_236:.*]] = llvm.xor %[[VAL_181]], %[[VAL_235]] : i32
    // CHECK:           %[[VAL_237:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK:           %[[VAL_238:.*]] = llvm.xor %[[VAL_182]], %[[VAL_237]] : i32
    // CHECK:           %[[VAL_239:.*]] = llvm.mlir.constant(12 : i32) : i32
    // CHECK:           %[[VAL_240:.*]] = llvm.xor %[[VAL_181]], %[[VAL_239]] : i32
    // CHECK:           %[[VAL_241:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK:           %[[VAL_242:.*]] = llvm.xor %[[VAL_182]], %[[VAL_241]] : i32
    // CHECK:           %[[VAL_243:.*]] = llvm.mlir.constant(13 : i32) : i32
    // CHECK:           %[[VAL_244:.*]] = llvm.xor %[[VAL_181]], %[[VAL_243]] : i32
    // CHECK:           %[[VAL_245:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK:           %[[VAL_246:.*]] = llvm.xor %[[VAL_182]], %[[VAL_245]] : i32

    // COM: Offsets of rep [1, 0].
    // CHECK:           %[[VAL_247:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_248:.*]] = llvm.xor %[[VAL_181]], %[[VAL_247]] : i32
    // CHECK:           %[[VAL_249:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_250:.*]] = llvm.xor %[[VAL_182]], %[[VAL_249]] : i32
    // CHECK:           %[[VAL_251:.*]] = llvm.mlir.constant(17 : i32) : i32
    // CHECK:           %[[VAL_252:.*]] = llvm.xor %[[VAL_181]], %[[VAL_251]] : i32
    // CHECK:           %[[VAL_253:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_254:.*]] = llvm.xor %[[VAL_182]], %[[VAL_253]] : i32
    // CHECK:           %[[VAL_255:.*]] = llvm.mlir.constant(20 : i32) : i32
    // CHECK:           %[[VAL_256:.*]] = llvm.xor %[[VAL_181]], %[[VAL_255]] : i32
    // CHECK:           %[[VAL_257:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_258:.*]] = llvm.xor %[[VAL_182]], %[[VAL_257]] : i32
    // CHECK:           %[[VAL_259:.*]] = llvm.mlir.constant(21 : i32) : i32
    // CHECK:           %[[VAL_260:.*]] = llvm.xor %[[VAL_181]], %[[VAL_259]] : i32
    // CHECK:           %[[VAL_261:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_262:.*]] = llvm.xor %[[VAL_182]], %[[VAL_261]] : i32
    // CHECK:           %[[VAL_263:.*]] = llvm.mlir.constant(24 : i32) : i32
    // CHECK:           %[[VAL_264:.*]] = llvm.xor %[[VAL_181]], %[[VAL_263]] : i32
    // CHECK:           %[[VAL_265:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_266:.*]] = llvm.xor %[[VAL_182]], %[[VAL_265]] : i32
    // CHECK:           %[[VAL_267:.*]] = llvm.mlir.constant(25 : i32) : i32
    // CHECK:           %[[VAL_268:.*]] = llvm.xor %[[VAL_181]], %[[VAL_267]] : i32
    // CHECK:           %[[VAL_269:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_270:.*]] = llvm.xor %[[VAL_182]], %[[VAL_269]] : i32
    // CHECK:           %[[VAL_271:.*]] = llvm.mlir.constant(28 : i32) : i32
    // CHECK:           %[[VAL_272:.*]] = llvm.xor %[[VAL_181]], %[[VAL_271]] : i32
    // CHECK:           %[[VAL_273:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_274:.*]] = llvm.xor %[[VAL_182]], %[[VAL_273]] : i32
    // CHECK:           %[[VAL_275:.*]] = llvm.mlir.constant(29 : i32) : i32
    // CHECK:           %[[VAL_276:.*]] = llvm.xor %[[VAL_181]], %[[VAL_275]] : i32
    // CHECK:           %[[VAL_277:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_278:.*]] = llvm.xor %[[VAL_182]], %[[VAL_277]] : i32

    // COM: Offsets of rep [1, 1].
    // CHECK:           %[[VAL_279:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_280:.*]] = llvm.xor %[[VAL_181]], %[[VAL_279]] : i32
    // CHECK:           %[[VAL_281:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK:           %[[VAL_282:.*]] = llvm.xor %[[VAL_182]], %[[VAL_281]] : i32
    // CHECK:           %[[VAL_283:.*]] = llvm.mlir.constant(17 : i32) : i32
    // CHECK:           %[[VAL_284:.*]] = llvm.xor %[[VAL_181]], %[[VAL_283]] : i32
    // CHECK:           %[[VAL_285:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK:           %[[VAL_286:.*]] = llvm.xor %[[VAL_182]], %[[VAL_285]] : i32
    // CHECK:           %[[VAL_287:.*]] = llvm.mlir.constant(20 : i32) : i32
    // CHECK:           %[[VAL_288:.*]] = llvm.xor %[[VAL_181]], %[[VAL_287]] : i32
    // CHECK:           %[[VAL_289:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK:           %[[VAL_290:.*]] = llvm.xor %[[VAL_182]], %[[VAL_289]] : i32
    // CHECK:           %[[VAL_291:.*]] = llvm.mlir.constant(21 : i32) : i32
    // CHECK:           %[[VAL_292:.*]] = llvm.xor %[[VAL_181]], %[[VAL_291]] : i32
    // CHECK:           %[[VAL_293:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK:           %[[VAL_294:.*]] = llvm.xor %[[VAL_182]], %[[VAL_293]] : i32
    // CHECK:           %[[VAL_295:.*]] = llvm.mlir.constant(24 : i32) : i32
    // CHECK:           %[[VAL_296:.*]] = llvm.xor %[[VAL_181]], %[[VAL_295]] : i32
    // CHECK:           %[[VAL_297:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK:           %[[VAL_298:.*]] = llvm.xor %[[VAL_182]], %[[VAL_297]] : i32
    // CHECK:           %[[VAL_299:.*]] = llvm.mlir.constant(25 : i32) : i32
    // CHECK:           %[[VAL_300:.*]] = llvm.xor %[[VAL_181]], %[[VAL_299]] : i32
    // CHECK:           %[[VAL_301:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK:           %[[VAL_302:.*]] = llvm.xor %[[VAL_182]], %[[VAL_301]] : i32
    // CHECK:           %[[VAL_303:.*]] = llvm.mlir.constant(28 : i32) : i32
    // CHECK:           %[[VAL_304:.*]] = llvm.xor %[[VAL_181]], %[[VAL_303]] : i32
    // CHECK:           %[[VAL_305:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK:           %[[VAL_306:.*]] = llvm.xor %[[VAL_182]], %[[VAL_305]] : i32
    // CHECK:           %[[VAL_307:.*]] = llvm.mlir.constant(29 : i32) : i32
    // CHECK:           %[[VAL_308:.*]] = llvm.xor %[[VAL_181]], %[[VAL_307]] : i32
    // CHECK:           %[[VAL_309:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK:           %[[VAL_310:.*]] = llvm.xor %[[VAL_182]], %[[VAL_309]] : i32

    // COM: Offsets of rep [0, 2].
    // CHECK:           %[[VAL_311:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_312:.*]] = llvm.xor %[[VAL_181]], %[[VAL_311]] : i32
    // CHECK:           %[[VAL_313:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_314:.*]] = llvm.xor %[[VAL_182]], %[[VAL_313]] : i32
    // CHECK:           %[[VAL_315:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[VAL_316:.*]] = llvm.xor %[[VAL_181]], %[[VAL_315]] : i32
    // CHECK:           %[[VAL_317:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_318:.*]] = llvm.xor %[[VAL_182]], %[[VAL_317]] : i32
    // CHECK:           %[[VAL_319:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK:           %[[VAL_320:.*]] = llvm.xor %[[VAL_181]], %[[VAL_319]] : i32
    // CHECK:           %[[VAL_321:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_322:.*]] = llvm.xor %[[VAL_182]], %[[VAL_321]] : i32
    // CHECK:           %[[VAL_323:.*]] = llvm.mlir.constant(5 : i32) : i32
    // CHECK:           %[[VAL_324:.*]] = llvm.xor %[[VAL_181]], %[[VAL_323]] : i32
    // CHECK:           %[[VAL_325:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_326:.*]] = llvm.xor %[[VAL_182]], %[[VAL_325]] : i32
    // CHECK:           %[[VAL_327:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK:           %[[VAL_328:.*]] = llvm.xor %[[VAL_181]], %[[VAL_327]] : i32
    // CHECK:           %[[VAL_329:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_330:.*]] = llvm.xor %[[VAL_182]], %[[VAL_329]] : i32
    // CHECK:           %[[VAL_331:.*]] = llvm.mlir.constant(9 : i32) : i32
    // CHECK:           %[[VAL_332:.*]] = llvm.xor %[[VAL_181]], %[[VAL_331]] : i32
    // CHECK:           %[[VAL_333:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_334:.*]] = llvm.xor %[[VAL_182]], %[[VAL_333]] : i32
    // CHECK:           %[[VAL_335:.*]] = llvm.mlir.constant(12 : i32) : i32
    // CHECK:           %[[VAL_336:.*]] = llvm.xor %[[VAL_181]], %[[VAL_335]] : i32
    // CHECK:           %[[VAL_337:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_338:.*]] = llvm.xor %[[VAL_182]], %[[VAL_337]] : i32
    // CHECK:           %[[VAL_339:.*]] = llvm.mlir.constant(13 : i32) : i32
    // CHECK:           %[[VAL_340:.*]] = llvm.xor %[[VAL_181]], %[[VAL_339]] : i32
    // CHECK:           %[[VAL_341:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_342:.*]] = llvm.xor %[[VAL_182]], %[[VAL_341]] : i32

    // COM: Offsets of rep [0, 3].
    // CHECK:           %[[VAL_343:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_344:.*]] = llvm.xor %[[VAL_181]], %[[VAL_343]] : i32
    // CHECK:           %[[VAL_345:.*]] = llvm.mlir.constant(24 : i32) : i32
    // CHECK:           %[[VAL_346:.*]] = llvm.xor %[[VAL_182]], %[[VAL_345]] : i32
    // CHECK:           %[[VAL_347:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[VAL_348:.*]] = llvm.xor %[[VAL_181]], %[[VAL_347]] : i32
    // CHECK:           %[[VAL_349:.*]] = llvm.mlir.constant(24 : i32) : i32
    // CHECK:           %[[VAL_350:.*]] = llvm.xor %[[VAL_182]], %[[VAL_349]] : i32
    // CHECK:           %[[VAL_351:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK:           %[[VAL_352:.*]] = llvm.xor %[[VAL_181]], %[[VAL_351]] : i32
    // CHECK:           %[[VAL_353:.*]] = llvm.mlir.constant(24 : i32) : i32
    // CHECK:           %[[VAL_354:.*]] = llvm.xor %[[VAL_182]], %[[VAL_353]] : i32
    // CHECK:           %[[VAL_355:.*]] = llvm.mlir.constant(5 : i32) : i32
    // CHECK:           %[[VAL_356:.*]] = llvm.xor %[[VAL_181]], %[[VAL_355]] : i32
    // CHECK:           %[[VAL_357:.*]] = llvm.mlir.constant(24 : i32) : i32
    // CHECK:           %[[VAL_358:.*]] = llvm.xor %[[VAL_182]], %[[VAL_357]] : i32
    // CHECK:           %[[VAL_359:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK:           %[[VAL_360:.*]] = llvm.xor %[[VAL_181]], %[[VAL_359]] : i32
    // CHECK:           %[[VAL_361:.*]] = llvm.mlir.constant(24 : i32) : i32
    // CHECK:           %[[VAL_362:.*]] = llvm.xor %[[VAL_182]], %[[VAL_361]] : i32
    // CHECK:           %[[VAL_363:.*]] = llvm.mlir.constant(9 : i32) : i32
    // CHECK:           %[[VAL_364:.*]] = llvm.xor %[[VAL_181]], %[[VAL_363]] : i32
    // CHECK:           %[[VAL_365:.*]] = llvm.mlir.constant(24 : i32) : i32
    // CHECK:           %[[VAL_366:.*]] = llvm.xor %[[VAL_182]], %[[VAL_365]] : i32
    // CHECK:           %[[VAL_367:.*]] = llvm.mlir.constant(12 : i32) : i32
    // CHECK:           %[[VAL_368:.*]] = llvm.xor %[[VAL_181]], %[[VAL_367]] : i32
    // CHECK:           %[[VAL_369:.*]] = llvm.mlir.constant(24 : i32) : i32
    // CHECK:           %[[VAL_370:.*]] = llvm.xor %[[VAL_182]], %[[VAL_369]] : i32
    // CHECK:           %[[VAL_371:.*]] = llvm.mlir.constant(13 : i32) : i32
    // CHECK:           %[[VAL_372:.*]] = llvm.xor %[[VAL_181]], %[[VAL_371]] : i32
    // CHECK:           %[[VAL_373:.*]] = llvm.mlir.constant(24 : i32) : i32
    // CHECK:           %[[VAL_374:.*]] = llvm.xor %[[VAL_182]], %[[VAL_373]] : i32

    // COM: Offsets of rep [1, 2].
    // CHECK:           %[[VAL_375:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_376:.*]] = llvm.xor %[[VAL_181]], %[[VAL_375]] : i32
    // CHECK:           %[[VAL_377:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_378:.*]] = llvm.xor %[[VAL_182]], %[[VAL_377]] : i32
    // CHECK:           %[[VAL_379:.*]] = llvm.mlir.constant(17 : i32) : i32
    // CHECK:           %[[VAL_380:.*]] = llvm.xor %[[VAL_181]], %[[VAL_379]] : i32
    // CHECK:           %[[VAL_381:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_382:.*]] = llvm.xor %[[VAL_182]], %[[VAL_381]] : i32
    // CHECK:           %[[VAL_383:.*]] = llvm.mlir.constant(20 : i32) : i32
    // CHECK:           %[[VAL_384:.*]] = llvm.xor %[[VAL_181]], %[[VAL_383]] : i32
    // CHECK:           %[[VAL_385:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_386:.*]] = llvm.xor %[[VAL_182]], %[[VAL_385]] : i32
    // CHECK:           %[[VAL_387:.*]] = llvm.mlir.constant(21 : i32) : i32
    // CHECK:           %[[VAL_388:.*]] = llvm.xor %[[VAL_181]], %[[VAL_387]] : i32
    // CHECK:           %[[VAL_389:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_390:.*]] = llvm.xor %[[VAL_182]], %[[VAL_389]] : i32
    // CHECK:           %[[VAL_391:.*]] = llvm.mlir.constant(24 : i32) : i32
    // CHECK:           %[[VAL_392:.*]] = llvm.xor %[[VAL_181]], %[[VAL_391]] : i32
    // CHECK:           %[[VAL_393:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_394:.*]] = llvm.xor %[[VAL_182]], %[[VAL_393]] : i32
    // CHECK:           %[[VAL_395:.*]] = llvm.mlir.constant(25 : i32) : i32
    // CHECK:           %[[VAL_396:.*]] = llvm.xor %[[VAL_181]], %[[VAL_395]] : i32
    // CHECK:           %[[VAL_397:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_398:.*]] = llvm.xor %[[VAL_182]], %[[VAL_397]] : i32
    // CHECK:           %[[VAL_399:.*]] = llvm.mlir.constant(28 : i32) : i32
    // CHECK:           %[[VAL_400:.*]] = llvm.xor %[[VAL_181]], %[[VAL_399]] : i32
    // CHECK:           %[[VAL_401:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_402:.*]] = llvm.xor %[[VAL_182]], %[[VAL_401]] : i32
    // CHECK:           %[[VAL_403:.*]] = llvm.mlir.constant(29 : i32) : i32
    // CHECK:           %[[VAL_404:.*]] = llvm.xor %[[VAL_181]], %[[VAL_403]] : i32
    // CHECK:           %[[VAL_405:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_406:.*]] = llvm.xor %[[VAL_182]], %[[VAL_405]] : i32

    // COM: Offsets of rep [1, 3].
    // CHECK:           %[[VAL_407:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_408:.*]] = llvm.xor %[[VAL_181]], %[[VAL_407]] : i32
    // CHECK:           %[[VAL_409:.*]] = llvm.mlir.constant(24 : i32) : i32
    // CHECK:           %[[VAL_410:.*]] = llvm.xor %[[VAL_182]], %[[VAL_409]] : i32
    // CHECK:           %[[VAL_411:.*]] = llvm.mlir.constant(17 : i32) : i32
    // CHECK:           %[[VAL_412:.*]] = llvm.xor %[[VAL_181]], %[[VAL_411]] : i32
    // CHECK:           %[[VAL_413:.*]] = llvm.mlir.constant(24 : i32) : i32
    // CHECK:           %[[VAL_414:.*]] = llvm.xor %[[VAL_182]], %[[VAL_413]] : i32
    // CHECK:           %[[VAL_415:.*]] = llvm.mlir.constant(20 : i32) : i32
    // CHECK:           %[[VAL_416:.*]] = llvm.xor %[[VAL_181]], %[[VAL_415]] : i32
    // CHECK:           %[[VAL_417:.*]] = llvm.mlir.constant(24 : i32) : i32
    // CHECK:           %[[VAL_418:.*]] = llvm.xor %[[VAL_182]], %[[VAL_417]] : i32
    // CHECK:           %[[VAL_419:.*]] = llvm.mlir.constant(21 : i32) : i32
    // CHECK:           %[[VAL_420:.*]] = llvm.xor %[[VAL_181]], %[[VAL_419]] : i32
    // CHECK:           %[[VAL_421:.*]] = llvm.mlir.constant(24 : i32) : i32
    // CHECK:           %[[VAL_422:.*]] = llvm.xor %[[VAL_182]], %[[VAL_421]] : i32
    // CHECK:           %[[VAL_423:.*]] = llvm.mlir.constant(24 : i32) : i32
    // CHECK:           %[[VAL_424:.*]] = llvm.xor %[[VAL_181]], %[[VAL_423]] : i32
    // CHECK:           %[[VAL_425:.*]] = llvm.mlir.constant(24 : i32) : i32
    // CHECK:           %[[VAL_426:.*]] = llvm.xor %[[VAL_182]], %[[VAL_425]] : i32
    // CHECK:           %[[VAL_427:.*]] = llvm.mlir.constant(25 : i32) : i32
    // CHECK:           %[[VAL_428:.*]] = llvm.xor %[[VAL_181]], %[[VAL_427]] : i32
    // CHECK:           %[[VAL_429:.*]] = llvm.mlir.constant(24 : i32) : i32
    // CHECK:           %[[VAL_430:.*]] = llvm.xor %[[VAL_182]], %[[VAL_429]] : i32
    // CHECK:           %[[VAL_431:.*]] = llvm.mlir.constant(28 : i32) : i32
    // CHECK:           %[[VAL_432:.*]] = llvm.xor %[[VAL_181]], %[[VAL_431]] : i32
    // CHECK:           %[[VAL_433:.*]] = llvm.mlir.constant(24 : i32) : i32
    // CHECK:           %[[VAL_434:.*]] = llvm.xor %[[VAL_182]], %[[VAL_433]] : i32
    // CHECK:           %[[VAL_435:.*]] = llvm.mlir.constant(29 : i32) : i32
    // CHECK:           %[[VAL_436:.*]] = llvm.xor %[[VAL_181]], %[[VAL_435]] : i32
    // CHECK:           %[[VAL_437:.*]] = llvm.mlir.constant(24 : i32) : i32
    // CHECK:           %[[VAL_438:.*]] = llvm.xor %[[VAL_182]], %[[VAL_437]] : i32
    tt.print " x: " {hex = false, isSigned = array<i32: 0>} : %cst : tensor<32x32xf16, #dot_operand_b>
    tt.return
  }
}
