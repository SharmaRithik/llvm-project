// RUN: mlir-opt --split-input-file --verify-diagnostics \
// RUN:   --spirv-webgpu-prepare --cse %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.UMulExtended
//===----------------------------------------------------------------------===//

spirv.module Logical GLSL450 {

// CHECK-LABEL: func @umul_extended_i32
// CHECK-SAME:       ([[ARG0:%.+]]: i32, [[ARG1:%.+]]: i32)
// CHECK-DAG:        [[CSTMASK:%.+]] = spirv.Constant 65535 : i32
// CHECK-DAG:        [[CST16:%.+]]   = spirv.Constant 16 : i32
// CHECK-NEXT:       [[LHSLOW:%.+]]  = spirv.BitwiseAnd [[ARG0]], [[CSTMASK]] : i32
// CHECK-NEXT:       [[LHSHI:%.+]]   = spirv.ShiftRightLogical [[ARG0]], [[CST16]] : i32
// CHECK-NEXT:       [[RHSLOW:%.+]]  = spirv.BitwiseAnd [[ARG1]], [[CSTMASK]] : i32
// CHECK-NEXT:       [[RHSHI:%.+]]   = spirv.ShiftRightLogical [[ARG1]], [[CST16]] : i32
// CHECK-DAG:                          spirv.IMul [[LHSLOW]], [[RHSLOW]]
// CHECK-DAG:                          spirv.IMul [[LHSLOW]], [[RHSHI]]
// CHECK-DAG:                          spirv.IMul [[LHSHI]],  [[RHSLOW]]
// CHECK-DAG:                          spirv.IMul [[LHSHI]],  [[RHSHI]]
// CHECK-DAG:                          spirv.IAdd
// CHECK-DAG:                          spirv.IAdd
// CHECK-DAG:                          spirv.IAdd
// CHECK-DAG:                          spirv.IAdd
// CHECK:                              spirv.ShiftLeftLogical {{%.+}}, [[CST16]] : i32
// CHECK:                              spirv.BitwiseOr
// CHECK:                              spirv.ShiftLeftLogical {{%.+}}, [[CST16]] : i32
// CHECK:                              spirv.BitwiseOr
// CHECK:            [[RES:%.+]]     = spirv.CompositeConstruct [[RESLO:%.+]], [[RESHI:%.+]] : (i32, i32) -> !spirv.struct<(i32, i32)>
// CHECK-NEXT:       spirv.ReturnValue [[RES]] : !spirv.struct<(i32, i32)>
spirv.func @umul_extended_i32(%arg0 : i32, %arg1 : i32) -> !spirv.struct<(i32, i32)> "None" {
  %0 = spirv.UMulExtended %arg0, %arg1 : !spirv.struct<(i32, i32)>
  spirv.ReturnValue %0 : !spirv.struct<(i32, i32)>
}

// CHECK-LABEL: func @umul_extended_vector_i32
// CHECK-SAME:       ([[ARG0:%.+]]: vector<3xi32>, [[ARG1:%.+]]: vector<3xi32>)
// CHECK-DAG:        [[CSTMASK:%.+]] = spirv.Constant dense<65535> : vector<3xi32>
// CHECK-DAG:        [[CST16:%.+]]   = spirv.Constant dense<16> : vector<3xi32>
// CHECK-NEXT:       [[LHSLOW:%.+]]  = spirv.BitwiseAnd [[ARG0]], [[CSTMASK]] : vector<3xi32>
// CHECK-NEXT:       [[LHSHI:%.+]]   = spirv.ShiftRightLogical [[ARG0]], [[CST16]] : vector<3xi32>
// CHECK-NEXT:       [[RHSLOW:%.+]]  = spirv.BitwiseAnd [[ARG1]], [[CSTMASK]] : vector<3xi32>
// CHECK-NEXT:       [[RHSHI:%.+]]   = spirv.ShiftRightLogical [[ARG1]], [[CST16]] : vector<3xi32>
// CHECK-DAG:                          spirv.IMul [[LHSLOW]], [[RHSLOW]]
// CHECK-DAG:                          spirv.IMul [[LHSLOW]], [[RHSHI]]
// CHECK-DAG:                          spirv.IMul [[LHSHI]],  [[RHSLOW]]
// CHECK-DAG:                          spirv.IMul [[LHSHI]],  [[RHSHI]]
// CHECK-DAG:                          spirv.IAdd
// CHECK-DAG:                          spirv.IAdd
// CHECK-DAG:                          spirv.IAdd
// CHECK-DAG:                          spirv.IAdd
// CHECK:                              spirv.ShiftLeftLogical {{%.+}}, [[CST16]]
// CHECK:                              spirv.BitwiseOr
// CHECK:                              spirv.ShiftLeftLogical {{%.+}}, [[CST16]]
// CHECK:                              spirv.BitwiseOr
// CHECK-NEXT:       [[RES:%.+]]     = spirv.CompositeConstruct [[RESLOW:%.+]], [[RESHI:%.+]]
// CHECK-NEXT:       spirv.ReturnValue [[RES]] : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
spirv.func @umul_extended_vector_i32(%arg0 : vector<3xi32>, %arg1 : vector<3xi32>)
  -> !spirv.struct<(vector<3xi32>, vector<3xi32>)> "None" {
  %0 = spirv.UMulExtended %arg0, %arg1 : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
  spirv.ReturnValue %0 : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
}

// CHECK-LABEL: func @umul_extended_i16
// CHECK-NEXT:       spirv.UMulExtended
// CHECK-NEXT:       spirv.ReturnValue
spirv.func @umul_extended_i16(%arg : i16) -> !spirv.struct<(i16, i16)> "None" {
  %0 = spirv.UMulExtended %arg, %arg : !spirv.struct<(i16, i16)>
  spirv.ReturnValue %0 : !spirv.struct<(i16, i16)>
}

//===----------------------------------------------------------------------===//
// spirv.SMulExtended
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @smul_extended_i32
// CHECK-SAME:       ([[ARG0:%.+]]: i32, [[ARG1:%.+]]: i32)
// CHECK-DAG:        [[CSTMASK:%.+]] = spirv.Constant 65535 : i32
// CHECK-DAG:        [[CST16:%.+]]   = spirv.Constant 16 : i32
// CHECK-DAG:        [[LHSLOW:%.+]]  = spirv.BitwiseAnd [[ARG0]], [[CSTMASK]] : i32
// CHECK-DAG:        [[LHSHI:%.+]]   = spirv.ShiftRightLogical [[ARG0]], [[CST16]] : i32
// The arithmetic shift that extracts the sign is itself expanded (see the
// ShiftRightArithmetic tests below); after CSE its false branch is LHSHI.
// CHECK-DAG:        [[LHSSIGN:%.+]] = spirv.Select {{%.+}}, {{%.+}}, [[LHSHI]] : i1, i32
// CHECK-DAG:        [[LHSEXT:%.+]]  = spirv.ShiftRightLogical [[LHSSIGN]], [[CST16]] : i32
// CHECK-DAG:        [[RHSLOW:%.+]]  = spirv.BitwiseAnd [[ARG1]], [[CSTMASK]] : i32
// CHECK-DAG:        [[RHSHI:%.+]]   = spirv.ShiftRightLogical [[ARG1]], [[CST16]] : i32
// CHECK-DAG:        [[RHSSIGN:%.+]] = spirv.Select {{%.+}}, {{%.+}}, [[RHSHI]] : i1, i32
// CHECK-DAG:        [[RHSEXT:%.+]]  = spirv.ShiftRightLogical [[RHSSIGN]], [[CST16]] : i32
// CHECK-DAG:                          spirv.IMul [[LHSLOW]], [[RHSLOW]]
// CHECK-DAG:                          spirv.IMul [[LHSLOW]], [[RHSHI]]
// CHECK-DAG:                          spirv.IMul [[LHSLOW]], [[RHSEXT]]
// CHECK-DAG:                          spirv.IMul [[LHSHI]],  [[RHSLOW]]
// CHECK-DAG:                          spirv.IMul [[LHSHI]],  [[RHSHI]]
// CHECK-DAG:                          spirv.IMul [[LHSHI]],  [[RHSEXT]]
// CHECK-DAG:                          spirv.IMul [[LHSEXT]], [[RHSLOW]]
// CHECK-DAG:                          spirv.IMul [[LHSEXT]], [[RHSHI]]
// CHECK:                              spirv.ShiftLeftLogical {{%.+}}, [[CST16]] : i32
// CHECK:                              spirv.BitwiseOr
// CHECK:                              spirv.ShiftLeftLogical {{%.+}}, [[CST16]] : i32
// CHECK:                              spirv.BitwiseOr
// CHECK:            [[RES:%.+]]     = spirv.CompositeConstruct [[RESLO:%.+]], [[RESHI:%.+]] : (i32, i32) -> !spirv.struct<(i32, i32)>
// CHECK-NEXT:       spirv.ReturnValue [[RES]] : !spirv.struct<(i32, i32)>
spirv.func @smul_extended_i32(%arg0 : i32, %arg1 : i32) -> !spirv.struct<(i32, i32)> "None" {
  %0 = spirv.SMulExtended %arg0, %arg1 : !spirv.struct<(i32, i32)>
  spirv.ReturnValue %0 : !spirv.struct<(i32, i32)>
}

// CHECK-LABEL: func @smul_extended_vector_i32
// CHECK-SAME:       ([[ARG0:%.+]]: vector<3xi32>, [[ARG1:%.+]]: vector<3xi32>)
// CHECK-DAG:        [[CSTMASK:%.+]] = spirv.Constant dense<65535> : vector<3xi32>
// CHECK-DAG:        [[CST16:%.+]]   = spirv.Constant dense<16> : vector<3xi32>
// CHECK-DAG:        [[LHSLOW:%.+]]  = spirv.BitwiseAnd [[ARG0]], [[CSTMASK]] : vector<3xi32>
// CHECK-DAG:        [[LHSHI:%.+]]   = spirv.ShiftRightLogical [[ARG0]], [[CST16]] : vector<3xi32>
// The arithmetic shift that extracts the sign is itself expanded (see the
// ShiftRightArithmetic tests below); after CSE its false branch is LHSHI.
// CHECK-DAG:        [[LHSSIGN:%.+]] = spirv.Select {{%.+}}, {{%.+}}, [[LHSHI]] : vector<3xi1>, vector<3xi32>
// CHECK-DAG:        [[LHSEXT:%.+]]  = spirv.ShiftRightLogical [[LHSSIGN]], [[CST16]] : vector<3xi32>
// CHECK-DAG:        [[RHSLOW:%.+]]  = spirv.BitwiseAnd [[ARG1]], [[CSTMASK]] : vector<3xi32>
// CHECK-DAG:        [[RHSHI:%.+]]   = spirv.ShiftRightLogical [[ARG1]], [[CST16]] : vector<3xi32>
// CHECK-DAG:        [[RHSSIGN:%.+]] = spirv.Select {{%.+}}, {{%.+}}, [[RHSHI]] : vector<3xi1>, vector<3xi32>
// CHECK-DAG:        [[RHSEXT:%.+]]  = spirv.ShiftRightLogical [[RHSSIGN]], [[CST16]] : vector<3xi32>
// CHECK-DAG:                          spirv.IMul [[LHSLOW]], [[RHSLOW]]
// CHECK-DAG:                          spirv.IMul [[LHSLOW]], [[RHSHI]]
// CHECK-DAG:                          spirv.IMul [[LHSLOW]], [[RHSEXT]]
// CHECK-DAG:                          spirv.IMul [[LHSHI]],  [[RHSLOW]]
// CHECK-DAG:                          spirv.IMul [[LHSHI]],  [[RHSHI]]
// CHECK-DAG:                          spirv.IMul [[LHSHI]],  [[RHSEXT]]
// CHECK-DAG:                          spirv.IMul [[LHSEXT]], [[RHSLOW]]
// CHECK-DAG:                          spirv.IMul [[LHSEXT]], [[RHSHI]]
// CHECK:                              spirv.ShiftLeftLogical {{%.+}}, [[CST16]]
// CHECK:                              spirv.BitwiseOr
// CHECK:                              spirv.ShiftLeftLogical {{%.+}}, [[CST16]]
// CHECK:                              spirv.BitwiseOr
// CHECK-NEXT:       [[RES:%.+]]     = spirv.CompositeConstruct [[RESLOW:%.+]], [[RESHI:%.+]]
// CHECK-NEXT:       spirv.ReturnValue [[RES]] : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
spirv.func @smul_extended_vector_i32(%arg0 : vector<3xi32>, %arg1 : vector<3xi32>)
  -> !spirv.struct<(vector<3xi32>, vector<3xi32>)> "None" {
  %0 = spirv.SMulExtended %arg0, %arg1 : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
  spirv.ReturnValue %0 : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
}

// CHECK-LABEL: func @smul_extended_i16
// CHECK-NEXT:       spirv.SMulExtended
// CHECK-NEXT:       spirv.ReturnValue
spirv.func @smul_extended_i16(%arg : i16) -> !spirv.struct<(i16, i16)> "None" {
  %0 = spirv.SMulExtended %arg, %arg : !spirv.struct<(i16, i16)>
  spirv.ReturnValue %0 : !spirv.struct<(i16, i16)>
}

// CHECK-LABEL: func @iaddcarry_i32
// CHECK-SAME:       ([[A:%.+]]: i32, [[B:%.+]]: i32)
// CHECK-NEXT:       [[ONE:%.+]]    = spirv.Constant 1 : i32
// CHECK-NEXT:       [[ZERO:%.+]]   = spirv.Constant 0 : i32
// CHECK-NEXT:       [[OUT:%.+]]    = spirv.IAdd [[A]], [[B]]
// CHECK-NEXT:       [[CMP:%.+]]    = spirv.ULessThan [[OUT]], [[A]]
// CHECK-NEXT:       [[CARRY:%.+]]  = spirv.Select [[CMP]], [[ONE]], [[ZERO]]
// CHECK-NEXT:       [[RES:%.+]]     = spirv.CompositeConstruct [[OUT]], [[CARRY]] : (i32, i32) -> !spirv.struct<(i32, i32)>
// CHECK-NEXT:       spirv.ReturnValue [[RES]] : !spirv.struct<(i32, i32)>
spirv.func @iaddcarry_i32(%a : i32, %b : i32) -> !spirv.struct<(i32, i32)> "None" {
  %0 = spirv.IAddCarry %a, %b : !spirv.struct<(i32, i32)>
  spirv.ReturnValue %0 : !spirv.struct<(i32, i32)>
}

// CHECK-LABEL: func @iaddcarry_vector_i32
// CHECK-SAME:       ([[A:%.+]]: vector<3xi32>, [[B:%.+]]: vector<3xi32>)
// CHECK-NEXT:       [[ONE:%.+]]    = spirv.Constant dense<1> : vector<3xi32>
// CHECK-NEXT:       [[ZERO:%.+]]   = spirv.Constant dense<0> : vector<3xi32>
// CHECK-NEXT:       [[OUT:%.+]]    = spirv.IAdd [[A]], [[B]]
// CHECK-NEXT:       [[CMP:%.+]]    = spirv.ULessThan [[OUT]], [[A]]
// CHECK-NEXT:       [[CARRY:%.+]]  = spirv.Select [[CMP]], [[ONE]], [[ZERO]]
// CHECK-NEXT:       [[RES:%.+]]    = spirv.CompositeConstruct [[OUT]], [[CARRY]] : (vector<3xi32>, vector<3xi32>) -> !spirv.struct<(vector<3xi32>, vector<3xi32>)>
// CHECK-NEXT:       spirv.ReturnValue [[RES]] : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
spirv.func @iaddcarry_vector_i32(%a : vector<3xi32>, %b : vector<3xi32>)
  -> !spirv.struct<(vector<3xi32>, vector<3xi32>)> "None" {
  %0 = spirv.IAddCarry %a, %b : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
  spirv.ReturnValue %0 : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
}

// CHECK-LABEL: func @iaddcarry_i16
// CHECK-NEXT:       spirv.IAddCarry
// CHECK-NEXT:       spirv.ReturnValue
spirv.func @iaddcarry_i16(%a : i16, %b : i16) -> !spirv.struct<(i16, i16)> "None" {
  %0 = spirv.IAddCarry %a, %b : !spirv.struct<(i16, i16)>
  spirv.ReturnValue %0 : !spirv.struct<(i16, i16)>
}

// CHECK-LABEL: func @isubborrow_i32
// CHECK-SAME:       ([[A:%.+]]: i32, [[B:%.+]]: i32)
// CHECK-NEXT:       [[ONE:%.+]]    = spirv.Constant 1 : i32
// CHECK-NEXT:       [[ZERO:%.+]]   = spirv.Constant 0 : i32
// CHECK-NEXT:       [[OUT:%.+]]    = spirv.ISub [[A]], [[B]]
// CHECK-NEXT:       [[CMP:%.+]]    = spirv.ULessThan [[A]], [[B]]
// CHECK-NEXT:       [[BORROW:%.+]] = spirv.Select [[CMP]], [[ONE]], [[ZERO]]
// CHECK-NEXT:       [[RES:%.+]]    = spirv.CompositeConstruct [[OUT]], [[BORROW]] : (i32, i32) -> !spirv.struct<(i32, i32)>
// CHECK-NEXT:       spirv.ReturnValue [[RES]] : !spirv.struct<(i32, i32)>
spirv.func @isubborrow_i32(%a : i32, %b : i32) -> !spirv.struct<(i32, i32)> "None" {
  %0 = spirv.ISubBorrow %a, %b : !spirv.struct<(i32, i32)>
  spirv.ReturnValue %0 : !spirv.struct<(i32, i32)>
}

// CHECK-LABEL: func @isubborrow_vector_i32
// CHECK-SAME:       ([[A:%.+]]: vector<3xi32>, [[B:%.+]]: vector<3xi32>)
// CHECK-NEXT:       [[ONE:%.+]]    = spirv.Constant dense<1> : vector<3xi32>
// CHECK-NEXT:       [[ZERO:%.+]]   = spirv.Constant dense<0> : vector<3xi32>
// CHECK-NEXT:       [[OUT:%.+]]    = spirv.ISub [[A]], [[B]]
// CHECK-NEXT:       [[CMP:%.+]]    = spirv.ULessThan [[A]], [[B]]
// CHECK-NEXT:       [[BORROW:%.+]] = spirv.Select [[CMP]], [[ONE]], [[ZERO]]
// CHECK-NEXT:       [[RES:%.+]]    = spirv.CompositeConstruct [[OUT]], [[BORROW]] : (vector<3xi32>, vector<3xi32>) -> !spirv.struct<(vector<3xi32>, vector<3xi32>)>
// CHECK-NEXT:       spirv.ReturnValue [[RES]] : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
spirv.func @isubborrow_vector_i32(%a : vector<3xi32>, %b : vector<3xi32>)
  -> !spirv.struct<(vector<3xi32>, vector<3xi32>)> "None" {
  %0 = spirv.ISubBorrow %a, %b : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
  spirv.ReturnValue %0 : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
}

// CHECK-LABEL: func @isubborrow_i16
// CHECK-NEXT:       spirv.ISubBorrow
// CHECK-NEXT:       spirv.ReturnValue
spirv.func @isubborrow_i16(%a : i16, %b : i16) -> !spirv.struct<(i16, i16)> "None" {
  %0 = spirv.ISubBorrow %a, %b : !spirv.struct<(i16, i16)>
  spirv.ReturnValue %0 : !spirv.struct<(i16, i16)>
}

// CHECK-LABEL: func @is_inf_f32
// CHECK-NEXT:       [[FALSE:%.+]] = spirv.Constant false
// CHECK-NEXT:       spirv.ReturnValue [[FALSE]] : i1
spirv.func @is_inf_f32(%a : f32) -> i1 "None" {
  %0 = spirv.IsInf %a : f32
  spirv.ReturnValue %0 : i1
}

// CHECK-LABEL: func @is_inf_4xf32
// CHECK-NEXT:       [[FALSE:%.+]] = spirv.Constant dense<false> : vector<4xi1>
// CHECK-NEXT:       spirv.ReturnValue [[FALSE]] : vector<4xi1>
spirv.func @is_inf_4xf32(%a : vector<4xf32>) -> vector<4xi1> "None" {
  %0 = spirv.IsInf %a : vector<4xf32>
  spirv.ReturnValue %0 : vector<4xi1>
}

// CHECK-LABEL: func @is_nan_f32
// CHECK-NEXT:       [[FALSE:%.+]] = spirv.Constant false
// CHECK-NEXT:       spirv.ReturnValue [[FALSE]] : i1
spirv.func @is_nan_f32(%a : f32) -> i1 "None" {
  %0 = spirv.IsNan %a : f32
  spirv.ReturnValue %0 : i1
}

// CHECK-LABEL: func @is_nan_4xf32
// CHECK-NEXT:       [[FALSE:%.+]] = spirv.Constant dense<false> : vector<4xi1>
// CHECK-NEXT:       spirv.ReturnValue [[FALSE]] : vector<4xi1>
spirv.func @is_nan_4xf32(%a : vector<4xf32>) -> vector<4xi1> "None" {
  %0 = spirv.IsNan %a : vector<4xf32>
  spirv.ReturnValue %0 : vector<4xi1>
}

//===----------------------------------------------------------------------===//
// spirv.SNegate and spirv.GL.SAbs
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @snegate_i32
// CHECK-SAME:       ([[ARG:%.+]]: i32)
// CHECK-NEXT:       [[ZERO:%.+]] = spirv.Constant 0 : i32
// CHECK-NEXT:       [[RES:%.+]]  = spirv.ISub [[ZERO]], [[ARG]] : i32
// CHECK-NEXT:       spirv.ReturnValue [[RES]] : i32
spirv.func @snegate_i32(%a : i32) -> i32 "None" {
  %0 = spirv.SNegate %a : i32
  spirv.ReturnValue %0 : i32
}

// CHECK-LABEL: func @snegate_vector_i32
// CHECK-SAME:       ([[ARG:%.+]]: vector<2xi32>)
// CHECK-NEXT:       [[ZERO:%.+]] = spirv.Constant dense<0> : vector<2xi32>
// CHECK-NEXT:       [[RES:%.+]]  = spirv.ISub [[ZERO]], [[ARG]] : vector<2xi32>
// CHECK-NEXT:       spirv.ReturnValue [[RES]] : vector<2xi32>
spirv.func @snegate_vector_i32(%a : vector<2xi32>) -> vector<2xi32> "None" {
  %0 = spirv.SNegate %a : vector<2xi32>
  spirv.ReturnValue %0 : vector<2xi32>
}

// CHECK-LABEL: func @sabs_i32
// CHECK-SAME:       ([[ARG:%.+]]: i32)
// CHECK-NEXT:       [[ZERO:%.+]] = spirv.Constant 0 : i32
// CHECK-NEXT:       [[NEG:%.+]]  = spirv.ISub [[ZERO]], [[ARG]] : i32
// CHECK-NEXT:       [[LT:%.+]]   = spirv.SLessThan [[ARG]], [[ZERO]] : i32
// CHECK-NEXT:       [[RES:%.+]]  = spirv.Select [[LT]], [[NEG]], [[ARG]] : i1, i32
// CHECK-NEXT:       spirv.ReturnValue [[RES]] : i32
spirv.func @sabs_i32(%a : i32) -> i32 "None" {
  %0 = spirv.GL.SAbs %a : i32
  spirv.ReturnValue %0 : i32
}

// CHECK-LABEL: func @sabs_vector_i32
// CHECK-SAME:       ([[ARG:%.+]]: vector<3xi32>)
// CHECK-NEXT:       [[ZERO:%.+]] = spirv.Constant dense<0> : vector<3xi32>
// CHECK-NEXT:       [[NEG:%.+]]  = spirv.ISub [[ZERO]], [[ARG]] : vector<3xi32>
// CHECK-NEXT:       [[LT:%.+]]   = spirv.SLessThan [[ARG]], [[ZERO]] : vector<3xi32>
// CHECK-NEXT:       [[RES:%.+]]  = spirv.Select [[LT]], [[NEG]], [[ARG]] : vector<3xi1>, vector<3xi32>
// CHECK-NEXT:       spirv.ReturnValue [[RES]] : vector<3xi32>
spirv.func @sabs_vector_i32(%a : vector<3xi32>) -> vector<3xi32> "None" {
  %0 = spirv.GL.SAbs %a : vector<3xi32>
  spirv.ReturnValue %0 : vector<3xi32>
}

// CHECK-LABEL: func @sra_i32
// CHECK-SAME:       ([[X:%.+]]: i32, [[N:%.+]]: i32)
// CHECK-DAG:        [[ZERO:%.+]] = spirv.Constant 0 : i32
// CHECK-DAG:        [[LOG:%.+]]  = spirv.ShiftRightLogical [[X]], [[N]] : i32, i32
// CHECK-DAG:        [[NX:%.+]]   = spirv.Not [[X]] : i32
// CHECK-DAG:        [[NS:%.+]]   = spirv.ShiftRightLogical [[NX]], [[N]] : i32, i32
// CHECK-DAG:        [[NEG:%.+]]  = spirv.Not [[NS]] : i32
// CHECK-DAG:        [[LT:%.+]]   = spirv.SLessThan [[X]], [[ZERO]] : i32
// CHECK:            [[RES:%.+]]  = spirv.Select [[LT]], [[NEG]], [[LOG]] : i1, i32
// CHECK-NEXT:       spirv.ReturnValue [[RES]] : i32
spirv.func @sra_i32(%x : i32, %n : i32) -> i32 "None" {
  %0 = spirv.ShiftRightArithmetic %x, %n : i32, i32
  spirv.ReturnValue %0 : i32
}

// CHECK-LABEL: func @sdiv_i32
// CHECK-SAME:       ([[A:%.+]]: i32, [[B:%.+]]: i32)
// CHECK-DAG:        [[ZERO:%.+]]  = spirv.Constant 0 : i32
// CHECK-DAG:        [[ABSA:%.+]]  = spirv.Select {{%.+}}, {{%.+}}, [[A]] : i1, i32
// CHECK-DAG:        [[ABSB:%.+]]  = spirv.Select {{%.+}}, {{%.+}}, [[B]] : i1, i32
// CHECK:            [[Q:%.+]]     = spirv.UDiv [[ABSA]], [[ABSB]] : i32
// CHECK-DAG:        [[NEGQ:%.+]]  = spirv.ISub [[ZERO]], [[Q]] : i32
// CHECK-DAG:        [[DIFF:%.+]]  = spirv.LogicalNotEqual {{%.+}}, {{%.+}} : i1
// CHECK:            [[RES:%.+]]   = spirv.Select [[DIFF]], [[NEGQ]], [[Q]] : i1, i32
// CHECK-NEXT:       spirv.ReturnValue [[RES]] : i32
spirv.func @sdiv_i32(%a : i32, %b : i32) -> i32 "None" {
  %0 = spirv.SDiv %a, %b : i32
  spirv.ReturnValue %0 : i32
}

// CHECK-LABEL: func @smin_smax_i32
// CHECK-SAME:       ([[A:%.+]]: i32, [[B:%.+]]: i32)
// CHECK:            [[LT:%.+]]  = spirv.SLessThan [[A]], [[B]] : i32
// CHECK-DAG:        [[MIN:%.+]] = spirv.Select [[LT]], [[A]], [[B]] : i1, i32
// CHECK-DAG:        [[MAX:%.+]] = spirv.Select [[LT]], [[B]], [[A]] : i1, i32
// CHECK:            spirv.IAdd [[MIN]], [[MAX]] : i32
spirv.func @smin_smax_i32(%a : i32, %b : i32) -> i32 "None" {
  %0 = spirv.GL.SMin %a, %b : i32
  %1 = spirv.GL.SMax %a, %b : i32
  %2 = spirv.IAdd %0, %1 : i32
  spirv.ReturnValue %2 : i32
}

} // end module
