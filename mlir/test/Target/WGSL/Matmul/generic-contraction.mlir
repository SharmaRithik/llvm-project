// RUN: split-file %s %t
// RUN: mlir-wgsl-compile %t/named.mlir --opt-level=0 --mma > %t/named.wgsl
// RUN: mlir-wgsl-compile %t/generic.mlir --opt-level=0 --mma > %t/generic.wgsl
// RUN: mlir-wgsl-compile %t/permuted.mlir --opt-level=0 --mma > %t/permuted.wgsl
// RUN: diff %t/named.wgsl %t/generic.wgsl
// RUN: diff %t/named.wgsl %t/permuted.wgsl
// RUN: mlir-wgsl-compile %t/named.mlir --opt-level=0 > %t/named-register.wgsl
// RUN: mlir-wgsl-compile %t/generic.mlir --opt-level=0 > %t/generic-register.wgsl
// RUN: diff %t/named-register.wgsl %t/generic-register.wgsl
// RUN: mlir-wgsl-compile %t/named-f16.mlir --mma --nvidia-native --storage-alignment=4 --max-shared-memory=49152 > %t/named-f16.wgsl
// RUN: mlir-wgsl-compile %t/generic-f16.mlir --mma --nvidia-native --storage-alignment=4 --max-shared-memory=49152 > %t/generic-f16.wgsl
// RUN: diff %t/named-f16.wgsl %t/generic-f16.wgsl
// RUN: not mlir-wgsl-compile %t/unary.mlir --mma 2>&1 | FileCheck %s --check-prefix=REJECT
// RUN: not mlir-wgsl-compile %t/round-product.mlir --mma 2>&1 | FileCheck %s --check-prefix=REJECT
// RUN: not mlir-wgsl-compile %t/nonzero.mlir --mma 2>&1 | FileCheck %s --check-prefix=ZERO
// RUN: not mlir-wgsl-compile %t/extra.mlir --mma 2>&1 | FileCheck %s --check-prefix=EXTRA
// REJECT: expected return of a transpose-B linalg.matmul
// ZERO: matmul accumulator must be initialized to zero
// EXTRA: unsupported computation in matmul function

// Named and generic matmuls emit the same kernel. Unary ops and a rounded
// product are rejected.

//--- named.mlir
func.func @matmul(%a: tensor<33x19xf16>, %b: tensor<21x19xf16>) -> tensor<33x21xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<33x21xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<33x21xf32>) -> tensor<33x21xf32>
  %product = linalg.matmul indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, k) -> (m, n)>] ins(%a, %b : tensor<33x19xf16>, tensor<21x19xf16>) outs(%init : tensor<33x21xf32>) -> tensor<33x21xf32>
  return %product : tensor<33x21xf32>
}


//--- generic.mlir
func.func @matmul(%a: tensor<33x19xf16>, %b: tensor<21x19xf16>) -> tensor<33x21xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<33x21xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<33x21xf32>) -> tensor<33x21xf32>
  %product = linalg.generic {indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, k) -> (m, n)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%a, %b : tensor<33x19xf16>, tensor<21x19xf16>) outs(%init : tensor<33x21xf32>) {
  ^bb0(%av: f16, %bv: f16, %cv: f32):
    %af = arith.extf %av : f16 to f32
    %bf = arith.extf %bv : f16 to f32
    %mul = arith.mulf %af, %bf : f32
    %add = arith.addf %cv, %mul : f32
    linalg.yield %add : f32
  } -> tensor<33x21xf32>
  return %product : tensor<33x21xf32>
}


//--- permuted.mlir
func.func @matmul(%a: tensor<33x19xf16>, %b: tensor<21x19xf16>) -> tensor<33x21xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<33x21xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<33x21xf32>) -> tensor<33x21xf32>
  %product = linalg.generic {indexing_maps = [affine_map<(k, m, n) -> (m, k)>, affine_map<(k, m, n) -> (n, k)>, affine_map<(k, m, n) -> (m, n)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%a, %b : tensor<33x19xf16>, tensor<21x19xf16>) outs(%init : tensor<33x21xf32>) {
  ^bb0(%av: f16, %bv: f16, %cv: f32):
    %af = arith.extf %av : f16 to f32
    %bf = arith.extf %bv : f16 to f32
    %mul = arith.mulf %af, %bf : f32
    %add = arith.addf %cv, %mul : f32
    linalg.yield %add : f32
  } -> tensor<33x21xf32>
  return %product : tensor<33x21xf32>
}


//--- named-f16.mlir
#id = affine_map<(i, j) -> (i, j)>
func.func @matmul(%a: tensor<512x512xf16>, %b: tensor<256x512xf32>) -> tensor<512x256xf32> {
  %tmp = tensor.empty() : tensor<256x512xf16>
  %half = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%b : tensor<256x512xf32>) outs(%tmp : tensor<256x512xf16>) {
  ^bb0(%x: f32, %unused: f16):
    %h = arith.truncf %x : f32 to f16
    linalg.yield %h : f16
  } -> tensor<256x512xf16>
  %zero = arith.constant 0.0 : f16
  %empty = tensor.empty() : tensor<512x256xf16>
  %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<512x256xf16>) -> tensor<512x256xf16>
  %product = linalg.matmul indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, k) -> (m, n)>] ins(%a, %half : tensor<512x512xf16>, tensor<256x512xf16>) outs(%init : tensor<512x256xf16>) -> tensor<512x256xf16>
  %out = tensor.empty() : tensor<512x256xf32>
  %result = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%product : tensor<512x256xf16>) outs(%out : tensor<512x256xf32>) {
  ^bb0(%x: f16, %unused: f32):
    %f = arith.extf %x : f16 to f32
    linalg.yield %f : f32
  } -> tensor<512x256xf32>
  return %result : tensor<512x256xf32>
}


//--- generic-f16.mlir
#id = affine_map<(i, j) -> (i, j)>
func.func @matmul(%a: tensor<512x512xf16>, %b: tensor<256x512xf32>) -> tensor<512x256xf32> {
  %tmp = tensor.empty() : tensor<256x512xf16>
  %half = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%b : tensor<256x512xf32>) outs(%tmp : tensor<256x512xf16>) {
  ^bb0(%x: f32, %unused: f16):
    %h = arith.truncf %x : f32 to f16
    linalg.yield %h : f16
  } -> tensor<256x512xf16>
  %zero = arith.constant 0.0 : f16
  %empty = tensor.empty() : tensor<512x256xf16>
  %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<512x256xf16>) -> tensor<512x256xf16>
  %product = linalg.generic {indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, k) -> (m, n)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%a, %half : tensor<512x512xf16>, tensor<256x512xf16>) outs(%init : tensor<512x256xf16>) {
  ^bb0(%av: f16, %bv: f16, %cv: f16):
    %mul = arith.mulf %av, %bv : f16
    %add = arith.addf %cv, %mul : f16
    linalg.yield %add : f16
  } -> tensor<512x256xf16>
  %out = tensor.empty() : tensor<512x256xf32>
  %result = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%product : tensor<512x256xf16>) outs(%out : tensor<512x256xf32>) {
  ^bb0(%x: f16, %unused: f32):
    %f = arith.extf %x : f16 to f32
    linalg.yield %f : f32
  } -> tensor<512x256xf32>
  return %result : tensor<512x256xf32>
}


//--- unary.mlir
func.func @matmul(%a: tensor<33x19xf16>, %b: tensor<21x19xf16>) -> tensor<33x21xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<33x21xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<33x21xf32>) -> tensor<33x21xf32>
  %product = linalg.generic {indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, k) -> (m, n)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%a, %b : tensor<33x19xf16>, tensor<21x19xf16>) outs(%init : tensor<33x21xf32>) {
  ^bb0(%av: f16, %bv: f16, %cv: f32):
    %af = arith.extf %av : f16 to f32
    %bf = arith.extf %bv : f16 to f32
    %negative = arith.negf %af : f32
    %mul = arith.mulf %negative, %bf : f32
    %add = arith.addf %cv, %mul : f32
    linalg.yield %add : f32
  } -> tensor<33x21xf32>
  return %product : tensor<33x21xf32>
}


//--- round-product.mlir
func.func @matmul(%a: tensor<33x19xf16>, %b: tensor<21x19xf16>) -> tensor<33x21xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<33x21xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<33x21xf32>) -> tensor<33x21xf32>
  %product = linalg.generic {indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, k) -> (m, n)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%a, %b : tensor<33x19xf16>, tensor<21x19xf16>) outs(%init : tensor<33x21xf32>) {
  ^bb0(%av: f16, %bv: f16, %cv: f32):
    %mul = arith.mulf %av, %bv : f16
    %rounded = arith.extf %mul : f16 to f32
    %add = arith.addf %cv, %rounded : f32
    linalg.yield %add : f32
  } -> tensor<33x21xf32>
  return %product : tensor<33x21xf32>
}


//--- nonzero.mlir
func.func @matmul(%a: tensor<33x19xf16>, %b: tensor<21x19xf16>) -> tensor<33x21xf32> {
  %zero = arith.constant 1.0 : f32
  %empty = tensor.empty() : tensor<33x21xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<33x21xf32>) -> tensor<33x21xf32>
  %product = linalg.generic {indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, k) -> (m, n)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%a, %b : tensor<33x19xf16>, tensor<21x19xf16>) outs(%init : tensor<33x21xf32>) {
  ^bb0(%av: f16, %bv: f16, %cv: f32):
    %af = arith.extf %av : f16 to f32
    %bf = arith.extf %bv : f16 to f32
    %mul = arith.mulf %af, %bf : f32
    %add = arith.addf %cv, %mul : f32
    linalg.yield %add : f32
  } -> tensor<33x21xf32>
  return %product : tensor<33x21xf32>
}


//--- extra.mlir
func.func @matmul(%a: tensor<33x19xf16>, %b: tensor<21x19xf16>) -> tensor<33x21xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<33x21xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<33x21xf32>) -> tensor<33x21xf32>
  %product = linalg.generic {indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, k) -> (m, n)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%a, %b : tensor<33x19xf16>, tensor<21x19xf16>) outs(%init : tensor<33x21xf32>) {
  ^bb0(%av: f16, %bv: f16, %cv: f32):
    %af = arith.extf %av : f16 to f32
    %bf = arith.extf %bv : f16 to f32
    %mul = arith.mulf %af, %bf : f32
    %add = arith.addf %cv, %mul : f32
    linalg.yield %add : f32
  } -> tensor<33x21xf32>
  %unused = arith.negf %a : tensor<33x19xf16>
  return %product : tensor<33x21xf32>
}
