//===-- CIRStdOps.td - CIR standard library ops ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Defines ops representing standard library calls. These are raised from
/// plain calls by the cir-idiom-recognizer pass and lowered back to calls by
/// LoweringPrepare when no optimization consumed them.
///
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_IR_CIRSTDOPS_TD
#define CLANG_CIR_DIALECT_IR_CIRSTDOPS_TD

class CIR_StdOp<string functionName, dag args, dag res, list<Trait> traits = []>
    : CIR_Op<"std." # functionName, traits> {
  let arguments = !con(args, (ins FlatSymbolRefAttr:$original_fn));
  let results = res;
  let hasLLVMLowering = false;

  let extraClassDeclaration = [{
    static constexpr unsigned getNumArgs() {
      return }] # !size(args) # [{;
    }
    static llvm::StringRef getFunctionName() {
      return "}] # functionName # [{";
    }
  }];
}

def CIR_StdFindOp : CIR_StdOp<"find",
  (ins CIR_AnyType:$first, CIR_AnyType:$last, CIR_AnyType:$pattern),
  (outs CIR_AnyType:$result),
  [AllTypesMatch<["first", "last", "result"]>]> {
  let summary = "std::find()";
  let assemblyFormat = [{
    `(` $first `:` qualified(type($first)) `,`
        $last `:` qualified(type($last)) `,`
        $pattern `:` qualified(type($pattern)) `,`
        $original_fn `)` `->` qualified(type($result)) attr-dict
  }];
}

def CIR_StrLenOp : CIR_StdOp<"strlen",
  // The pointee is signed because ClangIR maps plain char to !s8i on all
  // targets, including those where char is unsigned in the source language.
  (ins CIR_PtrToType<CIR_SInt8>:$string),
  (outs CIR_AnyFundamentalIntType:$result)> {
  let summary = "C standard library strlen()";
  let assemblyFormat = [{
    `(` $string `:` qualified(type($string)) `,` $original_fn `)`
    `->` qualified(type($result)) attr-dict
  }];
}

#endif // CLANG_CIR_DIALECT_IR_CIRSTDOPS_TD
