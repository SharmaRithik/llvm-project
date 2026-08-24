//===-- CIRStdOps.td - CIR standard library ops ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Defines ops representing standard library calls, raised from plain calls
/// by the cir-idiom-recognizer pass and lowered back by LoweringPrepare.
///
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_IR_CIRSTDOPS_TD
#define CLANG_CIR_DIALECT_IR_CIRSTDOPS_TD

// knownKind names the CIR_KnownFuncKind entry this operation raises, or is
// empty for a C library function that has no such entry.
class CIR_StdOp<string functionName, dag args, dag res,
                list<Trait> traits = [], string knownKind = "">
    : CIR_Op<"std." # functionName, traits # [CIR_StdOpInterface]> {
  let arguments = !con(args, (ins FlatSymbolRefAttr:$original_fn));
  let results = res;
  let hasLLVMLowering = false;

  let extraClassDeclaration = [{
    static constexpr unsigned getNumArgs() {
      return }] # !size(args) # [{;
    }
    static llvm::StringLiteral getFunctionName() {
      return "}] # functionName # [{";
    }
    // True when matched by its KnownFuncKind tag, not the callee symbol.
    static constexpr bool hasKnownFuncKind() {
      return }] # !if(!empty(knownKind), "false", "true") # [{;
    }
    static bool signatureMatches(mlir::TypeRange operands,
                                 mlir::TypeRange results);
  }] # !if(!empty(knownKind), "", [{
    static cir::KnownFuncKind getFuncKind() {
      return cir::KnownFuncKind::}] # knownKind # [{;
    }
  }]);
}

// A C library op whose signatureMatches is generated from its type constraints.
class CIR_CStdLibOp<string functionName, dag args, dag res,
                    list<Trait> traits = []>
    : CIR_StdOp<functionName, args, res, traits> {
  assert !foldl(true, !listconcat(
                          !foreach(i, !range(!size(args)),
                                   !getdagarg<TypeConstraint>(args, i)),
                          !foreach(i, !range(!size(res)),
                                   !getdagarg<TypeConstraint>(res, i))),
                acc, c, !and(acc, !ne(c.cppFunctionName, ""))),
      "CIR_CStdLibOp operand and result types must set cppFunctionName";

  let extraClassDefinition = [{
    bool $cppClass::signatureMatches(mlir::TypeRange operands,
                                     mlir::TypeRange results) {
      return }] # !interleave(
          !listconcat(
              ["operands.size() == " # !size(args),
               "results.size() == " # !size(res)],
              !foreach(i, !range(!size(args)),
                       "::cir::" #
                       !getdagarg<TypeConstraint>(args, i).cppFunctionName #
                       "(operands[" # i # "])"),
              !foreach(i, !range(!size(res)),
                       "::cir::" #
                       !getdagarg<TypeConstraint>(res, i).cppFunctionName #
                       "(results[" # i # "])")),
          " && ") # [{;
    }
  }];
}

def CIR_StdFindOp : CIR_StdOp<"find",
  (ins CIR_AnyType:$first, CIR_AnyType:$last, CIR_AnyType:$pattern),
  (outs CIR_AnyType:$result),
  [AllTypesMatch<["first", "last", "result"]>], "StdFind"> {
  let summary = "std::find()";
  let description = [{
    A raised `std::find`. `cir.narrow_char_params` or `cir.wide_char_params`
    is carried here from the call when CIRGen recorded it. Both are
    discardable, so a pass that drops them costs the rewrite rather than
    making it wrong.
  }];
  let assemblyFormat = [{
    `(` $first `:` qualified(type($first)) `,`
        $last `:` qualified(type($last)) `,`
        $pattern `:` qualified(type($pattern)) `,`
        $original_fn `)` `->` qualified(type($result)) attr-dict
  }];
}

def CIR_StdSearchOp : CIR_StdOp<"search",
  (ins CIR_AnyType:$first1, CIR_AnyType:$last1, CIR_AnyType:$first2,
       CIR_AnyType:$last2),
  (outs CIR_AnyType:$result),
  [AllTypesMatch<["first1", "last1", "result"]>,
   AllTypesMatch<["first2", "last2"]>], "StdSearch"> {
  let summary = "std::search()";
  let description = [{
    A raised `std::search` over two iterator pairs. The
    `cir.narrow_char_params` attribute is carried here from the call when
    CIRGen recorded it.
  }];
  let assemblyFormat = [{
    `(` $first1 `:` qualified(type($first1)) `,`
        $last1 `:` qualified(type($last1)) `,`
        $first2 `:` qualified(type($first2)) `,`
        $last2 `:` qualified(type($last2)) `,`
        $original_fn `)` `->` qualified(type($result)) attr-dict
  }];
}

class CIR_StdFindPredOp<string functionName, string knownKind>
    : CIR_StdOp<functionName,
  (ins CIR_AnyType:$first, CIR_AnyType:$last, CIR_AnyType:$pred),
  (outs CIR_AnyType:$result),
  [AllTypesMatch<["first", "last", "result"]>], knownKind> {
  let summary = "std::" # functionName # "()";
  let description = [{
    A raised `std::}] # functionName # [{`. `cir.byte_eq_pred` is carried
    here from the call when CIRGen proved the predicate is a byte equality
    against the closure's single capture. `cir.byte_eq_pred_value` instead
    carries the typed constant from a capture-free predicate, and it takes
    precedence when both are present. Both are discardable, so a pass that
    drops them costs the rewrite rather than making it wrong.
  }];
  let assemblyFormat = [{
    `(` $first `:` qualified(type($first)) `,`
        $last `:` qualified(type($last)) `,`
        $pred `:` qualified(type($pred)) `,`
        $original_fn `)` `->` qualified(type($result)) attr-dict
  }];
}

def CIR_StdFindIfOp : CIR_StdFindPredOp<"find_if", "StdFindIf">;

def CIR_StdFindIfNotOp : CIR_StdFindPredOp<"find_if_not", "StdFindIfNot">;

def CIR_StdEqualOp : CIR_StdOp<"equal",
  (ins CIR_AnyType:$first1, CIR_AnyType:$last1, CIR_AnyType:$first2),
  (outs CIR_BoolType:$result),
  [AllTypesMatch<["first1", "last1"]>], "StdEqual"> {
  let summary = "std::equal()";
  let description = [{
    A raised `std::equal`. `cir.narrow_char_params` is carried here from the
    call when CIRGen recorded it. It is discardable, so a pass that drops it
    costs the rewrite rather than making it wrong.
  }];
  let assemblyFormat = [{
    `(` $first1 `:` qualified(type($first1)) `,`
        $last1 `:` qualified(type($last1)) `,`
        $first2 `:` qualified(type($first2)) `,`
        $original_fn `)` `->` qualified(type($result)) attr-dict
  }];
}

def CIR_StdEqualPredOp : CIR_StdOp<"equal_pred",
  (ins CIR_AnyType:$first1, CIR_AnyType:$last1, CIR_AnyType:$first2,
       CIR_AnyType:$pred),
  (outs CIR_BoolType:$result),
  [AllTypesMatch<["first1", "last1"]>], "StdEqual"> {
  let summary = "std::equal() with a predicate";
  let description = [{
    A raised call to the predicate overload of `std::equal`.
    `cir.elem_eq_binary_pred` is carried here from the call when CIRGen
    proved the predicate is a capture free equality of its two parameters.
    It is discardable, so a pass that drops it costs the rewrite rather
    than making it wrong. The predicate operand is carried so lowering
    back to the original call loses nothing. The four iterator overload
    also raises here when its iterators are wrapper records, with last2
    riding in the predicate operand and the iterator marker on the call,
    so only the predicate marker may license a rewrite.
  }];
  let assemblyFormat = [{
    `(` $first1 `:` qualified(type($first1)) `,`
        $last1 `:` qualified(type($last1)) `,`
        $first2 `:` qualified(type($first2)) `,`
        $pred `:` qualified(type($pred)) `,`
        $original_fn `)` `->` qualified(type($result)) attr-dict
  }];
}

def CIR_StdMismatchOp : CIR_StdOp<"mismatch",
  (ins CIR_AnyType:$first1, CIR_AnyType:$last1, CIR_AnyType:$first2),
  (outs CIR_AnyType:$result),
  [AllTypesMatch<["first1", "last1"]>], "StdMismatch"> {
  let summary = "std::mismatch()";
  let description = [{
    A raised `std::mismatch`. The result is the pair record the call
    returns, holding the first position where the ranges differ.
    `cir.narrow_char_params` is carried here from the call when CIRGen
    recorded it. It is discardable, so a pass that drops it costs the
    rewrite rather than making it wrong.
  }];
  let assemblyFormat = [{
    `(` $first1 `:` qualified(type($first1)) `,`
        $last1 `:` qualified(type($last1)) `,`
        $first2 `:` qualified(type($first2)) `,`
        $original_fn `)` `->` qualified(type($result)) attr-dict
  }];
}

def CIR_StdMismatchBoundedOp : CIR_StdOp<"mismatch_bounded",
  (ins CIR_AnyType:$first1, CIR_AnyType:$last1, CIR_AnyType:$first2,
       CIR_AnyType:$last2),
  (outs CIR_AnyType:$result),
  [AllTypesMatch<["first1", "last1"]>,
   AllTypesMatch<["first2", "last2"]>], "StdMismatch"> {
  let summary = "std::mismatch() over bounded ranges";
  let description = [{
    A raised mismatch call over two bounded iterator ranges. The result is
    the pair record holding the first position where the ranges differ.
    `cir.narrow_char_params` is carried here from the call when CIRGen
    recorded it. It is discardable, so a pass that drops it costs the
    rewrite rather than making it wrong. Both bounds remain as operands so
    lowering back to the original call loses nothing.
  }];
  let assemblyFormat = [{
    `(` $first1 `:` qualified(type($first1)) `,`
        $last1 `:` qualified(type($last1)) `,`
        $first2 `:` qualified(type($first2)) `,`
        $last2 `:` qualified(type($last2)) `,`
        $original_fn `)` `->` qualified(type($result)) attr-dict
  }];
}

def CIR_StdMismatchPredOp : CIR_StdOp<"mismatch_pred",
  (ins CIR_AnyType:$first1, CIR_AnyType:$last1, CIR_AnyType:$first2,
       CIR_AnyType:$pred),
  (outs CIR_AnyType:$result),
  [AllTypesMatch<["first1", "last1"]>], "StdMismatch"> {
  let summary = "std::mismatch() with a predicate";
  let description = [{
    A raised call to the predicate overload of `std::mismatch`.
    `cir.elem_eq_binary_pred` is carried here from the call when CIRGen
    proved the predicate is a capture free equality of its two
    parameters. It is discardable, so a pass that drops it costs the
    rewrite rather than making it wrong. The predicate operand is
    carried so lowering back to the original call loses nothing.
  }];
  let assemblyFormat = [{
    `(` $first1 `:` qualified(type($first1)) `,`
        $last1 `:` qualified(type($last1)) `,`
        $first2 `:` qualified(type($first2)) `,`
        $pred `:` qualified(type($pred)) `,`
        $original_fn `)` `->` qualified(type($result)) attr-dict
  }];
}

def CIR_StdMismatchBoundedPredOp : CIR_StdOp<"mismatch_bounded_pred",
  (ins CIR_AnyType:$first1, CIR_AnyType:$last1, CIR_AnyType:$first2,
       CIR_AnyType:$last2, CIR_AnyType:$pred),
  (outs CIR_AnyType:$result),
  [AllTypesMatch<["first1", "last1"]>,
   AllTypesMatch<["first2", "last2"]>], "StdMismatch"> {
  let summary = "std::mismatch() over bounded ranges with a predicate";
  let description = [{
    A raised predicate mismatch call over two bounded iterator ranges.
    `cir.elem_eq_binary_pred` is carried here from the call when CIRGen
    proved the predicate is a capture free equality of its two parameters.
    It is discardable, so a pass that drops it costs the rewrite rather than
    making it wrong. Both bounds and the predicate remain as operands so
    lowering back to the original call loses nothing.
  }];
  let assemblyFormat = [{
    `(` $first1 `:` qualified(type($first1)) `,`
        $last1 `:` qualified(type($last1)) `,`
        $first2 `:` qualified(type($first2)) `,`
        $last2 `:` qualified(type($last2)) `,`
        $pred `:` qualified(type($pred)) `,`
        $original_fn `)` `->` qualified(type($result)) attr-dict
  }];
}

def CIR_StdRangesFindOp : CIR_StdOp<"ranges.find",
  (ins CIR_AnyType:$cpo, CIR_AnyType:$first, CIR_AnyType:$last,
       CIR_AnyType:$pattern, CIR_AnyType:$proj),
  (outs CIR_AnyType:$result),
  [AllTypesMatch<["first", "last", "result"]>], "StdRangesFind"> {
  let summary = "std::ranges::find()";
  let description = [{
    A raised call to std::ranges::find's call operator over an iterator
    pair. The first operand is the customization point object the call was
    made through and the last is the projection, both carried so lowering
    back to a call loses nothing. Lowering back passes the operands through
    in order, so the object has to stay operand 0 where the member call
    expects it. `cir.narrow_char_params` is carried here from the call when
    CIRGen recorded it, which requires the projection to be std::identity.
  }];
  let assemblyFormat = [{
    `(` $cpo `:` qualified(type($cpo)) `,`
        $first `:` qualified(type($first)) `,`
        $last `:` qualified(type($last)) `,`
        $pattern `:` qualified(type($pattern)) `,`
        $proj `:` qualified(type($proj)) `,`
        $original_fn `)` `->` qualified(type($result)) attr-dict
  }];
}

class CIR_StdRangesFindPredOp<string functionName, string cxxName,
                              string knownKind>
    : CIR_StdOp<functionName,
  (ins CIR_AnyType:$cpo, CIR_AnyType:$first, CIR_AnyType:$last,
       CIR_AnyType:$pred, CIR_AnyType:$proj),
  (outs CIR_AnyType:$result),
  [AllTypesMatch<["first", "last", "result"]>], knownKind> {
  let summary = "std::" # cxxName # "()";
  let description = [{
    A raised call to `std::}] # cxxName # [{` over an iterator pair.
    The customization point object and projection stay as operands so generic
    lowering restores the original member call with its operands in order.
    The predicate markers carry the same source proofs as the corresponding
    `std` algorithm.
  }];
  let assemblyFormat = [{
    `(` $cpo `:` qualified(type($cpo)) `,`
        $first `:` qualified(type($first)) `,`
        $last `:` qualified(type($last)) `,`
        $pred `:` qualified(type($pred)) `,`
        $proj `:` qualified(type($proj)) `,`
        $original_fn `)` `->` qualified(type($result)) attr-dict
  }];
}

def CIR_StdRangesFindIfOp :
    CIR_StdRangesFindPredOp<"ranges.find_if", "ranges::find_if",
                            "StdRangesFindIf">;

def CIR_StdRangesFindIfNotOp :
    CIR_StdRangesFindPredOp<"ranges.find_if_not", "ranges::find_if_not",
                            "StdRangesFindIfNot">;

def CIR_StdRangesFindRangeOp : CIR_StdOp<"ranges.find_range",
  (ins CIR_AnyType:$cpo, CIR_AnyType:$range, CIR_AnyType:$pattern,
       CIR_AnyType:$proj),
  (outs CIR_AnyType:$result),
  [], "StdRangesFind"> {
  let summary = "std::ranges::find() over a whole range";
  let description = [{
    A raised call to std::ranges::find's whole range call operator. The
    range arrives by reference, so the operand is its address, and the
    result is the container's iterator, a raw pointer or a record wrapping
    one. The first operand is the customization point object and the last
    is the projection, both carried so lowering back rebuilds the exact
    call, passing the operands through in order.
    `cir.narrow_char_params` is carried here from the call when CIRGen
    recorded it, which requires the range's record to carry a
    `#cir.std_type_info` naming a container of the searched character and
    the projection to be std::identity.
  }];
  let assemblyFormat = [{
    `(` $cpo `:` qualified(type($cpo)) `,`
        $range `:` qualified(type($range)) `,`
        $pattern `:` qualified(type($pattern)) `,`
        $proj `:` qualified(type($proj)) `,`
        $original_fn `)` `->` qualified(type($result)) attr-dict
  }];
}

def CIR_StrLenOp : CIR_CStdLibOp<"strlen",
  (ins CIR_PtrToChar8Type:$string),
  (outs CIR_AnyFundamentalUIntType:$result)> {
  let summary = "C standard library strlen()";
  let assemblyFormat = [{
    `(` $string `:` qualified(type($string)) `,` $original_fn `)`
    `->` qualified(type($result)) attr-dict
  }];
}

#endif // CLANG_CIR_DIALECT_IR_CIRSTDOPS_TD
