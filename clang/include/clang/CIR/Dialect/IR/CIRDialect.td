//===- CIRDialect.td - CIR dialect -------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the CIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_IR_CIRDIALECT_TD
#define CLANG_CIR_DIALECT_IR_CIRDIALECT_TD

include "mlir/IR/OpBase.td"

def CIR_Dialect : Dialect {
  let name = "cir";

  // A short one-line summary of our dialect.
  let summary = "A high-level dialect for analyzing and optimizing Clang "
                "supported languages";

  let cppNamespace = "::cir";

  let useDefaultAttributePrinterParser = 1;

  // Enable constant materialization for the CIR dialect. This generates a
  // declaration for the cir::CIRDialect::materializeConstant function. This
  // hook is necessary for canonicalization to properly handle attributes
  // returned by fold methods, allowing them to be materialized as constant
  // operations in the IR.
  let hasConstantMaterializer = 1;

  let extraClassDeclaration = [{
    static llvm::StringRef getSourceLanguageAttrName() { return "cir.lang"; }
    static llvm::StringRef getTripleAttrName() { return "cir.triple"; }
    static llvm::StringRef getSizeTypeWidthAttrName() { return "cir.size_type_width"; }
    static llvm::StringRef getIntTypeWidthAttrName() { return "cir.int_type_width"; }
    // Set by CIRGen on a call to a recognized standard library function
    // when each parameter of the callee is a pointer
    // or reference to the same non-volatile narrow character type, meaning
    // char, signed char, unsigned char or char8_t. Equality between two such
    // character values cannot be overloaded, which is what licenses comparing
    // representations.
    static llvm::StringRef getNarrowCharParamsAttrName() {
      return "cir.narrow_char_params";
    }
    // Set by CIRGen on a call to std::find_if or std::find_if_not whose
    // iterators satisfy the narrow character rules above and whose
    // predicate is a lambda proven to be a single equality comparison of
    // the element against its only capture, with the polarity that makes
    // the search look for an equal byte.
    static llvm::StringRef getByteEqPredAttrName() {
      return "cir.byte_eq_pred";
    }
    // The capture-free variant of the marker above, carrying the compared
    // constant as a typed value.
    static llvm::StringRef getByteEqPredValueAttrName() {
      return "cir.byte_eq_pred_value";
    }
    // Marks a bit iterator predicate that seeks its single boolean capture
    static llvm::StringRef getBoolEqPredAttrName() {
      return "cir.bool_eq_pred";
    }
    // Marks a bit iterator predicate that seeks the carried boolean constant
    static llvm::StringRef getBoolEqPredValueAttrName() {
      return "cir.bool_eq_pred_value";
    }
    // Set by CIRGen on a call to the predicate overload of std::equal whose
    // three iterators satisfy the narrow character rules above and whose
    // predicate is a capture free lambda proven to be a lone return of a
    // builtin equality of its two parameters.
    static llvm::StringRef getElemEqBinaryPredAttrName() {
      return "cir.elem_eq_binary_pred";
    }
    // The wide sibling of cir.narrow_char_params, set when the same rules
    // hold over a wide element instead of a narrow character, meaning the
    // target's wchar_t or a plain int or unsigned int of its width, whose
    // builtin equality is the bit equality wmemchr computes.
    static llvm::StringRef getWideCharParamsAttrName() {
      return "cir.wide_char_params";
    }
    // The wide siblings of the two byte predicate markers, set when the
    // proven equality predicate runs over a wide element as defined for
    // cir.wide_char_params.
    static llvm::StringRef getWideEqPredAttrName() {
      return "cir.wide_eq_pred";
    }
    static llvm::StringRef getWideEqPredValueAttrName() {
      return "cir.wide_eq_pred_value";
    }
    static llvm::StringRef getWCharTypeWidthAttrName() {
      return "cir.wchar_type_width";
    }
    static llvm::StringRef getOptInfoAttrName() { return "cir.opt_info"; }
    static llvm::StringRef getCalleeAttrName() { return "callee"; }
    static llvm::StringRef getNoThrowAttrName() { return "nothrow"; }
    static llvm::StringRef getNoReturnAttrName() { return "noreturn"; }
    static llvm::StringRef getSideEffectAttrName() { return "side_effect"; }
    static llvm::StringRef getReturnsTwiceAttrName() { return "returns_twice"; }
    static llvm::StringRef getColdAttrName() { return "cold"; }
    static llvm::StringRef getHotAttrName() { return "hot"; }
    static llvm::StringRef getStrictFPAttrName() { return "strictfp"; }
    static llvm::StringRef getNoDuplicatesAttrName() { return "noduplicate"; }
    static llvm::StringRef getConvergentAttrName() { return "convergent"; }
    static llvm::StringRef getNoUnwindAttrName() { return "nounwind"; }
    static llvm::StringRef getModuleLevelAsmAttrName() { return "cir.module_asm"; }
    static llvm::StringRef getGlobalCtorsAttrName() { return "cir.global_ctors"; }
    static llvm::StringRef getGlobalDtorsAttrName() { return "cir.global_dtors"; }
    static llvm::StringRef getOperandSegmentSizesAttrName() { return "operandSegmentSizes"; }
    static llvm::StringRef getNoCallerSavedRegsAttrName() { return "no_caller_saved_registers"; }
    static llvm::StringRef getNoCallbackAttrName() { return "nocallback"; }
    static llvm::StringRef getAllocSizeAttrName() { return "allocsize"; }
    static llvm::StringRef getOptimizeForSizeAttrName() { return "optsize"; }
    static llvm::StringRef getMinSizeAttrName() { return "minsize"; }
    // Note: we have to name this with the underscore instead of the dash like
    // traditional LLVM-IR does, because the LLVM-IR-Dialect doesn't have a way
    // of forming names with a dash instead of underscore in its auto-generated
    // names. TGLexer.cpp(from tablegen) only allows tablegen-names
    // of a [a-zA-Z0-9_] character regex(numbers only if not first), so there is
    // no way to get an underscore into this, even with escaping.
    static llvm::StringRef getModularFormatAttrName() { return "modular_format"; }
    // NoBuiltins means "don't put builtins into my body", whereas "nobuiltin"
    // means "I'm not a builtin, so don't replace me".  This is a subtle
    // difference, but one that reflects Classic Codegen.
    static llvm::StringRef getNoBuiltinsAttrName() { return "nobuiltins"; }
    static llvm::StringRef getNoBuiltinAttrName() { return "nobuiltin"; }
    static llvm::StringRef getBuiltinAttrName() { return "builtin"; }
    static llvm::StringRef getTrapFuncNameAttrName() { return "trap_func_name"; }
    static llvm::StringRef getZeroCallUsedRegsAttrName() { return "zero_call_used_regs"; }
    static llvm::StringRef getSaveRegParamsAttrName() { return "save_reg_params"; }
    static llvm::StringRef getDefaultFuncAttrsAttrName() { return "default_func_attrs"; }
    static llvm::StringRef getResAttrsAttrName() { return "res_attrs"; }
    static llvm::StringRef getArgAttrsAttrName() { return "arg_attrs"; }
    static llvm::StringRef getRecordLayoutsAttrName() { return "cir.record_layouts"; }
    static llvm::StringRef getTargetCPUAttrName() { return "cir.target-cpu"; }
    static llvm::StringRef getTuneCPUAttrName() { return "cir.tune-cpu"; }
    static llvm::StringRef getTargetFeaturesAttrName() { return "cir.target-features"; }
    static llvm::StringRef getCUDABinaryHandleAttrName() { return "cir.cu.binary_handle"; }
    // Mangled symbol name of the C++20 named-module initializer function,
    // precomputed by CIRGen so later passes don't need a live ASTContext.
    static llvm::StringRef getCXXModuleInitFnNameAttrName() { return "cir.cxx_module_init_fn_name"; }
    static llvm::StringRef getMustTailAttrName() { return "musttail"; }
    static llvm::StringRef getCatchCopyThunkAttrName() { return "cir.eh.catch_copy_thunk"; }

    static llvm::StringRef getAMDGPUCodeObjectVersionAttrName() { return "cir.amdhsa_code_object_version"; }
    static llvm::StringRef getAMDGPUPrintfKindAttrName() { return "cir.amdgpu_printf_kind"; }
    static llvm::StringRef getAMDGPUXnackAttrName() { return "cir.amdgpu_xnack"; }
    static llvm::StringRef getAMDGPUSramEccAttrName() { return "cir.amdgpu_sramecc"; }
    static llvm::StringRef getOpenCLKernelArgMetadataAttrName() { return "cir.cl.kernel_arg_metadata"; }
    static llvm::StringRef getDefaultTlsModelAttrName() { return "cir.default_tls_model"; }

    void registerAttributes();
    void registerTypes();

    mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
    void printType(mlir::Type type,
                   mlir::DialectAsmPrinter &printer) const override;
  }];
}

#endif // CLANG_CIR_DIALECT_IR_CIRDIALECT_TD
