//===- LibOpt.cpp - Optimize CIR raised C/C++ library idioms --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass optimizes C/C++ standard library idioms in Clang IR.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Region.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/TargetParser/Triple.h"
#include <optional>

using cir::CIRBaseBuilderTy;
using namespace mlir;
using namespace cir;

namespace mlir {
#define GEN_PASS_DEF_LIBOPT
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

struct LibOptPass : public impl::LibOptBase<LibOptPass> {
  LibOptPass() = default;
  mlir::LogicalResult
  initializeOptions(llvm::StringRef options,
                    llvm::function_ref<mlir::LogicalResult(const llvm::Twine &)>
                        errorHandler) override;
  void runOnOperation() override;

  // Raw libopt option string forwarded by the frontend. This will later control
  // which optimizations the pass enables.
  std::string optimizationOptions;
};
} // namespace

mlir::LogicalResult LibOptPass::initializeOptions(
    llvm::StringRef options,
    llvm::function_ref<mlir::LogicalResult(const llvm::Twine &)>) {
  optimizationOptions = options.str();
  // TODO(cir): Parse options to select the active transformations for the
  // pass.
  return mlir::success();
}

// The facts every rewrite checks before it may introduce the named libcall.
struct LibCallEnv {
  cir::FuncOp enclosing;
  cir::GlobalOp enclosingGlobal;
  mlir::ModuleOp moduleOp;
  llvm::Triple triple;
};

// Checks the environment shared by every libcall rewrite. LibOpt runs before
// LoweringPrepare, so a global initializer is still a cir.global here, and
// anything else is not a shape CIRGen produces. No builtin state rides on the
// raised call and on the enclosing function, while a global initializer has
// no function to carry the list. The rewrite introduces a libcall the program
// never named, so the target has to provide it. TargetLibraryInfo is the
// availability oracle LLVM transforms consult before creating a libcall, and
// it answers from the triple. The ABI attributes of the introduced call stay
// with the general call ABI work. A libc that implements the routine with the
// raised algorithm would call itself forever, so the enclosing function must
// not be the libcall.
static std::optional<LibCallEnv> checkLibCallEnv(mlir::Operation *op,
                                                 llvm::StringRef libcallName,
                                                 llvm::LibFunc libFunc) {
  LibCallEnv env;
  env.enclosing = op->getParentOfType<cir::FuncOp>();
  env.enclosingGlobal = op->getParentOfType<cir::GlobalOp>();
  if (!env.enclosing && !env.enclosingGlobal)
    return std::nullopt;
  if (isNoBuiltin(op, libcallName) ||
      (env.enclosing && noBuiltinListDisables(env.enclosing, libcallName)))
    return std::nullopt;
  env.moduleOp = op->getParentOfType<mlir::ModuleOp>();
  if (!env.moduleOp)
    return std::nullopt;
  auto tripleAttr = env.moduleOp->getAttrOfType<mlir::StringAttr>(
      cir::CIRDialect::getTripleAttrName());
  if (!tripleAttr)
    return std::nullopt;
  env.triple = llvm::Triple(tripleAttr.getValue().str());
  llvm::TargetLibraryInfoImpl tliImpl(env.triple);
  if (!llvm::TargetLibraryInfo(tliImpl).has(libFunc))
    return std::nullopt;
  if (env.enclosing && env.enclosing.getName() == libcallName)
    return std::nullopt;
  return env;
}

// Whether the module lets the rewrite name the libcall. An existing symbol
// can be shared only when it is exactly the C function with the expected
// prototype. An alias forwards to some other symbol, so a matching prototype
// says nothing about what the call would reach.
static bool sharesLibCallSymbol(mlir::ModuleOp moduleOp,
                                mlir::SymbolTableCollection &symbolTables,
                                llvm::StringRef libcallName,
                                cir::FuncType libcallTy) {
  mlir::Operation *existing = symbolTables.lookupSymbolIn(
      moduleOp, mlir::StringAttr::get(moduleOp.getContext(), libcallName));
  if (!existing)
    return true;
  auto existingFn = mlir::dyn_cast<cir::FuncOp>(existing);
  return existingFn && !existingFn.getAliasee() &&
         existingFn.getFunctionType() == libcallTy &&
         existingFn.getCallingConv() == cir::CallingConv::C;
}

// A raised contiguous iterator is a raw pointer or a complete record wrapping
// the pointer as its only member, the shape the CIRGen marker licensed.
// wrapperTy reports the record when one wrapped the pointer.
static cir::PointerType unwrapContiguousIterator(mlir::Type ty,
                                                 cir::RecordType &wrapperTy) {
  auto iterTy = mlir::dyn_cast<cir::PointerType>(ty);
  wrapperTy = mlir::dyn_cast<cir::RecordType>(ty);
  if (!wrapperTy)
    return iterTy;
  if (wrapperTy.isUnion() || !wrapperTy.isComplete() ||
      wrapperTy.getMembers().size() != 1)
    return cir::PointerType();
  return mlir::dyn_cast<cir::PointerType>(wrapperTy.getMembers()[0]);
}

static mlir::Value emitMemCmpIsEqual(CIRBaseBuilderTy &builder,
                                     mlir::Location loc, mlir::Value lhs,
                                     mlir::Value rhs, mlir::Value len,
                                     cir::IntType intTy,
                                     mlir::Value zero = {}) {
  mlir::Value cmp = cir::MemCmpOp::create(builder, loc, intTy, lhs, rhs, len);
  if (!zero)
    zero = builder.getNullValue(intTy, loc);
  return builder.createCompare(loc, cir::CmpOpKind::eq, cmp, zero).getResult();
}

// The find_if family searches by an equality predicate, so its sought
// element and licensing marker live in different places than the find
// forms'.
template <typename OpT>
constexpr bool findsByEqPredicate =
    std::is_same_v<OpT, StdFindIfOp> || std::is_same_v<OpT, StdFindIfNotOp> ||
    std::is_same_v<OpT, StdRangesFindIfOp> ||
    std::is_same_v<OpT, StdRangesFindIfNotOp>;

// The sought element arrives behind the pattern reference for find. A
// predicate either holds it in its single capture, by value or through a
// reference, or carries it in the typed value attribute.
template <typename OpT>
static bool matchEqSoughtShape(OpT findOp, cir::IntType elemTy, bool wide,
                               cir::IntAttr &predicateValue,
                               bool &captureByRef) {
  if constexpr (findsByEqPredicate<OpT>) {
    predicateValue = findOp->template getAttrOfType<cir::IntAttr>(
        wide ? cir::CIRDialect::getWideEqPredValueAttrName()
             : cir::CIRDialect::getByteEqPredValueAttrName());
    if (predicateValue)
      return predicateValue.getType() == elemTy;
    auto closureTy =
        mlir::dyn_cast<cir::RecordType>(findOp.getPred().getType());
    if (!closureTy || closureTy.isUnion() || !closureTy.isComplete() ||
        closureTy.getMembers().size() != 1)
      return false;
    mlir::Type capTy = closureTy.getMembers()[0];
    if (auto capPtrTy = mlir::dyn_cast<cir::PointerType>(capTy)) {
      if (capPtrTy.getAddrSpace() || capPtrTy.getPointee() != elemTy)
        return false;
      captureByRef = true;
      return true;
    }
    return capTy == elemTy;
  } else {
    auto patternPtrTy =
        mlir::dyn_cast<cir::PointerType>(findOp.getPattern().getType());
    return patternPtrTy && patternPtrTy.getPointee() == elemTy;
  }
}

// Only the CIRGen facts license a rewrite, since an enum or atomic element
// also lowers to a byte-wide integer and a closure of the right shape can
// compute anything.
template <typename OpT>
static bool hasEqSearchMarker(OpT findOp, bool wide,
                              cir::IntAttr predicateValue) {
  if constexpr (findsByEqPredicate<OpT>) {
    if (predicateValue)
      return true;
    return static_cast<bool>(findOp->template getAttrOfType<mlir::UnitAttr>(
        wide ? cir::CIRDialect::getWideEqPredAttrName()
             : cir::CIRDialect::getByteEqPredAttrName()));
  } else {
    return static_cast<bool>(findOp->template getAttrOfType<mlir::UnitAttr>(
        wide ? cir::CIRDialect::getWideCharParamsAttrName()
             : cir::CIRDialect::getNarrowCharParamsAttrName()));
  }
}

// The introduced operands take the widths the module records, which is
// what the introduced operation's verifier checks them against, int and
// size_t for memchr, wchar_t and size_t for wmemchr. An element that is
// not the target's wide character has no libcall and declines here.
static bool resolveMemchrWidths(mlir::ModuleOp moduleOp,
                                const llvm::Triple &triple, bool wide,
                                cir::IntType elemTy, unsigned &patternWidth,
                                unsigned &sizeWidth) {
  std::optional<unsigned> recordedPattern = cir::getRecordedIntegerWidth(
      moduleOp, wide ? cir::CIRDialect::getWCharTypeWidthAttrName()
                     : cir::CIRDialect::getIntTypeWidthAttrName());
  std::optional<unsigned> recordedSize = cir::getRecordedIntegerWidth(
      moduleOp, cir::CIRDialect::getSizeTypeWidthAttrName());
  if (!recordedPattern || !recordedSize)
    return false;
  // A program built with a nonstandard wchar_t width still links the C
  // library selected by the triple, and that library's wmemchr walks the
  // triple's default width. The recorded module attribute answers the
  // program side question instead, so comparing both is deliberate rather
  // than a substitute for recording the width.
  if (wide && *recordedPattern != triple.getDefaultWCharSize() * 8)
    return false;
  // Hand written IR can claim the wide character marker while using an
  // element whose width disagrees with the recorded width.
  if (wide && *recordedPattern != elemTy.getWidth())
    return false;
  patternWidth = *recordedPattern;
  sizeWidth = *recordedSize;
  return true;
}

// The element the call is typed at. For wide it starts as the searched
// element, but a module may already declare wmemchr at the same width
// with the other signedness, as a real standard library header does
// whenever a plain wide find instantiates its own dispatch. Equality of
// wide elements is bit equality, which is what licensed the rewrite, so
// the call adopts the declared signedness and casts at the boundary
// rather than decline.
static bool resolveMemchrCallTypes(CIRBaseBuilderTy &builder,
                                   mlir::ModuleOp moduleOp,
                                   mlir::SymbolTableCollection &symbolTables,
                                   bool wide, llvm::StringRef libcallName,
                                   unsigned patternWidth, unsigned sizeWidth,
                                   cir::IntType elemTy, cir::PointerType iterTy,
                                   cir::IntType &callElemTy,
                                   cir::PointerType &callIterTy) {
  callElemTy = elemTy;
  callIterTy = iterTy;
  auto makeWideTy = [&](cir::IntType eTy, cir::PointerType pTy) {
    return cir::FuncType::get(
        {mlir::Type(pTy), mlir::Type(eTy), builder.getUIntNTy(sizeWidth)}, pTy);
  };
  cir::FuncType libcallTy =
      wide ? makeWideTy(callElemTy, callIterTy)
           : cir::FuncType::get({builder.getVoidPtrTy(),
                                 builder.getSIntNTy(patternWidth),
                                 builder.getUIntNTy(sizeWidth)},
                                builder.getVoidPtrTy());
  if (sharesLibCallSymbol(moduleOp, symbolTables, libcallName, libcallTy))
    return true;
  if (!wide)
    return false;
  cir::IntType flippedTy = elemTy.isSigned()
                               ? builder.getUIntNTy(elemTy.getWidth())
                               : builder.getSIntNTy(elemTy.getWidth());
  auto flippedPtrTy = cir::PointerType::get(flippedTy);
  if (!sharesLibCallSymbol(moduleOp, symbolTables, libcallName,
                           makeWideTy(flippedTy, flippedPtrTy)))
    return false;
  callElemTy = flippedTy;
  callIterTy = flippedPtrTy;
  return true;
}

// Rewrites cir.std.find, cir.std.ranges.find, cir.std.ranges.find_range,
// and the byte equality predicate forms of cir.std.find_if,
// cir.std.find_if_not, cir.std.ranges.find_if, and
// cir.std.ranges.find_if_not. All of them search for the first equal byte
// and share everything except where the byte and bounds come from. The find
// forms use a referenced pattern. The predicate forms use a closure capture or
// recorded constant. The whole range form loads bounds through the member paths
// its record's library identity carries. The cpo and proj operands of the
// ranges forms are carried only for lowering back and play no part here.
template <typename OpT>
static void
rewriteFindLikeToMemchr(OpT findOp, mlir::SymbolTableCollection &symbolTables,
                        mlir::Pass::Statistic &numFindLikeToMemchr,
                        mlir::Pass::Statistic &numFindLikeToWmemchr) {
  constexpr bool predOp = findsByEqPredicate<OpT>;
  constexpr bool rangeOp = std::is_same_v<OpT, StdRangesFindRangeOp>;
  // A std contiguous iterator wraps the pointer as its only member.  The
  // narrow character fact CIRGen recorded is what licenses reading that
  // member as the address, so the rewrite runs on the wrapped pointer and
  // rebuilds the record around its result.
  mlir::Type resultTy = findOp.getResult().getType();
  cir::RecordType wrapperTy;
  cir::PointerType iterTy = unwrapContiguousIterator(resultTy, wrapperTy);
  if (!iterTy || iterTy.getAddrSpace())
    return;
  auto elemTy = mlir::dyn_cast<cir::IntType>(iterTy.getPointee());
  if (!elemTy)
    return;
  // A wider element rewrites to wmemchr under the wide markers, except
  // through the whole range form, whose container identity proof remains
  // byte scoped. The recorded wchar_t width check below rejects every
  // width that is not the target's wide character.
  const bool wide = elemTy.getWidth() != 8;
  if (wide && rangeOp)
    return;
  const llvm::StringRef libcallName = wide ? "wmemchr" : "memchr";
  bool captureByRef = false;
  cir::IntAttr predicateValue;
  if (!matchEqSoughtShape(findOp, elemTy, wide, predicateValue, captureByRef))
    return;

  // The whole range form owns no bounds operands. Its record's standard
  // library identity names the element and the member paths to the two
  // pointers bounding the storage, and the rewrite re-proves that every
  // path lands on a pointer to the searched element.
  cir::StructType rangeRecTy;
  cir::StdTypeInfoAttr rangeInfo;
  if constexpr (rangeOp) {
    auto rangePtrTy =
        mlir::dyn_cast<cir::PointerType>(findOp.getRange().getType());
    if (!rangePtrTy || rangePtrTy.getAddrSpace())
      return;
    rangeRecTy = mlir::dyn_cast<cir::StructType>(rangePtrTy.getPointee());
    if (rangeRecTy)
      rangeInfo = rangeRecTy.getStdTypeInfo();
    if (!rangeInfo || rangeInfo.getKind() != cir::StdTypeKind::StdVector ||
        rangeInfo.getElement() != mlir::Type(elemTy))
      return;
    auto pathLandsOnIter = [&](llvm::ArrayRef<int32_t> path) {
      return cir::StdTypeInfoAttr::resolvePath(rangeRecTy, path) ==
             mlir::Type(iterTy);
    };
    if (!pathLandsOnIter(
            rangeInfo.getRolePath(cir::StdTypeInfoAttr::kRoleBegin)) ||
        !pathLandsOnIter(rangeInfo.getRolePath(cir::StdTypeInfoAttr::kRoleEnd)))
      return;
  }

  std::optional<LibCallEnv> env = checkLibCallEnv(
      findOp, libcallName, wide ? llvm::LibFunc_wmemchr : llvm::LibFunc_memchr);
  if (!env)
    return;

  if (!hasEqSearchMarker(findOp, wide, predicateValue))
    return;

  mlir::ModuleOp moduleOp = env->moduleOp;

  unsigned patternWidth, sizeWidth;
  if (!resolveMemchrWidths(moduleOp, env->triple, wide, elemTy, patternWidth,
                           sizeWidth))
    return;

  CIRBaseBuilderTy builder(*findOp.getContext());

  cir::IntType callElemTy;
  cir::PointerType callIterTy;
  if (!resolveMemchrCallTypes(builder, moduleOp, symbolTables, wide,
                              libcallName, patternWidth, sizeWidth, elemTy,
                              iterTy, callElemTy, callIterTy))
    return;

  mlir::Location loc = findOp.getLoc();
  builder.setInsertionPointAfter(findOp);
  mlir::Value first, last;
  if constexpr (rangeOp) {
    // The same walk that licensed the paths supplies each step's member
    // type here, so the emitted accesses cannot drift from the proof.
    auto loadBound = [&](llvm::ArrayRef<int32_t> path) {
      llvm::SmallVector<mlir::Type, 4> steps;
      cir::StdTypeInfoAttr::resolvePath(rangeRecTy, path, &steps);
      mlir::Value addr = findOp.getRange();
      for (auto [idx, fieldTy] : llvm::zip_equal(path, steps))
        addr = cir::GetMemberOp::create(builder, loc,
                                        cir::PointerType::get(fieldTy), addr,
                                        /*name=*/"", /*index=*/idx);
      return builder.createLoad(loc, addr);
    };
    first = loadBound(rangeInfo.getRolePath(cir::StdTypeInfoAttr::kRoleBegin));
    last = loadBound(rangeInfo.getRolePath(cir::StdTypeInfoAttr::kRoleEnd));
  } else {
    first = findOp.getFirst();
    last = findOp.getLast();
    if (wrapperTy) {
      first = cir::ExtractMemberOp::create(builder, loc, first, 0);
      last = cir::ExtractMemberOp::create(builder, loc, last, 0);
    }
  }

  // An empty std::find range is allowed to be a pair of null pointers, while
  // C requires the memchr pointer to be valid even when the length is zero.
  mlir::Value isEmpty =
      builder.createCompare(loc, cir::CmpOpKind::eq, first, last);
  mlir::Value result =
      cir::TernaryOp::create(
          builder, loc, isEmpty,
          [&](mlir::OpBuilder &, mlir::Location) {
            builder.createYield(loc, last);
          },
          [&](mlir::OpBuilder &, mlir::Location) {
            mlir::Value sought;
            if constexpr (predOp) {
              if (predicateValue) {
                sought = builder.getConstant(loc, predicateValue);
              } else {
                sought = cir::ExtractMemberOp::create(builder, loc,
                                                      findOp.getPred(), 0);
                if (captureByRef)
                  sought = builder.createLoad(loc, sought);
              }
            } else {
              sought = builder.createLoad(loc, findOp.getPattern());
            }
            mlir::Value len = cir::PtrDiffOp::create(
                builder, loc, builder.getUIntNTy(sizeWidth), last, first);
            mlir::Value res;
            if (wide) {
              // wmemchr takes and returns the wide type itself and len
              // counts wide characters, which is what ptr_diff yields. The
              // casts appear only when the call adopted a declared
              // signedness that differs from the element.
              mlir::Value src = first;
              if (callElemTy != elemTy) {
                src = builder.createBitcast(loc, first, callIterTy);
                sought = builder.createIntCast(sought, callElemTy);
              }
              res = cir::WMemChrOp::create(builder, loc, src, sought, len);
              if (callElemTy != elemTy)
                res = builder.createBitcast(loc, res, iterTy);
            } else {
              mlir::Value src =
                  builder.createBitcast(loc, first, builder.getVoidPtrTy());
              mlir::Value pattern = builder.createIntCast(
                  sought, builder.getSIntNTy(patternWidth));
              res = cir::MemChrOp::create(builder, loc, src, pattern, len);
              res = builder.createBitcast(loc, res, iterTy);
            }
            builder.createYield(
                loc, builder.createSelect(loc, builder.createPtrIsNull(res),
                                          last, res));
          })
          .getResult();
  if (wrapperTy) {
    // The whole range form has no incoming wrapper value to rebuild
    // around, so the result record starts from undef and the insert
    // defines its only member.
    mlir::Value seed;
    if constexpr (rangeOp)
      seed = builder.getConstant(loc, cir::UndefAttr::get(wrapperTy));
    else
      seed = findOp.getFirst();
    result =
        cir::InsertMemberOp::create(builder, loc, seed, /*index=*/0, result);
  }
  findOp.getResult().replaceAllUsesWith(result);
  findOp.erase();
  if (wide)
    ++numFindLikeToWmemchr;
  else
    ++numFindLikeToMemchr;
}

// Rewrites the four iterator form of std::search over licensed bytes.
static void
rewriteSearchToMemmem(StdSearchOp searchOp,
                      mlir::SymbolTableCollection &symbolTables,
                      mlir::Pass::Statistic &numSearchToMemmem,
                      mlir::Pass::Statistic &numSearchEqualLengthToMemcmp) {
  cir::RecordType haystackWrapperTy;
  cir::RecordType needleWrapperTy;
  cir::PointerType haystackIterTy = unwrapContiguousIterator(
      searchOp.getFirst1().getType(), haystackWrapperTy);
  cir::PointerType needleIterTy =
      unwrapContiguousIterator(searchOp.getFirst2().getType(), needleWrapperTy);
  if (!haystackIterTy || !needleIterTy ||
      static_cast<bool>(haystackWrapperTy) !=
          static_cast<bool>(needleWrapperTy) ||
      haystackIterTy.getAddrSpace() || needleIterTy.getAddrSpace())
    return;

  auto haystackElemTy =
      mlir::dyn_cast<cir::IntType>(haystackIterTy.getPointee());
  auto needleElemTy = mlir::dyn_cast<cir::IntType>(needleIterTy.getPointee());
  if (!haystackElemTy || haystackElemTy.getWidth() != 8 ||
      needleElemTy != haystackElemTy)
    return;

  constexpr llvm::StringLiteral libcallName = "memmem";
  std::optional<LibCallEnv> env =
      checkLibCallEnv(searchOp, libcallName, llvm::LibFunc_memmem);
  if (!env)
    return;

  if (!searchOp->getAttrOfType<mlir::UnitAttr>(
          cir::CIRDialect::getNarrowCharParamsAttrName()))
    return;

  std::optional<unsigned> sizeWidth = cir::getRecordedIntegerWidth(
      env->moduleOp, cir::CIRDialect::getSizeTypeWidthAttrName());
  if (!sizeWidth)
    return;

  CIRBaseBuilderTy builder(*searchOp.getContext());
  auto voidPtrTy = builder.getVoidPtrTy();
  auto sizeTy = builder.getUIntNTy(*sizeWidth);
  if (!sharesLibCallSymbol(
          env->moduleOp, symbolTables, libcallName,
          cir::FuncType::get({voidPtrTy, sizeTy, voidPtrTy, sizeTy},
                             voidPtrTy)))
    return;

  // Equal lengths admit exactly one candidate position, which one memcmp
  // decides. The guard arm is emitted only when the recorded widths, the
  // no builtin lists, and the symbol table all admit memcmp, otherwise
  // the emission keeps the memmem only shape.
  std::optional<unsigned> intWidth = cir::getRecordedIntegerWidth(
      env->moduleOp, cir::CIRDialect::getIntTypeWidthAttrName());
  const bool memCmpAdmitted =
      intWidth.has_value() &&
      checkLibCallEnv(searchOp, "memcmp", llvm::LibFunc_memcmp).has_value() &&
      sharesLibCallSymbol(env->moduleOp, symbolTables, "memcmp",
                          cir::FuncType::get({voidPtrTy, voidPtrTy, sizeTy},
                                             builder.getSIntNTy(*intWidth)));

  mlir::Location loc = searchOp.getLoc();
  builder.setInsertionPointAfter(searchOp);
  mlir::Value first1 = searchOp.getFirst1();
  mlir::Value last1 = searchOp.getLast1();
  mlir::Value first2 = searchOp.getFirst2();
  mlir::Value last2 = searchOp.getLast2();
  if (haystackWrapperTy) {
    first1 = cir::ExtractMemberOp::create(builder, loc, first1, 0);
    last1 = cir::ExtractMemberOp::create(builder, loc, last1, 0);
    first2 = cir::ExtractMemberOp::create(builder, loc, first2, 0);
    last2 = cir::ExtractMemberOp::create(builder, loc, last2, 0);
  }

  // The C++ empty needle rule returns first1.
  mlir::Value needleIsEmpty =
      builder.createCompare(loc, cir::CmpOpKind::eq, first2, last2);
  mlir::Value result =
      cir::TernaryOp::create(
          builder, loc, needleIsEmpty,
          [&](mlir::OpBuilder &, mlir::Location) {
            builder.createYield(loc, first1);
          },
          [&](mlir::OpBuilder &, mlir::Location) {
            // These pointer differences are byte counts because both element
            // types have width eight.
            mlir::Value haystackLen =
                cir::PtrDiffOp::create(builder, loc, sizeTy, last1, first1);
            mlir::Value needleLen =
                cir::PtrDiffOp::create(builder, loc, sizeTy, last2, first2);
            // The C++ short haystack rule returns last1. Together these guards
            // ensure memmem receives a nonempty needle and a haystack at least
            // as long, so every pointer and length passed to C is valid.
            mlir::Value needleIsLonger = builder.createCompare(
                loc, cir::CmpOpKind::gt, needleLen, haystackLen);
            auto emitMemMem = [&]() -> mlir::Value {
              mlir::Value haystack =
                  builder.createBitcast(loc, first1, builder.getVoidPtrTy());
              mlir::Value needle =
                  builder.createBitcast(loc, first2, builder.getVoidPtrTy());
              mlir::Value found = cir::MemMemOp::create(
                  builder, loc, haystack, haystackLen, needle, needleLen);
              found = builder.createBitcast(loc, found, haystackIterTy);
              return builder.createSelect(loc, builder.createPtrIsNull(found),
                                          last1, found);
            };
            mlir::Value nonEmptyResult =
                cir::TernaryOp::create(
                    builder, loc, needleIsLonger,
                    [&](mlir::OpBuilder &, mlir::Location) {
                      builder.createYield(loc, last1);
                    },
                    [&](mlir::OpBuilder &, mlir::Location) {
                      if (!memCmpAdmitted) {
                        builder.createYield(loc, emitMemMem());
                        return;
                      }
                      mlir::Value sameLength = builder.createCompare(
                          loc, cir::CmpOpKind::eq, needleLen, haystackLen);
                      mlir::Value guarded =
                          cir::TernaryOp::create(
                              builder, loc, sameLength,
                              [&](mlir::OpBuilder &, mlir::Location) {
                                mlir::Value lhs = builder.createBitcast(
                                    loc, first1, builder.getVoidPtrTy());
                                mlir::Value rhs = builder.createBitcast(
                                    loc, first2, builder.getVoidPtrTy());
                                mlir::Value equal = emitMemCmpIsEqual(
                                    builder, loc, lhs, rhs, needleLen,
                                    builder.getSIntNTy(*intWidth));
                                builder.createYield(
                                    loc, builder.createSelect(loc, equal,
                                                              first1, last1));
                              },
                              [&](mlir::OpBuilder &, mlir::Location) {
                                builder.createYield(loc, emitMemMem());
                              })
                              .getResult();
                      builder.createYield(loc, guarded);
                    })
                    .getResult();
            builder.createYield(loc, nonEmptyResult);
          })
          .getResult();

  if (haystackWrapperTy) {
    result = cir::InsertMemberOp::create(builder, loc, searchOp.getFirst1(),
                                         /*index=*/0, result);
  }
  searchOp.getResult().replaceAllUsesWith(result);
  searchOp.erase();
  ++numSearchToMemmem;
  if (memCmpAdmitted)
    ++numSearchEqualLengthToMemcmp;
}

template <typename OpT>
static void rewriteEqualToMemcmp(OpT equalOp,
                                 mlir::SymbolTableCollection &symbolTables,
                                 mlir::Pass::Statistic &numEqualToMemcmp) {
  constexpr bool predOp = std::is_same_v<OpT, StdEqualPredOp>;
  constexpr llvm::StringLiteral libcallName = "memcmp";
  std::optional<LibCallEnv> env =
      checkLibCallEnv(equalOp, libcallName, llvm::LibFunc_memcmp);
  if (!env)
    return;

  // Only the CIRGen fact licenses the rewrite because an enum or atomic
  // element also lowers to a byte wide integer and a closure of the right
  // shape can compute anything. The predicate form accepts only its own
  // marker since the proof covers the lambda body, and because the four
  // iterator overload raises as the predicate form when its iterators are
  // wrapper records, carrying last2 in the predicate operand and the
  // iterator marker on the call. Rewriting that shape would read only one
  // range's length.
  if (!equalOp->template getAttrOfType<mlir::UnitAttr>(
          predOp ? cir::CIRDialect::getElemEqBinaryPredAttrName()
                 : cir::CIRDialect::getNarrowCharParamsAttrName()))
    return;

  // The introduced operation uses the recorded int and size_t widths which
  // its verifier checks against the result and length.
  std::optional<unsigned> intWidth = cir::getRecordedIntegerWidth(
      env->moduleOp, cir::CIRDialect::getIntTypeWidthAttrName());
  std::optional<unsigned> sizeWidth = cir::getRecordedIntegerWidth(
      env->moduleOp, cir::CIRDialect::getSizeTypeWidthAttrName());
  if (!intWidth || !sizeWidth)
    return;

  CIRBaseBuilderTy builder(*equalOp.getContext());
  if (!sharesLibCallSymbol(
          env->moduleOp, symbolTables, libcallName,
          cir::FuncType::get({builder.getVoidPtrTy(), builder.getVoidPtrTy(),
                              builder.getUIntNTy(*sizeWidth)},
                             builder.getSIntNTy(*intWidth))))
    return;

  // The CIRGen marker licenses trusting a one member record as a contiguous
  // iterator that wraps its pointer. The two ranges are independent, so one
  // may be wrapped while the other is raw.
  cir::RecordType firstWrapperTy;
  cir::RecordType secondWrapperTy;
  cir::PointerType firstPtrTy =
      unwrapContiguousIterator(equalOp.getFirst1().getType(), firstWrapperTy);
  cir::PointerType secondPtrTy =
      unwrapContiguousIterator(equalOp.getFirst2().getType(), secondWrapperTy);
  // The libc interface cannot represent other address spaces and the two
  // byte sequences must share an element type.
  if (!firstPtrTy || !secondPtrTy || firstPtrTy.getAddrSpace() ||
      secondPtrTy.getAddrSpace() ||
      firstPtrTy.getPointee() != secondPtrTy.getPointee())
    return;
  // BitInt belongs to a separate type family and the rewrite is licensed only
  // for fundamental eight bit characters.
  auto elementTy = mlir::dyn_cast<cir::IntType>(firstPtrTy.getPointee());
  if (!elementTy || elementTy.isBitInt() || elementTy.getWidth() != 8)
    return;

  mlir::Location loc = equalOp.getLoc();
  builder.setInsertionPointAfter(equalOp);
  mlir::Value first1 = equalOp.getFirst1();
  mlir::Value last1 = equalOp.getLast1();
  mlir::Value first2 = equalOp.getFirst2();
  if (mlir::isa<cir::RecordType>(first1.getType())) {
    first1 = cir::ExtractMemberOp::create(builder, loc, first1, 0);
    last1 = cir::ExtractMemberOp::create(builder, loc, last1, 0);
  }
  if (mlir::isa<cir::RecordType>(first2.getType()))
    first2 = cir::ExtractMemberOp::create(builder, loc, first2, 0);

  // An empty range may use null pointers while C requires valid pointers even
  // when the memcmp length is zero.
  mlir::Value isEmpty =
      builder.createCompare(loc, cir::CmpOpKind::eq, first1, last1);
  mlir::Value result =
      cir::TernaryOp::create(
          builder, loc, isEmpty,
          [&](mlir::OpBuilder &, mlir::Location) {
            builder.createYield(loc, builder.getTrue(loc).getResult());
          },
          [&](mlir::OpBuilder &, mlir::Location) {
            mlir::Value lhs =
                builder.createBitcast(loc, first1, builder.getVoidPtrTy());
            mlir::Value rhs =
                builder.createBitcast(loc, first2, builder.getVoidPtrTy());
            mlir::Value len = cir::PtrDiffOp::create(
                builder, loc, builder.getUIntNTy(*sizeWidth), last1, first1);
            builder.createYield(
                loc, emitMemCmpIsEqual(builder, loc, lhs, rhs, len,
                                       builder.getSIntNTy(*intWidth)));
          })
          .getResult();
  equalOp.getResult().replaceAllUsesWith(result);
  equalOp.erase();
  ++numEqualToMemcmp;
}

// Rewrites a raised mismatch over licensed integers into a chunked
// memcmp loop. The offset and block size count elements. Each memcmp length
// is scaled to bytes and an unequal chunk is rescanned with typed loads.
template <typename OpT>
static void
rewriteMismatchToMemcmpLoop(OpT mismatchOp,
                            mlir::SymbolTableCollection &symbolTables,
                            mlir::Pass::Statistic &numMismatchToMemcmpLoop) {
  constexpr bool boundedOp = std::is_same_v<OpT, StdMismatchBoundedOp> ||
                             std::is_same_v<OpT, StdMismatchBoundedPredOp>;
  constexpr bool predOp = std::is_same_v<OpT, StdMismatchPredOp> ||
                          std::is_same_v<OpT, StdMismatchBoundedPredOp>;
  static_assert(std::is_same_v<OpT, StdMismatchOp> ||
                std::is_same_v<OpT, StdMismatchBoundedOp> ||
                std::is_same_v<OpT, StdMismatchPredOp> ||
                std::is_same_v<OpT, StdMismatchBoundedPredOp>);
  constexpr llvm::StringLiteral libcallName = "memcmp";
  // The chunk bounds the typed rescan after an unequal memcmp while
  // keeping the call overhead amortized, 256 elements keeps both costs
  // small without tuning per element width.
  constexpr uint64_t blockSize = 256;
  std::optional<LibCallEnv> env =
      checkLibCallEnv(mismatchOp, libcallName, llvm::LibFunc_memcmp);
  if (!env)
    return;
  // The loop carries its offset in an alloca, which has no home in a global
  // initializer region, so only function bodies rewrite.
  if (!env->enclosing)
    return;

  // The introduced operation uses the recorded int and size_t widths which
  // its verifier checks against the result and length.
  std::optional<unsigned> intWidth = cir::getRecordedIntegerWidth(
      env->moduleOp, cir::CIRDialect::getIntTypeWidthAttrName());
  std::optional<unsigned> sizeWidth = cir::getRecordedIntegerWidth(
      env->moduleOp, cir::CIRDialect::getSizeTypeWidthAttrName());
  if (!intWidth || !sizeWidth)
    return;

  CIRBaseBuilderTy builder(*mismatchOp.getContext());
  auto sizeTy = builder.getUIntNTy(*sizeWidth);
  auto intTy = builder.getSIntNTy(*intWidth);
  if (!sharesLibCallSymbol(
          env->moduleOp, symbolTables, libcallName,
          cir::FuncType::get(
              {builder.getVoidPtrTy(), builder.getVoidPtrTy(), sizeTy}, intTy)))
    return;

  // The CIRGen marker licenses trusting a one member record as a contiguous
  // iterator that wraps its pointer. The two ranges are independent, so one
  // may be wrapped while the other is raw.
  cir::RecordType firstWrapperTy;
  cir::RecordType secondWrapperTy;
  cir::PointerType firstPtrTy = unwrapContiguousIterator(
      mismatchOp.getFirst1().getType(), firstWrapperTy);
  cir::PointerType secondPtrTy = unwrapContiguousIterator(
      mismatchOp.getFirst2().getType(), secondWrapperTy);
  // The libc interface cannot represent other address spaces and the two
  // byte sequences must share an element type.
  if (!firstPtrTy || !secondPtrTy || firstPtrTy.getAddrSpace() ||
      secondPtrTy.getAddrSpace() ||
      firstPtrTy.getPointee() != secondPtrTy.getPointee())
    return;
  // BitInt belongs to a separate type family, and the element size
  // scaling below needs an exact byte width.
  auto elementTy = mlir::dyn_cast<cir::IntType>(firstPtrTy.getPointee());
  if (!elementTy || elementTy.isBitInt() || elementTy.getWidth() % 8)
    return;

  const bool wide = elementTy.getWidth() != 8;
  if (wide) {
    // CIRGen marks only elements of the recorded wchar_t width, and the
    // re check here defends against hand written IR whose marker lies
    // about the element.
    std::optional<unsigned> wcharWidth = cir::getRecordedIntegerWidth(
        env->moduleOp, cir::CIRDialect::getWCharTypeWidthAttrName());
    if (!wcharWidth || *wcharWidth != elementTy.getWidth() ||
        !mismatchOp->template getAttrOfType<mlir::UnitAttr>(
            cir::CIRDialect::getWideCharParamsAttrName()))
      return;
  } else if (!predOp && !mismatchOp->template getAttrOfType<mlir::UnitAttr>(
                            cir::CIRDialect::getNarrowCharParamsAttrName())) {
    return;
  }
  // Predicate calls also need the independently proven equality body.
  if constexpr (predOp)
    if (!mismatchOp->template getAttrOfType<mlir::UnitAttr>(
            cir::CIRDialect::getElemEqBinaryPredAttrName()))
      return;

  // The result is the pair record holding both final iterators. Hand
  // written IR can pair up anything, so the members are re proved against
  // the operand types before the rebuild trusts them.
  auto pairTy =
      mlir::dyn_cast<cir::RecordType>(mismatchOp.getResult().getType());
  if (!pairTy || pairTy.isUnion() || !pairTy.isComplete() ||
      pairTy.getMembers().size() != 2 ||
      pairTy.getMembers()[0] != mismatchOp.getFirst1().getType() ||
      pairTy.getMembers()[1] != mismatchOp.getFirst2().getType())
    return;

  mlir::Location loc = mismatchOp.getLoc();
  builder.setInsertionPointAfter(mismatchOp);
  mlir::Value first1 = mismatchOp.getFirst1();
  mlir::Value last1 = mismatchOp.getLast1();
  mlir::Value first2 = mismatchOp.getFirst2();
  mlir::Value last2;
  if constexpr (boundedOp)
    last2 = mismatchOp.getLast2();
  if (firstWrapperTy) {
    first1 = cir::ExtractMemberOp::create(builder, loc, first1, 0);
    last1 = cir::ExtractMemberOp::create(builder, loc, last1, 0);
  }
  if (secondWrapperTy) {
    first2 = cir::ExtractMemberOp::create(builder, loc, first2, 0);
    if constexpr (boundedOp)
      last2 = cir::ExtractMemberOp::create(builder, loc, last2, 0);
  }

  // The bounded forms compare the lesser of the two element counts.
  mlir::Value count =
      cir::PtrDiffOp::create(builder, loc, sizeTy, last1, first1);
  if constexpr (boundedOp) {
    mlir::Value secondCount =
        cir::PtrDiffOp::create(builder, loc, sizeTy, last2, first2);
    mlir::Value firstIsShorter =
        builder.createCompare(loc, cir::CmpOpKind::lt, count, secondCount);
    count = builder.createSelect(loc, firstIsShorter, count, secondCount);
  }
  mlir::Value offAddr =
      builder.createAlloca(loc, builder.getPointerTo(sizeTy), "mismatch_off",
                           builder.getAlignmentAttr(*sizeWidth / 8));
  builder.createStore(loc, builder.getUnsignedInt(loc, 0, *sizeWidth), offAddr);
  mlir::Value block = builder.getUnsignedInt(loc, blockSize, *sizeWidth);
  mlir::Value elementSize =
      builder.getUnsignedInt(loc, elementTy.getWidth() / 8, *sizeWidth);
  mlir::Value zero = builder.getNullValue(intTy, loc);

  auto advance = [&](mlir::Location l, mlir::Value base,
                     mlir::Value off) -> mlir::Value {
    return cir::PtrStrideOp::create(builder, l, base.getType(), base, off);
  };

  // Advance whole chunks while they compare equal. The memcmp sits behind
  // the remaining elements test, so an empty range never reaches C with a
  // possibly null pointer, and every call passes in bounds pointers with a
  // nonzero length.
  builder.createWhile(
      loc,
      [&](mlir::OpBuilder &, mlir::Location l) {
        mlir::Value off = builder.createLoad(l, offAddr);
        mlir::Value more =
            builder.createCompare(l, cir::CmpOpKind::lt, off, count);
        mlir::Value chunkEqual =
            cir::TernaryOp::create(
                builder, l, more,
                [&](mlir::OpBuilder &, mlir::Location ll) {
                  mlir::Value rem = builder.createSub(ll, count, off);
                  mlir::Value overBlock =
                      builder.createCompare(ll, cir::CmpOpKind::gt, rem, block);
                  mlir::Value chunk =
                      builder.createSelect(ll, overBlock, block, rem);
                  mlir::Value lhs = builder.createBitcast(
                      ll, advance(ll, first1, off), builder.getVoidPtrTy());
                  mlir::Value rhs = builder.createBitcast(
                      ll, advance(ll, first2, off), builder.getVoidPtrTy());
                  mlir::Value byteCount =
                      builder.createMul(ll, chunk, elementSize);
                  builder.createYield(ll, emitMemCmpIsEqual(builder, ll, lhs,
                                                            rhs, byteCount,
                                                            intTy, zero));
                },
                [&](mlir::OpBuilder &, mlir::Location ll) {
                  builder.createYield(ll,
                                      builder.getBool(false, ll).getResult());
                })
                .getResult();
        builder.createCondition(chunkEqual);
      },
      [&](mlir::OpBuilder &, mlir::Location l) {
        mlir::Value off = builder.createLoad(l, offAddr);
        mlir::Value rem = builder.createSub(l, count, off);
        mlir::Value overBlock =
            builder.createCompare(l, cir::CmpOpKind::gt, rem, block);
        mlir::Value chunk = builder.createSelect(l, overBlock, block, rem);
        builder.createStore(l, builder.createAdd(l, off, chunk), offAddr);
        builder.createYield(l);
      });

  // Rescan the failing chunk for the exact position. memcmp already
  // proved a differing byte exists there, and a padding free element
  // makes a differing byte a differing element, so the typed loop stops
  // within one chunk. The fully equal input skips it through the
  // remaining elements test that also guards the loads.
  builder.createWhile(
      loc,
      [&](mlir::OpBuilder &, mlir::Location l) {
        mlir::Value off = builder.createLoad(l, offAddr);
        mlir::Value more =
            builder.createCompare(l, cir::CmpOpKind::lt, off, count);
        mlir::Value elementsEqual =
            cir::TernaryOp::create(
                builder, l, more,
                [&](mlir::OpBuilder &, mlir::Location ll) {
                  mlir::Value b1 =
                      builder.createLoad(ll, advance(ll, first1, off));
                  mlir::Value b2 =
                      builder.createLoad(ll, advance(ll, first2, off));
                  builder.createYield(
                      ll, builder.createCompare(ll, cir::CmpOpKind::eq, b1, b2)
                              .getResult());
                },
                [&](mlir::OpBuilder &, mlir::Location ll) {
                  builder.createYield(ll,
                                      builder.getBool(false, ll).getResult());
                })
                .getResult();
        builder.createCondition(elementsEqual);
      },
      [&](mlir::OpBuilder &, mlir::Location l) {
        mlir::Value off = builder.createLoad(l, offAddr);
        builder.createStore(
            l,
            builder.createAdd(l, off, builder.getUnsignedInt(l, 1, *sizeWidth)),
            offAddr);
        builder.createYield(l);
      });

  mlir::Value finalOff = builder.createLoad(loc, offAddr);
  mlir::Value res1 = advance(loc, first1, finalOff);
  mlir::Value res2 = advance(loc, first2, finalOff);
  if (firstWrapperTy)
    res1 = cir::InsertMemberOp::create(builder, loc, mismatchOp.getFirst1(),
                                       /*index=*/0, res1);
  if (secondWrapperTy)
    res2 = cir::InsertMemberOp::create(builder, loc, mismatchOp.getFirst2(),
                                       /*index=*/0, res2);
  // The pair has no incoming value to rebuild around, so it starts from
  // undef and the two inserts define both members.
  mlir::Value pair = builder.getConstant(loc, cir::UndefAttr::get(pairTy));
  pair = cir::InsertMemberOp::create(builder, loc, pair, /*index=*/0, res1);
  pair = cir::InsertMemberOp::create(builder, loc, pair, /*index=*/1, res2);
  mismatchOp.getResult().replaceAllUsesWith(pair);
  mismatchOp.erase();
  ++numMismatchToMemcmpLoop;
}

// Both markers mean seek the recorded value, because CIRGen marks find_if
// only on equality bodies and find_if_not only on inequality bodies, so
// the two forms rewrite identically with no polarity flip here. Loading
// whole words is safe because dereferencing a bit iterator loads its
// whole word too, so the scan touches exactly the word set the source
// touched, and the last word only when its offset is nonzero. The
// verified identity guarantees offsets below the word width, which the
// mask shifts rely on.
template <typename OpT>
static bool
rewriteFindBitToWordScan(OpT findOp,
                         mlir::Pass::Statistic &numFindBitToWordScan) {
  static_assert(std::is_same_v<OpT, StdFindIfOp> ||
                std::is_same_v<OpT, StdFindIfNotOp> ||
                std::is_same_v<OpT, StdRangesFindIfOp> ||
                std::is_same_v<OpT, StdRangesFindIfNotOp>);

  // The loop state needs a function allocation scope
  cir::FuncOp enclosing = findOp->template getParentOfType<cir::FuncOp>();
  if (!enclosing)
    return false;

  llvm::StringRef algorithmName = OpT::getFunctionName();
  // The std and ranges forms share one nobuiltin disable key, the std
  // algorithm family name.
  algorithmName.consume_front("ranges.");
  if (isNoBuiltin(findOp, algorithmName) ||
      noBuiltinListDisables(enclosing, algorithmName))
    return false;

  mlir::BoolAttr predicateValue =
      findOp->template getAttrOfType<mlir::BoolAttr>(
          cir::CIRDialect::getBoolEqPredValueAttrName());

  bool captureByRef = false;
  if (!predicateValue) {
    if (!findOp->template getAttrOfType<mlir::UnitAttr>(
            cir::CIRDialect::getBoolEqPredAttrName()))
      return false;

    auto closureTy =
        mlir::dyn_cast<cir::RecordType>(findOp.getPred().getType());
    if (!closureTy || closureTy.isUnion() || !closureTy.isComplete() ||
        closureTy.getMembers().size() != 1)
      return false;

    mlir::Type captureTy = closureTy.getMembers()[0];
    if (auto capturePtrTy = mlir::dyn_cast<cir::PointerType>(captureTy)) {
      if (capturePtrTy.getAddrSpace() ||
          !mlir::isa<cir::BoolType>(capturePtrTy.getPointee()))
        return false;
      captureByRef = true;
    } else if (!mlir::isa<cir::BoolType>(captureTy)) {
      return false;
    }
  }

  mlir::Type iteratorType = findOp.getFirst().getType();
  auto iteratorTy = mlir::dyn_cast<cir::StructType>(iteratorType);
  if (!iteratorTy || findOp.getLast().getType() != iteratorType ||
      findOp.getResult().getType() != iteratorType)
    return false;

  cir::StdTypeInfoAttr iteratorInfo = iteratorTy.getStdTypeInfo();
  if (!iteratorInfo ||
      iteratorInfo.getKind() != cir::StdTypeKind::StdBitIterator ||
      !mlir::isa<cir::BoolType>(iteratorInfo.getElement()))
    return false;

  llvm::ArrayRef<int32_t> wordPath =
      iteratorInfo.getRolePath(cir::StdTypeInfoAttr::kRoleWordPointer);
  llvm::ArrayRef<int32_t> bitPath =
      iteratorInfo.getRolePath(cir::StdTypeInfoAttr::kRoleBitOffset);
  if (wordPath.empty() || bitPath.empty())
    return false;

  auto wordPtrTy = mlir::dyn_cast<cir::PointerType>(iteratorInfo.resolveRole(
      iteratorTy, cir::StdTypeInfoAttr::kRoleWordPointer));
  if (!wordPtrTy || wordPtrTy.getAddrSpace())
    return false;

  auto wordTy = mlir::dyn_cast<cir::IntType>(wordPtrTy.getPointee());
  if (!wordTy || !wordTy.isUnsigned() || !wordTy.isFundamental())
    return false;

  switch (wordTy.getWidth()) {
  case 8:
  case 16:
  case 32:
  case 64:
    break;
  default:
    return false;
  }

  auto bitOffsetTy = mlir::dyn_cast<cir::IntType>(iteratorInfo.resolveRole(
      iteratorTy, cir::StdTypeInfoAttr::kRoleBitOffset));
  if (!bitOffsetTy || !bitOffsetTy.isUnsigned() || !bitOffsetTy.isFundamental())
    return false;

  CIRBaseBuilderTy builder(*findOp.getContext());
  mlir::Location loc = findOp.getLoc();
  builder.setInsertionPointAfter(findOp);

  auto extractPath = [&](mlir::Location l, mlir::Value record,
                         llvm::ArrayRef<int32_t> path) {
    mlir::Value value = record;
    for (int32_t index : path)
      value = cir::ExtractMemberOp::create(builder, l, value,
                                           static_cast<uint64_t>(index));
    return value;
  };

  // Rebuild each nested parent after replacing the role value
  auto insertPath = [&](mlir::Location l, mlir::Value record,
                        llvm::ArrayRef<int32_t> path, mlir::Value replacement) {
    llvm::SmallVector<mlir::Value, 4> parents;
    mlir::Value nested = record;
    for (size_t depth = 0; depth + 1 < path.size(); ++depth) {
      parents.push_back(nested);
      nested = cir::ExtractMemberOp::create(builder, l, nested,
                                            static_cast<uint64_t>(path[depth]));
    }

    nested = cir::InsertMemberOp::create(
        builder, l, nested, static_cast<uint64_t>(path.back()), replacement);

    for (size_t depth = path.size() - 1; depth > 0; --depth)
      nested = cir::InsertMemberOp::create(
          builder, l, parents[depth - 1],
          static_cast<uint64_t>(path[depth - 1]), nested);

    return nested;
  };

  mlir::Value firstWord = extractPath(loc, findOp.getFirst(), wordPath);
  mlir::Value firstOffset = extractPath(loc, findOp.getFirst(), bitPath);
  mlir::Value lastWord = extractPath(loc, findOp.getLast(), wordPath);
  mlir::Value lastOffset = extractPath(loc, findOp.getLast(), bitPath);

  mlir::Value sameWord =
      builder.createCompare(loc, cir::CmpOpKind::eq, firstWord, lastWord);
  mlir::Value sameOffset =
      builder.createCompare(loc, cir::CmpOpKind::eq, firstOffset, lastOffset);
  mlir::Value empty = builder.createLogicalAnd(loc, sameWord, sameOffset);

  mlir::Value result =
      cir::TernaryOp::create(
          builder, loc, empty,
          [&](mlir::OpBuilder &, mlir::Location l) {
            builder.createYield(l, findOp.getLast());
          },
          [&](mlir::OpBuilder &, mlir::Location l) {
            mlir::Value sought;
            if (predicateValue) {
              sought = builder.getBool(predicateValue.getValue(), l);
            } else {
              sought =
                  cir::ExtractMemberOp::create(builder, l, findOp.getPred(), 0);
              if (captureByRef)
                sought = builder.createLoad(l, sought);
            }

            mlir::Value wordZero =
                builder.getUnsignedInt(l, 0, wordTy.getWidth());
            mlir::Value wordOne =
                builder.getUnsignedInt(l, 1, wordTy.getWidth());
            mlir::Value wordAllOnes = builder.getConstAPInt(
                l, wordTy, llvm::APInt::getAllOnes(wordTy.getWidth()));
            mlir::Value bitZero =
                builder.getUnsignedInt(l, 0, bitOffsetTy.getWidth());
            mlir::Value oneStride =
                builder.getUnsignedInt(l, 1, bitOffsetTy.getWidth());

            mlir::IntegerAttr ptrAlign = builder.getAlignmentAttr(8);
            mlir::IntegerAttr offsetAlign =
                builder.getAlignmentAttr(bitOffsetTy.getWidth() / 8);
            mlir::Value foundWordAddr = builder.createAlloca(
                l, builder.getPointerTo(wordPtrTy), "find_bit_word", ptrAlign);
            mlir::Value foundOffsetAddr =
                builder.createAlloca(l, builder.getPointerTo(bitOffsetTy),
                                     "find_bit_offset", offsetAlign);
            mlir::Value currentWordAddr =
                builder.createAlloca(l, builder.getPointerTo(wordPtrTy),
                                     "find_bit_current", ptrAlign);
            mlir::Value foundAddr = builder.createAlloca(
                l, builder.getPointerTo(builder.getBoolTy()), "find_bit_found",
                builder.getAlignmentAttr(1));
            builder.createStore(l, builder.getFalse(l), foundAddr);

            // The load reads the whole word the iterator would read, and
            // the mask discards every bit outside the searched range
            // before the candidates test.
            auto scanWord = [&](mlir::Location scanLoc, mlir::Value wordPointer,
                                mlir::Value validMask) {
              mlir::Value loaded = builder.createLoad(scanLoc, wordPointer);
              mlir::Value inverted = builder.createNot(scanLoc, loaded);
              mlir::Value searched =
                  builder.createSelect(scanLoc, sought, loaded, inverted);
              mlir::Value candidates =
                  builder.createAnd(scanLoc, searched, validMask);
              mlir::Value nonzero = builder.createCompare(
                  scanLoc, cir::CmpOpKind::ne, candidates, wordZero);

              cir::TernaryOp::create(
                  builder, scanLoc, nonzero,
                  [&](mlir::OpBuilder &, mlir::Location foundLoc) {
                    mlir::Value bit = cir::BitCtzOp::create(builder, foundLoc,
                                                            candidates, true);
                    bit = builder.createIntCast(bit, bitOffsetTy);
                    builder.createStore(foundLoc, wordPointer, foundWordAddr);
                    builder.createStore(foundLoc, bit, foundOffsetAddr);
                    builder.createStore(foundLoc, builder.getTrue(foundLoc),
                                        foundAddr);
                    builder.createYield(foundLoc);
                  },
                  [&](mlir::OpBuilder &, mlir::Location missLoc) {
                    builder.createYield(missLoc);
                  });
            };

            cir::TernaryOp::create(
                builder, l, sameWord,
                [&](mlir::OpBuilder &, mlir::Location sameLoc) {
                  mlir::Value headMask = builder.createShiftLeft(
                      sameLoc, wordAllOnes, firstOffset);
                  mlir::Value endMask = builder.createSub(
                      sameLoc,
                      builder.createShiftLeft(sameLoc, wordOne, lastOffset),
                      wordOne);
                  mlir::Value validMask =
                      builder.createAnd(sameLoc, headMask, endMask);
                  mlir::Value hasValidBits = builder.createCompare(
                      sameLoc, cir::CmpOpKind::ne, validMask, wordZero);

                  // A zero same word mask must guard the load
                  cir::TernaryOp::create(
                      builder, sameLoc, hasValidBits,
                      [&](mlir::OpBuilder &, mlir::Location scanLoc) {
                        scanWord(scanLoc, firstWord, validMask);
                        builder.createYield(scanLoc);
                      },
                      [&](mlir::OpBuilder &, mlir::Location missLoc) {
                        builder.createYield(missLoc);
                      });
                  builder.createYield(sameLoc);
                },
                [&](mlir::OpBuilder &, mlir::Location differentLoc) {
                  mlir::Value hasHead = builder.createCompare(
                      differentLoc, cir::CmpOpKind::ne, firstOffset, bitZero);
                  mlir::Value scanBegin =
                      cir::TernaryOp::create(
                          builder, differentLoc, hasHead,
                          [&](mlir::OpBuilder &, mlir::Location headLoc) {
                            mlir::Value headMask = builder.createShiftLeft(
                                headLoc, wordAllOnes, firstOffset);
                            scanWord(headLoc, firstWord, headMask);
                            mlir::Value foundHead =
                                builder.createLoad(headLoc, foundAddr);
                            mlir::Value next = cir::PtrStrideOp::create(
                                builder, headLoc, firstWord.getType(),
                                firstWord, oneStride);
                            builder.createYield(headLoc, builder.createSelect(
                                                             headLoc, foundHead,
                                                             lastWord, next));
                          },
                          [&](mlir::OpBuilder &, mlir::Location wholeLoc) {
                            builder.createYield(wholeLoc, firstWord);
                          })
                          .getResult();
                  builder.createStore(differentLoc, scanBegin, currentWordAddr);

                  builder.createWhile(
                      differentLoc,
                      [&](mlir::OpBuilder &, mlir::Location conditionLoc) {
                        mlir::Value current =
                            builder.createLoad(conditionLoc, currentWordAddr);
                        mlir::Value more = builder.createCompare(
                            conditionLoc, cir::CmpOpKind::ne, current,
                            lastWord);
                        builder.createCondition(more);
                      },
                      [&](mlir::OpBuilder &, mlir::Location bodyLoc) {
                        mlir::Value current =
                            builder.createLoad(bodyLoc, currentWordAddr);
                        scanWord(bodyLoc, current, wordAllOnes);
                        mlir::Value found =
                            builder.createLoad(bodyLoc, foundAddr);
                        mlir::Value next = cir::PtrStrideOp::create(
                            builder, bodyLoc, current.getType(), current,
                            oneStride);
                        mlir::Value updated = builder.createSelect(
                            bodyLoc, found, lastWord, next);
                        builder.createStore(bodyLoc, updated, currentWordAddr);
                        builder.createYield(bodyLoc);
                      });

                  mlir::Value foundBeforeTail =
                      builder.createLoad(differentLoc, foundAddr);
                  mlir::Value notFound =
                      builder.createNot(differentLoc, foundBeforeTail);
                  mlir::Value hasTail = builder.createCompare(
                      differentLoc, cir::CmpOpKind::ne, lastOffset, bitZero);
                  mlir::Value needTail =
                      builder.createLogicalAnd(differentLoc, notFound, hasTail);

                  cir::TernaryOp::create(
                      builder, differentLoc, needTail,
                      [&](mlir::OpBuilder &, mlir::Location tailLoc) {
                        mlir::Value tailMask =
                            builder.createSub(tailLoc,
                                              builder.createShiftLeft(
                                                  tailLoc, wordOne, lastOffset),
                                              wordOne);
                        scanWord(tailLoc, lastWord, tailMask);
                        builder.createYield(tailLoc);
                      },
                      [&](mlir::OpBuilder &, mlir::Location skipLoc) {
                        builder.createYield(skipLoc);
                      });
                  builder.createYield(differentLoc);
                });

            mlir::Value found = builder.createLoad(l, foundAddr);
            mlir::Value scanResult =
                cir::TernaryOp::create(
                    builder, l, found,
                    [&](mlir::OpBuilder &, mlir::Location foundLoc) {
                      mlir::Value rebuilt = findOp.getLast();
                      rebuilt = insertPath(
                          foundLoc, rebuilt, wordPath,
                          builder.createLoad(foundLoc, foundWordAddr));
                      rebuilt = insertPath(
                          foundLoc, rebuilt, bitPath,
                          builder.createLoad(foundLoc, foundOffsetAddr));
                      builder.createYield(foundLoc, rebuilt);
                    },
                    [&](mlir::OpBuilder &, mlir::Location missLoc) {
                      builder.createYield(missLoc, findOp.getLast());
                    })
                    .getResult();
            builder.createYield(l, scanResult);
          })
          .getResult();

  findOp.getResult().replaceAllUsesWith(result);
  findOp.erase();
  ++numFindBitToWordScan;
  return true;
}

void LibOptPass::runOnOperation() {
  mlir::SymbolTableCollection symbolTables;
  getOperation()->walk([&](mlir::Operation *op) {
    llvm::TypeSwitch<mlir::Operation *>(op)
        .Case<StdFindOp, StdRangesFindOp, StdRangesFindRangeOp>([&](auto find) {
          rewriteFindLikeToMemchr(find, symbolTables, numFindLikeToMemchr,
                                  numFindLikeToWmemchr);
        })
        .Case<StdFindIfOp, StdFindIfNotOp, StdRangesFindIfOp,
              StdRangesFindIfNotOp>([&](auto find) {
          // The bit and byte rewrites accept disjoint iterator shapes, so
          // whichever declines leaves the operation for the other.
          if (!rewriteFindBitToWordScan(find, numFindBitToWordScan))
            rewriteFindLikeToMemchr(find, symbolTables, numFindLikeToMemchr,
                                    numFindLikeToWmemchr);
        })
        .Case<StdSearchOp>([&](auto search) {
          rewriteSearchToMemmem(search, symbolTables, numSearchToMemmem,
                                numSearchEqualLengthToMemcmp);
        })
        .Case<StdEqualOp, StdEqualPredOp>([&](auto equal) {
          rewriteEqualToMemcmp(equal, symbolTables, numEqualToMemcmp);
        })
        .Case<StdMismatchOp, StdMismatchBoundedOp, StdMismatchPredOp,
              StdMismatchBoundedPredOp>([&](auto mismatch) {
          rewriteMismatchToMemcmpLoop(mismatch, symbolTables,
                                      numMismatchToMemcmpLoop);
        });
  });
}

std::unique_ptr<Pass> mlir::createLibOptPass() {
  return std::make_unique<LibOptPass>();
}
