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
#include "llvm/ADT/SmallSet.h"
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

// Rewrites cir.std.find and the equality predicate form of cir.std.find_if
// over standard deque iterators into a block walk that calls memchr or
// wmemchr once per block. The identity carries the member paths to the
// current element pointer and the map slot pointer plus the block element
// count, so the walk reaches each block through the same map load iterator
// increment performs. A hit rebuilds the result iterator from the finish
// iterator, a miss returns the finish iterator whole.
template <typename OpT>
static bool
rewriteFindDequeToBlockWalk(OpT findOp,
                            mlir::SymbolTableCollection &symbolTables,
                            mlir::Pass::Statistic &numFindDequeToMemchr,
                            mlir::Pass::Statistic &numFindDequeToWmemchr) {
  static_assert(std::is_same_v<OpT, StdFindOp> ||
                std::is_same_v<OpT, StdFindIfOp>);
  constexpr bool predOp = std::is_same_v<OpT, StdFindIfOp>;

  // The operation verifier makes first, last, and result one type, so this
  // type speaks for all three.
  auto iteratorTy =
      mlir::dyn_cast<cir::StructType>(findOp.getFirst().getType());
  if (!iteratorTy)
    return false;

  cir::StdTypeInfoAttr iteratorInfo = iteratorTy.getStdTypeInfo();
  if (!iteratorInfo ||
      iteratorInfo.getKind() != cir::StdTypeKind::StdDequeIterator)
    return false;

  auto elemTy = mlir::dyn_cast<cir::IntType>(iteratorInfo.getElement());
  if (!elemTy)
    return false;
  auto elemPtrTy = cir::PointerType::get(elemTy);
  auto currentPointerTy =
      mlir::dyn_cast<cir::PointerType>(iteratorInfo.resolveRole(
          iteratorTy, cir::StdTypeInfoAttr::kRoleCurrentPointer));
  auto mapPointerTy = mlir::dyn_cast<cir::PointerType>(iteratorInfo.resolveRole(
      iteratorTy, cir::StdTypeInfoAttr::kRoleMapPointer));
  auto blockPointerTy =
      mapPointerTy ? mlir::dyn_cast<cir::PointerType>(mapPointerTy.getPointee())
                   : cir::PointerType();
  if (currentPointerTy != elemPtrTy || !mapPointerTy ||
      mapPointerTy.getAddrSpace() || blockPointerTy != elemPtrTy)
    return false;

  llvm::ArrayRef<int32_t> currentPointerPath =
      iteratorInfo.getRolePath(cir::StdTypeInfoAttr::kRoleCurrentPointer);
  llvm::ArrayRef<int32_t> mapPointerPath =
      iteratorInfo.getRolePath(cir::StdTypeInfoAttr::kRoleMapPointer);
  llvm::ArrayRef<int32_t> blockFirstPointerPath =
      iteratorInfo.getRolePath(cir::StdTypeInfoAttr::kRoleBlockFirstPointer);
  llvm::ArrayRef<int32_t> blockLastPointerPath =
      iteratorInfo.getRolePath(cir::StdTypeInfoAttr::kRoleBlockLastPointer);
  if (currentPointerPath.empty() || mapPointerPath.empty())
    return false;
  // The hit result copies the finish iterator and replaces the role
  // members, so an identity whose roles name fewer members than the record
  // holds would leave stale per position state in the rebuilt result. Hand
  // written IR can carry such an identity, so the roles must land on
  // distinct members and cover the whole record before any rewrite.
  llvm::SmallVector<llvm::ArrayRef<int32_t>, 4> rolePaths{currentPointerPath,
                                                          mapPointerPath};
  if (!blockFirstPointerPath.empty())
    rolePaths.push_back(blockFirstPointerPath);
  if (!blockLastPointerPath.empty())
    rolePaths.push_back(blockLastPointerPath);
  llvm::SmallSet<int32_t, 4> coveredMembers;
  for (llvm::ArrayRef<int32_t> path : rolePaths)
    if (path.size() != 1 || !coveredMembers.insert(path[0]).second)
      return false;
  if (coveredMembers.size() != iteratorTy.getMembers().size())
    return false;
  if (!blockFirstPointerPath.empty() &&
      (iteratorInfo.resolveRole(iteratorTy,
                                cir::StdTypeInfoAttr::kRoleBlockFirstPointer) !=
           elemPtrTy ||
       iteratorInfo.resolveRole(iteratorTy,
                                cir::StdTypeInfoAttr::kRoleBlockLastPointer) !=
           elemPtrTy))
    return false;

  // The attribute verifier admits only a positive signless 64 bit block
  // size on a deque identity, so the entry reads unchecked here.
  uint64_t blockSize = static_cast<uint64_t>(
      mlir::cast<mlir::IntegerAttr>(
          iteratorInfo.getRoles().get(cir::StdTypeInfoAttr::kBlockSize))
          .getInt());

  const bool wide = elemTy.getWidth() != 8;
  llvm::StringRef libcallName = wide ? "wmemchr" : "memchr";
  bool captureByRef = false;
  cir::IntAttr predicateValue;
  if (!matchEqSoughtShape(findOp, elemTy, wide, predicateValue, captureByRef))
    return false;

  std::optional<LibCallEnv> env = checkLibCallEnv(
      findOp, libcallName, wide ? llvm::LibFunc_wmemchr : llvm::LibFunc_memchr);
  if (!env)
    return false;
  // The walk carries its state in allocas, which have no home in a global
  // initializer region, so only function bodies rewrite.
  if (!env->enclosing)
    return false;

  if (!hasEqSearchMarker(findOp, wide, predicateValue))
    return false;

  unsigned patternWidth, sizeWidth;
  if (!resolveMemchrWidths(env->moduleOp, env->triple, wide, elemTy,
                           patternWidth, sizeWidth))
    return false;
  // The span lengths the calls receive are element counts within one
  // block, so the block count itself must be representable in size_t.
  if (sizeWidth < 64 && blockSize >= (uint64_t{1} << sizeWidth))
    return false;

  CIRBaseBuilderTy builder(*findOp.getContext());
  cir::IntType callElemTy;
  cir::PointerType callIterTy;
  if (!resolveMemchrCallTypes(builder, env->moduleOp, symbolTables, wide,
                              libcallName, patternWidth, sizeWidth, elemTy,
                              elemPtrTy, callElemTy, callIterTy))
    return false;

  mlir::Location loc = findOp.getLoc();
  builder.setInsertionPointAfter(findOp);
  auto extractPath = [&](mlir::Location pathLoc, mlir::Value record,
                         llvm::ArrayRef<int32_t> path) {
    mlir::Value value = record;
    for (int32_t index : path)
      value = cir::ExtractMemberOp::create(builder, pathLoc, value,
                                           static_cast<uint64_t>(index));
    return value;
  };
  auto insertPath = [&](mlir::Location pathLoc, mlir::Value record,
                        llvm::ArrayRef<int32_t> path, mlir::Value replacement) {
    llvm::SmallVector<mlir::Value, 4> parents;
    mlir::Value nested = record;
    for (size_t depth = 0; depth + 1 < path.size(); ++depth) {
      parents.push_back(nested);
      nested = cir::ExtractMemberOp::create(builder, pathLoc, nested,
                                            static_cast<uint64_t>(path[depth]));
    }
    nested = cir::InsertMemberOp::create(builder, pathLoc, nested,
                                         static_cast<uint64_t>(path.back()),
                                         replacement);
    for (size_t depth = path.size() - 1; depth > 0; --depth)
      nested = cir::InsertMemberOp::create(
          builder, pathLoc, parents[depth - 1],
          static_cast<uint64_t>(path[depth - 1]), nested);
    return nested;
  };

  mlir::Value firstCurrent =
      extractPath(loc, findOp.getFirst(), currentPointerPath);
  mlir::Value lastCurrent =
      extractPath(loc, findOp.getLast(), currentPointerPath);
  mlir::Value firstMap = extractPath(loc, findOp.getFirst(), mapPointerPath);
  mlir::Value lastMap = extractPath(loc, findOp.getLast(), mapPointerPath);
  mlir::Value sameMap =
      builder.createCompare(loc, cir::CmpOpKind::eq, firstMap, lastMap);
  mlir::Value sameCurrent =
      builder.createCompare(loc, cir::CmpOpKind::eq, firstCurrent, lastCurrent);
  mlir::Value empty = builder.createLogicalAnd(loc, sameMap, sameCurrent);

  mlir::Value result =
      cir::TernaryOp::create(
          builder, loc, empty,
          [&](mlir::OpBuilder &, mlir::Location emptyLoc) {
            builder.createYield(emptyLoc, findOp.getLast());
          },
          [&](mlir::OpBuilder &, mlir::Location walkLoc) {
            mlir::Value sought;
            if constexpr (predOp) {
              if (predicateValue) {
                sought = builder.getConstant(walkLoc, predicateValue);
              } else {
                sought = cir::ExtractMemberOp::create(builder, walkLoc,
                                                      findOp.getPred(), 0);
                if (captureByRef)
                  sought = builder.createLoad(walkLoc, sought);
              }
            } else {
              sought = builder.createLoad(walkLoc, findOp.getPattern());
            }

            mlir::IntegerAttr pointerAlign = builder.getAlignmentAttr(8);
            mlir::Value foundPointerAddr =
                builder.createAlloca(walkLoc, builder.getPointerTo(elemPtrTy),
                                     "find_deque_pointer", pointerAlign);
            mlir::Value foundMapAddr = builder.createAlloca(
                walkLoc, builder.getPointerTo(mapPointerTy), "find_deque_map",
                pointerAlign);
            mlir::Value currentMapAddr = builder.createAlloca(
                walkLoc, builder.getPointerTo(mapPointerTy),
                "find_deque_current_map", pointerAlign);
            mlir::Value foundAddr = builder.createAlloca(
                walkLoc, builder.getPointerTo(builder.getBoolTy()),
                "find_deque_found", builder.getAlignmentAttr(1));
            builder.createStore(walkLoc, builder.getFalse(walkLoc), foundAddr);

            mlir::Type sizeTy = builder.getUIntNTy(sizeWidth);
            mlir::Value blockCount =
                builder.getUnsignedInt(walkLoc, blockSize, sizeWidth);
            mlir::Value one = builder.getUnsignedInt(walkLoc, 1, sizeWidth);
            auto advance = [&](mlir::Location advanceLoc, mlir::Value base,
                               mlir::Value offset) {
              return cir::PtrStrideOp::create(builder, advanceLoc,
                                              base.getType(), base, offset)
                  .getResult();
            };
            auto loadBlock = [&](mlir::Location loadLoc, mlir::Value mapSlot) {
              // The identity licenses this load because iterator increment
              // performs the same load whenever it enters a block
              return builder.createLoad(loadLoc, mapSlot);
            };
            auto scanSpan = [&](mlir::Location scanLoc, mlir::Value begin,
                                mlir::Value end, mlir::Value mapSlot) {
              mlir::Value hasElements = builder.createCompare(
                  scanLoc, cir::CmpOpKind::ne, begin, end);
              cir::TernaryOp::create(
                  builder, scanLoc, hasElements,
                  [&](mlir::OpBuilder &, mlir::Location callLoc) {
                    mlir::Value len = cir::PtrDiffOp::create(
                        builder, callLoc, sizeTy, end, begin);
                    mlir::Value hit;
                    if (wide) {
                      mlir::Value src = begin;
                      mlir::Value pattern = sought;
                      if (callElemTy != elemTy) {
                        src = builder.createBitcast(callLoc, begin, callIterTy);
                        pattern = builder.createIntCast(sought, callElemTy);
                      }
                      hit = cir::WMemChrOp::create(builder, callLoc, src,
                                                   pattern, len);
                      if (callElemTy != elemTy)
                        hit = builder.createBitcast(callLoc, hit, elemPtrTy);
                    } else {
                      mlir::Value src = builder.createBitcast(
                          callLoc, begin, builder.getVoidPtrTy());
                      mlir::Value pattern = builder.createIntCast(
                          sought, builder.getSIntNTy(patternWidth));
                      hit = cir::MemChrOp::create(builder, callLoc, src,
                                                  pattern, len);
                      hit = builder.createBitcast(callLoc, hit, elemPtrTy);
                    }
                    mlir::Value hasHit = builder.createNot(
                        callLoc, builder.createPtrIsNull(hit));
                    cir::TernaryOp::create(
                        builder, callLoc, hasHit,
                        [&](mlir::OpBuilder &, mlir::Location hitLoc) {
                          builder.createStore(hitLoc, hit, foundPointerAddr);
                          builder.createStore(hitLoc, mapSlot, foundMapAddr);
                          builder.createStore(hitLoc, builder.getTrue(hitLoc),
                                              foundAddr);
                          builder.createYield(hitLoc);
                        },
                        [&](mlir::OpBuilder &, mlir::Location missLoc) {
                          builder.createYield(missLoc);
                        });
                    builder.createYield(callLoc);
                  },
                  [&](mlir::OpBuilder &, mlir::Location skipLoc) {
                    builder.createYield(skipLoc);
                  });
            };

            cir::TernaryOp::create(
                builder, walkLoc, sameMap,
                [&](mlir::OpBuilder &, mlir::Location sameLoc) {
                  scanSpan(sameLoc, firstCurrent, lastCurrent, firstMap);
                  builder.createYield(sameLoc);
                },
                [&](mlir::OpBuilder &, mlir::Location differentLoc) {
                  mlir::Value firstBlock = loadBlock(differentLoc, firstMap);
                  mlir::Value firstBlockEnd =
                      advance(differentLoc, firstBlock, blockCount);
                  scanSpan(differentLoc, firstCurrent, firstBlockEnd, firstMap);

                  mlir::Value foundFirst =
                      builder.createLoad(differentLoc, foundAddr);
                  mlir::Value nextMap = advance(differentLoc, firstMap, one);
                  builder.createStore(differentLoc,
                                      builder.createSelect(differentLoc,
                                                           foundFirst, lastMap,
                                                           nextMap),
                                      currentMapAddr);
                  builder.createWhile(
                      differentLoc,
                      [&](mlir::OpBuilder &, mlir::Location conditionLoc) {
                        mlir::Value currentMap =
                            builder.createLoad(conditionLoc, currentMapAddr);
                        builder.createCondition(builder.createCompare(
                            conditionLoc, cir::CmpOpKind::ne, currentMap,
                            lastMap));
                      },
                      [&](mlir::OpBuilder &, mlir::Location bodyLoc) {
                        mlir::Value currentMap =
                            builder.createLoad(bodyLoc, currentMapAddr);
                        mlir::Value block = loadBlock(bodyLoc, currentMap);
                        scanSpan(bodyLoc, block,
                                 advance(bodyLoc, block, blockCount),
                                 currentMap);
                        mlir::Value found =
                            builder.createLoad(bodyLoc, foundAddr);
                        mlir::Value followingMap =
                            advance(bodyLoc, currentMap, one);
                        builder.createStore(bodyLoc,
                                            builder.createSelect(bodyLoc, found,
                                                                 lastMap,
                                                                 followingMap),
                                            currentMapAddr);
                        builder.createYield(bodyLoc);
                      });

                  mlir::Value foundBeforeLast =
                      builder.createLoad(differentLoc, foundAddr);
                  cir::TernaryOp::create(
                      builder, differentLoc,
                      builder.createNot(differentLoc, foundBeforeLast),
                      [&](mlir::OpBuilder &, mlir::Location lastLoc) {
                        mlir::Value lastBlock = loadBlock(lastLoc, lastMap);
                        scanSpan(lastLoc, lastBlock, lastCurrent, lastMap);
                        builder.createYield(lastLoc);
                      },
                      [&](mlir::OpBuilder &, mlir::Location skipLoc) {
                        builder.createYield(skipLoc);
                      });
                  builder.createYield(differentLoc);
                });

            mlir::Value found = builder.createLoad(walkLoc, foundAddr);
            mlir::Value walkResult =
                cir::TernaryOp::create(
                    builder, walkLoc, found,
                    [&](mlir::OpBuilder &, mlir::Location hitLoc) {
                      mlir::Value hitPointer =
                          builder.createLoad(hitLoc, foundPointerAddr);
                      mlir::Value hitMap =
                          builder.createLoad(hitLoc, foundMapAddr);
                      mlir::Value rebuilt = findOp.getLast();
                      rebuilt = insertPath(hitLoc, rebuilt, currentPointerPath,
                                           hitPointer);
                      rebuilt =
                          insertPath(hitLoc, rebuilt, mapPointerPath, hitMap);
                      if (!blockFirstPointerPath.empty()) {
                        mlir::Value block = loadBlock(hitLoc, hitMap);
                        rebuilt = insertPath(hitLoc, rebuilt,
                                             blockFirstPointerPath, block);
                        rebuilt =
                            insertPath(hitLoc, rebuilt, blockLastPointerPath,
                                       advance(hitLoc, block, blockCount));
                      }
                      builder.createYield(hitLoc, rebuilt);
                    },
                    [&](mlir::OpBuilder &, mlir::Location missLoc) {
                      builder.createYield(missLoc, findOp.getLast());
                    })
                    .getResult();
            builder.createYield(walkLoc, walkResult);
          })
          .getResult();

  findOp.getResult().replaceAllUsesWith(result);
  findOp.erase();
  if (wide)
    ++numFindDequeToWmemchr;
  else
    ++numFindDequeToMemchr;
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

struct LicensedDequeCompareIterator {
  cir::StdTypeInfoAttr identity;
  cir::PointerType elementPointerTy;
  cir::PointerType mapPointerTy;
  uint64_t currentPointerIndex;
  uint64_t mapPointerIndex;
  std::optional<uint64_t> blockFirstPointerIndex;
  std::optional<uint64_t> blockLastPointerIndex;
  uint64_t blockSize;
};

static std::optional<LicensedDequeCompareIterator>
licenseDequeCompareIterator(mlir::Type type) {
  auto iteratorTy = mlir::dyn_cast<cir::StructType>(type);
  if (!iteratorTy)
    return std::nullopt;

  cir::StdTypeInfoAttr identity = iteratorTy.getStdTypeInfo();
  if (!identity || identity.getKind() != cir::StdTypeKind::StdDequeIterator)
    return std::nullopt;

  auto elementTy = mlir::dyn_cast<cir::IntType>(identity.getElement());
  if (!elementTy)
    return std::nullopt;
  auto elementPointerTy = cir::PointerType::get(elementTy);
  auto currentPointerTy = mlir::dyn_cast<cir::PointerType>(identity.resolveRole(
      iteratorTy, cir::StdTypeInfoAttr::kRoleCurrentPointer));
  auto mapPointerTy = mlir::dyn_cast<cir::PointerType>(
      identity.resolveRole(iteratorTy, cir::StdTypeInfoAttr::kRoleMapPointer));
  auto blockPointerTy =
      mapPointerTy ? mlir::dyn_cast<cir::PointerType>(mapPointerTy.getPointee())
                   : cir::PointerType();
  if (currentPointerTy != elementPointerTy || !mapPointerTy ||
      mapPointerTy.getAddrSpace() || blockPointerTy != elementPointerTy)
    return std::nullopt;

  llvm::ArrayRef<int32_t> currentPointerPath =
      identity.getRolePath(cir::StdTypeInfoAttr::kRoleCurrentPointer);
  llvm::ArrayRef<int32_t> mapPointerPath =
      identity.getRolePath(cir::StdTypeInfoAttr::kRoleMapPointer);
  llvm::ArrayRef<int32_t> blockFirstPointerPath =
      identity.getRolePath(cir::StdTypeInfoAttr::kRoleBlockFirstPointer);
  llvm::ArrayRef<int32_t> blockLastPointerPath =
      identity.getRolePath(cir::StdTypeInfoAttr::kRoleBlockLastPointer);
  if (currentPointerPath.size() != 1 || mapPointerPath.size() != 1)
    return std::nullopt;

  // The result rebuild copies a seed iterator and replaces the role
  // members, so an identity whose roles name fewer members than the
  // record holds would leave stale per position state in the rebuilt
  // result. The roles must land on distinct members and cover the whole
  // record, the same rule the find walk enforces.
  llvm::SmallVector<llvm::ArrayRef<int32_t>, 4> rolePaths{currentPointerPath,
                                                          mapPointerPath};
  if (!blockFirstPointerPath.empty()) {
    rolePaths.push_back(blockFirstPointerPath);
    rolePaths.push_back(blockLastPointerPath);
  }
  llvm::SmallSet<int32_t, 4> coveredMembers;
  for (llvm::ArrayRef<int32_t> path : rolePaths)
    if (path.size() != 1 || !coveredMembers.insert(path[0]).second)
      return std::nullopt;
  if (coveredMembers.size() != iteratorTy.getMembers().size())
    return std::nullopt;

  if (!blockFirstPointerPath.empty() &&
      (identity.resolveRole(iteratorTy,
                            cir::StdTypeInfoAttr::kRoleBlockFirstPointer) !=
           elementPointerTy ||
       identity.resolveRole(iteratorTy,
                            cir::StdTypeInfoAttr::kRoleBlockLastPointer) !=
           elementPointerTy))
    return std::nullopt;

  // The attribute verifier admits only a positive signless 64 bit block
  // size on a deque identity, so the entry reads unchecked here.
  uint64_t blockSize = static_cast<uint64_t>(
      mlir::cast<mlir::IntegerAttr>(
          identity.getRoles().get(cir::StdTypeInfoAttr::kBlockSize))
          .getInt());

  LicensedDequeCompareIterator licensed;
  licensed.identity = identity;
  licensed.elementPointerTy = elementPointerTy;
  licensed.mapPointerTy = mapPointerTy;
  licensed.currentPointerIndex = static_cast<uint64_t>(currentPointerPath[0]);
  licensed.mapPointerIndex = static_cast<uint64_t>(mapPointerPath[0]);
  licensed.blockSize = blockSize;
  if (!blockFirstPointerPath.empty()) {
    licensed.blockFirstPointerIndex =
        static_cast<uint64_t>(blockFirstPointerPath[0]);
    licensed.blockLastPointerIndex =
        static_cast<uint64_t>(blockLastPointerPath[0]);
  }
  return licensed;
}

template <typename OpT>
static bool rewriteDequeCompareToBlockWalk(
    OpT compareOp, mlir::SymbolTableCollection &symbolTables,
    mlir::Pass::Statistic &numEqualDequeToMemcmp,
    mlir::Pass::Statistic &numMismatchDequeToMemcmp) {
  constexpr bool isMismatchForm = std::is_same_v<OpT, StdMismatchOp> ||
                                  std::is_same_v<OpT, StdMismatchBoundedOp> ||
                                  std::is_same_v<OpT, StdMismatchPredOp> ||
                                  std::is_same_v<OpT, StdMismatchBoundedPredOp>;
  constexpr bool boundedOp = std::is_same_v<OpT, StdMismatchBoundedOp> ||
                             std::is_same_v<OpT, StdMismatchBoundedPredOp>;
  constexpr bool predOp = std::is_same_v<OpT, StdEqualPredOp> ||
                          std::is_same_v<OpT, StdMismatchPredOp> ||
                          std::is_same_v<OpT, StdMismatchBoundedPredOp>;
  static_assert(std::is_same_v<OpT, StdEqualOp> ||
                std::is_same_v<OpT, StdEqualPredOp> || isMismatchForm);

  std::optional<LicensedDequeCompareIterator> firstDeque =
      licenseDequeCompareIterator(compareOp.getFirst1().getType());
  std::optional<LicensedDequeCompareIterator> secondDeque =
      licenseDequeCompareIterator(compareOp.getFirst2().getType());
  if (!firstDeque && !secondDeque)
    return false;
  if (firstDeque && secondDeque &&
      firstDeque->identity != secondDeque->identity)
    return false;

  cir::RecordType firstWrapperTy;
  cir::RecordType secondWrapperTy;
  cir::PointerType firstPointerTy =
      firstDeque ? firstDeque->elementPointerTy
                 : unwrapContiguousIterator(compareOp.getFirst1().getType(),
                                            firstWrapperTy);
  cir::PointerType secondPointerTy =
      secondDeque ? secondDeque->elementPointerTy
                  : unwrapContiguousIterator(compareOp.getFirst2().getType(),
                                             secondWrapperTy);
  if (!firstPointerTy || !secondPointerTy || firstPointerTy.getAddrSpace() ||
      secondPointerTy.getAddrSpace() ||
      firstPointerTy.getPointee() != secondPointerTy.getPointee())
    return false;

  auto elementTy = mlir::dyn_cast<cir::IntType>(firstPointerTy.getPointee());
  if (!elementTy || elementTy.isBitInt() || elementTy.getWidth() % 8)
    return false;

  // Only the CIRGen facts license the rewrite, since an enum element also
  // lowers to the same integer and a closure of the right shape can compute
  // anything. The predicate forms carry their real second bound inside the
  // pred operand, which this walk never inspects, so the predicate marker
  // is required in addition to the width facts
  const bool wide = elementTy.getWidth() != 8;
  constexpr llvm::StringLiteral libcallName = "memcmp";
  std::optional<LibCallEnv> env =
      checkLibCallEnv(compareOp, libcallName, llvm::LibFunc_memcmp);
  if (!env || !env->enclosing)
    return false;
  if (wide) {
    std::optional<unsigned> wcharWidth = cir::getRecordedIntegerWidth(
        env->moduleOp, cir::CIRDialect::getWCharTypeWidthAttrName());
    if (!wcharWidth || *wcharWidth != elementTy.getWidth() ||
        !compareOp->template getAttrOfType<mlir::UnitAttr>(
            cir::CIRDialect::getWideCharParamsAttrName()))
      return false;
  } else if (!predOp && !compareOp->template getAttrOfType<mlir::UnitAttr>(
                            cir::CIRDialect::getNarrowCharParamsAttrName())) {
    return false;
  }
  if constexpr (predOp)
    if (!compareOp->template getAttrOfType<mlir::UnitAttr>(
            cir::CIRDialect::getElemEqBinaryPredAttrName()))
      return false;

  std::optional<unsigned> intWidth = cir::getRecordedIntegerWidth(
      env->moduleOp, cir::CIRDialect::getIntTypeWidthAttrName());
  std::optional<unsigned> sizeWidth = cir::getRecordedIntegerWidth(
      env->moduleOp, cir::CIRDialect::getSizeTypeWidthAttrName());
  if (!intWidth || !sizeWidth)
    return false;
  // The memcmp length is the element span times the element byte width,
  // so the block count must stay representable after that multiply. Hand
  // written IR can carry a block size CIRGen never emits.
  auto blockSizeRepresentable = [&](uint64_t blockSize, uint64_t elemBytes) {
    if (*sizeWidth >= 64)
      return true;
    uint64_t maxCount = (uint64_t{1} << *sizeWidth) / elemBytes;
    return blockSize < maxCount;
  };
  uint64_t elementBytes = elementTy.getWidth() / 8;
  if ((firstDeque &&
       !blockSizeRepresentable(firstDeque->blockSize, elementBytes)) ||
      (secondDeque &&
       !blockSizeRepresentable(secondDeque->blockSize, elementBytes)))
    return false;

  CIRBaseBuilderTy builder(*compareOp.getContext());
  auto sizeTy = builder.getUIntNTy(*sizeWidth);
  auto intTy = builder.getSIntNTy(*intWidth);
  if (!sharesLibCallSymbol(
          env->moduleOp, symbolTables, libcallName,
          cir::FuncType::get(
              {builder.getVoidPtrTy(), builder.getVoidPtrTy(), sizeTy}, intTy)))
    return false;

  // Hand written IR can name any two member record as the result pair, so
  // the members are re proved against the operand types before the rebuilt
  // iterators are stored through them
  cir::RecordType pairTy;
  if constexpr (isMismatchForm) {
    pairTy = mlir::dyn_cast<cir::RecordType>(compareOp.getResult().getType());
    if (!pairTy || pairTy.isUnion() || !pairTy.isComplete() ||
        pairTy.getMembers().size() != 2 ||
        pairTy.getMembers()[0] != compareOp.getFirst1().getType() ||
        pairTy.getMembers()[1] != compareOp.getFirst2().getType())
      return false;
  }

  mlir::Location loc = compareOp.getLoc();
  builder.setInsertionPointAfter(compareOp);
  auto extractCurrent = [&](mlir::Value iterator,
                            const LicensedDequeCompareIterator &deque) {
    return cir::ExtractMemberOp::create(builder, loc, iterator,
                                        deque.currentPointerIndex);
  };
  auto extractMap = [&](mlir::Value iterator,
                        const LicensedDequeCompareIterator &deque) {
    return cir::ExtractMemberOp::create(builder, loc, iterator,
                                        deque.mapPointerIndex);
  };
  auto unwrap = [&](mlir::Value iterator, cir::RecordType wrapperTy) {
    return wrapperTy ? cir::ExtractMemberOp::create(builder, loc, iterator, 0)
                     : iterator;
  };
  auto advance = [&](mlir::Location stepLoc, mlir::Value base,
                     mlir::Value offset) {
    return cir::PtrStrideOp::create(builder, stepLoc, base.getType(), base,
                                    offset)
        .getResult();
  };
  auto loadBlock = [&](mlir::Location loadLoc, mlir::Value mapSlot) {
    // The identity licenses this load because iterator increment performs
    // the same load whenever it enters a block
    return builder.createLoad(loadLoc, mapSlot);
  };

  // The operation verifier makes first1 and last1 one type, so first1's
  // licensed member indices read last1 as well, and likewise for the
  // second range
  mlir::Value firstCurrent =
      firstDeque ? extractCurrent(compareOp.getFirst1(), *firstDeque)
                 : unwrap(compareOp.getFirst1(), firstWrapperTy);
  mlir::Value last1Current =
      firstDeque ? extractCurrent(compareOp.getLast1(), *firstDeque)
                 : unwrap(compareOp.getLast1(), firstWrapperTy);
  mlir::Value firstMap = firstDeque
                             ? extractMap(compareOp.getFirst1(), *firstDeque)
                             : mlir::Value();
  mlir::Value last1Map = firstDeque
                             ? extractMap(compareOp.getLast1(), *firstDeque)
                             : mlir::Value();

  mlir::Value secondCurrent =
      secondDeque ? extractCurrent(compareOp.getFirst2(), *secondDeque)
                  : unwrap(compareOp.getFirst2(), secondWrapperTy);
  mlir::Value last2Current;
  mlir::Value secondMap = secondDeque
                              ? extractMap(compareOp.getFirst2(), *secondDeque)
                              : mlir::Value();
  mlir::Value last2Map;
  if constexpr (boundedOp) {
    last2Current = secondDeque
                       ? extractCurrent(compareOp.getLast2(), *secondDeque)
                       : unwrap(compareOp.getLast2(), secondWrapperTy);
    if (secondDeque)
      last2Map = extractMap(compareOp.getLast2(), *secondDeque);
  }

  mlir::IntegerAttr pointerAlign = builder.getAlignmentAttr(8);
  mlir::Value firstCurrentAddr =
      builder.createAlloca(loc, builder.getPointerTo(firstPointerTy),
                           "compare_deque_first_pointer", pointerAlign);
  mlir::Value secondCurrentAddr =
      builder.createAlloca(loc, builder.getPointerTo(secondPointerTy),
                           "compare_deque_second_pointer", pointerAlign);
  builder.createStore(loc, firstCurrent, firstCurrentAddr);
  builder.createStore(loc, secondCurrent, secondCurrentAddr);

  mlir::Value firstMapAddr;
  if (firstDeque) {
    firstMapAddr = builder.createAlloca(
        loc, builder.getPointerTo(firstDeque->mapPointerTy),
        "compare_deque_first_map", pointerAlign);
    builder.createStore(loc, firstMap, firstMapAddr);
  }
  mlir::Value secondMapAddr;
  if (secondDeque) {
    secondMapAddr = builder.createAlloca(
        loc, builder.getPointerTo(secondDeque->mapPointerTy),
        "compare_deque_second_map", pointerAlign);
    builder.createStore(loc, secondMap, secondMapAddr);
  }

  mlir::Value equalAddr =
      builder.createAlloca(loc, builder.getPointerTo(builder.getBoolTy()),
                           "compare_deque_equal", builder.getAlignmentAttr(1));
  builder.createStore(loc, builder.getTrue(loc), equalAddr);
  mlir::Value one = builder.getUnsignedInt(loc, 1, *sizeWidth);
  mlir::Value elementSize =
      builder.getUnsignedInt(loc, elementTy.getWidth() / 8, *sizeWidth);
  mlir::Value zero = builder.getNullValue(intTy, loc);

  auto atEnd = [&](mlir::Location checkLoc,
                   const std::optional<LicensedDequeCompareIterator> &deque,
                   mlir::Value currentAddr, mlir::Value mapAddr,
                   mlir::Value lastCurrent, mlir::Value lastMap) {
    mlir::Value current = builder.createLoad(checkLoc, currentAddr);
    mlir::Value sameCurrent = builder.createCompare(
        checkLoc, cir::CmpOpKind::eq, current, lastCurrent);
    if (!deque)
      return sameCurrent;
    mlir::Value map = builder.createLoad(checkLoc, mapAddr);
    mlir::Value sameMap =
        builder.createCompare(checkLoc, cir::CmpOpKind::eq, map, lastMap);
    return builder.createLogicalAnd(checkLoc, sameMap, sameCurrent);
  };

  auto blockEnd = [&](mlir::Location endLoc,
                      const LicensedDequeCompareIterator &deque,
                      mlir::Value map) {
    mlir::Value block = loadBlock(endLoc, map);
    mlir::Value blockCount =
        builder.getUnsignedInt(endLoc, deque.blockSize, *sizeWidth);
    return advance(endLoc, block, blockCount);
  };

  auto remaining = [&](mlir::Location remainingLoc,
                       const std::optional<LicensedDequeCompareIterator> &deque,
                       mlir::Value current, mlir::Value map, bool hasBound,
                       mlir::Value lastCurrent, mlir::Value lastMap) {
    if (!deque)
      return cir::PtrDiffOp::create(builder, remainingLoc, sizeTy, lastCurrent,
                                    current)
          .getResult();
    mlir::Value end = blockEnd(remainingLoc, *deque, map);
    // When the logical bound lives in the current block the span must stop
    // at the bound rather than the block end, otherwise the memcmp would
    // read the tail of the final block past the range
    if (hasBound) {
      mlir::Value sameMap =
          builder.createCompare(remainingLoc, cir::CmpOpKind::eq, map, lastMap);
      end = builder.createSelect(remainingLoc, sameMap, lastCurrent, end);
    }
    return cir::PtrDiffOp::create(builder, remainingLoc, sizeTy, end, current)
        .getResult();
  };

  auto advanceState =
      [&](mlir::Location stepLoc,
          const std::optional<LicensedDequeCompareIterator> &deque,
          mlir::Value currentAddr, mlir::Value mapAddr, mlir::Value span) {
        mlir::Value current = builder.createLoad(stepLoc, currentAddr);
        mlir::Value next = advance(stepLoc, current, span);
        if (!deque) {
          builder.createStore(stepLoc, next, currentAddr);
          return;
        }

        mlir::Value map = builder.createLoad(stepLoc, mapAddr);
        mlir::Value end = blockEnd(stepLoc, *deque, map);
        mlir::Value crossed =
            builder.createCompare(stepLoc, cir::CmpOpKind::eq, next, end);
        cir::TernaryOp::create(
            builder, stepLoc, crossed,
            [&](mlir::OpBuilder &, mlir::Location crossedLoc) {
              mlir::Value followingMap = advance(crossedLoc, map, one);
              mlir::Value followingBlock = loadBlock(crossedLoc, followingMap);
              builder.createStore(crossedLoc, followingBlock, currentAddr);
              builder.createStore(crossedLoc, followingMap, mapAddr);
              builder.createYield(crossedLoc);
            },
            [&](mlir::OpBuilder &, mlir::Location withinLoc) {
              builder.createStore(withinLoc, next, currentAddr);
              builder.createYield(withinLoc);
            });
      };

  builder.createWhile(
      loc,
      [&](mlir::OpBuilder &, mlir::Location conditionLoc) {
        mlir::Value firstMore = builder.createNot(
            conditionLoc, atEnd(conditionLoc, firstDeque, firstCurrentAddr,
                                firstMapAddr, last1Current, last1Map));
        mlir::Value rangesRemain = firstMore;
        if constexpr (boundedOp) {
          mlir::Value secondMore = builder.createNot(
              conditionLoc, atEnd(conditionLoc, secondDeque, secondCurrentAddr,
                                  secondMapAddr, last2Current, last2Map));
          rangesRemain =
              builder.createLogicalAnd(conditionLoc, firstMore, secondMore);
        }
        mlir::Value equal = builder.createLoad(conditionLoc, equalAddr);
        builder.createCondition(
            builder.createLogicalAnd(conditionLoc, rangesRemain, equal));
      },
      [&](mlir::OpBuilder &, mlir::Location bodyLoc) {
        mlir::Value current1 = builder.createLoad(bodyLoc, firstCurrentAddr);
        mlir::Value current2 = builder.createLoad(bodyLoc, secondCurrentAddr);
        mlir::Value map1 = firstDeque
                               ? builder.createLoad(bodyLoc, firstMapAddr)
                               : mlir::Value();
        mlir::Value map2 = secondDeque
                               ? builder.createLoad(bodyLoc, secondMapAddr)
                               : mlir::Value();
        mlir::Value span = remaining(bodyLoc, firstDeque, current1, map1,
                                     /*hasBound=*/true, last1Current, last1Map);
        if (secondDeque || boundedOp) {
          mlir::Value secondRemaining =
              remaining(bodyLoc, secondDeque, current2, map2,
                        /*hasBound=*/boundedOp, last2Current, last2Map);
          mlir::Value secondIsShorter = builder.createCompare(
              bodyLoc, cir::CmpOpKind::lt, secondRemaining, span);
          span = builder.createSelect(bodyLoc, secondIsShorter, secondRemaining,
                                      span);
        }

        mlir::Value lhs =
            builder.createBitcast(bodyLoc, current1, builder.getVoidPtrTy());
        mlir::Value rhs =
            builder.createBitcast(bodyLoc, current2, builder.getVoidPtrTy());
        mlir::Value byteCount = builder.createMul(bodyLoc, span, elementSize);
        mlir::Value spanEqual = emitMemCmpIsEqual(builder, bodyLoc, lhs, rhs,
                                                  byteCount, intTy, zero);
        cir::TernaryOp::create(
            builder, bodyLoc, spanEqual,
            [&](mlir::OpBuilder &, mlir::Location equalLoc) {
              advanceState(equalLoc, firstDeque, firstCurrentAddr, firstMapAddr,
                           span);
              advanceState(equalLoc, secondDeque, secondCurrentAddr,
                           secondMapAddr, span);
              builder.createYield(equalLoc);
            },
            [&](mlir::OpBuilder &, mlir::Location unequalLoc) {
              if constexpr (isMismatchForm) {
                // A failing span cannot cross a block edge, and the memcmp
                // just proved a differing element exists inside it, so the
                // typed rescan terminates before either block end
                builder.createWhile(
                    unequalLoc,
                    [&](mlir::OpBuilder &, mlir::Location rescanConditionLoc) {
                      mlir::Value rescan1 = builder.createLoad(
                          rescanConditionLoc, firstCurrentAddr);
                      mlir::Value rescan2 = builder.createLoad(
                          rescanConditionLoc, secondCurrentAddr);
                      mlir::Value value1 =
                          builder.createLoad(rescanConditionLoc, rescan1);
                      mlir::Value value2 =
                          builder.createLoad(rescanConditionLoc, rescan2);
                      builder.createCondition(builder.createCompare(
                          rescanConditionLoc, cir::CmpOpKind::eq, value1,
                          value2));
                    },
                    [&](mlir::OpBuilder &, mlir::Location rescanLoc) {
                      mlir::Value rescan1 =
                          builder.createLoad(rescanLoc, firstCurrentAddr);
                      mlir::Value rescan2 =
                          builder.createLoad(rescanLoc, secondCurrentAddr);
                      builder.createStore(rescanLoc,
                                          advance(rescanLoc, rescan1, one),
                                          firstCurrentAddr);
                      builder.createStore(rescanLoc,
                                          advance(rescanLoc, rescan2, one),
                                          secondCurrentAddr);
                      builder.createYield(rescanLoc);
                    });
              }
              builder.createStore(unequalLoc, builder.getFalse(unequalLoc),
                                  equalAddr);
              builder.createYield(unequalLoc);
            });
        builder.createYield(bodyLoc);
      });

  if constexpr (!isMismatchForm) {
    mlir::Value equal = builder.createLoad(loc, equalAddr);
    compareOp.getResult().replaceAllUsesWith(equal);
  } else {
    auto rebuild = [&](const std::optional<LicensedDequeCompareIterator> &deque,
                       cir::RecordType wrapperTy, mlir::Value seed,
                       mlir::Value currentAddr, mlir::Value mapAddr) {
      mlir::Value current = builder.createLoad(loc, currentAddr);
      if (!deque)
        return wrapperTy
                   ? cir::InsertMemberOp::create(builder, loc, seed, 0, current)
                         .getResult()
                   : current;

      mlir::Value map = builder.createLoad(loc, mapAddr);
      mlir::Value rebuilt = cir::InsertMemberOp::create(
          builder, loc, seed, deque->currentPointerIndex, current);
      rebuilt = cir::InsertMemberOp::create(builder, loc, rebuilt,
                                            deque->mapPointerIndex, map);
      if (deque->blockFirstPointerIndex) {
        mlir::Value block = loadBlock(loc, map);
        rebuilt = cir::InsertMemberOp::create(
            builder, loc, rebuilt, *deque->blockFirstPointerIndex, block);
        mlir::Value count =
            builder.getUnsignedInt(loc, deque->blockSize, *sizeWidth);
        rebuilt = cir::InsertMemberOp::create(builder, loc, rebuilt,
                                              *deque->blockLastPointerIndex,
                                              advance(loc, block, count));
      }
      return rebuilt;
    };

    mlir::Value result1 =
        rebuild(firstDeque, firstWrapperTy, compareOp.getLast1(),
                firstCurrentAddr, firstMapAddr);
    mlir::Value secondSeed = compareOp.getFirst2();
    if constexpr (boundedOp)
      secondSeed = compareOp.getLast2();
    mlir::Value result2 = rebuild(secondDeque, secondWrapperTy, secondSeed,
                                  secondCurrentAddr, secondMapAddr);
    mlir::Value pair = builder.getConstant(loc, cir::UndefAttr::get(pairTy));
    pair = cir::InsertMemberOp::create(builder, loc, pair, 0, result1);
    pair = cir::InsertMemberOp::create(builder, loc, pair, 1, result2);
    compareOp.getResult().replaceAllUsesWith(pair);
  }

  compareOp.erase();
  if constexpr (isMismatchForm)
    ++numMismatchDequeToMemcmp;
  else
    ++numEqualDequeToMemcmp;
  return true;
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

// Complete words tested per iteration of the grouped scan loop. Eight
// 64 bit words fill two 256 bit vector registers.
static constexpr unsigned kScanGroupWords = 8;
template <typename OpT>
static bool
rewriteFindBitToWordScan(OpT findOp,
                         mlir::Pass::Statistic &numFindBitToWordScan) {
  static_assert(std::is_same_v<OpT, StdFindOp> ||
                std::is_same_v<OpT, StdFindIfOp> ||
                std::is_same_v<OpT, StdFindIfNotOp> ||
                std::is_same_v<OpT, StdRangesFindIfOp> ||
                std::is_same_v<OpT, StdRangesFindIfNotOp>);
  // The value form has no predicate, its sought bool arrives behind the
  // pattern reference and the cir.bool_params marker carries the proof.
  constexpr bool valueForm = std::is_same_v<OpT, StdFindOp>;

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

  mlir::BoolAttr predicateValue;
  bool captureByRef = false;
  if constexpr (valueForm) {
    if (!findOp->template getAttrOfType<mlir::UnitAttr>(
            cir::CIRDialect::getBoolParamsAttrName()))
      return false;
    auto patternPtrTy =
        mlir::dyn_cast<cir::PointerType>(findOp.getPattern().getType());
    if (!patternPtrTy || patternPtrTy.getAddrSpace() ||
        !mlir::isa<cir::BoolType>(patternPtrTy.getPointee()))
      return false;
  } else {
    predicateValue = findOp->template getAttrOfType<mlir::BoolAttr>(
        cir::CIRDialect::getBoolEqPredValueAttrName());

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
            if constexpr (valueForm) {
              sought = builder.createLoad(l, findOp.getPattern());
            } else if (predicateValue) {
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
            mlir::Value groupStride = builder.getUnsignedInt(
                l, kScanGroupWords, bitOffsetTy.getWidth());
            mlir::Value groupCount =
                builder.getUnsignedInt(l, kScanGroupWords, 64);

            // One select before the scan turns both polarities into the
            // same question, a zero mask for a sought true and an all
            // ones mask for a sought false, so each word costs one xor
            // instead of a per word select.
            mlir::Value invertMask =
                builder.createSelect(l, sought, wordZero, wordAllOnes);

            // The load reads the whole word the iterator would read, and
            // the mask discards every bit outside the searched range
            // before the candidates test.
            auto candidatesOf = [&](mlir::Location loadLoc, mlir::Value wordPtr,
                                    mlir::Value validMask) {
              mlir::Value loaded = builder.createLoad(loadLoc, wordPtr);
              mlir::Value searched =
                  builder.createXor(loadLoc, loaded, invertMask);
              return builder.createAnd(loadLoc, searched, validMask);
            };
            auto hitIterator = [&](mlir::Location rebuildLoc,
                                   mlir::Value wordPtr,
                                   mlir::Value candidates) {
              mlir::Value bit =
                  cir::BitCtzOp::create(builder, rebuildLoc, candidates, true);
              bit = builder.createIntCast(bit, bitOffsetTy);
              mlir::Value rebuilt = findOp.getLast();
              rebuilt = insertPath(rebuildLoc, rebuilt, wordPath, wordPtr);
              rebuilt = insertPath(rebuildLoc, rebuilt, bitPath, bit);
              return rebuilt;
            };
            // Each hit path yields the rebuilt iterator directly, so the
            // scan carries no found state and the loop below stays a
            // predictable branch instead of a flag driven select chain.
            auto yieldHitOrElse =
                [&](mlir::Location testLoc, mlir::Value wordPtr,
                    mlir::Value mask,
                    llvm::function_ref<void(mlir::Location)> onMiss) {
                  mlir::Value c = candidatesOf(testLoc, wordPtr, mask);
                  mlir::Value hit = builder.createCompare(
                      testLoc, cir::CmpOpKind::ne, c, wordZero);
                  mlir::Value v =
                      cir::TernaryOp::create(
                          builder, testLoc, hit,
                          [&](mlir::OpBuilder &, mlir::Location hitLoc) {
                            builder.createYield(
                                hitLoc, hitIterator(hitLoc, wordPtr, c));
                          },
                          [&](mlir::OpBuilder &, mlir::Location missLoc) {
                            onMiss(missLoc);
                          })
                          .getResult();
                  builder.createYield(testLoc, v);
                };
            // The loop keeps the zero test in its condition, so a zero
            // word costs one load, one xor, and one compare, and the ctz
            // runs only after the loop exits on a hit. The condition
            // guards the load behind the bound check because the one past
            // bound word is not dereferenceable when the finish offset is
            // zero.
            auto scanLoopAndTail = [&](mlir::Location scanLoc,
                                       mlir::Value beginPtr) {
              mlir::IntegerAttr ptrAlign = builder.getAlignmentAttr(8);
              mlir::Value currentAddr =
                  builder.createAlloca(scanLoc, builder.getPointerTo(wordPtrTy),
                                       "find_bit_current", ptrAlign);
              builder.createStore(scanLoc, beginPtr, currentAddr);
              // The guard keeps every group load strictly before the
              // last word, and on a group hit the single word loop below
              // retests the group in address order, so the earliest hit
              // wins.
              builder.createWhile(
                  scanLoc,
                  [&](mlir::OpBuilder &, mlir::Location guardLoc) {
                    mlir::Value current =
                        builder.createLoad(guardLoc, currentAddr);
                    // The count is 64 bit so no span length can wrap it
                    // below the group size.
                    mlir::Value remaining = cir::PtrDiffOp::create(
                        builder, guardLoc, builder.getUIntNTy(64), lastWord,
                        current);
                    mlir::Value enough = builder.createCompare(
                        guardLoc, cir::CmpOpKind::ge, remaining, groupCount);
                    mlir::Value keepScanning =
                        cir::TernaryOp::create(
                            builder, guardLoc, enough,
                            [&](mlir::OpBuilder &, mlir::Location groupLoc) {
                              llvm::SmallVector<mlir::Value, kScanGroupWords>
                                  group;
                              group.push_back(
                                  candidatesOf(groupLoc, current, wordAllOnes));
                              for (unsigned i = 1; i < kScanGroupWords; ++i) {
                                mlir::Value stride = builder.getUnsignedInt(
                                    groupLoc, i, bitOffsetTy.getWidth());
                                mlir::Value ptr = cir::PtrStrideOp::create(
                                    builder, groupLoc, current.getType(),
                                    current, stride);
                                group.push_back(
                                    candidatesOf(groupLoc, ptr, wordAllOnes));
                              }
                              while (group.size() > 1) {
                                llvm::SmallVector<mlir::Value, kScanGroupWords>
                                    reduced;
                                for (unsigned i = 0; i + 1 < group.size();
                                     i += 2)
                                  reduced.push_back(builder.createOr(
                                      groupLoc, group[i], group[i + 1]));
                                group = reduced;
                              }
                              mlir::Value allZero = builder.createCompare(
                                  groupLoc, cir::CmpOpKind::eq, group.front(),
                                  wordZero);
                              builder.createYield(groupLoc, allZero);
                            },
                            [&](mlir::OpBuilder &, mlir::Location stopLoc) {
                              mlir::Value stop = builder.getFalse(stopLoc);
                              builder.createYield(stopLoc, stop);
                            })
                            .getResult();
                    builder.createCondition(keepScanning);
                  },
                  [&](mlir::OpBuilder &, mlir::Location stepLoc) {
                    mlir::Value current =
                        builder.createLoad(stepLoc, currentAddr);
                    mlir::Value next = cir::PtrStrideOp::create(
                        builder, stepLoc, current.getType(), current,
                        groupStride);
                    builder.createStore(stepLoc, next, currentAddr);
                    builder.createYield(stepLoc);
                  });
              builder.createWhile(
                  scanLoc,
                  [&](mlir::OpBuilder &, mlir::Location condLoc) {
                    mlir::Value current =
                        builder.createLoad(condLoc, currentAddr);
                    mlir::Value more = builder.createCompare(
                        condLoc, cir::CmpOpKind::ne, current, lastWord);
                    mlir::Value keepScanning =
                        cir::TernaryOp::create(
                            builder, condLoc, more,
                            [&](mlir::OpBuilder &, mlir::Location moreLoc) {
                              mlir::Value c =
                                  candidatesOf(moreLoc, current, wordAllOnes);
                              mlir::Value allZero = builder.createCompare(
                                  moreLoc, cir::CmpOpKind::eq, c, wordZero);
                              builder.createYield(moreLoc, allZero);
                            },
                            [&](mlir::OpBuilder &, mlir::Location doneLoc) {
                              mlir::Value stop = builder.getFalse(doneLoc);
                              builder.createYield(doneLoc, stop);
                            })
                            .getResult();
                    builder.createCondition(keepScanning);
                  },
                  [&](mlir::OpBuilder &, mlir::Location advanceLoc) {
                    mlir::Value current =
                        builder.createLoad(advanceLoc, currentAddr);
                    mlir::Value next = cir::PtrStrideOp::create(
                        builder, advanceLoc, current.getType(), current,
                        oneStride);
                    builder.createStore(advanceLoc, next, currentAddr);
                    builder.createYield(advanceLoc);
                  });
              mlir::Value current = builder.createLoad(scanLoc, currentAddr);
              mlir::Value loopHit = builder.createCompare(
                  scanLoc, cir::CmpOpKind::ne, current, lastWord);
              mlir::Value v =
                  cir::TernaryOp::create(
                      builder, scanLoc, loopHit,
                      [&](mlir::OpBuilder &, mlir::Location foundLoc) {
                        // The word that stopped the loop reloads once to
                        // name the bit. The reread is legal because the
                        // source dereference already reads this word, so a
                        // racing write was undefined before the rewrite.
                        // The miss branch of the retest is unreachable
                        // because the loop exit just proved the word has
                        // candidates.
                        yieldHitOrElse(foundLoc, current, wordAllOnes,
                                       [&](mlir::Location missLoc) {
                                         builder.createYield(missLoc,
                                                             findOp.getLast());
                                       });
                      },
                      [&](mlir::OpBuilder &, mlir::Location tailLoc) {
                        mlir::Value hasTail = builder.createCompare(
                            tailLoc, cir::CmpOpKind::ne, lastOffset, bitZero);
                        mlir::Value tv =
                            cir::TernaryOp::create(
                                builder, tailLoc, hasTail,
                                [&](mlir::OpBuilder &, mlir::Location maskLoc) {
                                  mlir::Value tailMask = builder.createSub(
                                      maskLoc,
                                      builder.createShiftLeft(maskLoc, wordOne,
                                                              lastOffset),
                                      wordOne);
                                  yieldHitOrElse(maskLoc, lastWord, tailMask,
                                                 [&](mlir::Location missLoc) {
                                                   builder.createYield(
                                                       missLoc,
                                                       findOp.getLast());
                                                 });
                                },
                                [&](mlir::OpBuilder &,
                                    mlir::Location noTailLoc) {
                                  builder.createYield(noTailLoc,
                                                      findOp.getLast());
                                })
                                .getResult();
                        builder.createYield(tailLoc, tv);
                      })
                      .getResult();
              builder.createYield(scanLoc, v);
            };

            mlir::Value scanResult =
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
                      mlir::Value v =
                          cir::TernaryOp::create(
                              builder, sameLoc, hasValidBits,
                              [&](mlir::OpBuilder &,
                                  mlir::Location guardedLoc) {
                                yieldHitOrElse(guardedLoc, firstWord, validMask,
                                               [&](mlir::Location missLoc) {
                                                 builder.createYield(
                                                     missLoc, findOp.getLast());
                                               });
                              },
                              [&](mlir::OpBuilder &, mlir::Location noBitsLoc) {
                                builder.createYield(noBitsLoc,
                                                    findOp.getLast());
                              })
                              .getResult();
                      builder.createYield(sameLoc, v);
                    },
                    [&](mlir::OpBuilder &, mlir::Location diffLoc) {
                      mlir::Value hasHead = builder.createCompare(
                          diffLoc, cir::CmpOpKind::ne, firstOffset, bitZero);
                      mlir::Value v =
                          cir::TernaryOp::create(
                              builder, diffLoc, hasHead,
                              [&](mlir::OpBuilder &, mlir::Location headLoc) {
                                mlir::Value headMask = builder.createShiftLeft(
                                    headLoc, wordAllOnes, firstOffset);
                                yieldHitOrElse(headLoc, firstWord, headMask,
                                               [&](mlir::Location restLoc) {
                                                 mlir::Value next =
                                                     cir::PtrStrideOp::create(
                                                         builder, restLoc,
                                                         firstWord.getType(),
                                                         firstWord, oneStride);
                                                 scanLoopAndTail(restLoc, next);
                                               });
                              },
                              [&](mlir::OpBuilder &, mlir::Location wholeLoc) {
                                scanLoopAndTail(wholeLoc, firstWord);
                              })
                              .getResult();
                      builder.createYield(diffLoc, v);
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
          if constexpr (std::is_same_v<decltype(find), StdFindOp>) {
            if (rewriteFindBitToWordScan(find, numFindBitToWordScan))
              return;
            if (rewriteFindDequeToBlockWalk(find, symbolTables,
                                            numFindDequeToMemchr,
                                            numFindDequeToWmemchr))
              return;
          }
          rewriteFindLikeToMemchr(find, symbolTables, numFindLikeToMemchr,
                                  numFindLikeToWmemchr);
        })
        .Case<StdFindIfOp, StdFindIfNotOp, StdRangesFindIfOp,
              StdRangesFindIfNotOp>([&](auto find) {
          // The bit, deque, and byte rewrites accept disjoint iterator
          // shapes, so whichever declines leaves the operation for the next.
          if (!rewriteFindBitToWordScan(find, numFindBitToWordScan)) {
            if constexpr (std::is_same_v<decltype(find), StdFindIfOp>) {
              if (rewriteFindDequeToBlockWalk(find, symbolTables,
                                              numFindDequeToMemchr,
                                              numFindDequeToWmemchr))
                return;
            }
            rewriteFindLikeToMemchr(find, symbolTables, numFindLikeToMemchr,
                                    numFindLikeToWmemchr);
          }
        })
        .Case<StdSearchOp>([&](auto search) {
          rewriteSearchToMemmem(search, symbolTables, numSearchToMemmem,
                                numSearchEqualLengthToMemcmp);
        })
        .Case<StdEqualOp, StdEqualPredOp>([&](auto equal) {
          if (rewriteDequeCompareToBlockWalk(equal, symbolTables,
                                             numEqualDequeToMemcmp,
                                             numMismatchDequeToMemcmp))
            return;
          rewriteEqualToMemcmp(equal, symbolTables, numEqualToMemcmp);
        })
        .Case<StdMismatchOp, StdMismatchBoundedOp, StdMismatchPredOp,
              StdMismatchBoundedPredOp>([&](auto mismatch) {
          if (rewriteDequeCompareToBlockWalk(mismatch, symbolTables,
                                             numEqualDequeToMemcmp,
                                             numMismatchDequeToMemcmp))
            return;
          rewriteMismatchToMemcmpLoop(mismatch, symbolTables,
                                      numMismatchToMemcmpLoop);
        });
  });
}

std::unique_ptr<Pass> mlir::createLibOptPass() {
  return std::make_unique<LibOptPass>();
}
