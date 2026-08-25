//===- CIRLoopAnalysis.h - CIR loop analysis -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_ANALYSIS_CIRLOOPANALYSIS_H
#define CLANG_CIR_DIALECT_ANALYSIS_CIRLOOPANALYSIS_H

#include "mlir/Support/LLVM.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>

namespace mlir {
class AliasAnalysis;
}

namespace cir {

/// Normalized loop domain expression independent of CIR memory form
class LoopDomainExpr {
public:
  enum class Kind { Constant, Symbol, Induction, Add, Sub, Mul, Div };

  LoopDomainExpr(LoopDomainExpr &&) = default;
  LoopDomainExpr &operator=(LoopDomainExpr &&) = default;

  LoopDomainExpr(const LoopDomainExpr &) = delete;
  LoopDomainExpr &operator=(const LoopDomainExpr &) = delete;

  Kind getKind() const { return kind; }
  mlir::Value getSource() const { return source; }
  cir::AllocaOp getInduction() const { return induction; }
  const LoopDomainExpr *getLHS() const { return lhs.get(); }
  const LoopDomainExpr *getRHS() const { return rhs.get(); }

  bool dependsOn(cir::AllocaOp variable) const;
  bool isStructurallyEqual(const LoopDomainExpr &other) const;
  void print(llvm::raw_ostream &os) const;

private:
  friend mlir::FailureOr<LoopDomainExpr>
      buildLoopDomainExpr(mlir::Value, mlir::ArrayRef<cir::AllocaOp>);

  LoopDomainExpr(Kind kind, mlir::Value source, cir::AllocaOp induction = {})
      : kind(kind), source(source), induction(induction) {}

  LoopDomainExpr(Kind kind, mlir::Value source,
                 std::unique_ptr<LoopDomainExpr> lhs,
                 std::unique_ptr<LoopDomainExpr> rhs)
      : kind(kind), source(source), lhs(std::move(lhs)), rhs(std::move(rhs)) {}

  Kind kind;
  mlir::Value source;
  cir::AllocaOp induction;
  std::unique_ptr<LoopDomainExpr> lhs;
  std::unique_ptr<LoopDomainExpr> rhs;
};

mlir::FailureOr<LoopDomainExpr>
buildLoopDomainExpr(mlir::Value value,
                    mlir::ArrayRef<cir::AllocaOp> inductions);

/// Canonical memory form loop domain
struct LoopDomain {
  cir::ForOp loop;
  cir::AllocaOp induction;
  cir::StoreOp initialization;
  cir::LoadOp stepLoad;
  cir::IncOp increment;
  cir::StoreOp stepStore;
  cir::CmpOp comparison;
  LoopDomainExpr initial;
  LoopDomainExpr conditionLHS;
  LoopDomainExpr conditionRHS;
};

struct TwoLevelLoopNest {
  LoopDomain outer;
  LoopDomain inner;
};

/// Anchor loop and nested candidate loop pairs
struct ThreeLevelLoopBand {
  LoopDomain anchor;
  LoopDomain outer;
  llvm::SmallVector<LoopDomain, 2> innerCandidates;
};

enum class LoopMemoryLegality {
  Safe,
  UnsupportedOperation,
  UnsupportedAddress,
  PotentialDependence
};

llvm::StringRef stringifyLoopMemoryLegality(LoopMemoryLegality result);

struct LoopMemoryBase {
  mlir::Value pointer;
};

struct LoopMemoryAccess {
  mlir::Operation *operation;
  LoopMemoryBase base;
  llvm::SmallVector<LoopDomainExpr, 2> subscripts;
  bool isWrite;
};

/// Ordered floating point array element update
struct LoopElementRecurrence {
  LoopMemoryAccess target;
  cir::AllocaOp recurrenceInduction;
  llvm::SmallVector<cir::AllocaOp, 2> laneInductions;
  cir::LoadOp load;
  mlir::Operation *combiner;
  cir::StoreOp store;
};

/// Nonescaping private integer addition updated once per iteration
struct LoopReduction {
  cir::AllocaOp variable;
  cir::LoadOp load;
  cir::AddOp operation;
  cir::StoreOp store;
};

struct LoopMemoryAnalysis {
  LoopMemoryLegality result;
  llvm::SmallVector<LoopMemoryAccess, 8> accesses;
  llvm::SmallVector<LoopReduction, 2> reductions;
  llvm::SmallVector<LoopElementRecurrence, 2> recurrences;

  bool isSafe() const { return result == LoopMemoryLegality::Safe; }
};

struct LoopBandMemoryAnalysis {
  LoopMemoryLegality result;
  llvm::SmallVector<LoopMemoryAccess, 16> accesses;
  llvm::SmallVector<LoopElementRecurrence, 2> recurrences;

  bool isSafe() const { return result == LoopMemoryLegality::Safe; }
};

/// Canonical unit step induction variable and loop domain
mlir::FailureOr<LoopDomain>
analyzeLoopDomain(cir::ForOp loop,
                  mlir::ArrayRef<cir::AllocaOp> enclosingInductions = {});

/// Two level loop nest with transparent CIR scopes
mlir::FailureOr<TwoLevelLoopNest> analyzeTwoLevelLoopNest(cir::ForOp outerLoop);

mlir::FailureOr<ThreeLevelLoopBand>
analyzeThreeLevelLoopBand(cir::ForOp anchorLoop);

llvm::SmallVector<LoopElementRecurrence, 2>
analyzeLoopElementRecurrences(const ThreeLevelLoopBand &band,
                              const LoopDomain &inner,
                              mlir::AliasAnalysis &aliasAnalysis);

LoopBandMemoryAnalysis
analyzeLoopBandMemory(const ThreeLevelLoopBand &band,
                      mlir::AliasAnalysis &aliasAnalysis);

/// Restricted memory independence for pointwise array nests
LoopMemoryAnalysis analyzeLoopMemory(const TwoLevelLoopNest &nest,
                                     mlir::AliasAnalysis &aliasAnalysis);

} // namespace cir

#endif // CLANG_CIR_DIALECT_ANALYSIS_CIRLOOPANALYSIS_H
