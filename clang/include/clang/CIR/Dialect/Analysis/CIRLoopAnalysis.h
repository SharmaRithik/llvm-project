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
#include "llvm/Support/raw_ostream.h"

#include <memory>

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

/// Canonical unit step induction variable and loop domain
mlir::FailureOr<LoopDomain>
analyzeLoopDomain(cir::ForOp loop,
                  mlir::ArrayRef<cir::AllocaOp> enclosingInductions = {});

/// Two level loop nest with transparent CIR scopes
mlir::FailureOr<TwoLevelLoopNest> analyzeTwoLevelLoopNest(cir::ForOp outerLoop);

} // namespace cir

#endif // CLANG_CIR_DIALECT_ANALYSIS_CIRLOOPANALYSIS_H
