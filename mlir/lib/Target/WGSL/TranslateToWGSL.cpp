//===- TranslateToWGSL.cpp - Translate GPU kernels to WGSL ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Translates a gpu.func with structured control flow to a WGSL compute shader.
// Subgroup MMA ops map to the chromium_experimental_subgroup_matrix builtins.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/WGSL/WGSLEmitter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace mlir;

namespace {

/// Shader builtins, in entry point parameter order.
enum class Builtin {
  LocalInvocationId,
  WorkgroupId,
  NumWorkgroups,
  GlobalInvocationId,
  SubgroupId,
  SubgroupInvocationId,
  SubgroupSize,
  NumSubgroups,
};

struct BuiltinInfo {
  StringRef attr;
  StringRef name;
  StringRef type;
  bool needsSubgroups;
};

BuiltinInfo getBuiltinInfo(Builtin b) {
  switch (b) {
  case Builtin::LocalInvocationId:
    return {"local_invocation_id", "local_id", "vec3<u32>", false};
  case Builtin::WorkgroupId:
    return {"workgroup_id", "wg_id", "vec3<u32>", false};
  case Builtin::NumWorkgroups:
    return {"num_workgroups", "num_wg", "vec3<u32>", false};
  case Builtin::GlobalInvocationId:
    return {"global_invocation_id", "global_id", "vec3<u32>", false};
  case Builtin::SubgroupId:
    return {"subgroup_id", "sg_id", "u32", true};
  case Builtin::SubgroupInvocationId:
    return {"subgroup_invocation_id", "sg_lane", "u32", true};
  case Builtin::SubgroupSize:
    return {"subgroup_size", "sg_size", "u32", true};
  case Builtin::NumSubgroups:
    return {"num_subgroups", "num_sg", "u32", true};
  }
  llvm_unreachable("unknown builtin");
}

StringRef getComponent(gpu::Dimension dim) {
  switch (dim) {
  case gpu::Dimension::x:
    return "x";
  case gpu::Dimension::y:
    return "y";
  case gpu::Dimension::z:
    return "z";
  }
  llvm_unreachable("unknown dimension");
}

StringRef getComponent(int64_t idx) {
  static const char *names[] = {"x", "y", "z", "w"};
  assert(idx >= 0 && idx < 4 && "vector component out of range");
  return names[idx];
}

/// Index arithmetic follows the emitter's 32-bit WGSL index representation.
struct IndexExpr {
  std::string text;
  std::optional<uint32_t> constant;
};

static IndexExpr indexConstant(uint32_t value) {
  return {llvm::formatv("{0}u", value).str(), value};
}

static IndexExpr addIndex(IndexExpr lhs, IndexExpr rhs) {
  if (lhs.constant && rhs.constant)
    return indexConstant(*lhs.constant + *rhs.constant);
  if (lhs.constant == 0)
    return rhs;
  if (rhs.constant == 0)
    return lhs;
  return {"(" + lhs.text + " + " + rhs.text + ")", std::nullopt};
}

static IndexExpr multiplyIndex(IndexExpr lhs, IndexExpr rhs) {
  if (lhs.constant && rhs.constant)
    return indexConstant(*lhs.constant * *rhs.constant);
  if (lhs.constant == 0 || rhs.constant == 0)
    return indexConstant(0);
  if (lhs.constant == 1)
    return rhs;
  if (rhs.constant == 1)
    return lhs;
  return {"(" + lhs.text + " * " + rhs.text + ")", std::nullopt};
}

struct RuntimeLayout {
  IndexExpr offset;
  SmallVector<IndexExpr> strides;
};

/// How a memref value is accessed in the generated shader.
struct MemRefInfo {
  std::string name;
  bool isUniform = false;
  // A cast may erase layout information from the SSA type.
  MemRefType layoutType = {};
  // Dynamic views retain their SSA offset/stride expressions through casts.
  std::optional<RuntimeLayout> runtimeLayout = std::nullopt;
};

class WGSLEmitter {
public:
  explicit WGSLEmitter(raw_ostream &os) : out(os) {}

  LogicalResult emitKernel(gpu::GPUFuncOp func);

private:
  // Types and literals.
  FailureOr<std::string> getTypeName(Location loc, Type type);
  std::string getTypeName(Type type);
  std::string getSignedTypeName(Type type);
  FailureOr<std::string> getLiteral(Location loc, Attribute attr, Type type);
  FailureOr<std::string> getScalarLiteral(Location loc, Attribute attr);

  // Values.
  std::string getFreshName(StringRef prefix) {
    return (prefix + Twine(nextId++)).str();
  }
  std::string get(Value v) {
    auto it = exprs.find(v);
    assert(it != exprs.end() && "value used before it was emitted");
    return it->second;
  }
  std::string asSigned(Value v) {
    return "bitcast<" + getSignedTypeName(v.getType()) + ">(" + get(v) + ")";
  }
  std::string asUnsigned(Type type, StringRef expr) {
    return ("bitcast<" + getTypeName(type) + ">(" + expr + ")").str();
  }
  /// Binds `v` to `expr` through a new `let` declaration.
  void emitLet(Value v, StringRef expr) {
    std::string name = getFreshName("v");
    os() << "let " << name << ": " << getTypeName(v.getType()) << " = " << expr
         << ";\n";
    exprs[v] = name;
  }
  /// Declares a function-scope variable and returns its name.
  std::string emitVar(Type type, StringRef init, StringRef prefix = "r") {
    std::string name = getFreshName(prefix);
    os() << "var " << name << ": " << getTypeName(type);
    if (!init.empty())
      os() << " = " << init;
    os() << ";\n";
    return name;
  }
  MemRefType getLayoutType(Value memref) {
    MemRefType type = memrefs.lookup(memref).layoutType;
    return type ? type : cast<MemRefType>(memref.getType());
  }
  std::string getLinearIndex(Value memref, ValueRange indices,
                             int64_t extraOffset = 0);
  FailureOr<IndexExpr> getIndexExpr(Location loc, OpFoldResult value);
  FailureOr<RuntimeLayout> getRuntimeLayout(Location loc, Value memref);
  FailureOr<std::string> getAccess(Location loc, Value memref,
                                   ValueRange indices, int64_t extraOffset = 0);
  FailureOr<std::string> getElementBase(Location loc, Value memref,
                                        ValueRange indices);
  FailureOr<std::string> getMMAPointer(Location loc, Value memref,
                                       ValueRange indices,
                                       std::string &offset);
  std::string useBuiltin(Builtin b, std::optional<gpu::Dimension> dim) {
    builtins.insert(b);
    BuiltinInfo info = getBuiltinInfo(b);
    if (info.needsSubgroups)
      usesSubgroups = true;
    std::string expr = info.name.str();
    if (dim)
      expr += ("." + getComponent(*dim)).str();
    return expr;
  }

  // Statements.
  LogicalResult emitBlock(Block &block, ArrayRef<std::string> yieldTargets);
  LogicalResult emitYield(Operation *yield, ArrayRef<std::string> targets);
  LogicalResult emitOperation(Operation &op);
  LogicalResult emitFor(scf::ForOp op);
  LogicalResult emitIf(scf::IfOp op);
  LogicalResult emitWhile(scf::WhileOp op);
  LogicalResult emitBinary(Operation *op, StringRef opStr,
                           bool isSigned = false);
  LogicalResult emitCall(Operation *op, StringRef callee);
  LogicalResult emitCmpI(arith::CmpIOp op);
  LogicalResult emitCmpF(arith::CmpFOp op);
  LogicalResult emitCast(Operation *op);
  LogicalResult emitReduction(vector::ReductionOp op);

  raw_indented_ostream &os() { return *body; }

  raw_ostream &out;
  raw_indented_ostream *body = nullptr;
  DenseMap<Value, std::string> exprs;
  DenseMap<Value, MemRefInfo> memrefs;
  llvm::SmallSetVector<Builtin, 8> builtins;
  SmallVector<int32_t, 3> blockSize;
  unsigned nextId = 0;
  bool usesF16 = false;
  bool usesSubgroups = false;
  bool usesSubgroupMatrix = false;
};

} // namespace

//===----------------------------------------------------------------------===//
// Types and literals
//===----------------------------------------------------------------------===//

FailureOr<std::string> WGSLEmitter::getTypeName(Location loc, Type type) {
  if (type.isInteger(1))
    return std::string("bool");
  if (type.isIndex() || type.isInteger(32))
    return std::string("u32");
  if (type.isF32())
    return std::string("f32");
  if (type.isF16()) {
    usesF16 = true;
    return std::string("f16");
  }
  if (auto vecType = dyn_cast<VectorType>(type)) {
    int64_t n = vecType.getRank() == 1 ? vecType.getNumElements() : 0;
    if (vecType.isScalable() || n < 2 || n > 4)
      return emitError(loc) << "cannot emit " << type
                            << " to WGSL; only 1-D vectors of 2 to 4 "
                               "elements are supported";
    FailureOr<std::string> elem = getTypeName(loc, vecType.getElementType());
    if (failed(elem))
      return failure();
    return llvm::formatv("vec{0}<{1}>", n, *elem).str();
  }
  if (auto mmaType = dyn_cast<gpu::MMAMatrixType>(type)) {
    usesSubgroupMatrix = true;
    usesSubgroups = true;
    StringRef kind = llvm::StringSwitch<StringRef>(mmaType.getOperand())
                         .Case("AOp", "left")
                         .Case("BOp", "right")
                         .Default("result");
    FailureOr<std::string> elem = getTypeName(loc, mmaType.getElementType());
    if (failed(elem))
      return failure();
    // WGSL subgroup matrix types take <element, columns, rows>.
    ArrayRef<int64_t> shape = mmaType.getShape();
    return llvm::formatv("subgroup_matrix_{0}<{1}, {2}, {3}>", kind, *elem,
                         shape[1], shape[0])
        .str();
  }
  if (auto memrefType = dyn_cast<MemRefType>(type)) {
    FailureOr<std::string> elem = getTypeName(loc, memrefType.getElementType());
    if (failed(elem))
      return failure();
    if (!memrefType.getLayout().isIdentity())
      return emitError(loc) << "cannot emit " << type
                            << " to WGSL; only identity layouts are supported";
    if (memrefType.getRank() == 1 && memrefType.isDynamicDim(0))
      return "array<" + *elem + ">";
    if (!memrefType.hasStaticShape())
      return emitError(loc) << "cannot emit " << type
                            << " to WGSL; only static shapes or 1-D dynamic "
                               "shapes are supported";
    return llvm::formatv("array<{0}, {1}>", *elem,
                         memrefType.getNumElements())
        .str();
  }
  return emitError(loc) << "cannot emit type " << type << " to WGSL";
}

std::string WGSLEmitter::getTypeName(Type type) {
  // The type was already checked by the other overload.
  return *getTypeName(UnknownLoc::get(type.getContext()), type);
}

std::string WGSLEmitter::getSignedTypeName(Type type) {
  if (auto vecType = dyn_cast<VectorType>(type))
    return llvm::formatv("vec{0}<i32>", vecType.getNumElements()).str();
  return "i32";
}

static std::optional<std::string> formatFloat(const APFloat &value,
                                              StringRef suffix) {
  if (value.isNaN() || value.isInfinity())
    return std::nullopt;
  bool losesInfo = false;
  APFloat asFloat = value;
  asFloat.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven,
                  &losesInfo);
  float f = asFloat.convertToFloat();
  bool negative = std::signbit(f);
  float magnitude = std::fabs(f);
  // Print the shortest decimal that reads back to the same value.
  char buf[64];
  for (int precision = 1; precision <= 9; ++precision) {
    std::snprintf(buf, sizeof(buf), "%.*g", precision, magnitude);
    if (std::strtof(buf, nullptr) == magnitude)
      break;
  }
  std::string text = buf;
  if (text.find_first_of(".e") == std::string::npos)
    text += ".0";
  text += suffix;
  if (negative)
    return "(-" + text + ")";
  return text;
}

FailureOr<std::string> WGSLEmitter::getScalarLiteral(Location loc,
                                                     Attribute attr) {
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    Type type = intAttr.getType();
    if (type.isInteger(1))
      return std::string(intAttr.getValue().isZero() ? "false" : "true");
    APInt value = intAttr.getValue();
    if (value.getBitWidth() > 32) {
      if (!value.isSignedIntN(32) && !value.isIntN(32))
        return emitError(loc) << "integer constant " << attr
                              << " does not fit in 32 bits";
      value = value.trunc(32);
    }
    return llvm::formatv("{0}u", value.getZExtValue()).str();
  }
  if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
    StringRef suffix = floatAttr.getType().isF16() ? "h" : "f";
    if (floatAttr.getType().isF16())
      usesF16 = true;
    if (std::optional<std::string> text =
            formatFloat(floatAttr.getValue(), suffix))
      return *text;
    return emitError(loc) << "cannot emit non-finite constant " << attr
                          << " to WGSL";
  }
  return emitError(loc) << "cannot emit constant " << attr << " to WGSL";
}

FailureOr<std::string> WGSLEmitter::getLiteral(Location loc, Attribute attr,
                                               Type type) {
  if (failed(getTypeName(loc, type)))
    return failure();
  if (auto dense = dyn_cast<DenseElementsAttr>(attr)) {
    FailureOr<std::string> typeName = getTypeName(loc, type);
    if (failed(typeName))
      return failure();
    if (dense.isSplat()) {
      FailureOr<std::string> elem =
          getScalarLiteral(loc, dense.getSplatValue<Attribute>());
      if (failed(elem))
        return failure();
      return *typeName + "(" + *elem + ")";
    }
    SmallVector<std::string> elems;
    for (Attribute elemAttr : dense.getValues<Attribute>()) {
      FailureOr<std::string> elem = getScalarLiteral(loc, elemAttr);
      if (failed(elem))
        return failure();
      elems.push_back(*elem);
    }
    return *typeName + "(" + llvm::join(elems, ", ") + ")";
  }
  return getScalarLiteral(loc, attr);
}

//===----------------------------------------------------------------------===//
// Memory accesses
//===----------------------------------------------------------------------===//

std::string WGSLEmitter::getLinearIndex(Value memref, ValueRange indices,
                                        int64_t extraOffset) {
  if (auto &layout = memrefs[memref].runtimeLayout) {
    IndexExpr index = layout->offset;
    for (auto [value, stride] : llvm::zip_equal(indices, layout->strides)) {
      auto constant = getConstantIntValue(value);
      IndexExpr component = constant ? indexConstant(uint32_t(*constant))
                                     : IndexExpr{get(value), std::nullopt};
      index = addIndex(index, multiplyIndex(component, stride));
    }
    return addIndex(index, indexConstant(uint32_t(extraOffset))).text;
  }
  MemRefType type = getLayoutType(memref);
  // Memrefs are flat arrays. Views carry static strides and an offset.
  SmallVector<int64_t> strides;
  int64_t offset = 0;
  if (type.getLayout().isIdentity()) {
    strides.assign(type.getRank(), 1);
    for (int64_t d = type.getRank() - 2; d >= 0; --d)
      strides[d] = strides[d + 1] * type.getDimSize(d + 1);
  } else {
    // Checked by isEmittableView when the view was created.
    (void)type.getStridesAndOffset(strides, offset);
  }
  int64_t constant = extraOffset + offset;
  SmallVector<std::string> terms;
  for (auto [idx, stride] : llvm::zip(indices, strides)) {
    if (std::optional<int64_t> cst = getConstantIntValue(idx)) {
      constant += *cst * stride;
      continue;
    }
    terms.push_back(stride == 1 ? get(idx)
                                : llvm::formatv("{0} * {1}u", get(idx), stride)
                                      .str());
  }
  if (constant != 0 || terms.empty())
    terms.push_back(llvm::formatv("{0}u", constant).str());
  return llvm::join(terms, " + ");
}

FailureOr<IndexExpr> WGSLEmitter::getIndexExpr(Location loc, OpFoldResult value) {
  if (auto ssa = dyn_cast<Value>(value)) {
    if (auto constant = getConstantIntValue(ssa))
      return indexConstant(uint32_t(*constant));
    return IndexExpr{get(ssa), std::nullopt};
  }
  auto attr = cast<IntegerAttr>(cast<Attribute>(value));
  auto literal = getScalarLiteral(loc, attr);
  if (failed(literal))
    return failure();
  return IndexExpr{*literal, uint32_t(attr.getInt())};
}

FailureOr<RuntimeLayout> WGSLEmitter::getRuntimeLayout(Location loc,
                                                      Value memref) {
  const MemRefInfo &info = memrefs[memref];
  if (info.runtimeLayout)
    return *info.runtimeLayout;
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(getLayoutType(memref).getStridesAndOffset(strides, offset)) ||
      ShapedType::isDynamic(offset) ||
      llvm::any_of(strides, ShapedType::isDynamic))
    return emitError(loc) << "view source has unknown offset or strides";
  Builder builder(memref.getContext());
  RuntimeLayout result;
  auto offsetExpr = getIndexExpr(loc, builder.getIndexAttr(offset));
  if (failed(offsetExpr))
    return failure();
  result.offset = *offsetExpr;
  for (int64_t stride : strides) {
    auto strideExpr = getIndexExpr(loc, builder.getIndexAttr(stride));
    if (failed(strideExpr))
      return failure();
    result.strides.push_back(*strideExpr);
  }
  return result;
}

/// Returns true if the view's element positions are known statically.
static bool isEmittableView(MemRefType type) {
  if (type.getLayout().isIdentity())
    return type.hasStaticShape() || type.getRank() == 1;
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(type.getStridesAndOffset(strides, offset)))
    return false;
  return !ShapedType::isDynamic(offset) &&
         llvm::none_of(strides, ShapedType::isDynamic);
}

FailureOr<std::string> WGSLEmitter::getAccess(Location loc, Value memref,
                                              ValueRange indices,
                                              int64_t extraOffset) {
  auto it = memrefs.find(memref);
  if (it == memrefs.end())
    return emitError(loc) << "memref is not a kernel argument, workgroup "
                             "attribution or local allocation";
  const MemRefInfo &info = it->second;
  std::string idx = getLinearIndex(memref, indices, extraOffset);
  if (info.isUniform) {
    // Uniform buffers are emitted as arrays of 16-byte vectors.
    if (auto cst = getConstantIntValue(indices[0]))
      return llvm::formatv("{0}[{1}u].{2}", info.name, (*cst + extraOffset) / 4,
                           getComponent((*cst + extraOffset) % 4))
          .str();
    return llvm::formatv("{0}[({1}) / 4u][({1}) % 4u]", info.name, idx).str();
  }
  return info.name + "[" + idx + "]";
}

FailureOr<std::string> WGSLEmitter::getElementBase(Location loc, Value memref,
                                                   ValueRange indices) {
  auto it = memrefs.find(memref);
  if (it == memrefs.end() || it->second.isUniform)
    return emitError(loc) << "vector accesses need a storage, workgroup or "
                             "local memref";
  auto layout = getRuntimeLayout(loc, memref);
  if (failed(layout))
    return failure();
  if (layout->strides.empty() || layout->strides.back().constant != 1)
    return emitError(loc) << "vector memory accesses require a unit innermost "
                             "stride";
  std::string name = getFreshName("b");
  os() << "let " << name << ": u32 = "
       << getLinearIndex(memref, indices) << ";\n";
  return name;
}

FailureOr<std::string> WGSLEmitter::getMMAPointer(Location loc, Value memref,
                                                  ValueRange indices,
                                                  std::string &offset) {
  auto it = memrefs.find(memref);
  if (it == memrefs.end() || it->second.isUniform)
    return emitError(loc) << "subgroup matrix operands must be storage or "
                             "workgroup memrefs";
  offset = getLinearIndex(memref, indices);
  return "&" + it->second.name;
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

LogicalResult WGSLEmitter::emitBlock(Block &block,
                                     ArrayRef<std::string> yieldTargets) {
  for (Operation &op : block.without_terminator())
    if (failed(emitOperation(op)))
      return failure();
  Operation *terminator = block.getTerminator();
  if (isa<scf::YieldOp>(terminator))
    return emitYield(terminator, yieldTargets);
  if (isa<gpu::ReturnOp>(terminator)) {
    os() << "return;\n";
    return success();
  }
  return terminator->emitError("cannot emit terminator to WGSL");
}

LogicalResult WGSLEmitter::emitYield(Operation *yield,
                                     ArrayRef<std::string> targets) {
  assert(yield->getNumOperands() == targets.size() &&
         "yield does not match its targets");
  // Snapshot values that an earlier assignment in this yield overwrites.
  SmallVector<std::string> values;
  for (auto [idx, operand] : llvm::enumerate(yield->getOperands())) {
    std::string expr = get(operand);
    auto pos = llvm::find(targets, expr);
    if (pos != targets.end() && pos != targets.begin() + idx) {
      std::string name = getFreshName("t");
      os() << "let " << name << " = " << expr << ";\n";
      expr = name;
    }
    values.push_back(expr);
  }
  for (auto [target, value] : llvm::zip(targets, values))
    if (target != value)
      os() << target << " = " << value << ";\n";
  return success();
}

LogicalResult WGSLEmitter::emitFor(scf::ForOp op) {
  SmallVector<std::string> iterVars;
  for (auto [init, result] : llvm::zip(op.getInitArgs(), op.getResults())) {
    if (failed(getTypeName(op.getLoc(), result.getType())))
      return failure();
    iterVars.push_back(emitVar(result.getType(), get(init), "a"));
  }
  // An "i" prefix can produce i32 and shadow the WGSL integer type.
  std::string iv = getFreshName("iv");
  std::string ub = get(op.getUpperBound());
  std::string cond =
      op.getUnsignedCmp()
          ? iv + " < " + ub
          : "bitcast<i32>(" + iv + ") < " + asSigned(op.getUpperBound());
  os() << "for (var " << iv << ": u32 = " << get(op.getLowerBound()) << "; "
       << cond << "; " << iv << " += " << get(op.getStep()) << ") {\n";
  os().indent();
  exprs[op.getInductionVar()] = iv;
  for (auto [arg, var] : llvm::zip(op.getRegionIterArgs(), iterVars))
    exprs[arg] = var;
  if (failed(emitBlock(*op.getBody(), iterVars)))
    return failure();
  os().unindent();
  os() << "}\n";
  for (auto [result, var] : llvm::zip(op.getResults(), iterVars))
    exprs[result] = var;
  return success();
}

LogicalResult WGSLEmitter::emitIf(scf::IfOp op) {
  SmallVector<std::string> resultVars;
  for (Value result : op.getResults()) {
    if (failed(getTypeName(op.getLoc(), result.getType())))
      return failure();
    resultVars.push_back(emitVar(result.getType(), ""));
  }
  os() << "if (" << get(op.getCondition()) << ") {\n";
  os().indent();
  if (failed(emitBlock(op.getThenRegion().front(), resultVars)))
    return failure();
  os().unindent();
  if (!op.getElseRegion().empty()) {
    os() << "} else {\n";
    os().indent();
    if (failed(emitBlock(op.getElseRegion().front(), resultVars)))
      return failure();
    os().unindent();
  }
  os() << "}\n";
  for (auto [result, var] : llvm::zip(op.getResults(), resultVars))
    exprs[result] = var;
  return success();
}

LogicalResult WGSLEmitter::emitWhile(scf::WhileOp op) {
  // Values carried into the "before" region.
  SmallVector<std::string> beforeVars;
  for (auto [init, arg] : llvm::zip(op.getInits(), op.getBeforeArguments())) {
    if (failed(getTypeName(op.getLoc(), arg.getType())))
      return failure();
    beforeVars.push_back(emitVar(arg.getType(), get(init), "a"));
  }
  // Values forwarded by `scf.condition` to the "after" region and results.
  SmallVector<std::string> resultVars;
  for (Value result : op.getResults()) {
    if (failed(getTypeName(op.getLoc(), result.getType())))
      return failure();
    resultVars.push_back(emitVar(result.getType(), ""));
  }
  os() << "loop {\n";
  os().indent();
  Block &before = op.getBefore().front();
  for (auto [arg, var] : llvm::zip(before.getArguments(), beforeVars))
    exprs[arg] = var;
  for (Operation &nested : before.without_terminator())
    if (failed(emitOperation(nested)))
      return failure();
  auto condition = cast<scf::ConditionOp>(before.getTerminator());
  for (auto [var, value] : llvm::zip(resultVars, condition.getArgs()))
    os() << var << " = " << get(value) << ";\n";
  os() << "if (!" << get(condition.getCondition()) << ") {\n";
  os().indent();
  os() << "break;\n";
  os().unindent();
  os() << "}\n";
  Block &after = op.getAfter().front();
  for (auto [arg, var] : llvm::zip(after.getArguments(), resultVars))
    exprs[arg] = var;
  if (failed(emitBlock(after, beforeVars)))
    return failure();
  os().unindent();
  os() << "}\n";
  for (auto [result, var] : llvm::zip(op.getResults(), resultVars))
    exprs[result] = var;
  return success();
}

LogicalResult WGSLEmitter::emitBinary(Operation *op, StringRef opStr,
                                      bool isSigned) {
  Value lhs = op->getOperand(0), rhs = op->getOperand(1);
  Value result = op->getResult(0);
  if (failed(getTypeName(op->getLoc(), result.getType())))
    return failure();
  if (!isSigned) {
    emitLet(result, get(lhs) + " " + opStr.str() + " " + get(rhs));
    return success();
  }
  // Shift amounts stay unsigned in WGSL.
  std::string rhsExpr = isa<arith::ShRSIOp>(op) ? get(rhs) : asSigned(rhs);
  emitLet(result, asUnsigned(result.getType(), asSigned(lhs) + " " +
                                                   opStr.str() + " " +
                                                   rhsExpr));
  return success();
}

LogicalResult WGSLEmitter::emitCall(Operation *op, StringRef callee) {
  if (failed(getTypeName(op->getLoc(), op->getResult(0).getType())))
    return failure();
  SmallVector<std::string> args;
  for (Value operand : op->getOperands())
    args.push_back(get(operand));
  emitLet(op->getResult(0),
          (callee + "(" + llvm::join(args, ", ") + ")").str());
  return success();
}

LogicalResult WGSLEmitter::emitCmpI(arith::CmpIOp op) {
  if (failed(getTypeName(op.getLoc(), op.getType())) ||
      failed(getTypeName(op.getLoc(), op.getLhs().getType())))
    return failure();
  StringRef opStr;
  bool isSigned = false;
  switch (op.getPredicate()) {
  case arith::CmpIPredicate::eq:
    opStr = "==";
    break;
  case arith::CmpIPredicate::ne:
    opStr = "!=";
    break;
  case arith::CmpIPredicate::ult:
    opStr = "<";
    break;
  case arith::CmpIPredicate::ule:
    opStr = "<=";
    break;
  case arith::CmpIPredicate::ugt:
    opStr = ">";
    break;
  case arith::CmpIPredicate::uge:
    opStr = ">=";
    break;
  case arith::CmpIPredicate::slt:
    opStr = "<", isSigned = true;
    break;
  case arith::CmpIPredicate::sle:
    opStr = "<=", isSigned = true;
    break;
  case arith::CmpIPredicate::sgt:
    opStr = ">", isSigned = true;
    break;
  case arith::CmpIPredicate::sge:
    opStr = ">=", isSigned = true;
    break;
  }
  // Booleans only support equality.
  if (getElementTypeOrSelf(op.getLhs().getType()).isInteger(1))
    isSigned = false;
  std::string lhs = isSigned ? asSigned(op.getLhs()) : get(op.getLhs());
  std::string rhs = isSigned ? asSigned(op.getRhs()) : get(op.getRhs());
  emitLet(op.getResult(), lhs + " " + opStr.str() + " " + rhs);
  return success();
}

LogicalResult WGSLEmitter::emitCmpF(arith::CmpFOp op) {
  if (failed(getTypeName(op.getLoc(), op.getType())) ||
      failed(getTypeName(op.getLoc(), op.getLhs().getType())))
    return failure();
  StringRef opStr;
  switch (op.getPredicate()) {
  case arith::CmpFPredicate::OEQ:
  case arith::CmpFPredicate::UEQ:
    opStr = "==";
    break;
  case arith::CmpFPredicate::ONE:
  case arith::CmpFPredicate::UNE:
    opStr = "!=";
    break;
  case arith::CmpFPredicate::OLT:
  case arith::CmpFPredicate::ULT:
    opStr = "<";
    break;
  case arith::CmpFPredicate::OLE:
  case arith::CmpFPredicate::ULE:
    opStr = "<=";
    break;
  case arith::CmpFPredicate::OGT:
  case arith::CmpFPredicate::UGT:
    opStr = ">";
    break;
  case arith::CmpFPredicate::OGE:
  case arith::CmpFPredicate::UGE:
    opStr = ">=";
    break;
  default:
    return op.emitError("cannot emit predicate to WGSL");
  }
  emitLet(op.getResult(), get(op.getLhs()) + " " + opStr.str() + " " +
                              get(op.getRhs()));
  return success();
}

LogicalResult WGSLEmitter::emitCast(Operation *op) {
  Value in = op->getOperand(0);
  Value result = op->getResult(0);
  FailureOr<std::string> resultType = getTypeName(op->getLoc(), result.getType());
  if (failed(resultType) || failed(getTypeName(op->getLoc(), in.getType())))
    return failure();
  Type inElem = getElementTypeOrSelf(in.getType());
  Type outElem = getElementTypeOrSelf(result.getType());
  auto is32 = [](Type t) { return t.isIndex() || t.isInteger(32); };

  return TypeSwitch<Operation *, LogicalResult>(op)
      .Case<arith::ExtFOp, arith::TruncFOp, arith::UIToFPOp, arith::FPToUIOp>(
          [&](auto) {
            emitLet(result, *resultType + "(" + get(in) + ")");
            return success();
          })
      .Case([&](arith::SIToFPOp) {
        emitLet(result, *resultType + "(" + asSigned(in) + ")");
        return success();
      })
      .Case([&](arith::FPToSIOp) {
        emitLet(result, asUnsigned(result.getType(),
                                   getSignedTypeName(result.getType()) + "(" +
                                       get(in) + ")"));
        return success();
      })
      .Case([&](arith::BitcastOp) {
        emitLet(result, "bitcast<" + *resultType + ">(" + get(in) + ")");
        return success();
      })
      .Case<arith::IndexCastOp, arith::IndexCastUIOp, arith::ExtUIOp,
            arith::ExtSIOp, arith::TruncIOp>([&](Operation *) -> LogicalResult {
        if (is32(inElem) && is32(outElem)) {
          exprs[result] = get(in);
          return success();
        }
        if (inElem.isInteger(1) && is32(outElem)) {
          StringRef ones = isa<arith::ExtSIOp, arith::IndexCastOp>(op)
                               ? "4294967295u"
                               : "1u";
          emitLet(result, llvm::formatv("select({0}(0u), {0}({1}), {2})",
                                        *resultType, ones, get(in))
                              .str());
          return success();
        }
        if (is32(inElem) && outElem.isInteger(1)) {
          std::string inType = getTypeName(in.getType());
          emitLet(result, llvm::formatv("({0} & {1}(1u)) != {1}(0u)", get(in),
                                        inType)
                              .str());
          return success();
        }
        return op->emitError("cannot emit integer cast to WGSL");
      })
      .Default([&](Operation *) {
        return op->emitError("cannot emit cast to WGSL");
      });
}

LogicalResult WGSLEmitter::emitReduction(vector::ReductionOp op) {
  if (failed(getTypeName(op.getLoc(), op.getVector().getType())))
    return failure();
  int64_t n = op.getSourceVectorType().getNumElements();
  std::string vec = get(op.getVector());
  SmallVector<std::string> elems;
  for (int64_t i = 0; i < n; ++i)
    elems.push_back(vec + "." + getComponent(i).str());
  if (op.getAcc())
    elems.push_back(get(op.getAcc()));
  auto combine = [&](StringRef fn, StringRef opStr) {
    std::string acc = elems[0];
    for (StringRef e : ArrayRef(elems).drop_front())
      acc = fn.empty() ? "(" + acc + " " + opStr.str() + " " + e.str() + ")"
                       : (fn + "(" + acc + ", " + e + ")").str();
    return acc;
  };
  std::string expr;
  switch (op.getKind()) {
  case vector::CombiningKind::ADD:
    expr = combine("", "+");
    break;
  case vector::CombiningKind::MUL:
    expr = combine("", "*");
    break;
  case vector::CombiningKind::MINUI:
  case vector::CombiningKind::MINNUMF:
  case vector::CombiningKind::MINIMUMF:
    expr = combine("min", "");
    break;
  case vector::CombiningKind::MAXUI:
  case vector::CombiningKind::MAXNUMF:
  case vector::CombiningKind::MAXIMUMF:
    expr = combine("max", "");
    break;
  case vector::CombiningKind::AND:
    expr = combine("", "&");
    break;
  case vector::CombiningKind::OR:
    expr = combine("", "|");
    break;
  case vector::CombiningKind::XOR:
    expr = combine("", "^");
    break;
  default:
    return op.emitError("cannot emit reduction kind to WGSL");
  }
  emitLet(op.getResult(), expr);
  return success();
}

LogicalResult WGSLEmitter::emitOperation(Operation &op) {
  Location loc = op.getLoc();
  return TypeSwitch<Operation *, LogicalResult>(&op)
      // Constants.
      .Case([&](arith::ConstantOp cst) -> LogicalResult {
        FailureOr<std::string> lit =
            getLiteral(loc, cst.getValue(), cst.getType());
        if (failed(lit))
          return failure();
        exprs[cst.getResult()] = *lit;
        return success();
      })
      .Case([&](ub::PoisonOp poison) -> LogicalResult {
        FailureOr<std::string> type = getTypeName(loc, poison.getType());
        if (failed(type))
          return failure();
        exprs[poison.getResult()] = *type + "()";
        return success();
      })
      // Integer arithmetic.
      .Case([&](arith::AddIOp) { return emitBinary(&op, "+"); })
      .Case([&](arith::SubIOp) { return emitBinary(&op, "-"); })
      .Case([&](arith::MulIOp) { return emitBinary(&op, "*"); })
      .Case([&](arith::DivUIOp) { return emitBinary(&op, "/"); })
      .Case([&](arith::RemUIOp) { return emitBinary(&op, "%"); })
      .Case([&](arith::DivSIOp) { return emitBinary(&op, "/", true); })
      .Case([&](arith::RemSIOp) { return emitBinary(&op, "%", true); })
      .Case([&](arith::ShLIOp) { return emitBinary(&op, "<<"); })
      .Case([&](arith::ShRUIOp) { return emitBinary(&op, ">>"); })
      .Case([&](arith::ShRSIOp) { return emitBinary(&op, ">>", true); })
      .Case([&](arith::AndIOp) { return emitBinary(&op, "&"); })
      .Case([&](arith::OrIOp) { return emitBinary(&op, "|"); })
      .Case([&](arith::XOrIOp a) {
        return emitBinary(
            &op, getElementTypeOrSelf(a.getType()).isInteger(1) ? "!=" : "^");
      })
      .Case([&](arith::MaxUIOp) { return emitCall(&op, "max"); })
      .Case([&](arith::MinUIOp) { return emitCall(&op, "min"); })
      .Case<arith::MaxSIOp, arith::MinSIOp>([&](Operation *) -> LogicalResult {
        if (failed(getTypeName(loc, op.getResult(0).getType())))
          return failure();
        StringRef fn = isa<arith::MaxSIOp>(op) ? "max" : "min";
        emitLet(op.getResult(0),
                asUnsigned(op.getResult(0).getType(),
                           (fn + "(" + asSigned(op.getOperand(0)) + ", " +
                            asSigned(op.getOperand(1)) + ")")
                               .str()));
        return success();
      })
      .Case([&](arith::CeilDivUIOp div) -> LogicalResult {
        if (failed(getTypeName(loc, div.getType())))
          return failure();
        std::string a = get(div.getLhs()), b = get(div.getRhs());
        std::string type = getTypeName(div.getType());
        emitLet(div.getResult(),
                llvm::formatv("select(({0} - {2}(1u)) / {1} + {2}(1u), "
                              "{2}(0u), {0} == {2}(0u))",
                              a, b, type)
                    .str());
        return success();
      })
      .Case([&](arith::CmpIOp cmp) { return emitCmpI(cmp); })
      .Case([&](arith::CmpFOp cmp) { return emitCmpF(cmp); })
      .Case([&](arith::SelectOp sel) -> LogicalResult {
        if (failed(getTypeName(loc, sel.getType())))
          return failure();
        emitLet(sel.getResult(),
                "select(" + get(sel.getFalseValue()) + ", " +
                    get(sel.getTrueValue()) + ", " + get(sel.getCondition()) +
                    ")");
        return success();
      })
      // Floating-point arithmetic.
      .Case([&](arith::AddFOp) { return emitBinary(&op, "+"); })
      .Case([&](arith::SubFOp) { return emitBinary(&op, "-"); })
      .Case([&](arith::MulFOp) { return emitBinary(&op, "*"); })
      .Case([&](arith::DivFOp) { return emitBinary(&op, "/"); })
      .Case([&](arith::RemFOp) { return emitBinary(&op, "%"); })
      .Case([&](arith::NegFOp neg) -> LogicalResult {
        if (failed(getTypeName(loc, neg.getType())))
          return failure();
        emitLet(neg.getResult(), "-" + get(neg.getOperand()));
        return success();
      })
      .Case<arith::MaxNumFOp, arith::MaximumFOp>(
          [&](Operation *) { return emitCall(&op, "max"); })
      .Case<arith::MinNumFOp, arith::MinimumFOp>(
          [&](Operation *) { return emitCall(&op, "min"); })
      .Case<arith::ExtFOp, arith::TruncFOp, arith::UIToFPOp, arith::SIToFPOp,
            arith::FPToUIOp, arith::FPToSIOp, arith::BitcastOp,
            arith::IndexCastOp, arith::IndexCastUIOp, arith::ExtUIOp,
            arith::ExtSIOp, arith::TruncIOp>(
          [&](Operation *) { return emitCast(&op); })
      // Math.
      .Case([&](math::FmaOp) { return emitCall(&op, "fma"); })
      .Case([&](math::ExpOp) { return emitCall(&op, "exp"); })
      .Case([&](math::Exp2Op) { return emitCall(&op, "exp2"); })
      .Case([&](math::LogOp) { return emitCall(&op, "log"); })
      .Case([&](math::Log2Op) { return emitCall(&op, "log2"); })
      .Case([&](math::SqrtOp) { return emitCall(&op, "sqrt"); })
      .Case([&](math::RsqrtOp) { return emitCall(&op, "inverseSqrt"); })
      .Case([&](math::TanhOp) { return emitCall(&op, "tanh"); })
      .Case([&](math::SinOp) { return emitCall(&op, "sin"); })
      .Case([&](math::CosOp) { return emitCall(&op, "cos"); })
      .Case([&](math::AbsFOp) { return emitCall(&op, "abs"); })
      .Case([&](math::FloorOp) { return emitCall(&op, "floor"); })
      .Case([&](math::CeilOp) { return emitCall(&op, "ceil"); })
      .Case([&](math::TruncOp) { return emitCall(&op, "trunc"); })
      .Case([&](math::RoundEvenOp) { return emitCall(&op, "round"); })
      .Case([&](math::PowFOp) { return emitCall(&op, "pow"); })
      // Vectors.
      .Case([&](vector::FMAOp) { return emitCall(&op, "fma"); })
      .Case([&](vector::BroadcastOp bcast) -> LogicalResult {
        FailureOr<std::string> type = getTypeName(loc, bcast.getType());
        if (failed(type))
          return failure();
        if (bcast.getSourceType() == bcast.getType()) {
          exprs[bcast.getResult()] = get(bcast.getSource());
          return success();
        }
        if (isa<VectorType>(bcast.getSourceType()))
          return bcast.emitError("only scalar broadcasts can be emitted");
        emitLet(bcast.getResult(), *type + "(" + get(bcast.getSource()) + ")");
        return success();
      })
      .Case([&](vector::FromElementsOp from) -> LogicalResult {
        FailureOr<std::string> type = getTypeName(loc, from.getType());
        if (failed(type))
          return failure();
        SmallVector<std::string> elems;
        for (Value e : from.getElements())
          elems.push_back(get(e));
        emitLet(from.getResult(), *type + "(" + llvm::join(elems, ", ") + ")");
        return success();
      })
      .Case([&](vector::ExtractOp extract) -> LogicalResult {
        if (extract.getSourceVectorType().getRank() != 1 ||
            extract.getNumIndices() != 1)
          return extract.emitError("only 1-D element extracts can be emitted");
        if (failed(getTypeName(loc, extract.getSourceVectorType())) ||
            failed(getTypeName(loc, extract.getType())))
          return failure();
        OpFoldResult pos = extract.getMixedPosition()[0];
        std::string src = get(extract.getSource());
        if (auto cst = getConstantIntValue(pos))
          emitLet(extract.getResult(), src + "." + getComponent(*cst).str());
        else
          emitLet(extract.getResult(),
                  src + "[" + get(cast<Value>(pos)) + "]");
        return success();
      })
      .Case([&](vector::InsertOp insert) -> LogicalResult {
        if (insert.getDestVectorType().getRank() != 1 ||
            insert.getNumIndices() != 1)
          return insert.emitError("only 1-D element inserts can be emitted");
        if (failed(getTypeName(loc, insert.getDestVectorType())))
          return failure();
        OpFoldResult pos = insert.getMixedPosition()[0];
        std::string var =
            emitVar(insert.getDestVectorType(), get(insert.getDest()));
        std::string idx = getConstantIntValue(pos)
                              ? "." + getComponent(*getConstantIntValue(pos)).str()
                              : "[" + get(cast<Value>(pos)) + "]";
        os() << var << idx << " = " << get(insert.getValueToStore()) << ";\n";
        exprs[insert.getResult()] = var;
        return success();
      })
      .Case([&](vector::ShuffleOp shuffle) -> LogicalResult {
        FailureOr<std::string> type = getTypeName(loc, shuffle.getType());
        if (failed(type) || shuffle.getV1VectorType().getRank() != 1)
          return failure();
        int64_t n1 = shuffle.getV1VectorType().getNumElements();
        SmallVector<std::string> elems;
        for (int64_t m : shuffle.getMask()) {
          Value src = m < n1 ? shuffle.getV1() : shuffle.getV2();
          int64_t idx = m < n1 ? m : m - n1;
          if (m < 0) {
            elems.push_back(
                getTypeName(shuffle.getType().getElementType()) + "()");
            continue;
          }
          elems.push_back(get(src) + "." + getComponent(idx).str());
        }
        emitLet(shuffle.getResult(),
                *type + "(" + llvm::join(elems, ", ") + ")");
        return success();
      })
      .Case([&](vector::ExtractStridedSliceOp slice) -> LogicalResult {
        if (slice.getSourceVectorType().getRank() != 1 ||
            failed(getTypeName(loc, slice.getSourceVectorType())) ||
            failed(getTypeName(loc, slice.getType())))
          return slice.emitError("only 1-D strided slices can be emitted");
        int64_t offset = cast<IntegerAttr>(slice.getOffsets()[0]).getInt();
        int64_t size = cast<IntegerAttr>(slice.getSizes()[0]).getInt();
        std::string swizzle;
        for (int64_t i = 0; i < size; ++i)
          swizzle += getComponent(offset + i);
        emitLet(slice.getResult(), get(slice.getSource()) + "." + swizzle);
        return success();
      })
      .Case([&](vector::InsertStridedSliceOp slice) -> LogicalResult {
        if (slice.getDestVectorType().getRank() != 1 ||
            failed(getTypeName(loc, slice.getDestVectorType())))
          return slice.emitError("only 1-D strided slices can be emitted");
        int64_t offset = cast<IntegerAttr>(slice.getOffsets()[0]).getInt();
        int64_t size = slice.getSourceVectorType().getNumElements();
        std::string var =
            emitVar(slice.getDestVectorType(), get(slice.getDest()));
        std::string src = get(slice.getValueToStore());
        for (int64_t i = 0; i < size; ++i)
          os() << var << "." << getComponent(offset + i) << " = " << src << "."
               << getComponent(i) << ";\n";
        exprs[slice.getResult()] = var;
        return success();
      })
      .Case([&](vector::ReductionOp red) { return emitReduction(red); })
      .Case([&](vector::LoadOp load) -> LogicalResult {
        FailureOr<std::string> type = getTypeName(loc, load.getVectorType());
        if (failed(type))
          return failure();
        if (load.getMemRefType().getElementType() == load.getVectorType()) {
          FailureOr<std::string> access =
              getAccess(loc, load.getBase(), load.getIndices());
          if (failed(access))
            return failure();
          emitLet(load.getResult(), *access);
          return success();
        }
        // Gather the elements along the innermost dimension.
        FailureOr<std::string> base =
            getElementBase(loc, load.getBase(), load.getIndices());
        if (failed(base))
          return failure();
        SmallVector<std::string> elems;
        for (int64_t i = 0; i < load.getVectorType().getNumElements(); ++i)
          elems.push_back(llvm::formatv("{0}[{1} + {2}u]",
                                        memrefs[load.getBase()].name, *base, i));
        emitLet(load.getResult(), *type + "(" + llvm::join(elems, ", ") + ")");
        return success();
      })
      .Case([&](vector::StoreOp store) -> LogicalResult {
        if (failed(getTypeName(loc, store.getVectorType())))
          return failure();
        std::string value = get(store.getValueToStore());
        if (store.getMemRefType().getElementType() == store.getVectorType()) {
          FailureOr<std::string> access =
              getAccess(loc, store.getBase(), store.getIndices());
          if (failed(access))
            return failure();
          os() << *access << " = " << value << ";\n";
          return success();
        }
        FailureOr<std::string> base =
            getElementBase(loc, store.getBase(), store.getIndices());
        if (failed(base))
          return failure();
        for (int64_t i = 0; i < store.getVectorType().getNumElements(); ++i)
          os() << memrefs[store.getBase()].name << "[" << *base << " + " << i
               << "u] = " << value << "." << getComponent(i) << ";\n";
        return success();
      })
      // Memory.
      .Case([&](memref::LoadOp load) -> LogicalResult {
        if (failed(getTypeName(loc, load.getType())))
          return failure();
        FailureOr<std::string> access =
            getAccess(loc, load.getMemRef(), load.getIndices());
        if (failed(access))
          return failure();
        emitLet(load.getResult(), *access);
        return success();
      })
      .Case([&](memref::StoreOp store) -> LogicalResult {
        if (memrefs.lookup(store.getMemRef()).isUniform)
          return store.emitError("cannot store to a uniform buffer");
        FailureOr<std::string> access =
            getAccess(loc, store.getMemRef(), store.getIndices());
        if (failed(access))
          return failure();
        os() << *access << " = " << get(store.getValueToStore()) << ";\n";
        return success();
      })
      .Case([&](memref::AllocaOp alloca) -> LogicalResult {
        FailureOr<std::string> type = getTypeName(loc, alloca.getType());
        if (failed(type) || !alloca.getType().hasStaticShape())
          return alloca.emitError("only static allocas can be emitted");
        std::string name = getFreshName("p");
        os() << "var " << name << ": " << *type << ";\n";
        memrefs[alloca.getResult()] = {name, false};
        return success();
      })
      .Case<memref::CastOp, memref::AssumeAlignmentOp>(
          [&](Operation *) -> LogicalResult {
            auto it = memrefs.find(op.getOperand(0));
            if (it == memrefs.end())
              return op.emitError("unknown memref");
            MemRefInfo info = it->second;
            info.layoutType = getLayoutType(op.getOperand(0));
            if (info.runtimeLayout) {
              // A cast can make a dynamic offset or stride static.
              auto resultType = cast<MemRefType>(op.getResult(0).getType());
              SmallVector<int64_t> strides;
              int64_t offset;
              if (succeeded(resultType.getStridesAndOffset(strides, offset))) {
                if (!ShapedType::isDynamic(offset)) {
                  auto expr = getIndexExpr(loc, Builder(op.getContext())
                                                   .getIndexAttr(offset));
                  if (failed(expr))
                    return failure();
                  info.runtimeLayout->offset = *expr;
                }
                for (auto [dim, stride] : llvm::enumerate(strides)) {
                  if (ShapedType::isDynamic(stride))
                    continue;
                  auto expr = getIndexExpr(loc, Builder(op.getContext())
                                                   .getIndexAttr(stride));
                  if (failed(expr))
                    return failure();
                  info.runtimeLayout->strides[dim] = *expr;
                }
              }
            }
            memrefs[op.getResult(0)] = info;
            return success();
          })
      // Views alias the array of their base memref.
      .Case<memref::SubViewOp, memref::ReinterpretCastOp>(
          [&](Operation *) -> LogicalResult {
            auto it = memrefs.find(op.getOperand(0));
            if (it == memrefs.end() || it->second.isUniform)
              return op.emitError("views need a storage, workgroup or local "
                                  "base memref");
            auto viewType = cast<MemRefType>(op.getResult(0).getType());
            MemRefInfo info = it->second;
            info.layoutType = viewType;
            if (auto reinterpret = dyn_cast<memref::ReinterpretCastOp>(op)) {
              // Absolute positions in the underlying allocation.
              info.runtimeLayout.reset();
              if (!isEmittableView(viewType)) {
                RuntimeLayout layout;
                auto offset = getIndexExpr(loc, reinterpret.getMixedOffsets()[0]);
                if (failed(offset))
                  return failure();
                layout.offset = *offset;
                for (OpFoldResult stride : reinterpret.getMixedStrides()) {
                  auto expr = getIndexExpr(loc, stride);
                  if (failed(expr))
                    return failure();
                  layout.strides.push_back(*expr);
                }
                info.runtimeLayout = std::move(layout);
              }
            } else {
              auto subview = cast<memref::SubViewOp>(op);
              // Keep the static form when the types describe both layouts.
              if (!info.runtimeLayout && isEmittableView(viewType) &&
                  getLayoutType(subview.getSource()) == subview.getSourceType()) {
                memrefs[op.getResult(0)] = std::move(info);
                return success();
              }
              auto source = getRuntimeLayout(loc, subview.getSource());
              if (failed(source))
                return failure();
              RuntimeLayout layout{source->offset, {}};
              auto dropped = subview.getDroppedDims();
              for (auto [dim, offset, stride] : llvm::enumerate(
                       subview.getMixedOffsets(), subview.getMixedStrides())) {
                auto offsetExpr = getIndexExpr(loc, offset);
                auto strideExpr = getIndexExpr(loc, stride);
                if (failed(offsetExpr) || failed(strideExpr))
                  return failure();
                layout.offset = addIndex(
                    layout.offset, multiplyIndex(*offsetExpr, source->strides[dim]));
                if (!dropped.test(dim))
                  layout.strides.push_back(
                      multiplyIndex(source->strides[dim], *strideExpr));
              }
              info.runtimeLayout = std::move(layout);
            }
            memrefs[op.getResult(0)] = info;
            return success();
          })
      // GPU.
      .Case([&](gpu::ThreadIdOp id) {
        exprs[id.getResult()] =
            useBuiltin(Builtin::LocalInvocationId, id.getDimension());
        return success();
      })
      .Case([&](gpu::BlockIdOp id) {
        exprs[id.getResult()] =
            useBuiltin(Builtin::WorkgroupId, id.getDimension());
        return success();
      })
      .Case([&](gpu::GridDimOp dim) {
        exprs[dim.getResult()] =
            useBuiltin(Builtin::NumWorkgroups, dim.getDimension());
        return success();
      })
      .Case([&](gpu::GlobalIdOp id) {
        exprs[id.getResult()] =
            useBuiltin(Builtin::GlobalInvocationId, id.getDimension());
        return success();
      })
      .Case([&](gpu::BlockDimOp dim) {
        exprs[dim.getResult()] =
            llvm::formatv("{0}u",
                          blockSize[static_cast<int>(dim.getDimension())])
                .str();
        return success();
      })
      .Case([&](gpu::SubgroupIdOp id) {
        exprs[id.getResult()] = useBuiltin(Builtin::SubgroupId, std::nullopt);
        return success();
      })
      .Case([&](gpu::LaneIdOp id) {
        exprs[id.getResult()] =
            useBuiltin(Builtin::SubgroupInvocationId, std::nullopt);
        return success();
      })
      .Case([&](gpu::SubgroupSizeOp size) {
        exprs[size.getResult()] =
            useBuiltin(Builtin::SubgroupSize, std::nullopt);
        return success();
      })
      .Case([&](gpu::NumSubgroupsOp num) {
        exprs[num.getResult()] =
            useBuiltin(Builtin::NumSubgroups, std::nullopt);
        return success();
      })
      .Case([&](gpu::BarrierOp barrier) -> LogicalResult {
        if (barrier.getScope() != gpu::BarrierScope::Workgroup ||
            barrier->getNumOperands() != 0)
          return barrier.emitError("only workgroup barriers can be emitted");
        bool workgroup = true, storage = true;
        if (std::optional<ArrayAttr> spaces = barrier.getAddressSpaces()) {
          workgroup = storage = false;
          for (auto space : spaces->getAsRange<gpu::AddressSpaceAttr>()) {
            workgroup |= space.getValue() == gpu::AddressSpace::Workgroup;
            storage |= space.getValue() == gpu::AddressSpace::Global;
          }
        }
        if (storage)
          os() << "storageBarrier();\n";
        if (workgroup || !storage)
          os() << "workgroupBarrier();\n";
        return success();
      })
      .Case([&](gpu::SubgroupReduceOp reduce) -> LogicalResult {
        if (reduce.getClusterSize())
          return reduce.emitError("clustered reductions cannot be emitted");
        if (failed(getTypeName(loc, reduce.getType())))
          return failure();
        StringRef fn;
        switch (reduce.getOp()) {
        case gpu::AllReduceOperation::ADD:
          fn = "subgroupAdd";
          break;
        case gpu::AllReduceOperation::MUL:
          fn = "subgroupMul";
          break;
        case gpu::AllReduceOperation::MINUI:
        case gpu::AllReduceOperation::MINNUMF:
        case gpu::AllReduceOperation::MINIMUMF:
          fn = "subgroupMin";
          break;
        case gpu::AllReduceOperation::MAXUI:
        case gpu::AllReduceOperation::MAXNUMF:
        case gpu::AllReduceOperation::MAXIMUMF:
          fn = "subgroupMax";
          break;
        default:
          return reduce.emitError("cannot emit reduction kind to WGSL");
        }
        usesSubgroups = true;
        emitLet(reduce.getResult(),
                (fn + "(" + get(reduce.getValue()) + ")").str());
        return success();
      })
      // Subgroup matrices.
      .Case([&](gpu::SubgroupMmaLoadMatrixOp load) -> LogicalResult {
        FailureOr<std::string> type = getTypeName(loc, load.getType());
        std::string offset;
        FailureOr<std::string> ptr = getMMAPointer(
            loc, load.getSrcMemref(), load.getIndices(), offset);
        if (failed(type) || failed(ptr))
          return failure();
        emitLet(load.getResult(),
                llvm::formatv("subgroupMatrixLoad<{0}>({1}, {2}, {3}, {4}u)",
                              *type, *ptr, offset,
                              load.getTranspose() ? "true" : "false",
                              load.getLeadDimension().getZExtValue())
                    .str());
        return success();
      })
      .Case([&](gpu::SubgroupMmaStoreMatrixOp store) -> LogicalResult {
        std::string offset;
        FailureOr<std::string> ptr = getMMAPointer(
            loc, store.getDstMemref(), store.getIndices(), offset);
        if (failed(ptr))
          return failure();
        os() << llvm::formatv("subgroupMatrixStore({0}, {1}, {2}, {3}, {4}u);\n",
                              *ptr, offset, get(store.getSrc()),
                              store.getTranspose() ? "true" : "false",
                              store.getLeadDimension().getZExtValue());
        return success();
      })
      .Case([&](gpu::SubgroupMmaComputeOp mma) -> LogicalResult {
        if (mma.getATranspose() || mma.getBTranspose())
          return mma.emitError("transposed MMA operands cannot be emitted");
        if (failed(getTypeName(loc, mma.getType())))
          return failure();
        emitLet(mma.getResult(),
                "subgroupMatrixMultiplyAccumulate(" + get(mma.getOpA()) +
                    ", " + get(mma.getOpB()) + ", " + get(mma.getOpC()) + ")");
        return success();
      })
      .Case([&](gpu::SubgroupMmaConstantMatrixOp cst) -> LogicalResult {
        FailureOr<std::string> type = getTypeName(loc, cst.getType());
        if (failed(type))
          return failure();
        if (!matchPattern(cst.getValue(), m_AnyZeroFloat()) &&
            !matchPattern(cst.getValue(), m_Zero()))
          return cst.emitError("only zero constant matrices can be emitted");
        exprs[cst.getResult()] = *type + "()";
        return success();
      })
      // Structured control flow.
      .Case([&](scf::ForOp forOp) { return emitFor(forOp); })
      .Case([&](scf::IfOp ifOp) { return emitIf(ifOp); })
      .Case([&](scf::WhileOp whileOp) { return emitWhile(whileOp); })
      .Default([&](Operation *) {
        return op.emitError("cannot emit operation to WGSL");
      });
}

//===----------------------------------------------------------------------===//
// Kernels
//===----------------------------------------------------------------------===//

LogicalResult WGSLEmitter::emitKernel(gpu::GPUFuncOp func) {
  if (!func.getBody().hasOneBlock())
    return func.emitError("kernels must have a single block; use structured "
                          "control flow");
  std::optional<ArrayRef<int32_t>> knownBlockSize = func.getKnownBlockSize();
  if (!knownBlockSize)
    return func.emitError("kernels need a known_block_size to set the WGSL "
                          "workgroup size");
  blockSize.assign(knownBlockSize->begin(), knownBlockSize->end());

  std::string declsStr, bodyStr;
  llvm::raw_string_ostream declsOs(declsStr), bodyOs(bodyStr);
  raw_indented_ostream decls(declsOs), bodyIndented(bodyOs);
  body = &bodyIndented;

  // Resource variables.
  for (unsigned idx = 0, e = func.getNumArguments(); idx < e; ++idx) {
    BlockArgument arg = func.getArgument(idx);
    auto memrefType = dyn_cast<MemRefType>(arg.getType());
    if (!memrefType)
      return func.emitError("kernel argument ")
             << idx << " must be a memref; pass scalars through a uniform";
    auto getIntAttr = [&](StringRef name, int64_t defaultValue) {
      if (auto attr = func.getArgAttrOfType<IntegerAttr>(idx, name))
        return attr.getInt();
      return defaultValue;
    };
    auto getStrAttr = [&](StringRef name, StringRef defaultValue) {
      if (auto attr = func.getArgAttrOfType<StringAttr>(idx, name))
        return attr.getValue();
      return defaultValue;
    };
    MemRefInfo info;
    info.name = getStrAttr("wgsl.name", "").str();
    if (info.name.empty())
      info.name = llvm::formatv("arg{0}", idx).str();
    info.isUniform = getStrAttr("wgsl.address_space", "storage") == "uniform";
    std::string type;
    if (info.isUniform) {
      Type elem = memrefType.getElementType();
      if (memrefType.getRank() != 1 || !memrefType.hasStaticShape() ||
          !(elem.isInteger(32) || elem.isF32()))
        return func.emitError("uniform arguments must be static 1-D memrefs "
                              "of i32 or f32");
      type = llvm::formatv("array<vec4<{0}>, {1}>", getTypeName(elem),
                           llvm::divideCeil(memrefType.getNumElements(), 4))
                 .str();
    } else {
      FailureOr<std::string> typeName = getTypeName(func.getLoc(), memrefType);
      if (failed(typeName))
        return failure();
      type = *typeName;
    }
    decls << llvm::formatv("@group({0}) @binding({1}) ",
                           getIntAttr("wgsl.group", 0),
                           getIntAttr("wgsl.binding", idx));
    if (info.isUniform)
      decls << "var<uniform> ";
    else
      decls << "var<storage, " << getStrAttr("wgsl.access", "read_write")
            << "> ";
    decls << info.name << ": " << type << ";\n";
    memrefs[arg] = info;
  }

  // Workgroup memory.
  for (auto [idx, arg] : llvm::enumerate(func.getWorkgroupAttributionBBArgs())) {
    FailureOr<std::string> type = getTypeName(func.getLoc(), arg.getType());
    if (failed(type))
      return failure();
    std::string name = llvm::formatv("shmem{0}", idx).str();
    if (auto attr = func.getWorkgroupAttributionAttr(idx, "wgsl.name"))
      name = cast<StringAttr>(attr).getValue().str();
    decls << "var<workgroup> " << name << ": " << *type << ";\n";
    memrefs[arg] = {name, false};
  }

  // Function body.
  bodyIndented.indent();
  for (BlockArgument arg : func.getPrivateAttributions()) {
    FailureOr<std::string> type = getTypeName(func.getLoc(), arg.getType());
    if (failed(type))
      return failure();
    std::string name = getFreshName("p");
    bodyIndented << "var " << name << ": " << *type << ";\n";
    memrefs[arg] = {name, false};
  }
  if (failed(emitBlock(func.getBody().front(), {})))
    return failure();

  // Directives depend on what the body used, so they are emitted last.
  out << "// Generated by mlir-translate --mlir-to-wgsl.\n";
  if (usesSubgroupMatrix)
    out << "diagnostic(off, chromium.subgroup_matrix_uniformity);\n";
  if (usesF16)
    out << "enable f16;\n";
  if (usesSubgroups)
    out << "enable subgroups;\n";
  if (usesSubgroupMatrix)
    out << "enable chromium_experimental_subgroup_matrix;\n";
  out << "\n" << declsStr << "\n";

  SmallVector<std::string> params;
  for (Builtin b : llvm::to_vector(builtins)) {
    BuiltinInfo info = getBuiltinInfo(b);
    params.push_back(llvm::formatv("@builtin({0}) {1}: {2}", info.attr,
                                   info.name, info.type));
  }
  llvm::sort(params);
  out << llvm::formatv("@compute @workgroup_size({0}, {1}, {2})\n",
                       blockSize[0], blockSize[1], blockSize[2]);
  out << "fn " << func.getName() << "(" << llvm::join(params, ", ")
      << ") {\n";
  out << bodyStr << "}\n";
  return success();
}

LogicalResult mlir::wgsl::translateToWGSL(Operation *op, raw_ostream &os,
                                          StringRef kernelName) {
  SmallVector<gpu::GPUFuncOp> kernels;
  op->walk([&](gpu::GPUFuncOp func) {
    if (func.isKernel() && (kernelName.empty() || func.getName() == kernelName))
      kernels.push_back(func);
  });
  if (kernels.size() != 1)
    return op->emitError("expected exactly one kernel to translate, found ")
           << kernels.size();
  WGSLEmitter emitter(os);
  return emitter.emitKernel(kernels.front());
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void mlir::registerToWGSLTranslation() {
  static llvm::cl::opt<std::string> kernelName(
      "wgsl-kernel", llvm::cl::desc("Name of the gpu.func kernel to emit"),
      llvm::cl::init(""));

  TranslateFromMLIRRegistration reg(
      "mlir-to-wgsl", "translate a gpu.func kernel to WGSL",
      [](Operation *op, raw_ostream &output) {
        return wgsl::translateToWGSL(op, output, kernelName);
      },
      [](DialectRegistry &registry) {
        registry.insert<arith::ArithDialect, gpu::GPUDialect,
                        math::MathDialect, memref::MemRefDialect,
                        scf::SCFDialect, ub::UBDialect,
                        vector::VectorDialect>();
      });
}
