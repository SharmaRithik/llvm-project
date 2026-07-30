add_subdirectory(TargetLowering)

add_clang_library(MLIRCIRTransforms
  CallConvLoweringPass.cpp
  CIRCanonicalize.cpp
  CIRSimplify.cpp
  CIRTransformUtils.cpp
  CXXABILowering.cpp
  EHABILowering.cpp
  TargetLowering.cpp
  FlattenCFG.cpp
  HoistAllocas.cpp
  LoweringPrepare.cpp
  LoopInterchange.cpp
  GotoSolver.cpp
  IdiomRecognizer.cpp
  LibOpt.cpp

  DEPENDS
  MLIRCIRPassIncGen

  LINK_LIBS PUBLIC
  clangAST
  clangBasic

  LLVMABI
  MLIRABI
  MLIRAnalysis
  MLIRDLTIDialect
  MLIRIR
  MLIRPass
  MLIRTransformUtils

  MLIRCIR
  MLIRCIRAnalysis
  MLIRCIRInterfaces
  MLIRCIRTargetLowering
  MLIROpenACCDialect
  MLIROpenMPDialect
)
