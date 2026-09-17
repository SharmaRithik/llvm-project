add_mlir_conversion_library(MLIRGPUToSPIRV
  GPUToSPIRV.cpp
  GPUToSPIRVPass.cpp
  WmmaOpsToSPIRV.cpp

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithToSPIRV
  MLIRGPUDialect
  MLIRFuncToSPIRV
  MLIRMathToSPIRV
  MLIRIR
  MLIRPass
  MLIRSCFToSPIRV
  MLIRVectorToSPIRV
  MLIRSPIRVDialect
  MLIRSPIRVConversion
  MLIRSupport
  MLIRTransforms
  )
