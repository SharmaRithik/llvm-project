// RUN: mlir-translate --mlir-to-wgsl --split-input-file --verify-diagnostics %s

module attributes {gpu.container_module} {
  gpu.module @kernels {
    // expected-error @below {{kernels need a known_block_size to set the WGSL workgroup size}}
    gpu.func @no_block_size(%buf: memref<?xf32>) kernel {
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  gpu.module @kernels {
    // expected-error @below {{only 1-D vectors of 2 to 4 elements are supported}}
    gpu.func @wide_vector(%buf: memref<?xvector<8xf32>>) kernel
        attributes {known_block_size = array<i32: 32, 1, 1>} {
      %c0 = arith.constant 0 : index
      %v = memref.load %buf[%c0] : memref<?xvector<8xf32>>
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @subgroup_barrier(%buf: memref<?xf32>) kernel
        attributes {known_block_size = array<i32: 32, 1, 1>} {
      // expected-error @below {{only workgroup barriers can be emitted}}
      gpu.barrier scope <subgroup>
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  gpu.module @kernels {
    // expected-error @below {{kernel argument 0 must be a memref; pass scalars through a uniform}}
    gpu.func @scalar_arg(%x: f32) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
      gpu.return
    }
  }
}
