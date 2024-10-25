// clang/test/SemaHLSL/BuiltIns/ConstantBuffers.hlsl
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -fsyntax-only -verify %s

typedef vector<float, 3> float3;

struct Constants {
    float3 position;
};

ConstantBuffer<Constants> Buffer;

// expected-error@+2 {{class template 'ConstantBuffer' requires template arguments}}
// expected-note@*:* {{template declaration from hidden source: template <class element_type> class ConstantBuffer}}
ConstantBuffer BufferErr1;

// expected-error@+2 {{too few template arguments for class template 'ConstantBuffer'}}
// expected-note@*:* {{template declaration from hidden source: template <class element_type> class ConstantBuffer}}
ConstantBuffer<> BufferErr2;

[numthreads(1,1,1)]
void main() {
    // expected-error@+1 {{no member named 'position' in 'hlsl::ConstantBuffer<Constants>'}}
    Buffer.position;
    
    // expected-error@+1 {{no member named 'invalid' in 'hlsl::ConstantBuffer<Constants>'}}
    Buffer.invalid;
}
