#include <cuda_runtime.h>
#include <math.h>

//------------------------------------------------------------------------------
// Operator overloads for float2, float3, float4
//------------------------------------------------------------------------------

__host__ __device__
inline float2 operator+(const float2 &a, const float2 &b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

__host__ __device__
inline float2 operator*(float s, const float2 &v) {
    return make_float2(s * v.x, s * v.y);
}

// float3
__host__ __device__
inline float3 operator+(const float3 &a, const float3 &b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__
inline float3 operator*(float s, const float3 &v) {
    return make_float3(s * v.x, s * v.y, s * v.z);
}

// float4
__host__ __device__
inline float4 operator+(const float4 &a, const float4 &b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__host__ __device__
inline float4 operator*(float s, const float4 &v) {
    return make_float4(s * v.x, s * v.y, s * v.z, s * v.w);
}

//------------------------------------------------------------------------------
// Math functions: dot, normalize, etc.
//------------------------------------------------------------------------------

__host__ __device__
inline float dot(const float3 &a, const float3 &b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__host__ __device__
inline float3 normalize(const float3 &v) {
    float len = sqrtf(dot(v, v));
    return (len > 1e-8f) ? make_float3(v.x/len, v.y/len, v.z/len) : make_float3(0.0f,0.0f,0.0f);
}

// Helper to extract the xyz from a float4
__host__ __device__
inline float3 xyz(const float4 &v) {
    return make_float3(v.x, v.y, v.z);
}

