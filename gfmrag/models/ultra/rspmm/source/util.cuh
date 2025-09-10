#pragma once

#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>

namespace at {

const unsigned kFullMask = 0xFFFFFFFF;

template <class scalar_t>
__device__ scalar_t warp_reduce(scalar_t value) {
#pragma unroll
    for (int delta = 1; delta < warpSize; delta *= 2)
#if __CUDACC_VER_MAJOR__ >= 9
        value += __shfl_down_sync(kFullMask, value, delta);
#else
        value += __shfl_down(value, delta);
#endif
    return value;
}

// Specialization for c10::Half to resolve __shfl_down_sync ambiguity
template <>
__device__ c10::Half warp_reduce<c10::Half>(c10::Half value) {
#pragma unroll
    for (int delta = 1; delta < warpSize; delta *= 2) {
#if __CUDACC_VER_MAJOR__ >= 9
        __half val = __half(value);
        val = __shfl_down_sync(kFullMask, val, delta);
        value += c10::Half(val);
#else
        __half val = __half(value);
        val = __shfl_down(val, delta);
        value += c10::Half(val);
#endif
    }
    return value;
}

// Specialization for c10::BFloat16
template <>
__device__ c10::BFloat16 warp_reduce<c10::BFloat16>(c10::BFloat16 value) {
#pragma unroll
    for (int delta = 1; delta < warpSize; delta *= 2) {
#if __CUDACC_VER_MAJOR__ >= 9
        __nv_bfloat16 val = __nv_bfloat16(value);
        val = __shfl_down_sync(kFullMask, val, delta);
        value += c10::BFloat16(val);
#else
        float val = float(value);
        val = __shfl_down(val, delta);
        value += c10::BFloat16(val);
#endif
    }
    return value;
}

// Specialization for float (explicit for mixed precision kernels)
template <>
__device__ float warp_reduce<float>(float value) {
#pragma unroll
    for (int delta = 1; delta < warpSize; delta *= 2)
#if __CUDACC_VER_MAJOR__ >= 9
        value += __shfl_down_sync(kFullMask, value, delta);
#else
        value += __shfl_down(value, delta);
#endif
    return value;
}

template<class scalar_t>
__device__ scalar_t warp_broadcast(scalar_t value, int lane_id) {
#if __CUDACC_VER_MAJOR__ >= 9
    return __shfl_sync(kFullMask, value, lane_id);
#else
    return __shfl(value, lane_id);
#endif
}

} // namespace at
