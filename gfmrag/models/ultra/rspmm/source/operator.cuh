#pragma once

#include <limits>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>

#ifdef __CUDA_ARCH__
    #define HOST_DEVICE __host__ __device__
#else
    #define HOST_DEVICE
#endif

// Macro to get zero value for any type
#define GET_ZERO(scalar_t) \
  (std::is_same_v<scalar_t, c10::Half> ? scalar_t(0.0f) : \
   std::is_same_v<scalar_t, c10::BFloat16> ? scalar_t(0.0f) : \
   scalar_t(0))

namespace at {

// Helper function to get zero value for NaryOp
template<typename NaryOp, typename scalar_t>
HOST_DEVICE constexpr scalar_t get_nary_zero() {
    if constexpr (std::is_same_v<scalar_t, c10::Half>) {
        return c10::Half(0.0f);
    } else if constexpr (std::is_same_v<scalar_t, c10::BFloat16>) {
        return c10::BFloat16(0.0f);
    } else {
        return NaryOp::zero;
    }
}

template <class scalar_t>
struct BinaryAdd {
    HOST_DEVICE static scalar_t forward(scalar_t x, scalar_t y) {
        return x + y;
    }

    HOST_DEVICE static scalar_t backward_lhs(scalar_t x, scalar_t y) {
        return 1;
    }

    HOST_DEVICE static scalar_t backward_rhs(scalar_t x, scalar_t y) {
        return 1;
    }
};

template <class scalar_t>
struct BinaryMul {
    HOST_DEVICE static scalar_t forward(scalar_t x, scalar_t y) {
        return x * y;
    }

    HOST_DEVICE static scalar_t backward_lhs(scalar_t x, scalar_t y) {
        return y;
    }

    HOST_DEVICE static scalar_t backward_rhs(scalar_t x, scalar_t y) {
        return x;
    }
};

template <class scalar_t>
struct NaryAdd {
    HOST_DEVICE static scalar_t forward(scalar_t result, scalar_t x) {
        return result + x;
    }

    HOST_DEVICE static scalar_t backward(scalar_t result, scalar_t x) {
        return 1;
    }

    static constexpr scalar_t zero = 0;
};

// Specialization for c10::Half
template <>
struct NaryAdd<c10::Half> {
    HOST_DEVICE static c10::Half forward(c10::Half result, c10::Half x) {
        return result + x;
    }

    HOST_DEVICE static c10::Half backward(c10::Half result, c10::Half x) {
        return c10::Half(1.0f);
    }

    HOST_DEVICE static c10::Half zero_value() {
        return c10::Half(0.0f);
    }
};

// Specialization for c10::BFloat16
template <>
struct NaryAdd<c10::BFloat16> {
    HOST_DEVICE static c10::BFloat16 forward(c10::BFloat16 result, c10::BFloat16 x) {
        return result + x;
    }

    HOST_DEVICE static c10::BFloat16 backward(c10::BFloat16 result, c10::BFloat16 x) {
        return c10::BFloat16(1.0f);
    }

    HOST_DEVICE static c10::BFloat16 zero_value() {
        return c10::BFloat16(0.0f);
    }
};

template <class scalar_t>
struct NaryMin {
    HOST_DEVICE static scalar_t forward(scalar_t result, scalar_t x) {
        return result < x ? result : x;
    }

    HOST_DEVICE static scalar_t backward(scalar_t result, scalar_t x) {
        return result == x ? 1 : 0;
    }

    static constexpr scalar_t zero = std::numeric_limits<scalar_t>::max();
};

template <class scalar_t>
struct NaryMax {
    HOST_DEVICE static scalar_t forward(scalar_t result, scalar_t x) {
        return result > x ? result : x;
    }

    HOST_DEVICE static scalar_t backward(scalar_t result, scalar_t x) {
        return result == x ? 1 : 0;
    }

    static constexpr scalar_t zero = std::numeric_limits<scalar_t>::lowest();
};

} // namespace at
