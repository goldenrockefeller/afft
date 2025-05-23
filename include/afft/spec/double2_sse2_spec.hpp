#ifndef AFFT_DOUBLE2_SSE2_SPEC_HPP
#define AFFT_DOUBLE2_SSE2_SPEC_HPP

#include <cstddef>

#include "xsimd/xsimd.hpp"
#include "afft/spec/std_spec.hpp"

namespace afft
{
    struct Double2Sse2Spec
    {
        using sample = double;
        using arch = typename xsimd::sse2;
        using operand = xsimd::batch<sample, arch>;
        using fallback_spec = StdSpec<sample>;
        static constexpr std::size_t n_samples_per_operand = 2;

        static inline void load(operand &x, const sample *ptr)
        {
            x = xsimd::batch<sample, arch>::load_unaligned(ptr);
        }

        static inline void store(sample *ptr, const operand &x)
        {
            x.store_unaligned(ptr);
        }

        template <std::size_t LogInterleaveFactor>
        static inline void interleave4(
            operand &out_a, 
            operand &out_b, 
            operand &out_c,
            operand &out_d,
            operand in_a, 
            operand in_b, 
            operand in_c,
            operand in_d
        ) {
            // Do Nothing
        }

        template <std::size_t LogInterleaveFactor>
        static inline void interleave2(
            operand &out_a, 
            operand &out_b, 
            operand in_a, 
            operand in_b
        ) {
            // Do Nothing
        }

        template <>
        void interleave4<0>(
            operand &out_a,
            operand &out_b,
            operand &out_c,
            operand &out_d,
            operand in_a,
            operand in_b,
            operand in_c,
            operand in_d)
        {
            out_a = _mm_unpacklo_pd(in_a, in_b);
            out_b = _mm_unpacklo_pd(in_c, in_d);
            out_c = _mm_unpackhi_pd(in_a, in_b);
            out_d = _mm_unpackhi_pd(in_c, in_d);
        }

        template <>
        void interleave2<0>(
            operand &out_a, 
            operand &out_b, 
            operand in_a, 
            operand in_b
        ) {
            out_a = _mm_unpacklo_pd(in_a, in_b);
            out_b = _mm_unpackhi_pd(in_a, in_b);
        }

    };
}

#endif