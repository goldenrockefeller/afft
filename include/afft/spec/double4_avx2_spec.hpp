#ifndef AFFT_DOUBLE4_AVX2_SPEC_HPP
#define AFFT_DOUBLE4_AVX2_SPEC_HPP

#include <cstddef>

#include "xsimd/xsimd.hpp"
#include "afft/spec/double2_sse2_spec.hpp"
#include <immintrin.h>

namespace afft
{
    struct Double4Avx2Spec
    {
        using sample = double;
        using arch = typename xsimd::avx2;
        using operand = xsimd::batch<sample, arch>;
        using fallback_spec = Double2Sse2Spec;
        static constexpr std::size_t n_samples_per_operand = 4;
        static constexpr std::size_t prefetch_lookahead = 4;
        static constexpr std::size_t min_partition_len = 512;

        static inline void load(operand &x, const sample *ptr)
        {
            x = xsimd::batch<sample, arch>::load_unaligned(ptr);
        }

        static inline void store(sample *ptr, const operand &x)
        {
            x.store_unaligned(ptr);
        }

        static inline void prefetch(const sample *ptr){
            _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T0);
        }

        static inline void prefetch(const std::size_t *ptr){
            _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T0);
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
            operand in_d)
        {
            // Do Nothing
        }

        template <std::size_t LogInterleaveFactor>
        static inline void interleave2(
            operand &out_a,
            operand &out_b,
            operand in_a,
            operand in_b)
        {
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

    //         // Lower halves
    // __m128d a_lo = _mm256_castpd256_pd128(in_a);
    // __m128d b_lo = _mm256_castpd256_pd128(in_b);
    // __m128d c_lo = _mm256_castpd256_pd128(in_c);
    // __m128d d_lo = _mm256_castpd256_pd128(in_d);

    // // Upper halves
    // __m128d a_hi = _mm256_extractf128_pd(in_a, 1);
    // __m128d b_hi = _mm256_extractf128_pd(in_b, 1);
    // __m128d c_hi = _mm256_extractf128_pd(in_c, 1);
    // __m128d d_hi = _mm256_extractf128_pd(in_d, 1);

    // // out_a = shuffle and insert lower lanes
    // __m128d ab_lo = _mm_unpacklo_pd(a_lo, b_lo);
    // __m256d abcd_lo = _mm256_insertf128_pd(_mm256_castpd128_pd256(ab_lo), c_lo, 1);
    // __m256d abcd_lo2 = _mm256_insertf128_pd(_mm256_castpd128_pd256(ab_lo), d_lo, 1);
    // out_a = _mm256_shuffle_pd(abcd_lo, abcd_lo2, 0b0010); // control = 0x2

    // // out_b = blend upper half
    // __m128d ab_hi = _mm_unpackhi_pd(a_lo, b_lo);
    // __m256d c_perm = _mm256_permute4x64_pd(in_c, 0x55); // replicate lane 1
    // __m256d tmp1 = _mm256_blend_pd(c_perm, _mm256_castpd128_pd256(ab_hi), 0b1100);
    // __m256d ab_hi_insert = _mm256_insertf128_pd(_mm256_castpd128_pd256(ab_hi), d_lo, 1);
    // out_b = _mm256_blend_pd(tmp1, ab_hi_insert, 0b1010);

    // // out_c = interleave upper halves of A and B with broadcasted C and D
    // __m128d ab2_lo = _mm_unpacklo_pd(a_hi, b_hi);
    // __m256d ab2 = _mm256_castpd128_pd256(ab2_lo);
    // ab2 = _mm256_blend_pd(ab2, in_c, 0b1100);
    // __m256d d_broadcast = _mm256_broadcast_sd(&d_lo[0]);
    // out_c = _mm256_blend_pd(ab2, d_broadcast, 0b1000);

    // // out_d = upper hi interleave and shuffle
    // __m128d ab2_hi = _mm_unpackhi_pd(a_hi, b_hi);
    // __m256d c_shuffle = _mm256_shuffle_pd(in_c, in_c, 0b0100);
    // __m256d blend1 = _mm256_blend_pd(_mm256_castpd128_pd256(ab2_hi), c_shuffle, 0b1100);
    // out_d = _mm256_blend_pd(blend1, in_d, 0b1000);
            // Unpack lower and upper halves of the input vectors
            __m256d lo0 = _mm256_unpacklo_pd(in_a, in_b);
            __m256d hi0 = _mm256_unpackhi_pd(in_a, in_b);
            __m256d lo1 = _mm256_unpacklo_pd(in_c, in_d);
            __m256d hi1 = _mm256_unpackhi_pd(in_c, in_d);

            // Transpose the 4x4 matrix
            out_a = _mm256_permute2f128_pd(lo0, lo1, 0x20); // [0,1,2,3]
            out_b = _mm256_permute2f128_pd(hi0, hi1, 0x20); // [4,5,6,7]
            out_c = _mm256_permute2f128_pd(lo0, lo1, 0x31); // [8,9,10,11]
            out_d = _mm256_permute2f128_pd(hi0, hi1, 0x31); // [12,13,14,15]
        }

        template <>
        void interleave4<1>(
            operand &out_a,
            operand &out_b,
            operand &out_c,
            operand &out_d,
            operand in_a,
            operand in_b,
            operand in_c,
            operand in_d)
        {
            // ymm4 = insert xmm1 (low 128 bits of in_b) into ymm0 (in_a) upper 128 bits
            __m256d ymm4 = _mm256_insertf128_pd(in_a, _mm256_castpd256_pd128(in_b), 1);
            out_a = ymm4;

            // ymm4 = insert xmm3 (low 128 bits of in_d) into ymm2 (in_c) upper 128 bits
            ymm4 = _mm256_insertf128_pd(in_c, _mm256_castpd256_pd128(in_d), 1);
            out_b = ymm4;

            // ymm0 = perm2f128 ymm0 (in_a) and ymm1 (in_b) with immediate 49 (0x31)
            __m256d ymm0 = _mm256_permute2f128_pd(in_a, in_b, 0x31);
            out_c = ymm0;

            // ymm0 = perm2f128 ymm0 (in_c) and ymm3 (in_d) with immediate 49 (0x31)
            ymm0 = _mm256_permute2f128_pd(in_c, in_d, 0x31);
            // vshufpd ymm0, ymm0, ymm0, 15 (0xF = 0b1111)
            ymm0 = _mm256_shuffle_pd(ymm0, ymm0, 0xF);
            out_d = ymm0;
        }

        template <>
        void interleave2<0>(
            operand &out_a,
            operand &out_b,
            operand in_a,
            operand in_b)
        {
            // Permute in_b: 0, 2, x, x (via immediate 96 = 0b01100000)
            __m256d perm_b = _mm256_permute4x64_pd(in_b, 0x60);

            // Permute in_a: 0, 3, 1, 2 (via immediate 212 = 0b11010100)
            __m256d perm_a = _mm256_permute4x64_pd(in_a, 0xD4);

            // Blend: select bits from perm_a and perm_b using mask 0b1010
            out_a = _mm256_blend_pd(perm_a, perm_b, 0b1010);

            // Permute both inputs: 2,3,0,1 (via immediate 230 = 0b11100110)
            __m256d in_a2 = _mm256_permute4x64_pd(in_a, 0xE6);
            __m256d in_b2 = _mm256_permute4x64_pd(in_b, 0xE6);

            // Shuffle pairs: interleave across 128-bit lanes with mask 0b1100
            out_b = _mm256_shuffle_pd(in_a2, in_b2, 0b1100);
        }

        template <>
        void interleave2<1>(
            operand &out_a,
            operand &out_b,
            operand in_a,
            operand in_b)
        {
            // Insert the lower 128 bits of in_b into the upper half of in_a
            out_a = _mm256_insertf128_pd(in_a, _mm256_castpd256_pd128(in_b), 1);

            // Permute the 128-bit lanes: lower from in_a, upper from in_b (control = 0x31)
            out_b = _mm256_permute2f128_pd(in_a, in_b, 0x31);
        }
    };
}

#endif