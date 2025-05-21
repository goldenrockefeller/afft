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

        static inline void load(operand &x, const sample *ptr)
        {
            x = xsimd::batch<sample, arch>::load_unaligned(ptr);
        }

        static inline void store(sample *ptr, const operand &x)
        {
            x.store_unaligned(ptr);
        }

        static inline void transpose_diagonal(sample *out_real, sample *out_imag, const sample *in_real, const sample *in_imag, const std::size_t *offsets)
        {
            // from https://gist.github.com/nanaHa1003/b13b6d927b7997d5b7c9c72c0fc17a53

            operand row0_real, row1_real, row2_real, row3_real;
            operand row0_imag, row1_imag, row2_imag, row3_imag;

            std::size_t offsets0 = offsets[0];
            std::size_t offsets1 = offsets[1];
            std::size_t offsets2 = offsets[2];
            std::size_t offsets3 = offsets[3];

            row0_real = _mm256_load_pd(in_real + offsets0);
            row1_real = _mm256_load_pd(in_real + offsets1);
            row2_real = _mm256_load_pd(in_real + offsets2);
            row3_real = _mm256_load_pd(in_real + offsets3);
            row0_imag = _mm256_load_pd(in_imag + offsets0);
            row1_imag = _mm256_load_pd(in_imag + offsets1);
            row2_imag = _mm256_load_pd(in_imag + offsets2);
            row3_imag = _mm256_load_pd(in_imag + offsets3);

            transpose4x4(
                row0_real, row0_imag,
                row1_real, row1_imag,
                row2_real, row2_imag,
                row3_real, row3_imag,
                row0_real, row0_imag,
                row1_real, row1_imag,
                row2_real, row2_imag,
                row3_real, row3_imag
            );

            _mm256_store_pd(out_real + offsets0, row0_real);
            _mm256_store_pd(out_real + offsets1, row1_real);
            _mm256_store_pd(out_real + offsets2, row2_real);
            _mm256_store_pd(out_real + offsets3, row3_real);
            _mm256_store_pd(out_imag + offsets0, row0_imag);
            _mm256_store_pd(out_imag + offsets1, row1_imag);
            _mm256_store_pd(out_imag + offsets2, row2_imag);
            _mm256_store_pd(out_imag + offsets3, row3_imag);
        }

        static inline void transpose_off_diagonal(sample *out_real, sample *out_imag, const sample *in_real, const sample *in_imag, const std::size_t *offsets)
        {
            // from https://gist.github.com/nanaHa1003/b13b6d927b7997d5b7c9c72c0fc17a53

            operand row0_a_real, row1_a_real, row2_a_real, row3_a_real;
            operand row0_b_real, row1_b_real, row2_b_real, row3_b_real;

            operand row0_a_imag, row1_a_imag, row2_a_imag, row3_a_imag;
            operand row0_b_imag, row1_b_imag, row2_b_imag, row3_b_imag;

            std::size_t offsets0 = offsets[0];
            std::size_t offsets1 = offsets[1];
            std::size_t offsets2 = offsets[2];
            std::size_t offsets3 = offsets[3];
            std::size_t offsets4 = offsets[4];
            std::size_t offsets5 = offsets[5];
            std::size_t offsets6 = offsets[6];
            std::size_t offsets7 = offsets[7];

            row0_a_real = _mm256_load_pd(in_real + offsets0);
            row1_a_real = _mm256_load_pd(in_real + offsets1);
            row2_a_real = _mm256_load_pd(in_real + offsets2);
            row3_a_real = _mm256_load_pd(in_real + offsets3);

            row0_a_imag = _mm256_load_pd(in_imag + offsets0);
            row1_a_imag = _mm256_load_pd(in_imag + offsets1);
            row2_a_imag = _mm256_load_pd(in_imag + offsets2);
            row3_a_imag = _mm256_load_pd(in_imag + offsets3);

            row0_b_real = _mm256_load_pd(in_real + offsets4);
            row1_b_real = _mm256_load_pd(in_real + offsets5);
            row2_b_real = _mm256_load_pd(in_real + offsets6);
            row3_b_real = _mm256_load_pd(in_real + offsets7);

            row0_b_imag = _mm256_load_pd(in_imag + offsets4);
            row1_b_imag = _mm256_load_pd(in_imag + offsets5);
            row2_b_imag = _mm256_load_pd(in_imag + offsets6);
            row3_b_imag = _mm256_load_pd(in_imag + offsets7);

            transpose4x4(
                row0_a_real, row0_a_imag,
                row1_a_real, row1_a_imag,
                row2_a_real, row2_a_imag,
                row3_a_real, row3_a_imag,
                row0_a_real, row0_a_imag,
                row1_a_real, row1_a_imag,
                row2_a_real, row2_a_imag,
                row3_a_real, row3_a_imag
            );

            transpose4x4(
                row0_b_real, row0_b_imag,
                row1_b_real, row1_b_imag,
                row2_b_real, row2_b_imag,
                row3_b_real, row3_b_imag,
                row0_b_real, row0_b_imag,
                row1_b_real, row1_b_imag,
                row2_b_real, row2_b_imag,
                row3_b_real, row3_b_imag
            );

            _mm256_store_pd(out_real + offsets4, row0_a_real);
            _mm256_store_pd(out_real + offsets5, row1_a_real);
            _mm256_store_pd(out_real + offsets6, row2_a_real);
            _mm256_store_pd(out_real + offsets7, row3_a_real);

            _mm256_store_pd(out_imag + offsets4, row0_a_imag);
            _mm256_store_pd(out_imag + offsets5, row1_a_imag);
            _mm256_store_pd(out_imag + offsets6, row2_a_imag);
            _mm256_store_pd(out_imag + offsets7, row3_a_imag);

            _mm256_store_pd(out_real + offsets0, row0_b_real);
            _mm256_store_pd(out_real + offsets1, row1_b_real);
            _mm256_store_pd(out_real + offsets2, row2_b_real);
            _mm256_store_pd(out_real + offsets3, row3_b_real);

            _mm256_store_pd(out_imag + offsets0, row0_b_imag);
            _mm256_store_pd(out_imag + offsets1, row1_b_imag);
            _mm256_store_pd(out_imag + offsets2, row2_b_imag);
            _mm256_store_pd(out_imag + offsets3, row3_b_imag);
        }

        static inline void interleave2(
            operand &out_real_a, operand &out_imag_a,
            operand &out_real_b, operand &out_imag_b,
            operand in_real_a, operand in_imag_a,
            operand in_real_b, operand in_imag_b)
        {
            // ymm0, ymm1, ymm2, ymm3 are inputs
            __m256d ymm0 = in_real_a;
            __m256d ymm1 = in_imag_a;
            __m256d ymm2 = in_real_b;
            __m256d ymm3 = in_imag_b;

            // vpermpd ymm4, ymm0, 68  ; 0b01000100
            __m256d ymm4 = _mm256_permute4x64_pd(ymm0, 0x44);

            // vpermpd ymm5, ymm2, 68
            __m256d ymm5 = _mm256_permute4x64_pd(ymm2, 0x44);

            // vpermpd ymm0, ymm0, 238 ; 0b11101110
            ymm0 = _mm256_permute4x64_pd(ymm0, 0xEE);

            // vpermpd ymm2, ymm2, 238
            ymm2 = _mm256_permute4x64_pd(ymm2, 0xEE);

            // vshufpd ymm4, ymm4, ymm5, 12 ; 0b1100
            out_real_a = _mm256_shuffle_pd(ymm4, ymm5, 0xC);

            // vshufpd ymm0, ymm0, ymm2, 12
            out_real_b = _mm256_shuffle_pd(ymm0, ymm2, 0xC);

            // vpermpd ymm2, ymm3, 68
            ymm2 = _mm256_permute4x64_pd(ymm3, 0x44);

            // vpermpd ymm3, ymm3, 238
            ymm3 = _mm256_permute4x64_pd(ymm3, 0xEE);

            // vpermpd ymm0, ymm1, 68
            ymm0 = _mm256_permute4x64_pd(ymm1, 0x44);

            // vpermpd ymm1, ymm1, 238
            ymm1 = _mm256_permute4x64_pd(ymm1, 0xEE);

            // vshufpd ymm0, ymm0, ymm2, 12
            out_imag_a = _mm256_shuffle_pd(ymm0, ymm2, 0xC);

            // vshufpd ymm1, ymm1, ymm3, 12
            out_imag_b = _mm256_shuffle_pd(ymm1, ymm3, 0xC);
        }

        static inline void deinterleave2(
            operand &out_real_a, operand &out_imag_a,
            operand &out_real_b, operand &out_imag_b,
            operand in_real_a, operand in_imag_a,
            operand in_real_b, operand in_imag_b)
        {
            // ymm0, ymm1, ymm2, ymm3 are inputs
            __m256d ymm0 = in_real_a;
            __m256d ymm1 = in_imag_a;
            __m256d ymm2 = in_real_b;
            __m256d ymm3 = in_imag_b;

            // vperm2f128 ymm4, ymm0, ymm2, 49
            __m256d ymm4 = _mm256_permute2f128_pd(ymm0, ymm2, 49);

            // vinsertf128 ymm0, ymm0, xmm2, 1
            __m128d xmm2 = _mm256_extractf128_pd(ymm2, 0);
            ymm0 = _mm256_insertf128_pd(ymm0, xmm2, 1);

            // vunpcklpd ymm2, ymm0, ymm4
            out_real_a = _mm256_unpacklo_pd(ymm0, ymm4);

            // vunpckhpd ymm0, ymm0, ymm4
            out_real_b = _mm256_unpackhi_pd(ymm0, ymm4);

            // vperm2f128 ymm0, ymm1, ymm3, 49
            ymm0 = _mm256_permute2f128_pd(ymm1, ymm3, 49);

            // vinsertf128 ymm1, ymm1, xmm3, 1
            __m128d xmm3 = _mm256_extractf128_pd(ymm3, 0);
            ymm1 = _mm256_insertf128_pd(ymm1, xmm3, 1);

            // vunpcklpd ymm2, ymm1, ymm0
            out_imag_a = _mm256_unpacklo_pd(ymm1, ymm0);

            // vunpckhpd ymm0, ymm1, ymm0
            out_imag_b = _mm256_unpackhi_pd(ymm1, ymm0);
        }

        static inline void deinterleave4(
            operand &out_real_a, operand &out_imag_a,
            operand &out_real_b, operand &out_imag_b,
            operand &out_real_c, operand &out_imag_c,
            operand &out_real_d, operand &out_imag_d,
            operand in_real_a, operand in_imag_a,
            operand in_real_b, operand in_imag_b,
            operand in_real_c, operand in_imag_c,
            operand in_real_d, operand in_imag_d)
        {
            transpose4x4(
                out_real_a, out_imag_a,
                out_real_b, out_imag_b,
                out_real_c, out_imag_c,
                out_real_d, out_imag_d,
                in_real_a, in_imag_a,
                in_real_b, in_imag_b,
                in_real_c, in_imag_c,
                in_real_d, in_imag_d
            );
        }

        static inline void interleave4(
            operand &out_real_a, operand &out_imag_a,
            operand &out_real_b, operand &out_imag_b,
            operand &out_real_c, operand &out_imag_c,
            operand &out_real_d, operand &out_imag_d,
            operand in_real_a, operand in_imag_a,
            operand in_real_b, operand in_imag_b,
            operand in_real_c, operand in_imag_c,
            operand in_real_d, operand in_imag_d)
        {
            transpose4x4(
                out_real_a, out_imag_a,
                out_real_b, out_imag_b,
                out_real_c, out_imag_c,
                out_real_d, out_imag_d,
                in_real_a, in_imag_a,
                in_real_b, in_imag_b,
                in_real_c, in_imag_c,
                in_real_d, in_imag_d
            );
        }

        static inline void transpose4x4(
            operand &out_real_a, operand &out_imag_a,
            operand &out_real_b, operand &out_imag_b,
            operand &out_real_c, operand &out_imag_c,
            operand &out_real_d, operand &out_imag_d,
            operand in_real_a, operand in_imag_a,
            operand in_real_b, operand in_imag_b,
            operand in_real_c, operand in_imag_c,
            operand in_real_d, operand in_imag_d)
        {
            __m256d row0_real, row1_real, row2_real, row3_real;
            __m256d tmp3_real, tmp2_real, tmp1_real, tmp0_real;
            __m256d row0_imag, row1_imag, row2_imag, row3_imag;
            __m256d tmp3_imag, tmp2_imag, tmp1_imag, tmp0_imag;

            row0_real = in_real_a;
            row1_real = in_real_b;
            row2_real = in_real_c;
            row3_real = in_real_d;
            row0_imag = in_imag_a;
            row1_imag = in_imag_b;
            row2_imag = in_imag_c;
            row3_imag = in_imag_d;

            tmp0_real = _mm256_shuffle_pd(row0_real, row1_real, 0x0);
            tmp2_real = _mm256_shuffle_pd(row0_real, row1_real, 0xF);
            tmp1_real = _mm256_shuffle_pd(row2_real, row3_real, 0x0);
            tmp3_real = _mm256_shuffle_pd(row2_real, row3_real, 0xF);
            tmp0_imag = _mm256_shuffle_pd(row0_imag, row1_imag, 0x0);
            tmp2_imag = _mm256_shuffle_pd(row0_imag, row1_imag, 0xF);
            tmp1_imag = _mm256_shuffle_pd(row2_imag, row3_imag, 0x0);
            tmp3_imag = _mm256_shuffle_pd(row2_imag, row3_imag, 0xF);

            out_real_a = _mm256_permute2f128_pd(tmp0_real, tmp1_real, 0x20);
            out_real_b = _mm256_permute2f128_pd(tmp2_real, tmp3_real, 0x20);
            out_real_c = _mm256_permute2f128_pd(tmp0_real, tmp1_real, 0x31);
            out_real_d = _mm256_permute2f128_pd(tmp2_real, tmp3_real, 0x31);
            out_imag_a = _mm256_permute2f128_pd(tmp0_imag, tmp1_imag, 0x20);
            out_imag_b = _mm256_permute2f128_pd(tmp2_imag, tmp3_imag, 0x20);
            out_imag_c = _mm256_permute2f128_pd(tmp0_imag, tmp1_imag, 0x31);
            out_imag_d = _mm256_permute2f128_pd(tmp2_imag, tmp3_imag, 0x31);
        }
    };
}

#endif