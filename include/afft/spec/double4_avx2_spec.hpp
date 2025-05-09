#ifndef AFFT_DOUBLE4_AVX2_SPEC_HPP
#define AFFT_DOUBLE4_AVX2_SPEC_HPP

#include <cstddef>

#include "xsimd/xsimd.hpp"
#include "afft/spec/double2_sse2_spec.hpp"
#include <immintrin.h>

namespace afft {
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

            __m256d row0_real, row1_real, row2_real, row3_real;
            __m256d tmp3_real, tmp2_real, tmp1_real, tmp0_real;
            __m256d row0_imag, row1_imag, row2_imag, row3_imag;
            __m256d tmp3_imag, tmp2_imag, tmp1_imag, tmp0_imag;

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

            
            tmp0_real = _mm256_shuffle_pd(row0_real, row1_real, 0x0);
            tmp2_real = _mm256_shuffle_pd(row0_real, row1_real, 0xF);
            tmp1_real = _mm256_shuffle_pd(row2_real, row3_real, 0x0);
            tmp3_real = _mm256_shuffle_pd(row2_real, row3_real, 0xF);
            tmp0_imag = _mm256_shuffle_pd(row0_imag, row1_imag, 0x0);
            tmp2_imag = _mm256_shuffle_pd(row0_imag, row1_imag, 0xF);
            tmp1_imag = _mm256_shuffle_pd(row2_imag, row3_imag, 0x0);
            tmp3_imag = _mm256_shuffle_pd(row2_imag, row3_imag, 0xF);

            row0_real = _mm256_permute2f128_pd(tmp0_real, tmp1_real, 0x20);
            row1_real = _mm256_permute2f128_pd(tmp2_real, tmp3_real, 0x20);
            row2_real = _mm256_permute2f128_pd(tmp0_real, tmp1_real, 0x31);
            row3_real = _mm256_permute2f128_pd(tmp2_real, tmp3_real, 0x31);
            row0_imag = _mm256_permute2f128_pd(tmp0_imag, tmp1_imag, 0x20);
            row1_imag = _mm256_permute2f128_pd(tmp2_imag, tmp3_imag, 0x20);
            row2_imag = _mm256_permute2f128_pd(tmp0_imag, tmp1_imag, 0x31);
            row3_imag = _mm256_permute2f128_pd(tmp2_imag, tmp3_imag, 0x31);

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

            __m256d row0_a_real, row1_a_real, row2_a_real, row3_a_real;
            __m256d row0_b_real, row1_b_real, row2_b_real, row3_b_real;
            __m256d tmp3_a_real, tmp2_a_real, tmp1_a_real, tmp0_a_real;
            __m256d tmp3_b_real, tmp2_b_real, tmp1_b_real, tmp0_b_real;

            __m256d row0_a_imag, row1_a_imag, row2_a_imag, row3_a_imag;
            __m256d row0_b_imag, row1_b_imag, row2_b_imag, row3_b_imag;
            __m256d tmp3_a_imag, tmp2_a_imag, tmp1_a_imag, tmp0_a_imag;
            __m256d tmp3_b_imag, tmp2_b_imag, tmp1_b_imag, tmp0_b_imag;

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

            tmp0_a_real = _mm256_shuffle_pd(row0_a_real, row1_a_real, 0x0);
            tmp2_a_real = _mm256_shuffle_pd(row0_a_real, row1_a_real, 0xF);
            tmp1_a_real = _mm256_shuffle_pd(row2_a_real, row3_a_real, 0x0);
            tmp3_a_real = _mm256_shuffle_pd(row2_a_real, row3_a_real, 0xF);

            tmp0_a_imag = _mm256_shuffle_pd(row0_a_imag, row1_a_imag, 0x0);
            tmp2_a_imag = _mm256_shuffle_pd(row0_a_imag, row1_a_imag, 0xF);
            tmp1_a_imag = _mm256_shuffle_pd(row2_a_imag, row3_a_imag, 0x0);
            tmp3_a_imag = _mm256_shuffle_pd(row2_a_imag, row3_a_imag, 0xF);

            tmp0_b_real = _mm256_shuffle_pd(row0_b_real, row1_b_real, 0x0);
            tmp2_b_real = _mm256_shuffle_pd(row0_b_real, row1_b_real, 0xF);
            tmp1_b_real = _mm256_shuffle_pd(row2_b_real, row3_b_real, 0x0);
            tmp3_b_real = _mm256_shuffle_pd(row2_b_real, row3_b_real, 0xF);

            tmp0_b_imag = _mm256_shuffle_pd(row0_b_imag, row1_b_imag, 0x0);
            tmp2_b_imag = _mm256_shuffle_pd(row0_b_imag, row1_b_imag, 0xF);
            tmp1_b_imag = _mm256_shuffle_pd(row2_b_imag, row3_b_imag, 0x0);
            tmp3_b_imag = _mm256_shuffle_pd(row2_b_imag, row3_b_imag, 0xF);

            row0_a_real = _mm256_permute2f128_pd(tmp0_a_real, tmp1_a_real, 0x20);
            row1_a_real = _mm256_permute2f128_pd(tmp2_a_real, tmp3_a_real, 0x20);
            row2_a_real = _mm256_permute2f128_pd(tmp0_a_real, tmp1_a_real, 0x31);
            row3_a_real = _mm256_permute2f128_pd(tmp2_a_real, tmp3_a_real, 0x31);

            row0_a_imag = _mm256_permute2f128_pd(tmp0_a_imag, tmp1_a_imag, 0x20);
            row1_a_imag = _mm256_permute2f128_pd(tmp2_a_imag, tmp3_a_imag, 0x20);
            row2_a_imag = _mm256_permute2f128_pd(tmp0_a_imag, tmp1_a_imag, 0x31);
            row3_a_imag = _mm256_permute2f128_pd(tmp2_a_imag, tmp3_a_imag, 0x31);

            row0_b_real = _mm256_permute2f128_pd(tmp0_b_real, tmp1_b_real, 0x20);
            row1_b_real = _mm256_permute2f128_pd(tmp2_b_real, tmp3_b_real, 0x20);
            row2_b_real = _mm256_permute2f128_pd(tmp0_b_real, tmp1_b_real, 0x31);
            row3_b_real = _mm256_permute2f128_pd(tmp2_b_real, tmp3_b_real, 0x31);

            row0_b_imag = _mm256_permute2f128_pd(tmp0_b_imag, tmp1_b_imag, 0x20);
            row1_b_imag = _mm256_permute2f128_pd(tmp2_b_imag, tmp3_b_imag, 0x20);
            row2_b_imag = _mm256_permute2f128_pd(tmp0_b_imag, tmp1_b_imag, 0x31);
            row3_b_imag = _mm256_permute2f128_pd(tmp2_b_imag, tmp3_b_imag, 0x31);

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
    };
}

#endif