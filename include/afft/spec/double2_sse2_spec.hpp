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

        static inline void transpose2x2(
            operand &out_real_a, operand &out_imag_a,
            operand &out_real_b, operand &out_imag_b,
            operand in_real_a, operand in_imag_a,
            operand in_real_b, operand in_imag_b)
        {
            auto u_real = in_real_a;
            auto v_real = in_real_b;
            auto u_imag = in_imag_a;
            auto v_imag = in_imag_b;

            out_real_a = xsimd::zip_lo(u_real, v_real);
            out_real_b = xsimd::zip_hi(u_real, v_real);
            out_imag_a = xsimd::zip_lo(u_imag, v_imag);
            out_imag_b = xsimd::zip_hi(u_imag, v_imag);
        }

        static inline void transpose_diagonal(sample *out_real, sample *out_imag, const sample *in_real, const sample *in_imag, const std::size_t *offsets)
        {
            operand x_real, y_real, u_real, v_real;
            operand x_imag, y_imag, u_imag, v_imag;

            std::size_t offsets0 = offsets[0];
            std::size_t offsets1 = offsets[1];

            load(u_real, in_real + offsets0);
            load(v_real, in_real + offsets1);
            load(u_imag, in_imag + offsets0);
            load(v_imag, in_imag + offsets1);

            transpose2x2(
                x_real, x_imag,
                y_real, y_imag,
                u_real, u_imag,
                v_real, v_imag);

            store(out_real + offsets0, x_real);
            store(out_real + offsets1, y_real);
            store(out_imag + offsets0, x_imag);
            store(out_imag + offsets1, y_imag);
        }

        static inline void transpose_off_diagonal(sample *out_real, sample *out_imag, const sample *in_real, const sample *in_imag, const std::size_t *offsets)
        {
            operand x_a_real, y_a_real, u_a_real, v_a_real;
            operand x_b_real, y_b_real, u_b_real, v_b_real;
            operand x_a_imag, y_a_imag, u_a_imag, v_a_imag;
            operand x_b_imag, y_b_imag, u_b_imag, v_b_imag;

            std::size_t offsets0 = offsets[0];
            std::size_t offsets1 = offsets[1];
            std::size_t offsets2 = offsets[2];
            std::size_t offsets3 = offsets[3];

            load(u_a_real, in_real + offsets0);
            load(v_a_real, in_real + offsets1);
            load(u_b_real, in_real + offsets2);
            load(v_b_real, in_real + offsets3);
            load(u_a_imag, in_imag + offsets0);
            load(v_a_imag, in_imag + offsets1);
            load(u_b_imag, in_imag + offsets2);
            load(v_b_imag, in_imag + offsets3);

            transpose2x2(
                x_a_real, x_a_imag,
                y_a_real, y_a_imag,
                u_a_real, u_a_imag,
                v_a_real, v_a_imag);

            transpose2x2(
                x_b_real, x_b_imag,
                y_b_real, y_b_imag,
                u_b_real, u_b_imag,
                v_b_real, v_b_imag);

            store(out_real + offsets2, x_a_imag);
            store(out_real + offsets3, y_a_imag);
            store(out_real + offsets0, x_b_imag);
            store(out_real + offsets1, y_b_imag);
            store(out_imag + offsets2, x_a_imag);
            store(out_imag + offsets3, y_a_imag);
            store(out_imag + offsets0, x_b_imag);
            store(out_imag + offsets1, y_b_imag);
        }

        static inline void interleave2(
            operand &out_real_a, operand &out_imag_a,
            operand &out_real_b, operand &out_imag_b,
            operand in_real_a, operand in_imag_a,
            operand in_real_b, operand in_imag_b)
        {
            transpose2x2(
                out_real_a, out_imag_a,
                out_real_b, out_imag_a,
                in_real_a, in_imag_a,
                in_real_b, in_imag_b);
        }

        static inline void deinterleave2(
            operand &out_real_a, operand &out_imag_a,
            operand &out_real_b, operand &out_imag_b,
            operand in_real_a, operand in_imag_a,
            operand in_real_b, operand in_imag_b)
        {
            transpose2x2(
                out_real_a, out_imag_a,
                out_real_b, out_imag_a,
                in_real_a, in_imag_a,
                in_real_b, in_imag_b);
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
            out_real_a = xsimd::zip_lo(in_real_a, in_real_c);
            out_real_b = xsimd::zip_hi(in_real_a, in_real_c);
            out_real_c = xsimd::zip_lo(in_real_b, in_real_d);
            out_real_d = xsimd::zip_hi(in_real_b, in_real_d);

            out_imag_a = xsimd::zip_lo(in_imag_a, in_imag_c);
            out_imag_b = xsimd::zip_hi(in_imag_a, in_imag_c);
            out_imag_c = xsimd::zip_lo(in_imag_b, in_imag_d);
            out_imag_d = xsimd::zip_hi(in_imag_b, in_imag_d);
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
            out_real_a = xsimd::zip_lo(in_real_a, in_real_b);
            out_real_b = xsimd::zip_lo(in_real_c, in_real_d);
            out_real_c = xsimd::zip_hi(in_real_a, in_real_b);
            out_real_d = xsimd::zip_hi(in_real_c, in_real_d);

            out_imag_a = xsimd::zip_lo(in_imag_a, in_imag_b);
            out_imag_b = xsimd::zip_lo(in_imag_c, in_imag_d);
            out_imag_c = xsimd::zip_hi(in_imag_a, in_imag_b);
            out_imag_d = xsimd::zip_hi(in_imag_c, in_real_d);
        }
    };
}

#endif