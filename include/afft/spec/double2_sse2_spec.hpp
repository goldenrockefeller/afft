#ifndef AFFT_DOUBLE2_SSE2_SPEC_HPP
#define AFFT_DOUBLE2_SSE2_SPEC_HPP

#include <cstddef>

#include "xsimd/xsimd.hpp"
#include "afft/spec/std_spec.hpp"

namespace afft{        
    struct Double2Sse2Spec{
        using sample = double;
        using arch = typename xsimd::sse2;
        using operand = xsimd::batch<sample, arch>;
        using fallback_spec = StdSpec<sample>;
        static constexpr std::size_t n_samples_per_operand = 2;

        static inline void load(operand& x, const sample* ptr) {
            x = xsimd::batch<sample, arch>::load_unaligned(ptr);
        }

        static inline void store(sample* ptr, const operand& x) {
            x.store_unaligned(ptr);
        }

        static inline void transpose_diagonal(sample *out_real, sample *out_imag, const sample *in_real, const sample *in_imag, const std::size_t *offsets) {
            operand x_real, y_real, u_real, v_real;
            operand x_imag, y_imag, u_imag, v_imag;

            std::size_t offsets0 = offsets[0];
            std::size_t offsets1 = offsets[1];

            load(u_real, in_real + offsets0);
            load(v_real, in_real + offsets1);
            load(u_imag, in_imag + offsets0);
            load(v_imag, in_imag + offsets1);

            x_real = xsimd::zip_lo(u_real, v_real);
            y_real = xsimd::zip_hi(u_real, v_real);
            x_imag = xsimd::zip_lo(u_imag, v_imag);
            y_imag = xsimd::zip_hi(u_imag, v_imag);

            store(out_real + offsets0, x_real);
            store(out_real + offsets1, y_real);
            store(out_imag + offsets0, x_imag);
            store(out_imag + offsets1, y_imag);
        }

        static inline void transpose_off_diagonal(sample *out_real, sample *out_imag, const sample *in_real, const sample *in_imag, const std::size_t *offsets) {
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

            x_a_real = xsimd::zip_lo(u_a_real, v_a_real);
            y_a_real = xsimd::zip_hi(u_a_real, v_a_real);
            x_b_real = xsimd::zip_lo(u_b_real, v_b_real);
            y_b_real = xsimd::zip_hi(u_b_real, v_b_real);
            x_a_imag = xsimd::zip_lo(u_a_imag, v_a_imag);
            y_a_imag = xsimd::zip_hi(u_a_imag, v_a_imag);
            x_b_imag = xsimd::zip_lo(u_b_imag, v_b_imag);
            y_b_imag = xsimd::zip_hi(u_b_imag, v_b_imag);

            store(out_real + offsets2, x_a_imag);
            store(out_real + offsets3, y_a_imag);
            store(out_real + offsets0, x_b_imag);
            store(out_real + offsets1, y_b_imag);
            store(out_imag + offsets2, x_a_imag);
            store(out_imag + offsets3, y_a_imag);
            store(out_imag + offsets0, x_b_imag);
            store(out_imag + offsets1, y_b_imag);
        }
    };
}

#endif