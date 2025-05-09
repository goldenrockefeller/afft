#ifndef AFFT_STD_SPEC_HPP
#define AFFT_STD_SPEC_HPP

#include <cstddef>

namespace afft{   
    template <typename Sample>    
    struct StdSpec{
        using fallback_spec = std::nullptr_t;
        using sample = Sample;
        using operand = Sample;
        static constexpr std::size_t n_samples_per_operand = 1;

        static inline void load(operand& x, const sample* ptr) {
            x = *ptr;
        }

        static inline void store(double* ptr, const operand& x) {
            *ptr = x;
        }

        static inline void transpose_diagonal(sample *out_real, sample *out_imag, const sample *in_real, const sample *in_imag, const std::size_t *offsets) {
            std::size_t offsets0 = offsets[0];
            auto tmp_real = *(in_real + offsets0);
            auto tmp_imag = *(in_imag + offsets0);
            *(out_real + offsets0) = tmp_real;
            *(out_imag + offsets0) = tmp_imag;
        }

        static inline void transpose_off_diagonal(sample *out_real, sample *out_imag, const sample *in_real, const sample *in_imag, const std::size_t *offsets) {
            std::size_t offsets0 = offsets[0];
            std::size_t offsets1 = offsets[1];

            auto tmp0_real = *(in_real + offsets0);
            auto tmp0_imag = *(in_imag + offsets0);
            auto tmp1_real = *(in_real + offsets1);
            auto tmp1_imag = *(in_imag + offsets1);

            *(out_real + offsets1) = tmp0_real;
            *(out_imag + offsets1) = tmp0_imag;
            *(out_real + offsets0) = tmp1_real;
            *(out_imag + offsets0) = tmp1_imag;
        }
    };
}

#endif