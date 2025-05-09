#ifndef AFFT_VAL_ARRAY_SPEC_HPP
#define AFFT_VAL_ARRAY_SPEC_HPP

#include <cstddef>
#include <valarray>
#include "afft/spec/std_spec.hpp"

namespace afft{   
    template <std::size_t Size>
    struct ValArrayOperand : public std::valarray<double> {
        template<typename T>
        explicit ValArrayOperand(T x) : std::valarray<double>(x, Size) {}

        ValArrayOperand() : ValArrayOperand(double{}) {}

        ValArrayOperand(const valarray& other) noexcept : std::valarray<double>(other)  {}

        ValArrayOperand(ValArrayOperand&& other) noexcept : std::valarray<double>(other)  {}
    };

    template <std::size_t Size>    
    struct ValArraySpec{
        using fallback_spec = ValArraySpec<Size/2>;
        using sample = double;
        using operand = ValArrayOperand<Size>;
        static constexpr std::size_t n_samples_per_operand = Size;

        static inline void load(operand& x, const sample* ptr) {
            for (size_t i = 0; i < Size; i++){ 
                x[i] = *(ptr + i);
            }
        }

        static inline void store(double* ptr, const operand& x) {
            for (size_t i = 0; i < Size; i++){ 
                *(ptr + i) = x[i];
            }
        }

        static inline void transpose_diagonal(sample *out_real, sample *out_imag, const sample *in_real, const sample *in_imag, const std::size_t *offsets) {
            for (std::size_t i = 0; i < Size; ++i) {
                std::size_t row_i = offsets[i];
                for (std::size_t j = i + 1; j < Size; ++j) {
                    std::size_t row_j = offsets[j];
                    std::size_t idx_ij = row_i + j;
                    std::size_t idx_ji = row_j + i;

                    auto in_real_idx_ij = in_real[idx_ij];
                    auto in_real_idx_ji = in_real[idx_ji];

                    auto in_imag_idx_ij = in_imag[idx_ij];
                    auto in_imag_idx_ji = in_imag[idx_ji];

                    out_real[idx_ij] = in_real_idx_ji;
                    out_real[idx_ji] = in_real_idx_ij;

                    out_imag[idx_ij] = in_imag_idx_ji;
                    out_imag[idx_ji] = in_imag_idx_ij;
                }
            }
        }

        static inline void transpose_off_diagonal(sample *out_real, sample *out_imag, const sample *in_real, const sample *in_imag, const std::size_t *offsets) {
            for (std::size_t i = 0; i < Size; ++i) {
                std::size_t row_i_a = offsets[i];
                std::size_t row_i_b = offsets[i + Size];
                for (std::size_t j = i + 1; j < Size; ++j) {
                    std::size_t row_j_a = offsets[j];
                    std::size_t row_j_b = offsets[j + Size];
                    std::size_t idx_aij = row_i_a + j;
                    std::size_t idx_bij = row_i_b + j;
                    std::size_t idx_aji = row_j_a + i;
                    std::size_t idx_bji = row_j_b + i;

                    auto in_real_idx_aij = in_real[idx_aij];
                    auto in_real_idx_bij = in_real[idx_bij];
                    auto in_real_idx_aji = in_real[idx_aji];
                    auto in_real_idx_bji = in_real[idx_bji];

                    auto in_imag_idx_aij = in_imag[idx_aij];
                    auto in_imag_idx_bij = in_imag[idx_bij];
                    auto in_imag_idx_aji = in_imag[idx_aji];
                    auto in_imag_idx_bji = in_imag[idx_bji];

                    out_real[idx_aij] = in_real_idx_bji;
                    out_real[idx_bij] = in_real_idx_aji;
                    out_real[idx_bji] = in_real_idx_aij;
                    out_real[idx_aji] = in_real_idx_bij;

                    out_imag[idx_aij] = in_imag_idx_bji;
                    out_imag[idx_bij] = in_imag_idx_aji;
                    out_imag[idx_bji] = in_imag_idx_aij;
                    out_imag[idx_aji] = in_imag_idx_bij;
                }
            }

            for (std::size_t i = 0; i < Size; ++i) {
                std::size_t row_i_a = offsets[i];
                std::size_t row_i_b = offsets[i + Size];
                std::size_t idx_aii = row_i_a + i;
                std::size_t idx_bii = row_i_b + i;

                auto in_real_idx_aii = in_real[idx_aii];
                auto in_real_idx_bii = in_real[idx_bii];

                auto in_imag_idx_aii = in_imag[idx_aii];
                auto in_imag_idx_bii = in_imag[idx_bii];

                out_real[idx_aii] = in_real_idx_bii;
                out_real[idx_bii] = in_real_idx_aii;

                out_imag[idx_aii] = in_imag_idx_bii;
                out_imag[idx_bii] = in_imag_idx_aii;
            }
        }
    };

    template <>   
    struct ValArraySpec<1> : StdSpec<double>{
        // Nothing to add
    };
}

#endif