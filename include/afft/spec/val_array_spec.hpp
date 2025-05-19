#ifndef AFFT_VAL_ARRAY_SPEC_HPP
#define AFFT_VAL_ARRAY_SPEC_HPP

#include <cstddef>
#include <valarray>
#include "afft/spec/std_spec.hpp"

namespace afft
{
    template <std::size_t Size>
    struct ValArrayOperand : public std::valarray<double>
    {
        template <typename T>
        explicit ValArrayOperand(T x) : std::valarray<double>(x, Size) {}

        ValArrayOperand() : ValArrayOperand(double{}) {}

        ~ValArrayOperand() = default;

        ValArrayOperand(const valarray &other) noexcept : std::valarray<double>(other) {}
        ValArrayOperand(const ValArrayOperand &other) noexcept : std::valarray<double>(other) {}

        ValArrayOperand(valarray &&other) noexcept : std::valarray<double>(other) {}
        ValArrayOperand(ValArrayOperand &&other) noexcept : std::valarray<double>(other) {}

        ValArrayOperand &operator=(const ValArrayOperand &other)
        {
            if (this != &other)
            {
                std::valarray<double>::operator=(other);
            }
            return *this;
        }

        // Copy assignment from std::valarray<double>
        ValArrayOperand &operator=(const std::valarray<double> &other)
        {
            std::valarray<double>::operator=(other);
            return *this;
        }

        // Move assignment operator (optional, for completeness)
        ValArrayOperand &operator=(ValArrayOperand &&other) noexcept
        {
            std::valarray<double>::operator=(std::move(other));
            return *this;
        }
    };

    template <std::size_t Size>
    struct ValArraySpec
    {
        using fallback_spec = ValArraySpec<Size / 2>;
        using sample = double;
        using operand = ValArrayOperand<Size>;
        static constexpr std::size_t n_samples_per_operand = Size;

        static inline void load(operand &x, const sample *ptr)
        {
            for (size_t i = 0; i < Size; i++)
            {
                x[i] = *(ptr + i);
            }
        }

        static inline void store(double *ptr, const operand &x)
        {
            for (size_t i = 0; i < Size; i++)
            {
                *(ptr + i) = x[i];
            }
        }

        static inline void transpose_diagonal(sample *out_real, sample *out_imag, const sample *in_real, const sample *in_imag, const std::size_t *offsets)
        {
            for (std::size_t i = 0; i < Size; ++i)
            {
                std::size_t row_i = offsets[i];
                for (std::size_t j = i + 1; j < Size; ++j)
                {
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

        static inline void transpose_off_diagonal(sample *out_real, sample *out_imag, const sample *in_real, const sample *in_imag, const std::size_t *offsets)
        {
            for (std::size_t i = 0; i < Size; ++i)
            {
                std::size_t row_i_a = offsets[i];
                std::size_t row_i_b = offsets[i + Size];
                for (std::size_t j = i + 1; j < Size; ++j)
                {
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

            for (std::size_t i = 0; i < Size; ++i)
            {
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

        static inline void interleave2(
            operand &out_real_a, operand &out_imag_a,
            operand &out_real_b, operand &out_imag_b,
            const operand &in_real_a, const operand &in_imag_a,
            const operand &in_real_b, const operand &in_imag_b)
        {
            std::size_t in_select_id = 0;
            std::size_t j = 0;

            operand tmp_out_real_a = out_real_a;
            operand tmp_out_imag_a = out_imag_a;
            operand tmp_out_real_b = out_real_b;
            operand tmp_out_imag_b = out_imag_b;

            for (std::size_t out_select_id = 0; out_select_id < 2; out_select_id++)
            {
                operand *out_real_ptr;
                operand *out_imag_ptr;

                switch (out_select_id)
                {
                case 0:
                    out_real_ptr = &tmp_out_real_a;
                    out_imag_ptr = &tmp_out_imag_a;
                    break;
                case 1:
                    out_real_ptr = &tmp_out_real_b;
                    out_imag_ptr = &tmp_out_imag_b;
                    break;
                }

                auto &out_real = *out_real_ptr;
                auto &out_imag = *out_imag_ptr;

                for (std::size_t i = 0; i < Size; i++)
                {
                    switch (in_select_id)
                    {
                    case 0:
                        out_real[i] = in_real_a[j];
                        out_imag[i] = in_imag_a[j];
                        in_select_id++;
                        break;
                    case 1:
                        out_real[i] = in_real_b[j];
                        out_imag[i] = in_imag_b[j];
                        in_select_id = 0;
                        j++;
                        break;
                    }
                }
            }

            out_real_a = tmp_out_real_a;
            out_imag_a = tmp_out_imag_a;
            out_real_b = tmp_out_real_b;
            out_imag_b = tmp_out_imag_b;
        }

        static inline void deinterleave2(
            operand &out_real_a, operand &out_imag_a,
            operand &out_real_b, operand &out_imag_b,
            const operand &in_real_a, const operand &in_imag_a,
            const operand &in_real_b, const operand &in_imag_b)
        {
            operand tmp_out_real_a = out_real_a;
            operand tmp_out_imag_a = out_imag_a;
            operand tmp_out_real_b = out_real_b;
            operand tmp_out_imag_b = out_imag_b;

            std::size_t out_index = 0;

            for (std::size_t in_select_id = 0; in_select_id < 2; in_select_id++)
            {
                const operand *in_real_ptr;
                const operand *in_imag_ptr;

                switch (in_select_id)
                {
                case 0:
                    in_real_ptr = &in_real_a;
                    in_imag_ptr = &in_imag_a;
                    break;
                case 1:
                    in_real_ptr = &in_real_b;
                    in_imag_ptr = &in_imag_b;
                    break;
                }

                const auto &in_real = *in_real_ptr;
                const auto &in_imag = *in_imag_ptr;

                for (std::size_t i = 0; i < Size; i++)
                {
                    if (i % 2 == 0)
                    {
                        tmp_out_real_a[out_index] = in_real[i];
                        tmp_out_imag_a[out_index] = in_imag[i];
                    }
                    else
                    {
                        tmp_out_real_b[out_index] = in_real[i];
                        tmp_out_imag_b[out_index] = in_imag[i];
                        out_index++;
                    }
                }
            }

            out_real_a = tmp_out_real_a;
            out_imag_a = tmp_out_imag_a;
            out_real_b = tmp_out_real_b;
            out_imag_b = tmp_out_imag_b;
        }

        static inline void interleave4(
            operand &out_real_a, operand &out_imag_a,
            operand &out_real_b, operand &out_imag_b,
            operand &out_real_c, operand &out_imag_c,
            operand &out_real_d, operand &out_imag_d,
            const operand &in_real_a, const operand &in_imag_a,
            const operand &in_real_b, const operand &in_imag_b,
            const operand &in_real_c, const operand &in_imag_c,
            const operand &in_real_d, const operand &in_imag_d)
        {
            operand tmp_out_real_a = out_real_a;
            operand tmp_out_imag_a = out_imag_a;
            operand tmp_out_real_b = out_real_b;
            operand tmp_out_imag_b = out_imag_b;
            operand tmp_out_real_c = out_real_c;
            operand tmp_out_imag_c = out_imag_c;
            operand tmp_out_real_d = out_real_d;
            operand tmp_out_imag_d = out_imag_d;

            std::size_t in_select_id = 0;
            std::size_t j = 0;
            for (std::size_t out_select_id = 0; out_select_id < 4; out_select_id++)
            {
                operand *out_real_ptr;
                operand *out_imag_ptr;

                switch (out_select_id)
                {
                case 0:
                    out_real_ptr = &tmp_out_real_a;
                    out_imag_ptr = &tmp_out_imag_a;
                    break;
                case 1:
                    out_real_ptr = &tmp_out_real_b;
                    out_imag_ptr = &tmp_out_imag_b;
                    break;
                case 2:
                    out_real_ptr = &tmp_out_real_c;
                    out_imag_ptr = &tmp_out_imag_c;
                    break;
                case 3:
                    out_real_ptr = &tmp_out_real_d;
                    out_imag_ptr = &tmp_out_imag_d;
                    break;
                }

                auto &out_real = *out_real_ptr;
                auto &out_imag = *out_imag_ptr;

                for (std::size_t i = 0; i < Size; i++)
                {
                    switch (in_select_id)
                    {
                    case 0:
                        out_real[i] = in_real_a[j];
                        out_imag[i] = in_imag_a[j];
                        in_select_id++;
                        break;
                    case 1:
                        out_real[i] = in_real_b[j];
                        out_imag[i] = in_imag_b[j];
                        in_select_id++;
                        break;
                    case 2:
                        out_real[i] = in_real_c[j];
                        out_imag[i] = in_imag_c[j];
                        in_select_id++;
                        break;
                    case 3:
                        out_real[i] = in_real_d[j];
                        out_imag[i] = in_imag_d[j];
                        in_select_id = 0;
                        j++;
                        break;
                    }
                }
            }

            out_real_a = tmp_out_real_a;
            out_imag_a = tmp_out_imag_a;
            out_real_b = tmp_out_real_b;
            out_imag_b = tmp_out_imag_b;
            out_real_c = tmp_out_real_c;
            out_imag_c = tmp_out_imag_c;
            out_real_d = tmp_out_real_d;
            out_imag_d = tmp_out_imag_d;
        }

        static inline void deinterleave4(
            operand &out_real_a, operand &out_imag_a,
            operand &out_real_b, operand &out_imag_b,
            operand &out_real_c, operand &out_imag_c,
            operand &out_real_d, operand &out_imag_d,
            const operand &in_real_a, const operand &in_imag_a,
            const operand &in_real_b, const operand &in_imag_b,
            const operand &in_real_c, const operand &in_imag_c,
            const operand &in_real_d, const operand &in_imag_d)
        {
            
            operand tmp_out_real_a = out_real_a;
            operand tmp_out_imag_a = out_imag_a;
            operand tmp_out_real_b = out_real_b;
            operand tmp_out_imag_b = out_imag_b;
            operand tmp_out_real_c = out_real_c;
            operand tmp_out_imag_c = out_imag_c;
            operand tmp_out_real_d = out_real_d;
            operand tmp_out_imag_d = out_imag_d;

            std::size_t out_select_id = 0;
            std::size_t j = 0;

            for (std::size_t in_select_id = 0; in_select_id < 4; in_select_id++)
            {
                
                const operand *in_real_ptr;
                const operand *in_imag_ptr;

                switch (in_select_id)
                {
                case 0:
                    in_real_ptr = &in_real_a;
                    in_imag_ptr = &in_imag_a;
                    break;
                case 1:
                    in_real_ptr = &in_real_b;
                    in_imag_ptr = &in_imag_b;
                    break;
                case 2:
                    in_real_ptr = &in_real_c;
                    in_imag_ptr = &in_imag_c;
                    break;
                case 3:
                    in_real_ptr = &in_real_d;
                    in_imag_ptr = &in_imag_d;
                    break;
                }

                const operand &in_real = *in_real_ptr;
                const operand &in_imag = *in_imag_ptr;

                for (std::size_t i = 0; i < Size; i++)
                {
                    switch (out_select_id)
                    {
                    case 0:
                        tmp_out_real_a[j] = in_real[i];
                        tmp_out_imag_a[j] = in_imag[i];
                        out_select_id++;
                        break;
                    case 1:
                        tmp_out_real_b[j] = in_real[i];
                        tmp_out_imag_b[j] = in_imag[i];
                        out_select_id++;
                        break;
                    case 2:
                        tmp_out_real_c[j] = in_real[i];
                        tmp_out_imag_c[j] = in_imag[i];
                        out_select_id++;
                        break;
                    case 3:
                        tmp_out_real_d[j] = in_real[i];
                        tmp_out_imag_d[j] = in_imag[i];
                        out_select_id = 0;
                        j++; // advance to next index in destination arrays
                        break;
                    }
                }
            }
            
            out_real_a = tmp_out_real_a;
            out_imag_a = tmp_out_imag_a;
            out_real_b = tmp_out_real_b;
            out_imag_b = tmp_out_imag_b;
            out_real_c = tmp_out_real_c;
            out_imag_c = tmp_out_imag_c;
            out_real_d = tmp_out_real_d;
            out_imag_d = tmp_out_imag_d;
        }
    };

    template <>
    struct ValArraySpec<1> : StdSpec<double>
    {
        using fallback_spec = std::nullptr_t;
        using sample = double;
        using operand = double;
        static constexpr std::size_t n_samples_per_operand = 1;
    };
}

#endif