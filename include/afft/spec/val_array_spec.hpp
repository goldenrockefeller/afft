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
        static constexpr std::size_t prefetch_lookahead = 16;
        static constexpr std::size_t min_partition_len = 256;

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

        static inline void prefetch(const sample *ptr){
            // Do Nothing
        }

        static inline void prefetch(const std::size_t *ptr){
            _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T0);
        }

        template <std::size_t LogInterleaveFactor>
        static inline void interleave2(
            operand &out_a,
            operand &out_b,
            operand in_a,
            operand in_b)
        {
            const std::size_t interleave_factor = 1 << LogInterleaveFactor;

            // First half interleaving into out_a
            for (std::size_t i = 0; i < Size / interleave_factor / 2; ++i)
            {
                for (std::size_t j = 0; j < interleave_factor; ++j)
                {
                    out_a[interleave_factor * i * 2 + j] = in_a[i * interleave_factor + j];
                }
                for (std::size_t j = 0; j < interleave_factor; ++j)
                {
                    out_a[interleave_factor * i * 2 + j + interleave_factor] = in_b[i * interleave_factor + j];
                }
            }

            // Second half interleaving into out_b
            for (std::size_t i = 0; i < Size / interleave_factor / 2; ++i)
            {
                for (std::size_t j = 0; j < interleave_factor; ++j)
                {
                    out_b[interleave_factor * i * 2 + j] = in_a[Size / 2 + i * interleave_factor + j];
                }
                for (std::size_t j = 0; j < interleave_factor; ++j)
                {
                    out_b[interleave_factor * i * 2 + j + interleave_factor] = in_b[Size / 2 + i * interleave_factor + j];
                }
            }
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
            
            const std::size_t interleave_factor = 1 << LogInterleaveFactor;

            if (Size == 2)
            {
                out_a[0] = in_a[0];
                out_a[1] = in_b[0];
                out_b[0] = in_c[0];
                out_b[1] = in_d[0];
                out_c[0] = in_a[1];
                out_c[1] = in_b[1];
                out_d[0] = in_c[1];
                out_d[1] = in_d[1];
                return;
            }

            // First quarter -> out_a
            for (std::size_t i = 0; i < Size / interleave_factor / 4; ++i)
            {
                for (std::size_t j = 0; j < interleave_factor; ++j)
                {
                    out_a[interleave_factor * i * 4 + j] = in_a[i * interleave_factor + j];
                    out_a[interleave_factor * i * 4 + j + interleave_factor] = in_b[i * interleave_factor + j];
                    out_a[interleave_factor * i * 4 + j + 2 * interleave_factor] = in_c[i * interleave_factor + j];
                    out_a[interleave_factor * i * 4 + j + 3 * interleave_factor] = in_d[i * interleave_factor + j];
                }
            }

            // Second quarter -> out_b
            std::size_t offset = Size / 4;
            for (std::size_t i = 0; i < Size / interleave_factor / 4; ++i)
            {
                for (std::size_t j = 0; j < interleave_factor; ++j)
                {
                    out_b[interleave_factor * i * 4 + j] = in_a[offset + i * interleave_factor + j];
                    out_b[interleave_factor * i * 4 + j + interleave_factor] = in_b[offset + i * interleave_factor + j];
                    out_b[interleave_factor * i * 4 + j + 2 * interleave_factor] = in_c[offset + i * interleave_factor + j];
                    out_b[interleave_factor * i * 4 + j + 3 * interleave_factor] = in_d[offset + i * interleave_factor + j];
                }
            }

            // Third quarter -> out_c
            offset = Size / 2;
            for (std::size_t i = 0; i < Size / interleave_factor / 4; ++i)
            {
                for (std::size_t j = 0; j < interleave_factor; ++j)
                {
                    out_c[interleave_factor * i * 4 + j] = in_a[offset + i * interleave_factor + j];
                    out_c[interleave_factor * i * 4 + j + interleave_factor] = in_b[offset + i * interleave_factor + j];
                    out_c[interleave_factor * i * 4 + j + 2 * interleave_factor] = in_c[offset + i * interleave_factor + j];
                    out_c[interleave_factor * i * 4 + j + 3 * interleave_factor] = in_d[offset + i * interleave_factor + j];
                }
            }

            // Fourth quarter -> out_d
            offset = 3 * Size / 4;
            for (std::size_t i = 0; i < Size / interleave_factor / 4; ++i)
            {
                for (std::size_t j = 0; j < interleave_factor; ++j)
                {
                    out_d[interleave_factor * i * 4 + j] = in_a[offset + i * interleave_factor + j];
                    out_d[interleave_factor * i * 4 + j + interleave_factor] = in_b[offset + i * interleave_factor + j];
                    out_d[interleave_factor * i * 4 + j + 2 * interleave_factor] = in_c[offset + i * interleave_factor + j];
                    out_d[interleave_factor * i * 4 + j + 3 * interleave_factor] = in_d[offset + i * interleave_factor + j];
                }
            }
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