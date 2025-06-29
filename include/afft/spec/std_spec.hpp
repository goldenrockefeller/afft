#ifndef AFFT_STD_SPEC_HPP
#define AFFT_STD_SPEC_HPP

#include <cstddef>
#include <cmath>

namespace afft
{
    template <typename Sample>
    struct StdSpec
    {
        using fallback_spec = std::nullptr_t;
        using sample = Sample;
        using operand = Sample;
        static constexpr std::size_t n_samples_per_operand = 1;
        static constexpr std::size_t prefetch_lookahead = 4;
        static constexpr std::size_t min_partition_len = 64;

        static inline void load(operand &x, const sample *ptr)
        {
            x = *ptr;
        }

        static inline void store(double *ptr, const operand &x)
        {
            *ptr = x;
        }

        static inline void prefetch(const sample *ptr){
            // Do Nothing
        }

        static inline void prefetch(const std::size_t *ptr){
            _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T0);
        }

        static inline sample sin(const sample &x) {
            return std::sin(x);
        }

        static inline sample cos(const sample &x) {
            return std::cos(x);
        }
        
        static inline sample pi() {
            return sample(3.14159265358979323846L);
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

    };
}

#endif