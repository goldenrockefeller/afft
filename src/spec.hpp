#ifndef GOLDENROCEKEFELLER_AFFT_SPEC
#define GOLDENROCEKEFELLER_AFFT_SPEC

#include <cmath>

#include "xsimd/xsimd.hpp"

namespace goldenrockefeller{ namespace afft{

    template <typename ValueT>
    struct StdSpec{
        using Value = ValueT;

        static inline Value Cos(const Value& x) {
            return std::cos(x);
        }

        static inline Value Sin(const Value& x) {
            return std::sin(x);
        }

        static inline Value Pi() {
            return Value(3.14159265358979323846264338327950288L);
        }

        template <typename SampleValue>
        static inline void Load(const SampleValue* t, Value& x) {
            x = *t;
        }

        static inline void Store(Value* t, const Value& x) {
            *t = x;
        }
    };

    struct Double4Spec{
        using Value = xsimd::batch<double, xsimd::avx>;

        static inline void Load(const double* ptr, Value& x) {
            x = xsimd::batch<double, xsimd::avx>::load_unaligned(ptr);
        }

        static inline void Store(double* ptr, const Value& x) {
            x.store_unaligned(ptr);
        }
    };
}}



#endif