#ifndef GOLDENROCEKEFELLER_AFFT_SPEC
#define GOLDENROCEKEFELLER_AFFT_SPEC

#include <cmath>

#include "xsimd/xsimd.hpp"

namespace afft{

    template <typename ValueT>
    struct StdSpec{
        using Value = ValueT;

        static inline Value cos(const Value& x) {
            return std::cos(x);
        }

        static inline Value sin(const Value& x) {
            return std::sin(x);
        }

        static inline Value fma(const Value& x, const Value& y, const Value& z) {
            return x*y + z;
        }

        static inline Value fms(const Value& x, const Value& y, const Value& z) {
            return x*y - z;
        }

        static inline Value pi() {
            return Value(3.14159265358979323846264338327950288L);
        }

        template <typename SampleValue>
        static inline void load(const SampleValue* t, Value& x) {
            x = *t;
        }

        static inline void store(Value* t, const Value& x) {
            *t = x;
        }
    };

    struct Double4Spec{
        using Value = xsimd::batch<double, xsimd::avx>;

        static inline void load(const double* ptr, Value& x) {
            x = xsimd::batch<double, xsimd::avx>::load_unaligned(ptr);
        }

        static inline void store(double* ptr, const Value& x) {
            x.store_unaligned(ptr);
        }
    
        static inline Value fma(const Value& x, const Value& y, const Value& z) {
            return xsimd::fma(x, y, z);
        }

        static inline Value fms(const Value& x, const Value& y, const Value& z) {
            return xsimd::fms(x, y, z);
        }

    };
}



#endif