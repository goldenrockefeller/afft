#ifndef GOLDENROCEKEFELLER_AFFT_SPEC
#define GOLDENROCEKEFELLER_AFFT_SPEC

#include <cmath>

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
    };
}}



#endif