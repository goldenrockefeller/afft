#ifndef AFFT_COMMON_MATH_HPP
#define AFFT_COMMON_MATH_HPP

#include <cstddef>

namespace afft{ namespace common_math {
    inline std::size_t int_log_2(std::size_t n) {
        if (n == 0) {
            return 0;
        }
        std::size_t res = 0;
        while (n > 0) {
            res ++;
            n = n >> 1;
        }
        return res - 1;
    }
}}

#endif