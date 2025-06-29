#ifndef AFFT_CT_RADIX2_PARAMS_HPP
#define AFFT_CT_RADIX2_PARAMS_HPP

#include <cstddef>

namespace afft{
    template <typename Spec>
    struct CtRadix2Params {
        typename Spec::sample* twiddles; 
        std::size_t subtwiddle_len;
        std::size_t subtwiddle_start;
        std::size_t subtwiddle_end;
    };
}
#endif