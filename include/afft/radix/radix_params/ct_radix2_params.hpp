#ifndef AFFT_CT_RADIX2_PARAMS_HPP
#define AFFT_CT_RADIX2_PARAMS_HPP

#include <cstddef>

namespace afft{
    template <typename Spec>
    struct CtRadix2Params {
        typename Spec::sample* tw_real_b_0; 
        typename Spec::sample* tw_imag_b_0; 
        std::size_t subtwiddle_len;
        std::size_t subtwiddle_start;
        std::size_t subtwiddle_end;
    };
}
#endif