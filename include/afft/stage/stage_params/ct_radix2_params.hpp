#ifndef AFFT_CT_RADIX2_PARAMS_HPP
#define AFFT_CT_RADIX2_PARAMS_HPP

#include <cstddef>

namespace afft{
    template <typename Sample>
    struct CtRadix2Params {
        std::size_t inout_real_id;
        std::size_t inout_imag_id;
        const Sample* twiddles; 
        std::size_t subtwiddle_len;
        std::size_t subtwiddle_start;
        std::size_t subtwiddle_end;
    };
}
#endif