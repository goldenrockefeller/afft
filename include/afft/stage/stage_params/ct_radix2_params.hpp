#ifndef AFFT_CT_RADIX2_PARAMS_HPP
#define AFFT_CT_RADIX2_PARAMS_HPP

#include <cstddef>

namespace afft{
    template <typename Sample>
    struct CtRadix2Params {
        const Sample* twiddles; 
        std::size_t subtwiddle_len;
        std::size_t subtwiddle_start;
        std::size_t subtwiddle_end;
    };
}
#endif