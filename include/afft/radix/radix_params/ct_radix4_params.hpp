#ifndef AFFT_CT_RADIX4_PARAMS_HPP
#define AFFT_CT_RADIX4_PARAMS_HPP

#include <cstddef>

namespace afft{
    template <typename Spec>
    struct CtRadix4Params {
        typename Spec::sample* twiddles; 
        std::size_t subfft_id_start;
        std::size_t subfft_id_end;
        std::size_t subtwiddle_len;
        std::size_t subtwiddle_start;
        std::size_t subtwiddle_end;
    };
}
#endif