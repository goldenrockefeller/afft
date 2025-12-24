#ifndef AFFT_CT_RADIX4_PARAMS_HPP
#define AFFT_CT_RADIX4_PARAMS_HPP

#include <cstddef>

namespace afft{
    template <typename Sample>
    struct CtRadix4Params {
        std::size_t inout_real_id;
        std::size_t inout_imag_id;
        const Sample* twiddles; 
        std::size_t subfft_id_start;
        std::size_t subfft_id_end;
        std::size_t subtwiddle_len;
        std::size_t subtwiddle_start;
        std::size_t subtwiddle_end;
    };
}
#endif