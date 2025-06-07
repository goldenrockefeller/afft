#ifndef AFFT_CT_RADIX4_PARAMS_HPP
#define AFFT_CT_RADIX4_PARAMS_HPP

#include <cstddef>

namespace afft{
    template <typename Spec>
    struct CtRadix4Params {
        typename Spec::sample* tw_real_b_0; 
        typename Spec::sample* tw_imag_b_0; 
        typename Spec::sample* tw_real_c_0; 
        typename Spec::sample* tw_imag_c_0; 
        typename Spec::sample* tw_real_d_0; 
        typename Spec::sample* tw_imag_d_0;
        std::size_t subfft_id_start;
        std::size_t subfft_id_end;
        std::size_t subtwiddle_len;
        std::size_t subtwiddle_start;
        std::size_t subtwiddle_end;
    };
}
#endif