#ifndef AFFT_RADIX_STAGE_HPP
#define AFFT_RADIX_STAGE_HPP

#include <cstddef>
#include "afft/radix/radix_type.hpp"

namespace afft{
    template <typename Spec>
    struct RadixStage {
        RadixType type;
        Spec::sample* tw_real_b; 
        Spec::sample* tw_imag_b; 
        Spec::sample* tw_real_c; 
        Spec::sample* tw_imag_c; 
        Spec::sample* tw_real_d; 
        Spec::sample* tw_imag_d;
        std::size_t subfft_id_start;
        std::size_t subfft_id_end;
        std::size_t subtwiddle_len;
        std::size_t subtwiddle_start;
        std::size_t subtwiddle_end;
    };
}

#endif