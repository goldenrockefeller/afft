#ifndef AFFT_RADIX_STAGE_HPP
#define AFFT_RADIX_STAGE_HPP

#include <cstddef>
#include "afft/radix/radix_type.hpp"

namespace afft{
    template <typename Spec>
    struct RadixStage {
        RadixType type;
        typename Spec::sample* tw_real_b; 
        typename Spec::sample* tw_imag_b; 
        typename Spec::sample* tw_real_c; 
        typename Spec::sample* tw_imag_c; 
        typename Spec::sample* tw_real_d; 
        typename Spec::sample* tw_imag_d;
        std::size_t *out_indexes;
        std::size_t *in_indexes;
        std::size_t subfft_id_start;
        std::size_t subfft_id_end;
        std::size_t subtwiddle_len;
        std::size_t log_subtwiddle_len;
        std::size_t subtwiddle_start;
        std::size_t subtwiddle_end;
        std::size_t n_samples;
        bool is_first_ct_radix_stage;
    };
}
#endif