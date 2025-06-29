#ifndef AFFT_S_RADIX2_PARAMS_HPP
#define AFFT_S_RADIX2_PARAMS_HPP

#include <cstddef>
#include "afft/radix/radix_params/s_radix4_params.hpp"

namespace afft{
    template <typename Spec>
    struct SRadix2Params {
        typename Spec::sample *twiddles;
        std::size_t *out_indexes;
        std::size_t *in_indexes;
        std::size_t subfft_id_start;
        std::size_t subfft_id_end;
        LogInterleavePermute log_interleave_permute;
        std::size_t output_id;
        std::size_t input_id;
    };
}
#endif