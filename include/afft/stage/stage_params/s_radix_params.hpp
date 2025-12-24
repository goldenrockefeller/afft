#ifndef AFFT_S_RADIX_PARAMS_HPP
#define AFFT_S_RADIX_PARAMS_HPP

#include <cstddef>
#include "afft/stage/stage_params/log_interleave_permute.hpp"

namespace afft{
    template <typename Sample>
    struct SRadixParams {
        std::size_t out_real_id;
        std::size_t out_imag_id;
        std::size_t in_real_id;
        std::size_t in_imag_id;
        const Sample *twiddles;
        const std::size_t *out_permute_indexes;
        const std::size_t *in_permute_indexes;
        std::size_t subfft_id_start;
        std::size_t subfft_id_end;
        std::size_t subtwiddle_len;
        LogInterleavePermute log_interleave_permute;
    };
}
#endif