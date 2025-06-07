#ifndef AFFT_S_RADIX4_PARAMS_HPP
#define AFFT_S_RADIX4_PARAMS_HPP

#include <cstddef>

namespace afft{
    template <typename Spec>
    struct SRadix4Params {
        typename Spec::sample *tw_real_b_0;
        typename Spec::sample *tw_imag_b_0;
        typename Spec::sample *tw_real_c_0;
        typename Spec::sample *tw_imag_c_0;
        typename Spec::sample *tw_real_d_0;
        typename Spec::sample *tw_imag_d_0;
        std::size_t *out_indexes;
        std::size_t *in_indexes;
        std::size_t subfft_id_start;
        std::size_t subfft_id_end;
        std::size_t log_subtwiddle_len;
        std::size_t output_id;
        std::size_t input_id;
    };
}
#endif