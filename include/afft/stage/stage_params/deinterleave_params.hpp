#ifndef AFFT_DEINTERLEAVE_PARAMS_HPP
#define AFFT_DEINTERLEAVE_PARAMS_HPP

#include <cstddef>

namespace afft{
    template <typename Sample>
    struct DeinterleaveParams {
        std::size_t out_real_id;
        std::size_t out_imag_id;
        std::size_t in_id;
        std::size_t id_start;
        std::size_t id_end;
    };
}
#endif
