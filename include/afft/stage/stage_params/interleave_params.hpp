#ifndef AFFT_INTERLEAVE_PARAMS_HPP
#define AFFT_INTERLEAVE_PARAMS_HPP

#include <cstddef>

namespace afft{
    template <typename Sample>
    struct InterleaveParams {
        std::size_t out_id;
        std::size_t in_real_id;
        std::size_t in_imag_id;
        std::size_t id_start;
        std::size_t id_end;
    };
}
#endif
