#ifndef AFFT_CONJ_REVERSE_PARAMS_HPP
#define AFFT_CONJ_REVERSE_PARAMS_HPP

#include <cstddef>

namespace afft{
    template <typename Sample>
    struct ConjReverseParams {
        std::size_t out_real_id;
        std::size_t out_imag_id;
        std::size_t spectra_real_id;
        std::size_t spectra_imag_id;
        std::size_t spectra_len;
        std::size_t id_start;
        std::size_t id_end;
    };
}
#endif
