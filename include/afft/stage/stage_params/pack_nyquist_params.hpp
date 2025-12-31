#ifndef AFFT_PACK_NYQUIST_PARAMS_HPP
#define AFFT_PACK_NYQUIST_PARAMS_HPP

#include <cstddef>

namespace afft{
    template <typename Sample>
    struct PackNyquistParams {
        std::size_t out_real_id;
        std::size_t out_imag_id;
        std::size_t in_real_id;
        std::size_t in_imag_id;
        std::size_t spectra_len;
        bool using_hermitian_packed_form;
    };
}
#endif
