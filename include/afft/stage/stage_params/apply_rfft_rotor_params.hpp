#ifndef AFFT_APPLY_RFFT_ROTOR_PARAMS_HPP
#define AFFT_APPLY_RFFT_ROTOR_PARAMS_HPP

#include <cstddef>

namespace afft{
    template <typename Sample>
    struct ApplyRfftRotorParams {
        std::size_t out_real_id;
        std::size_t out_imag_id;
        std::size_t spectra_real_id;
        std::size_t spectra_imag_id;
        std::size_t reversed_real_id;
        std::size_t reversed_imag_id;
        std::size_t id_start;
        std::size_t id_end;
        std::size_t spectra_len;
        bool using_hermitian_packed_form;
    };
}
#endif
