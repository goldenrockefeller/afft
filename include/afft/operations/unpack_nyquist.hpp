
#ifndef AFFT_UNPACK_NYQUIST_HPP
#define AFFT_UNPACK_NYQUIST_HPP

#include <cstddef>

namespace afft {
        template <typename Sample>
        void unpack_nyquist(
            Sample* out_real,
            Sample* out_imag,
            const Sample* in_real,
            const Sample* in_imag,
            std::size_t spectra_len,
            bool using_hermitian_packed_form
        )
        {
            if (!using_hermitian_packed_form) {
                auto unpacked_spectra_len = spectra_len + 1;
                Sample ampl_at_zero = in_real[0] + in_imag[0];
                Sample ampl_at_nyquist = in_real[0] - in_imag[0];
                out_real[0] = ampl_at_zero;
                out_real[unpacked_spectra_len - 1] = ampl_at_nyquist;
                out_imag[0] = Sample(0);
                out_imag[unpacked_spectra_len - 1] = Sample(0);
            }
            else {
                Sample ampl_at_zero = in_real[0] + in_imag[0];
                Sample ampl_at_nyquist = in_real[0] - in_imag[0];
                out_real[0] = ampl_at_zero;
                out_imag[0] = ampl_at_nyquist;
            }
        }
}

#endif