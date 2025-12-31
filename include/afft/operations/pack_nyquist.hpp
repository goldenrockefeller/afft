
#ifndef AFFT_PACK_NYQUIST_HPP
#define AFFT_PACK_NYQUIST_HPP

#include <cstddef>

namespace afft {
    template <typename Sample>
    void pack_nyquist(
        Sample* out_real,
        Sample* out_imag,
        const Sample* in_real,
        const Sample* in_imag,
        std::size_t spectra_len,
        bool using_hermitian_packed_form)
    {
        if (!using_hermitian_packed_form)
        {
            auto unpacked_spectra_len = spectra_len + 1;
            Sample ampl_at_zero = in_real[0];
            Sample ampl_at_nyquist = in_real[unpacked_spectra_len - 1];
            out_real[0] = Sample(0.5) * (ampl_at_zero + ampl_at_nyquist);
            out_imag[0] = Sample(0.5) * (ampl_at_zero - ampl_at_nyquist);
        }
        else
        {
            Sample ampl_at_zero = in_real[0];
            Sample ampl_at_nyquist = in_imag[0];
            out_real[0] = Sample(0.5) * (ampl_at_zero + ampl_at_nyquist);
            out_imag[0] = Sample(0.5) * (ampl_at_zero - ampl_at_nyquist);
        }
    }
}

#endif
