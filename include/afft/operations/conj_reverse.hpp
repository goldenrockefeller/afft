#ifndef AFFT_CONJ_REVERSE_HPP
#define AFFT_CONJ_REVERSE_HPP

#include <cstddef>

namespace afft
{
    template <typename Sample>
    void conj_reverse(
        Sample* reversed_real,
        Sample* reversed_imag,
        const Sample* spectra_real,
        const Sample* spectra_imag,
        std::size_t spectra_len,
        std::size_t id_start,
        std::size_t id_end)
    {
        std::size_t half_spectra_len = spectra_len >> 1;

        // Reverse
        if (id_start == 0) {
            reversed_real[0] = spectra_real[0];
            reversed_imag[0] = spectra_imag[0]; //Not negative in case this is Nyquist frequency

            reversed_real[half_spectra_len] = spectra_real[half_spectra_len];
            reversed_imag[half_spectra_len] = -spectra_imag[half_spectra_len];
            id_start += 1;
        }

        // GET Conjugate REVERSED SPECTRA
        for (std::size_t i = id_start; i < id_end; i++)
        {
            auto spectra_real_tmp = spectra_real[i];
            auto spectra_imag_tmp = spectra_imag[i];

            reversed_real[i] = spectra_real[spectra_len - i];
            reversed_imag[i] = -spectra_imag[spectra_len - i];

            reversed_real[spectra_len - i] = spectra_real_tmp;
            reversed_imag[spectra_len - i] = -spectra_imag_tmp;
        }
    }
}

#endif
              