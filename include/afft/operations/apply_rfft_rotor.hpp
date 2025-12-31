
#ifndef AFFT_APPLY_RFFT_ROTOR_HPP
#define AFFT_APPLY_RFFT_ROTOR_HPP

#include <cstddef>

namespace afft {
    template <typename Spec>
    void apply_rfft_rotor(
        typename Spec::sample* out_real,
        typename Spec::sample* out_imag,
        const typename Spec::sample* spectra_real,
        const typename Spec::sample* spectra_imag,
        const typename Spec::sample* reversed_spectra_real,
        const typename Spec::sample* reversed_spectra_imag,
        const typename Spec::sample* rotor_real,
        const typename Spec::sample* rotor_imag,
        std::size_t id_start,
        std::size_t id_end,
        std::size_t spectra_len,
        bool using_hermitian_packed_form)
    {
        using operand = typename Spec::operand;
        operand half(0.5);
        const bool handle_edge_freqs = (id_start == 0);
        typename Spec::sample unpacked_zero{};
        typename Spec::sample unpacked_nyquist{};

        if (handle_edge_freqs)
        {
            // Recreate the unpacked DC and Nyquist amplitudes that were previously written
            // by the separate unpack_nyquist stage so they can be applied after the rotor work.
            unpacked_zero = spectra_real[0] + spectra_imag[0];
            unpacked_nyquist = spectra_real[0] - spectra_imag[0];
        }

        for (std::size_t i = id_start; i < id_end; i += Spec::n_samples_per_operand) {
            operand s_re, rs_re, r_re, s_im, rs_im, r_im, diff_re, diff_im, rot_re, rot_im;
            Spec::load(s_re, spectra_real + i);
            Spec::load(rs_re, reversed_spectra_real + i);
            Spec::load(r_re, rotor_real + i);
            Spec::load(s_im, spectra_imag + i);
            Spec::load(rs_im, reversed_spectra_imag + i);
            Spec::load(r_im, rotor_imag + i);

            diff_re = s_re - rs_re;
            diff_im = s_im - rs_im;
            rot_re = diff_re * r_im + diff_im * r_re;
            rot_im = diff_im * r_im - diff_re * r_re;

            s_re = half * (s_re + rs_re + rot_re);
            s_im = half * (s_im + rs_im + rot_im);

            Spec::store(out_real + i, s_re);
            Spec::store(out_imag + i, s_im);
        }

        if (handle_edge_freqs)
        {
            if (using_hermitian_packed_form)
            {
                out_real[0] = unpacked_zero;
                out_imag[0] = unpacked_nyquist;
            }
            else
            {
                out_real[0] = unpacked_zero;
                out_imag[0] = typename Spec::sample(0);
                if (spectra_len > 0)
                {
                    out_real[spectra_len] = unpacked_nyquist;
                    out_imag[spectra_len] = typename Spec::sample(0);
                }
            }
        }
    }
}

#endif