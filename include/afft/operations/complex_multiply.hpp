#ifndef AFFT_COMPLEX_MULTIPLY_HPP
#define AFFT_COMPLEX_MULTIPLY_HPP

#include <cstddef>

namespace afft {
    template <typename Spec>
    void complex_multiply(
        typename Spec::sample *out_real,
        typename Spec::sample *out_imag,
        const typename Spec::sample *in_real_a,
        const typename Spec::sample *in_imag_a,
        const typename Spec::sample *in_real_b,
        const typename Spec::sample *in_imag_b,
        std::size_t id_start,
        std::size_t id_end,
        bool using_hermitian_packed_form
    )
    {
        using operand = typename Spec::operand;

        typename Spec::sample ampl_at_zero;
        typename Spec::sample ampl_at_nyquist;
        bool using_hermitian_packed_correction = false;


        if (using_hermitian_packed_form && id_start == 0) {
            ampl_at_zero = in_real_a[0] * in_real_b[0];
            ampl_at_nyquist = in_imag_a[0] * in_imag_b[0];

            using_hermitian_packed_correction = true;
        }

        for (std::size_t i = id_start; i < id_end; i += Spec::n_samples_per_operand)
        {
            operand a_re, a_im, b_re, b_im, out_re, out_im;
            Spec::load(a_re, in_real_a + i);
            Spec::load(a_im, in_imag_a + i);
            Spec::load(b_re, in_real_b + i);
            Spec::load(b_im, in_imag_b + i);

            out_re = a_re * b_re - a_im * b_im;
            out_im = a_re * b_im + a_im * b_re;

            Spec::store(out_real + i, out_re);
            Spec::store(out_imag + i, out_im);
        }

        if (using_hermitian_packed_correction) {
            out_real[0] = ampl_at_zero;
            out_imag[0] = ampl_at_nyquist;
        }
    }
}

#endif
