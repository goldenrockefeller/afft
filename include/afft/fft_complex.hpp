#ifndef AFFT_FFT_COMPLEX_HPP
#define AFFT_FFT_COMPLEX_HPP

#include <cstddef>
#include <vector>
#include "afft/butterfly/butterfly.hpp"

namespace afft
{
    template <typename Spec, class Allocator = std::allocator<typename Spec::sample>>
    class FftComplex
    {
        Butterfly<Spec, Allocator> butterfly_;

    public:
        explicit FftComplex(std::size_t n_samples) : butterfly_(n_samples) {}

        const Butterfly<Spec, Allocator> &butterfly() const
        {
            return butterfly_;
        }

        const std::size_t &log_n_samples_per_operand() const
        {
            return butterfly_.log_n_samples_per_operand();
        }

        const std::size_t &n_samples() const
        {
            return butterfly_.n_samples();
        }


        template <bool Rescaling = false>
        void eval(
            typename Spec::sample *out_real,
            typename Spec::sample *out_imag,
            const typename Spec::sample *in_real,
            const typename Spec::sample *in_imag) const
        {
            butterfly_.template eval<Rescaling>(out_real, out_imag, in_real, in_imag);
        }

        void fft(
            typename Spec::sample *out_real,
            typename Spec::sample *out_imag,
            const typename Spec::sample *in_real,
            const typename Spec::sample *in_imag) const
        {
            butterfly_.template eval<false>(out_real, out_imag, in_real, in_imag);
        }

        void fft_normalized(
            typename Spec::sample *out_real,
            typename Spec::sample *out_imag,
            const typename Spec::sample *in_real,
            const typename Spec::sample *in_imag) const
        {
            butterfly_.template eval<true>(out_real, out_imag, in_real, in_imag);
        }

        void ifft(
            typename Spec::sample *out_real,
            typename Spec::sample *out_imag,
            const typename Spec::sample *in_real,
            const typename Spec::sample *in_imag) const
        {
            butterfly_.template eval<true>(in_real, in_imag, out_real, out_imag);
        }

        void ifft_normalized(
            typename Spec::sample *out_real,
            typename Spec::sample *out_imag,
            const typename Spec::sample *in_real,
            const typename Spec::sample *in_imag) const
        {
            butterfly_.template eval<true>(in_real, in_imag, out_real, out_imag);
        }
    };
}

#endif