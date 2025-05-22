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
        explicit FftComplex(std::size_t n_samples, std::size_t min_partition_len) : butterfly_(n_samples, min_partition_len) {}

        const Butterfly<Spec, Allocator> &butterfly() const
        {
            return butterfly_;
        }

        template <bool Rescaling = false>
        void eval_ditime(
            typename Spec::sample *out_real,
            typename Spec::sample *out_imag,
            const typename Spec::sample *in_real,
            const typename Spec::sample *in_imag)
        {
            butterfly_.template eval_ditime<Rescaling>(out_real, out_imag, out_real, out_imag);
        }

        template <bool Rescaling = false>
        void eval_difreq(
            typename Spec::sample *out_real,
            typename Spec::sample *out_imag,
            const typename Spec::sample *in_real,
            const typename Spec::sample *in_imag)
        {
            butterfly_.template eval_difreq<Rescaling>(out_real, out_imag, in_real, in_imag);
        }
    };
}

#endif