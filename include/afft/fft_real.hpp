#ifndef AFFT_FFT_REAL_HPP
#define AFFT_FFT_REAL_HPP

#include <cstddef>
#include <vector>
#include "afft/fft_complex.hpp"
#include "afft/fft_real_impl.hpp"

namespace afft
{
    template <typename Spec, class Allocator = std::allocator<typename Spec::sample>>
    class FftReal {
        using sample = typename Spec::sample;
        using operand = typename Spec::operand;
        using vector = typename std::vector<sample, Allocator>;
        
        constexpr cache_optimizing_offset = 1024 /sizeof(sample);
        constexpr std::size_t n_samples_per_operand = Spec::n_samples_per_operand;

        private:
            FftComplex<Spec, Allocator> fft_complex_;

            vector rotor_real_;
            vector rotor_imag_;

            mutable vector work_real_;
            mutable vector work_imag_;

        public:
            const std::size_t &log_n_samples_per_operand() const
            {
                return fft_complex_.log_n_samples_per_operand();
            }

            const std::size_t &n_samples() const
            {
                return fft_complex_.n_samples();
            }

            FftReal() : 
                signal_len_(0)
            {}

            const std::size_t &log_n_samples_per_operand() const
            {
                return fft_complex_.log_n_samples_per_operand();
            }

            FftReal() : 
                signal_len_(0)
            {}

            explicit FftReal(std::size_t signal_len) : 
                fft_complex_(signal_len >> 1),
                rotor_real_(rotor<true>(signal_len >> 1)),
                rotor_imag_(rotor<false>(signal_len >> 1)),
                work_real_(spectra_len_),
                work_imag_(spectra_len_ + cache_optimizing_offset)
            {
                // Nothing else to do.
            }

            template <bool Normalizing>
            static inline void eval_fft (
                sample* spectra_real, 
                sample* spectra_imag,
                const sample* signal_real,
                const sample *rotor_real,
                const sample *rotor_imag,
                const FftComplex<Spec, Allocator>& fft_complex,
                sample *work_real,
                sample *work_imag)
            {
                switch (log_n_samples_per_operand())
                { //
                case 0:
                    FftRealImpl<typename BoundedSpec<Spec, 0>::spec, Allocator>::template eval_fft<Normalizing>(
                        spectra_real,
                        spectra_imag,
                        signal_real,
                        rotor_real,
                        rotor_imag,
                        fft_complex,
                        work_real,
                        work_imag
                    );
                    break;
                case 1:
                    FftRealImpl<typename BoundedSpec<Spec, 1>::spec, Allocator>::template eval_fft<Normalizing>(
                        spectra_real,
                        spectra_imag,
                        signal_real,
                        rotor_real,
                        rotor_imag,
                        fft_complex,
                        work_real,
                        work_imag);
                    break;
                case 2:
                    FftRealImpl<typename BoundedSpec<Spec, 2>::spec, Allocator>::template eval_fft<Normalizing>(
                        spectra_real,
                        spectra_imag,
                        signal_real,
                        rotor_real,
                        rotor_imag,
                        fft_complex,
                        work_real,
                        work_imag);
                    break;
                case 3:
                    FftRealImpl<typename BoundedSpec<Spec, 3>::spec, Allocator>::template eval_fft<Normalizing>(
                        spectra_real,
                        spectra_imag,
                        signal_real,
                        rotor_real,
                        rotor_imag,
                        fft_complex,
                        work_real,
                        work_imag);
                    break;
                case 4:
                    FftRealImpl<typename BoundedSpec<Spec, 4>::spec, Allocator>::template eval_fft<Normalizing>(
                        spectra_real,
                        spectra_imag,
                        signal_real,
                        rotor_real,
                        rotor_imag,
                        fft_complex,
                        work_real,
                        work_imag);
                    break;
                case 5:
                    FftRealImpl<typename BoundedSpec<Spec, 5>::spec, Allocator>::template eval_fft<Normalizing>(
                        spectra_real,
                        spectra_imag,
                        signal_real,
                        rotor_real,
                        rotor_imag,
                        fft_complex,
                        work_real,
                        work_imag);
                    break;
                case 6:
                    FftRealImpl<typename BoundedSpec<Spec, 6>::spec, Allocator>::template eval_fft<Normalizing>(
                        spectra_real,
                        spectra_imag,
                        signal_real,
                        rotor_real,
                        rotor_imag,
                        fft_complex,
                        work_real,
                        work_imag);
                    break;
                case 7:
                    FftRealImpl<typename BoundedSpec<Spec, 7>::spec, Allocator>::template eval_fft<Normalizing>(
                        spectra_real,
                        spectra_imag,
                        signal_real,
                        rotor_real,
                        rotor_imag,
                        fft_complex,
                        work_real,
                        work_imag);
                    break;
                case 8: // Maximum support number of samples per operand is 256!
                    FftRealImpl<typename BoundedSpec<Spec, 8>::spec, Allocator>::template eval_fft<Normalizing>(
                        spectra_real,
                        spectra_imag,
                        signal_real,
                        rotor_real,
                        rotor_imag,
                        fft_complex,
                        work_real,
                        work_imag);
                    break;
                }
            }

            template <bool Normalizing>
            static inline void eval_ifft (
                sample* signal_real,
                const sample* spectra_real, 
                const sample* spectra_imag,
                const sample *rotor_real,
                const sample *rotor_imag,
                const FftComplex<Spec, Allocator>& fft_complex,
                sample *work_real,
                sample *work_imag)
            {
                switch (log_n_samples_per_operand())
                { //
                case 0:
                    FftRealImpl<typename BoundedSpec<Spec, 0>::spec, Allocator>::template eval_ifft<Normalizing>(
                        signal_real,
                        spectra_real,
                        spectra_imag,
                        rotor_real,
                        rotor_imag,
                        fft_complex,
                        work_real,
                        work_imag
                    );
                    break;
                case 1:
                    FftRealImpl<typename BoundedSpec<Spec, 1>::spec, Allocator>::template eval_ifft<Normalizing>(
                        signal_real,
                        spectra_real,
                        spectra_imag,
                        rotor_real,
                        rotor_imag,
                        fft_complex,
                        work_real,
                        work_imag);
                    break;
                case 2:
                    FftRealImpl<typename BoundedSpec<Spec, 2>::spec, Allocator>::template eval_ifft<Normalizing>(
                        signal_real,
                        spectra_real,
                        spectra_imag,
                        rotor_real,
                        rotor_imag,
                        fft_complex,
                        work_real,
                        work_imag);
                    break;
                case 3:
                    FftRealImpl<typename BoundedSpec<Spec, 3>::spec, Allocator>::template eval_ifft<Normalizing>(
                        signal_real,
                        spectra_real,
                        spectra_imag,
                        rotor_real,
                        rotor_imag,
                        fft_complex,
                        work_real,
                        work_imag);
                    break;
                case 4:
                    FftRealImpl<typename BoundedSpec<Spec, 4>::spec, Allocator>::template eval_ifft<Normalizing>(
                        signal_real,
                        spectra_real,
                        spectra_imag,
                        rotor_real,
                        rotor_imag,
                        fft_complex,
                        work_real,
                        work_imag);
                    break;
                case 5:
                    FftRealImpl<typename BoundedSpec<Spec, 5>::spec, Allocator>::template eval_ifft<Normalizing>(
                        signal_real,
                        spectra_real,
                        spectra_imag,
                        rotor_real,
                        rotor_imag,
                        fft_complex,
                        work_real,
                        work_imag);
                    break;
                case 6:
                    FftRealImpl<typename BoundedSpec<Spec, 6>::spec, Allocator>::template eval_ifft<Normalizing>(
                        signal_real,
                        spectra_real,
                        spectra_imag,
                        rotor_real,
                        rotor_imag,
                        fft_complex,
                        work_real,
                        work_imag);
                    break;
                case 7:
                    FftRealImpl<typename BoundedSpec<Spec, 7>::spec, Allocator>::template eval_ifft<Normalizing>(
                        signal_real,
                        spectra_real,
                        spectra_imag,
                        rotor_real,
                        rotor_imag,
                        fft_complex,
                        work_real,
                        work_imag);
                    break;
                case 8: // Maximum support number of samples per operand is 256!
                    FftRealImpl<typename BoundedSpec<Spec, 8>::spec, Allocator>::template eval_ifft<Normalizing>(
                        signal_real,
                        spectra_real,
                        spectra_imag,
                        rotor_real,
                        rotor_imag,
                        fft_complex,
                        work_real,
                        work_imag);
                    break;
                }
            }

            void fft(
                sample* spectra_real, 
                sample* spectra_imag,
                const sample* signal_real) const
            {
                eval_fft<false>(
                    spectra_real,
                    spectra_imag,
                    signal_real,
                    rotor_real_.data(),
                    rotor_imag_.data(),
                    fft_complex_,
                    work_real_.data(),
                    work_imag_.data() + cache_optimizing_offset);
            }

            void fft_normalized(
                sample* spectra_real, 
                sample* spectra_imag,
                const sample* signal_real) const
            {
                eval_fft<true>(
                    spectra_real,
                    spectra_imag,
                    signal_real,
                    rotor_real_.data(),
                    rotor_imag_.data(),
                    fft_complex_,
                    work_real_.data(),
                    work_imag_.data() + cache_optimizing_offset);
            }

            void ifft(
                sample* signal_real
                const sample* spectra_real, 
                const sample* spectra_imag) const
            {
                eval_ifft<false>(
                    spectra_real,
                    spectra_imag,
                    signal_real,
                    rotor_real_.data(),
                    rotor_imag_.data(),
                    fft_complex_,
                    work_real_.data(),
                    work_imag_.data() + cache_optimizing_offset);
            }

            void ifft_normalized(
                sample* signal_real
                const sample* spectra_real, 
                const sample* spectra_imag) const
            {
                eval_ifft<true>(
                    spectra_real,
                    spectra_imag,
                    signal_real,
                    rotor_real_.data(),
                    rotor_imag_.data(),
                    fft_complex_,
                    work_real_.data(),
                    work_imag_.data() + cache_optimizing_offset);
            }

        private:
            static std::size_t checked_signal_len(std::size_t signal_len) {

                if ((signal_len & (signal_len-1)) != 0) {
                    throw std::invalid_argument(
                        "The transform length must be a power of 2"
                    );
                }

                if (signal_len < 2) {
                    throw std::invalid_argument(
                        "The signal length must be 2 or greater."
                    );
                }

                return signal_len;
            }

            template <bool ForReal>
            vector rotor (
                std::size_t rotor_len
            ) {
                vector rotor(rotor_len);

                for (
                    std::size_t i = 0;
                    i < rotor_len;
                    i ++
                ) {
                    if (ForReal) {
                        rotor[i] 
                            = cos(
                                pi() 
                                * rem_2(sample(i) / sample(rotor_len))
                            );
                    }
                    else {
                        rotor[i] 
                            = -sin(
                                pi() 
                                * rem_2(sample(i) / sample(rotor_len))
                            );
                    }
                }

                return rotor;
            }
    };

}
#endif