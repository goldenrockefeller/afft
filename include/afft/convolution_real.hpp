#ifndef GOLDENROCEKEFELLER_AFFT_CONVOLUTION_HPP
#define GOLDENROCEKEFELLER_AFFT_CONVOLUTION_HPP

#include "afft/fft_real.hpp"

namespace goldenrockefeller{ namespace afft{
    template<
        typename SampleSpec, 
        typename OperandSpec
    >
    class ConvolutionReal {
        using Sample = typename SampleSpec::Value;
        using Operand = typename OperandSpec::Value;
        using size_t = std::size_t;

        static constexpr size_t k_N_SAMPLES_PER_OPERAND 
            = sizeof(Operand) / sizeof(Sample);

        static inline void Load(const Sample* t, Operand& x) {
            OperandSpec::Load(t, x);
        }

        static inline void Store(Sample* t, const Operand& x) {
            OperandSpec::Store(t, x);
        }

        private:

            size_t signal_len;
            size_t spectra_len;
            FftReal<SampleSpec, OperandSpec> fft_real;

            mutable std::vector<Sample> spectra_a_real;
            mutable std::vector<Sample> spectra_b_real;
            mutable std::vector<Sample> spectra_a_imag;
            mutable std::vector<Sample> spectra_b_imag;

        public:

            ConvolutionReal(size_t signal_len) : 
                signal_len(signal_len),
                spectra_len( (signal_len >> 1) + 1 ),
                fft_real(signal_len),
                spectra_a_real(spectra_len),
                spectra_b_real(spectra_len),
                spectra_a_imag(spectra_len),
                spectra_b_imag(spectra_len)
            {
                // Nothing else to do.
            }

        void ComputeConvolution (
            Sample* convolution,
            Sample* signal_a,
            Sample* signal_b
        ) {
            fft_real.ComputeSpectra(
                spectra_a_real.data(),
                spectra_a_imag.data(),
                signal_a,
                false, /*rescaling*/
                true /*computing for convolution*/
            );
            // std::cout << "----------------"  << std::endl;
            // for(size_t i = 0; i < spectra_len; i++) {
            //     std::cout << spectra_a_real[i] << ", " << spectra_a_imag[i] << std::endl;
            // }
            // std::cout << "----------------"  << std::endl;


            fft_real.ComputeSpectra(
                spectra_b_real.data(),
                spectra_b_imag.data(),
                signal_b,
                false, /*rescaling*/
                true /*computing for convolution*/
            );

            size_t half_signal_len = signal_len >> 1;
            Sample* a_real = spectra_a_real.data();
            Sample* b_real = spectra_b_real.data();
            Sample* a_imag = spectra_a_imag.data();
            Sample* b_imag = spectra_b_imag.data();

            for (
                size_t i = 0;
                i < half_signal_len;
                i += k_N_SAMPLES_PER_OPERAND
            ) {
                Operand a_real_operand;
                Operand b_real_operand;
                Operand a_imag_operand;
                Operand b_imag_operand;                
                Operand a_real_operand_copy;

                Load(a_real, a_real_operand);
                Load(a_real, a_real_operand_copy);
                Load(b_real, b_real_operand);
                Load(a_imag, a_imag_operand);
                Load(b_imag, b_imag_operand);

                a_real_operand =
                    a_real_operand * b_real_operand
                    - a_imag_operand * b_imag_operand;

                a_imag_operand =
                    a_real_operand_copy * b_imag_operand
                    + a_imag_operand * b_real_operand;

                Store(a_real, a_real_operand);
                Store(a_imag, a_imag_operand);

                a_real += k_N_SAMPLES_PER_OPERAND;
                a_imag += k_N_SAMPLES_PER_OPERAND;
                b_real += k_N_SAMPLES_PER_OPERAND;
                b_imag += k_N_SAMPLES_PER_OPERAND;
            }

            auto a_real_copy = *a_real;
            *a_real = 
                *a_real * *b_real
                - *a_imag * *b_imag;

            *a_imag =
                a_real_copy * *b_imag
                + *a_imag * *b_real;

            fft_real.ComputeSignal(
                convolution, 
                spectra_a_real.data(), 
                spectra_a_imag.data(),
                true, /*rescaling*/
                true /*compute for convolution*/
            );
        }
    };
}}

#endif


