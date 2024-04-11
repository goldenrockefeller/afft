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

            mutable std::vector<Sample> spectre_a_real;
            mutable std::vector<Sample> spectre_b_real;
            mutable std::vector<Sample> spectre_a_imag;
            mutable std::vector<Sample> spectre_b_imag;

        public:

            ConvolutionReal(size_t signal_len) : 
                signal_len(signal_len),
                spectra_len( (signal_len >> 1) + 1 ),
                fft_real(signal_len),
                spectre_a_real(spectra_len),
                spectre_b_real(spectra_len),
                spectre_a_imag(spectra_len),
                spectre_b_imag(spectra_len)
            {
                // Nothing else to do.
            }

        void ComputeConvolution (
            Sample* convolution,
            Sample* signal_a,
            Sample* signal_b,
            bool fast
        ) {
            fft_real.ComputeSpectra(
                spectre_a_real.data(),
                spectre_a_imag.data(),
                signal_a,
                false, /*rescaling*/
                fast /*computing for convolution*/
            );


            fft_real.ComputeSpectra(
                spectre_b_real.data(),
                spectre_b_imag.data(),
                signal_b,
                false, /*rescaling*/
                fast /*computing for convolution*/
            );

            size_t half_signal_len = signal_len >> 1;
            Sample* a_real = spectre_a_real.data();
            Sample* b_real = spectre_b_real.data();
            Sample* a_imag = spectre_a_imag.data();
            Sample* b_imag = spectre_b_imag.data();

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
                spectre_a_real.data(), 
                spectre_a_imag.data(),
                true, /*rescaling*/
                fast /*compute for convolution*/
            );
        }
    };
}}

#endif


