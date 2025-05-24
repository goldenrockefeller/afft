#ifndef GOLDENROCEKEFELLER_AFFT_CONVOLUTION_HPP
#define GOLDENROCEKEFELLER_AFFT_CONVOLUTION_HPP

#include "afft/fft_real.hpp"

namespace afft{
    template<
        typename SampleSpec, 
        typename OperandSpec
    >
    class ConvolutionReal {
        using Sample = typename SampleSpec::Value;
        using Operand = typename OperandSpec::Value;
        using size_t = std::size_t;

        using Vector = typename std::vector<Sample, xsimd::aligned_allocator<Sample, 128>>;

        static constexpr size_t k_N_SAMPLES_PER_OPERAND 
            = sizeof(Operand) / sizeof(Sample);

        static inline void load(const Sample* t, Operand& x) {
            OperandSpec::load(t, x);
        }

        static inline void store(Sample* t, const Operand& x) {
            OperandSpec::store(t, x);
        }

        private:

            size_t signal_len_;
            size_t spectra_len_;
            FftReal<SampleSpec, OperandSpec> fft_real_;

            mutable Vector spectra_a_real_;
            mutable Vector spectra_b_real_;
            mutable Vector spectra_a_imag_;
            mutable Vector spectra_b_imag_;

        public:
            ConvolutionReal() : 
                signal_len_(0)    
            {}

            explicit ConvolutionReal(size_t signal_len) : 
                signal_len_(signal_len),
                spectra_len_( (signal_len >> 1) + 1 ),
                fft_real_(signal_len),
                spectra_a_real_(spectra_len_),
                spectra_b_real_(spectra_len_),
                spectra_a_imag_(spectra_len_),
                spectra_b_imag_(spectra_len_)
            {
                // Nothing else to do.
            }

            void compute_convolution (
                Sample* convolution,
                Sample* signal_a,
                Sample* signal_b,
                bool fast = false
            ) {
                compute_convolution (
                    convolution,
                    signal_a,
                    signal_b,
                    spectra_b_real_.data(),
                    spectra_b_imag_.data(),
                    fast
                );
            }

            void compute_convolution (
                Sample* convolution,
                Sample* signal_a,
                bool fast = false
            ) {
                compute_convolution (
                    convolution,
                    signal_a,
                    spectra_b_real_.data(),
                    spectra_b_imag_.data(),
                    fast
                );
            }

            void compute_convolution (
                Sample* convolution,
                Sample* signal_a,
                Sample* signal_b,
                Sample* spectre_b_real_cache,
                Sample* spectre_b_imag_cache,
                bool fast = false
            ) {
                if (signal_len_ == 0) {
                    return;
                }

                fft_real_.compute_spectra(
                    spectre_b_real_cache,
                    spectre_b_imag_cache,
                    signal_b,
                    false, /*rescaling*/
                    fast /*computing for convolution*/
                );

                compute_convolution(
                    convolution,
                    signal_a,
                    spectre_b_real_cache,
                    spectre_b_imag_cache,
                    fast
                );
            }

            void compute_convolution (
                Sample* convolution,
                Sample* signal_a,
                Sample* spectre_b_real_cache,
                Sample* spectre_b_imag_cache,
                bool fast = false
            ) {
                if (signal_len_ == 0) {
                    return;
                }
                fft_real_.compute_spectra(
                    spectra_a_real_.data(),
                    spectra_a_imag_.data(),
                    signal_a,
                    false, /*rescaling*/
                    fast /*computing for convolution*/
                );

                size_t half_signal_len = signal_len_ >> 1;
                Sample* a_real = spectra_a_real_.data();
                Sample* b_real = spectre_b_real_cache;
                Sample* a_imag = spectra_a_imag_.data();
                Sample* b_imag = spectre_b_imag_cache;

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

                    load(a_real, a_real_operand);
                    load(a_real, a_real_operand_copy);
                    load(b_real, b_real_operand);
                    load(a_imag, a_imag_operand);
                    load(b_imag, b_imag_operand);

                    a_real_operand =
                        a_real_operand * b_real_operand
                        - a_imag_operand * b_imag_operand;

                    a_imag_operand =
                        a_real_operand_copy * b_imag_operand
                        + a_imag_operand * b_real_operand;

                    store(a_real, a_real_operand);
                    store(a_imag, a_imag_operand);

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

                fft_real_.compute_signal(
                    convolution, 
                    spectra_a_real_.data(), 
                    spectra_a_imag_.data(),
                    true, /*rescaling*/
                    fast /*compute for convolution*/
                );
            }
    };
}

#endif


