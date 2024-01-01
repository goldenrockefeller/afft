#ifndef GOLDENROCEKEFELLER_AFFT_FFT_REAL_HPP
#define GOLDENROCEKEFELLER_AFFT_FFT_REAL_HPP

#include "fft_complex.hpp"

namespace goldenrockefeller{ namespace afft{

    template<
        typename SampleSpec, 
        typename OperandSpec
    >
    class FftReal {
        using Sample = typename SampleSpec::Value;
        using Operand = typename OperandSpec::Value;

        static constexpr size_t k_N_SAMPLES_PER_OPERAND 
            = sizeof(Operand) / sizeof(Sample);

        static inline Sample Pi(){
            return SampleSpec::Pi();
        }

        static inline Sample Cos(const Sample& x){
            return SampleSpec::Cos(x);
        }

        static inline Sample Sin(const Sample& x){
            return SampleSpec::Sin(x);
        }

        static inline void Load(const Sample* t, Operand& x) {
            OperandSpec::Load(t, x);
        }

        static inline void Store(Sample* t, const Operand& x) {
            OperandSpec::Store(t, x);
        }

        static_assert(
            ((k_N_SAMPLES_PER_OPERAND & (k_N_SAMPLES_PER_OPERAND-1)) == 0), 
            "The number of samples per operand must be a power of 2"
        );


        static_assert(
            (k_N_SAMPLES_PER_OPERAND>=1), 
            "The number of samples per operand must be positive"
        );

        private:

            std::size_t signal_len;
            std::size_t spectra_len;
            FftComplex fft_complex;

            std::vector<Sample> rotor_real;
            std::vector<Sample> rotor_imag;

            mutable std::vector<Sample> work_real;
            mutable std::vector<Sample> work_imag;

        public:
            FftReal(std::size_t signal_len) : 
                signal_len(CheckedSignalLen(signal_len)),
                spectra_len( (signal_len >> 1) + 1 )
                fft_complex(signal_len >> 1),
                work_real(spectra_len),
                work_imag(spectra_len),
                rotor_real(Rotor(signal_len >> 1, Cos, 1)),
                rotor_imag(Rotor(signal_len >> 1, Sin, -1))
            {
                // Nothing else to do.
            }

            ComputeSpectra (
                Sample* spectra_real, 
                Sample* spectra_imag,
                Sample* signal_real, 
                bool rescaling = false,
                bool processing_for_convolution = false
            ) {
                Sample* combined_signal_real = work_real.data();
                Sample* combined_signal_imag = work_imag.data();
                auto half_signal_len = signal_len >> 1;

                for (
                    std::size_t i = 0;
                    i < (half_signal_len);
                    i++
                ) {
                    combined_signal_real[i] = signal_real[2 * i];
                    combined_signal_imag[i] = signal_imag[2 * i + 1];
                }

                fft_complex.Process(
                    spectra_real,
                    spectra_imag,
                    combined_signal_real,
                    combined_signal_imag,
                    false, /* calculating_inverse */
                    rescaling,
                    processing_for_convolution
                );

                if (processing_for_convolution == false) {
                    Sample* reversed_spectra_real = work_real.data();
                    Sample* reversed_spectra_imag = work_imag.data();

                    // GET REVERSED SPECTRA

                    reversed_spectra_real[0] = spectra_real[0];
                    reversed_spectra_imag[0] = spectra_imag[0];

                    auto half_spectra_len = half_signal_len >> 1;
                    auto reversed_spectra_len = spectra_len - 1;

                    reversed_spectra_real[half_spectra_len] 
                        = spectra_real[half_spectra_len];

                    reversed_spectra_imag[half_spectra_len] 
                        = spectra_imag[half_spectra_len];

                    for (
                        std::size_t i = 1;
                        i < (half_spectra_len);
                        i++
                    ) {
                        reversed_spectra_real[i] 
                            = spectra_real[reversed_spectra_len - i];

                        reversed_spectra_imag[i]
                            = - spectra_imag[reversed_spectra_len - i];

                        reversed_spectra_real[reversed_spectra_len - i] 
                            = spectra_real[i];

                        reversed_spectra_imag[reversed_spectra_len - i]
                            = - spectra_imag[i];
                    }

                    // Calculate Spectra

                    spectra_real[spectra_len - 1] = 
                        Sample(0.5) * (
                            spectra_real[0] + reversed_spectra_real[0]
                            - spectra_imag[0] + reversed_spectra_imag[0]
                        );

                    spectra_imag[spectra_len - 1] = Sample(0.); 

                    Operand half(0.5);

                    for (
                        std::size_t i = 0;
                        i < reversed_spectra_len;
                        i += k_N_SAMPLES_PER_OPERAND
                    ) {
                        Operand spectra_operand_real;
                        Operand rspectra_operand_real;
                        Operand rotor_operand_real;

                        Operand spectra_operand_imag;
                        Operand rspectra_operand_imag;
                        Operand rotor_operand_imag;   

                        Load(spectra_real + i, spectra_operand_real);
                        Load(reversed_spectra_real + i, rspectra_operand_real);
                        Load(rotor_real.data() + i, rotor_operand_real);  

                        Load(spectra_imag + i, spectra_operand_imag);
                        Load(reversed_spectra_imag + i, rspectra_operand_imag);
                        Load(rotor_imag.data() + i, rotor_operand_imag);                  

                        Operand difference_operand_real
                            = spectra_operand_real - rspectra_operand_real;

                        Operand difference_operand_imag
                            = spectra_operand_imag - rspectra_operand_imag;

                        Operand rotated_operand_real
                            = -difference_operand_real * rotor_operand_imag
                            - difference_operand_imag * rotor_operand_real;

                        Operand rotated_operand_imag
                            = difference_operand_real * rotor_operand_real
                            - difference_operand_imag * rotor_operand_imag;
                        
                        spectra_operand_real  = 
                            half * (
                                spectra_operand_real 
                                + rspectra_operand_real
                                - rotated_operand_real
                            );

                        spectra_operand_imag  = 
                            half * (
                                spectra_operand_imag 
                                + rspectra_operand_imag
                                - rotated_operand_imag
                            );

                        Store(spectra_real + i, spectra_operand_real); 
                        Store(spectra_imag + i, spectra_operand_imag); 
                    }
                }
            }

        private:
            static std::size_t CheckedSignalLen(std::size_t signal_len) {

                if ((signal_len & (signal_len-1)) != 0) {
                    throw std::invalid_argument(
                        "The transform length must be a power of 2"
                    );
                }

                if (signal_len < 2 * k_N_SAMPLES_PER_OPERAND) {
                    throw std::invalid_argument(
                        "The signal length must not be "
                        "less than twice the operand size."
                    );
                }

                if (signal_len < 8) {
                    throw std::invalid_argument(
                        "The signal length must be 8 or greater."
                    );
                }

                return signal_len;
            }

            std::vector<Sample> Rotor (
                std::size_t rotor_len,
                const std::function<Sample(Sample)>& trig_fn,
                int multiplier
            ) {
                std::vector<Sample> rotor;

                rotor.resize(rotor_len);

                for (
                    std::size_t i = 0;
                    i < rotor_len;
                    i ++
                ) {
                    rotor[i] = 
                        = multiplier 
                        * trig_fn(
                            Pi() 
                            * Rem2(Sample(i) / Sample(rotor_len))
                        );
                }
            }

    };

}}