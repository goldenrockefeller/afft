#ifndef GOLDENROCEKEFELLER_AFFT_FFT_REAL_HPP
#define GOLDENROCEKEFELLER_AFFT_FFT_REAL_HPP

#include "afft/fft_complex.hpp"

namespace goldenrockefeller{ namespace afft{
    template <typename vector_t>
    vector_t BitReversedOrder(vector_t input){
        auto bit_reversed_indexes = BitReversedIndexes(input.size());
        vector_t bit_reversed_order(input.size());

        for (std::size_t i = 0; i < input.size(); i++) {
            bit_reversed_order[i] = input[bit_reversed_indexes[i]];
        }

        return bit_reversed_order;
    }

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
            FftComplex<SampleSpec, OperandSpec> fft_complex;

            std::vector<Sample> rotor_real;
            std::vector<Sample> rotor_imag;

            std::vector<Sample> rotor_bit_reversed_real;
            std::vector<Sample> rotor_bit_reversed_imag;

            mutable std::vector<Sample> work_real;
            mutable std::vector<Sample> work_imag;

        public:
            FftReal(std::size_t signal_len) : 
                signal_len(CheckedSignalLen(signal_len)),
                spectra_len( (signal_len >> 1) + 1 ),
                fft_complex(signal_len >> 1),
                work_real(spectra_len),
                work_imag(spectra_len),
                rotor_real(Rotor(signal_len >> 1, Cos, 1)),
                rotor_imag(Rotor(signal_len >> 1, Sin, -1)),
                rotor_bit_reversed_real(BitReversedOrder(rotor_real)),
                rotor_bit_reversed_imag(BitReversedOrder(rotor_imag))
            {
                // Nothing else to do.
            }

            void ComputeSpectra (
                Sample* spectra_real, 
                Sample* spectra_imag,
                Sample* signal_real, 
                bool rescaling = false,
                bool computing_for_convolution = false
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
                    combined_signal_imag[i] = signal_real[2 * i + 1];
                }

                fft_complex.ProcessDif(
                    spectra_real,
                    spectra_imag,
                    combined_signal_real,
                    combined_signal_imag,
                    false, /* calculating_inverse */
                    rescaling,
                    computing_for_convolution
                );

                
                // Calculate Spectra

                spectra_real[spectra_len - 1] = 
                        spectra_real[0]
                        - spectra_imag[0];

                spectra_imag[spectra_len - 1] = Sample(0.); 

                Sample* reversed_spectra_real = work_real.data();
                Sample* reversed_spectra_imag = work_imag.data();
                Sample* rotor_variant_real;
                Sample* rotor_variant_imag;

                auto half_spectra_len = half_signal_len >> 1;
                auto reversed_spectra_len = spectra_len - 1;

                reversed_spectra_real[0] = spectra_real[0];
                reversed_spectra_imag[0] = - spectra_imag[0];

                reversed_spectra_real[half_spectra_len] 
                    = spectra_real[half_spectra_len];

                reversed_spectra_imag[half_spectra_len] 
                    = - spectra_imag[half_spectra_len];
            
                // GET REVERSED SPECTRA

                if (computing_for_convolution == false) { 
                    rotor_variant_real = rotor_real.data();
                    rotor_variant_imag = rotor_imag.data();

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
                }
                else {
                    rotor_variant_real = rotor_bit_reversed_real.data();
                    rotor_variant_imag = rotor_bit_reversed_imag.data();

                    reversed_spectra_real[1] =  spectra_real[1];
                    reversed_spectra_imag[1] =  -spectra_imag[1];

                    std::size_t start_id = 2;
                    std::size_t section_len = 2;

                    while (start_id < half_signal_len) {
                        auto half_section_len = section_len >> 1;
                        auto a = start_id;
                        auto b = start_id + section_len - 1;
                        for (
                            std::size_t i = 0;
                            i < half_section_len;
                            i++
                        ) {
                            reversed_spectra_real[a] =  spectra_real[b];
                            reversed_spectra_imag[a] =  -spectra_imag[b];
                            reversed_spectra_real[b] =  spectra_real[a];
                            reversed_spectra_imag[b] =  -spectra_imag[a];
                            a += 1;
                            b -= 1;
                        }
                        start_id += section_len;
                        section_len = section_len << 1;
                    }
                }

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
                    Load(rotor_variant_real + i, rotor_operand_real);  

                    Load(spectra_imag + i, spectra_operand_imag);
                    Load(reversed_spectra_imag + i, rspectra_operand_imag);
                    Load(rotor_variant_imag + i, rotor_operand_imag);                  

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
            
            void ComputeSignal (                
                Sample* signal_real, 
                Sample* spectra_real, 
                Sample* spectra_imag,
                bool rescaling = false,
                bool computing_for_convolution = false
            ) {
                Sample ampl_at_zero = spectra_real[0];
                Sample ampl_at_zero_imag = spectra_imag[0];
                Sample ampl_at_nyquist = spectra_real[spectra_len - 1];
                Sample* compact_spectra_real = spectra_real;
                Sample* compact_spectra_imag = spectra_imag;

                // Put spectra in compact form (zero and nyquist frequencies are
                // stored in the first value). 
                spectra_real[0] 
                    = Sample(0.5) * (ampl_at_zero + ampl_at_nyquist);

                spectra_imag[0] 
                    = Sample(0.5) * (ampl_at_zero - ampl_at_nyquist);

                Sample* reversed_spectra_real = work_real.data();
                Sample* reversed_spectra_imag = work_imag.data();
                Sample* rotor_variant_real;
                Sample* rotor_variant_imag;

                auto half_signal_len = signal_len >> 1;
                auto half_spectra_len = signal_len >> 2;
                auto reversed_spectra_len = spectra_len - 1;

                reversed_spectra_real[0] = spectra_real[0];
                reversed_spectra_imag[0] = - spectra_imag[0];

                reversed_spectra_real[half_spectra_len] 
                    = spectra_real[half_spectra_len];

                reversed_spectra_imag[half_spectra_len] 
                    = - spectra_imag[half_spectra_len];

                // GET REVERSED SPECTRA

                if (computing_for_convolution == false) { 
                    rotor_variant_real = rotor_real.data();
                    rotor_variant_imag = rotor_imag.data();

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
                } else {
                    rotor_variant_real = rotor_bit_reversed_real.data();
                    rotor_variant_imag = rotor_bit_reversed_imag.data();

                    reversed_spectra_real[1] =  spectra_real[1];
                    reversed_spectra_imag[1] =  -spectra_imag[1];

                    std::size_t start_id = 2;
                    std::size_t section_len = 2;

                    while (start_id < half_signal_len) {
                        auto half_section_len = section_len >> 1;
                        auto a = start_id;
                        auto b = start_id + section_len - 1;
                        for (
                            std::size_t i = 0;
                            i < half_section_len;
                            i++
                        ) {
                            reversed_spectra_real[a] =  spectra_real[b];
                            reversed_spectra_imag[a] =  -spectra_imag[b];
                            reversed_spectra_real[b] =  spectra_real[a];
                            reversed_spectra_imag[b] =  -spectra_imag[a];
                            a += 1;
                            b -= 1;
                        }
                        start_id += section_len;
                        section_len = section_len << 1;
                    }
                }

                Operand half(0.5);

                compact_spectra_real = work_real.data();
                compact_spectra_imag = work_imag.data();

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
                    Load(rotor_variant_real + i, rotor_operand_real);  

                    Load(spectra_imag + i, spectra_operand_imag);
                    Load(reversed_spectra_imag + i, rspectra_operand_imag);
                    Load(rotor_variant_imag + i, rotor_operand_imag);                  

                    Operand difference_operand_real
                        = spectra_operand_real - rspectra_operand_real;

                    Operand difference_operand_imag
                        = spectra_operand_imag - rspectra_operand_imag;

                    Operand rotated_operand_real
                        = difference_operand_real * rotor_operand_imag
                        - difference_operand_imag * rotor_operand_real;

                    Operand rotated_operand_imag
                        = difference_operand_real * rotor_operand_real
                        + difference_operand_imag * rotor_operand_imag;
                    
                    spectra_operand_real  = 
                        half * (
                            spectra_operand_real 
                            + rspectra_operand_real
                            + rotated_operand_real
                        );

                    spectra_operand_imag  = 
                        half * (
                            spectra_operand_imag 
                            + rspectra_operand_imag
                            + rotated_operand_imag
                        );

                    Store(compact_spectra_real + i, spectra_operand_real); 
                    Store(compact_spectra_imag + i, spectra_operand_imag); 
                }

                compact_spectra_real[0] 
                    = Sample(0.5) * (ampl_at_zero + ampl_at_nyquist);

                compact_spectra_imag[0] 
                    = Sample(0.5) * (ampl_at_zero - ampl_at_nyquist);

                Sample* raw_combined_signal_real = signal_real;
                Sample* raw_combined_signal_imag 
                    = signal_real + half_signal_len;

                fft_complex.ProcessDit(
                    raw_combined_signal_real,
                    raw_combined_signal_imag,
                    compact_spectra_real,
                    compact_spectra_imag,
                    true, 
                    rescaling,
                    computing_for_convolution
                );

                // Interweave the combined signal;

                Sample* combined_signal_real = work_real.data();
                Sample* combined_signal_imag = work_imag.data();

                for (
                    std::size_t i = 0;
                    i < (half_signal_len);
                    i += k_N_SAMPLES_PER_OPERAND
                ) {
                    Operand real;
                    Operand imag;

                    Load(raw_combined_signal_real + i, real);
                    Load(raw_combined_signal_imag + i, imag);

                    Store(combined_signal_real + i, real);
                    Store(combined_signal_imag + i, imag);
                }

                for (
                    std::size_t i = 0;
                    i < (half_signal_len);
                    i++
                ) {
                    signal_real[2 * i] = combined_signal_real[i] ;
                    signal_real[2 * i + 1] = combined_signal_imag[i];
                }

                // Restore spectra to its original state 
                spectra_real[0] = ampl_at_zero;
                spectra_real[0] = ampl_at_zero_imag;
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
                std::vector<Sample> rotor(rotor_len);

                for (
                    std::size_t i = 0;
                    i < rotor_len;
                    i ++
                ) {
                    rotor[i] 
                        = multiplier 
                        * trig_fn(
                            Pi() 
                            * Rem2(Sample(i) / Sample(rotor_len))
                        );
                }

                return rotor;
            }
    };
}}

#endif