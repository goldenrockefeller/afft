#ifndef GOLDENROCEKEFELLER_AFFT_FFT_REAL_HPP
#define GOLDENROCEKEFELLER_AFFT_FFT_REAL_HPP

#include "afft/fft_complex.hpp"

namespace afft{
    template <typename vector_t>
    vector_t bit_reversed_order(vector_t input){
        auto bit_reversed_indexes_ = bit_reversed_indexes(input.size());
        vector_t bit_reversed_order(input.size());

        for (std::size_t i = 0; i < input.size(); i++) {
            bit_reversed_order[i] = input[bit_reversed_indexes_[i]];
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
        using Vector = typename std::vector<Sample, xsimd::aligned_allocator<Sample, 128>>;

        static constexpr size_t k_N_SAMPLES_PER_OPERAND 
            = sizeof(Operand) / sizeof(Sample);

        static inline Sample pi(){
            return SampleSpec::pi();
        }

        static inline Sample cos(const Sample& x){
            return SampleSpec::cos(x);
        }

        static inline Sample sin(const Sample& x){
            return SampleSpec::sin(x);
        }

        static inline void load(const Sample* t, Operand& x) {
            OperandSpec::load(t, x);
        }

        static inline void store(Sample* t, const Operand& x) {
            OperandSpec::store(t, x);
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

            std::size_t signal_len_;
            std::size_t spectra_len_;
            FftComplex<SampleSpec, OperandSpec> fft_complex_;

            Vector rotor_real_;
            Vector rotor_imag_;

            Vector rotor_bit_reversed_real_;
            Vector rotor_bit_reversed_imag_;

            mutable Vector work_real_;
            mutable Vector work_imag_;

        public:
            FftReal() : 
                signal_len_(0)
            {}

            explicit FftReal(std::size_t signal_len) : 
                signal_len_(checked_signal_len(signal_len)),
                spectra_len_( (signal_len >> 1) + 1 ),
                fft_complex_(signal_len >> 1),
                rotor_real_(rotor(signal_len >> 1, cos, 1)),
                rotor_imag_(rotor(signal_len >> 1, sin, -1)),
                rotor_bit_reversed_real_(bit_reversed_order(rotor_real_)),
                rotor_bit_reversed_imag_(bit_reversed_order(rotor_imag_)),
                work_real_(spectra_len_),
                work_imag_(spectra_len_)
            {
                // Nothing else to do.
            }

            void compute_spectra (
                Sample* spectra_real, 
                Sample* spectra_imag,
                Sample* signal_real, 
                bool rescaling = false,
                bool computing_for_convolution = false
            ) {
                if (signal_len_ == 0) {
                    return;
                }

                Sample* combined_signal_real = work_real_.data();
                Sample* combined_signal_imag = work_imag_.data();
                auto half_signal_len = signal_len_ >> 1;

                for (
                    std::size_t i = 0;
                    i < (half_signal_len);
                    i++
                ) {
                    combined_signal_real[i] = signal_real[2 * i];
                    combined_signal_imag[i] = signal_real[2 * i + 1];
                }

                fft_complex_.process_dif(
                    spectra_real,
                    spectra_imag,
                    combined_signal_real,
                    combined_signal_imag,
                    false, /* calculating_inverse */
                    rescaling,
                    computing_for_convolution
                );

                
                // Calculate Spectra

                spectra_real[spectra_len_ - 1] = 
                        spectra_real[0]
                        - spectra_imag[0];

                spectra_imag[spectra_len_ - 1] = Sample(0.); 

                Sample* reversed_spectra_real = work_real_.data();
                Sample* reversed_spectra_imag = work_imag_.data();
                Sample* rotor_variant_real;
                Sample* rotor_variant_imag;

                auto half_spectra_len = half_signal_len >> 1;
                auto reversed_spectra_len = spectra_len_ - 1;

                reversed_spectra_real[0] = spectra_real[0];
                reversed_spectra_imag[0] = - spectra_imag[0];

                reversed_spectra_real[half_spectra_len] 
                    = spectra_real[half_spectra_len];

                reversed_spectra_imag[half_spectra_len] 
                    = - spectra_imag[half_spectra_len];
            
                // GET REVERSED SPECTRA

                if (computing_for_convolution == false) { 
                    rotor_variant_real = rotor_real_.data();
                    rotor_variant_imag = rotor_imag_.data();

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
                    rotor_variant_real = rotor_bit_reversed_real_.data();
                    rotor_variant_imag = rotor_bit_reversed_imag_.data();

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

                    load(spectra_real + i, spectra_operand_real);
                    load(reversed_spectra_real + i, rspectra_operand_real);
                    load(rotor_variant_real + i, rotor_operand_real);  

                    load(spectra_imag + i, spectra_operand_imag);
                    load(reversed_spectra_imag + i, rspectra_operand_imag);
                    load(rotor_variant_imag + i, rotor_operand_imag);                  

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

                    store(spectra_real + i, spectra_operand_real); 
                    store(spectra_imag + i, spectra_operand_imag); 
                }
            }
            
            void compute_signal (                
                Sample* signal_real, 
                Sample* spectra_real, 
                Sample* spectra_imag,
                bool rescaling = false,
                bool computing_for_convolution = false
            ) {
                Sample ampl_at_zero = spectra_real[0];
                Sample ampl_at_zero_imag = spectra_imag[0];
                Sample ampl_at_nyquist = spectra_real[spectra_len_ - 1];

                if (signal_len_ == 0) {
                    return;
                }

                // Put spectra in compact form (zero and nyquist frequencies are
                // stored in the first value). 
                spectra_real[0] 
                    = Sample(0.5) * (ampl_at_zero + ampl_at_nyquist);

                spectra_imag[0] 
                    = Sample(0.5) * (ampl_at_zero - ampl_at_nyquist);

                Sample* reversed_spectra_real = work_real_.data();
                Sample* reversed_spectra_imag = work_imag_.data();
                Sample* rotor_variant_real;
                Sample* rotor_variant_imag;

                auto half_signal_len = signal_len_ >> 1;
                auto half_spectra_len = signal_len_ >> 2;
                auto reversed_spectra_len = spectra_len_ - 1;

                reversed_spectra_real[0] = spectra_real[0];
                reversed_spectra_imag[0] = - spectra_imag[0];

                reversed_spectra_real[half_spectra_len] 
                    = spectra_real[half_spectra_len];

                reversed_spectra_imag[half_spectra_len] 
                    = - spectra_imag[half_spectra_len];

                // GET REVERSED SPECTRA

                if (computing_for_convolution == false) { 
                    rotor_variant_real = rotor_real_.data();
                    rotor_variant_imag = rotor_imag_.data();

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
                    rotor_variant_real = rotor_bit_reversed_real_.data();
                    rotor_variant_imag = rotor_bit_reversed_imag_.data();

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

                Sample* compact_spectra_real = work_real_.data();
                Sample* compact_spectra_imag = work_imag_.data();

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

                    load(spectra_real + i, spectra_operand_real);
                    load(reversed_spectra_real + i, rspectra_operand_real);
                    load(rotor_variant_real + i, rotor_operand_real);  

                    load(spectra_imag + i, spectra_operand_imag);
                    load(reversed_spectra_imag + i, rspectra_operand_imag);
                    load(rotor_variant_imag + i, rotor_operand_imag);                  

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

                    store(compact_spectra_real + i, spectra_operand_real); 
                    store(compact_spectra_imag + i, spectra_operand_imag); 
                }

                compact_spectra_real[0] 
                    = Sample(0.5) * (ampl_at_zero + ampl_at_nyquist);

                compact_spectra_imag[0] 
                    = Sample(0.5) * (ampl_at_zero - ampl_at_nyquist);

                Sample* raw_combined_signal_real = signal_real;
                Sample* raw_combined_signal_imag 
                    = signal_real + half_signal_len;

                fft_complex_.process_dit(
                    raw_combined_signal_real,
                    raw_combined_signal_imag,
                    compact_spectra_real,
                    compact_spectra_imag,
                    true, 
                    rescaling,
                    computing_for_convolution
                );

                // Interweave the combined signal;

                Sample* combined_signal_real = work_real_.data();
                Sample* combined_signal_imag = work_imag_.data();

                for (
                    std::size_t i = 0;
                    i < (half_signal_len);
                    i += k_N_SAMPLES_PER_OPERAND
                ) {
                    Operand real;
                    Operand imag;

                    load(raw_combined_signal_real + i, real);
                    load(raw_combined_signal_imag + i, imag);

                    store(combined_signal_real + i, real);
                    store(combined_signal_imag + i, imag);
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
            static std::size_t checked_signal_len(std::size_t signal_len) {

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

            Vector rotor (
                std::size_t rotor_len,
                const std::function<Sample(Sample)>& trig_fn,
                int multiplier
            ) {
                Vector rotor(rotor_len);

                for (
                    std::size_t i = 0;
                    i < rotor_len;
                    i ++
                ) {
                    rotor[i] 
                        = multiplier 
                        * trig_fn(
                            pi() 
                            * rem_2(Sample(i) / Sample(rotor_len))
                        );
                }

                return rotor;
            }
    };
}

#endif