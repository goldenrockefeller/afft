#ifndef AFFT_FFT_REAL_IMPL_HPP
#define AFFT_FFT_REAL_IMPL_HPP

#include <cstddef>
#include <vector>
#include "afft/fft_complex.hpp"

namespace afft
{
    template <typename Spec, class Allocator = std::allocator<typename Spec::sample>>
    class FftRealImpl {
        using sample = typename Spec::sample;
        using operand = typename Spec::operand;
        using vector = typename std::vector<sample, Allocator>;
        
        constexpr std::size_t n_samples_per_operand = Spec::n_samples_per_operand;

        static inline sample pi(){
            return Spec::pi();
        }

        static inline sample cos(const sample& x){
            return Spec::cos(x);
        }

        static inline sample sin(const sample& x){
            return Spec::sin(x);
        }

        static inline void load(operand& x, const sample* t) {
            Spec::load(t, x);
        }

        static inline void store(sample* t, const operand& x) {
            Spec::store(t, x);
        }

        public:
            
            template <bool Normalizing>
            static inline void eval_fft (
                sample* spectra_real, 
                sample* spectra_imag,
                const sample* signal_real,
                const sample *rotor_real,
                const sample *rotor_imag,
                const FftComplex<Spec, Allocator>& fft_complex,
                sample *work_real,
                sample *work_imag
            ) {
                std::size_t signal_len_ = 2 * fft_complex.n_samples(),
                auto spectra_len_ =  (signal_len >> 1) + 1 ;

                if (signal_len_ == 0) {
                    return;
                }
                

                sample* combined_signal_real = work_real_;
                sample* combined_signal_imag = work_imag_;
                auto half_signal_len = signal_len_ >> 1;

                for (
                    std::size_t i = 0;
                    i < (half_signal_len);
                    i++
                ) {
                    combined_signal_real[i] = signal_real[2 * i];
                    combined_signal_imag[i] = signal_real[2 * i + 1];
                }

                if (Normalizing)
                {
                    fft_complex.fft_normalized(
                        spectra_real,
                        spectra_imag,
                        combined_signal_real,
                        combined_signal_imag
                    );
                }
                else {
                    fft_complex.fft(
                        spectra_real,
                        spectra_imag,
                        combined_signal_real,
                        combined_signal_imag
                    );
                }   
                // Calculate Spectra

                spectra_real[spectra_len_ - 1] = 
                        spectra_real[0]
                        - spectra_imag[0];

                spectra_imag[spectra_len_ - 1] = sample(0.); 

                sample* reversed_spectra_real = work_real;
                sample* reversed_spectra_imag = work_imag;

                auto half_spectra_len = half_signal_len >> 1;
                auto reversed_spectra_len = half_signal_len;

                reversed_spectra_real[0] = spectra_real[0];
                reversed_spectra_imag[0] = - spectra_imag[0];

                reversed_spectra_real[half_spectra_len] 
                    = spectra_real[half_spectra_len];

                reversed_spectra_imag[half_spectra_len] 
                    = - spectra_imag[half_spectra_len];
            
                // GET REVERSED SPECTRA
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
              
                operand half(0.5);
                for (
                    std::size_t i = 0;
                    i < reversed_spectra_len;
                    i += n_samples_per_operand
                ) {
                    operand spectra_operand_real;
                    operand rspectra_operand_real;
                    operand rotor_operand_real;

                    operand spectra_operand_imag;
                    operand rspectra_operand_imag;
                    operand rotor_operand_imag;   

                    load(spectra_operand_real, spectra_real + i);
                    load(rspectra_operand_real, reversed_spectra_real + i);
                    load(rotor_operand_real, rotor_real + i);  

                    load(spectra_operand_imag, spectra_imag + i);
                    load(rspectra_operand_imag, reversed_spectra_imag + i);
                    load(rotor_operand_imag, rotor_imag + i);                  

                    operand difference_operand_real
                        = spectra_operand_real - rspectra_operand_real;

                    operand difference_operand_imag
                        = spectra_operand_imag - rspectra_operand_imag;

                    operand rotated_operand_real
                        = -difference_operand_real * rotor_operand_imag
                        - difference_operand_imag * rotor_operand_real;

                    operand rotated_operand_imag
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

            template <bool Normalizing>
            static inline void eval_ifft (                
                sample* signal_real, 
                const sample* spectra_real, 
                const sample* spectra_imag,
                const sample *rotor_real,
                const sample *rotor_imag,
                const FftComplex<Spec, Allocator>& fft_complex,
                sample *work_real,
                sample *work_imag
            ) {
                std::size_t signal_len_ = 2 * fft_complex.n_samples(),
                auto spectra_len_ =  (signal_len >> 1) + 1 ;
                if (signal_len_ == 0) {
                    return;
                }

                sample ampl_at_zero = spectra_real[0];
                sample ampl_at_nyquist = spectra_real[spectra_len_ - 1];


                // Put spectra in compact form (zero and nyquist frequencies are
                // stored in the first value). 
                auto spectra_real_0
                    = sample(0.5) * (ampl_at_zero + ampl_at_nyquist);

                auto spectra_imag_0
                    = sample(0.5) * (ampl_at_zero - ampl_at_nyquist);

                sample* reversed_spectra_real = work_real;
                sample* reversed_spectra_imag = work_imag;
                auto half_signal_len = signal_len_ >> 1;
                auto half_spectra_len = signal_len_ >> 2;
                auto reversed_spectra_len = half_signal_len;

                reversed_spectra_real[0] = spectra_real_0;
                reversed_spectra_imag[0] = - spectra_imag_0;

                // GET REVERSED SPECTRA
                reversed_spectra_real[half_spectra_len] 
                    = spectra_real[half_spectra_len];

                reversed_spectra_imag[half_spectra_len] 
                    = - spectra_imag[half_spectra_len];

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
             
                operand half(0.5);

                sample* compact_spectra_real = work_real;
                sample* compact_spectra_imag = work_imag;

                for (
                    std::size_t i = 0;
                    i < reversed_spectra_len;
                    i += n_samples_per_operand
                ) {
                    operand spectra_operand_real;
                    operand rspectra_operand_real;
                    operand rotor_operand_real;

                    operand spectra_operand_imag;
                    operand rspectra_operand_imag;
                    operand rotor_operand_imag;   

                    load(spectra_operand_real, spectra_real + i);
                    load(rspectra_operand_real, reversed_spectra_real + i);
                    load(rotor_operand_real, rotor_real + i);  

                    load(spectra_operand_imag, spectra_imag + i);
                    load(rspectra_operand_imag, reversed_spectra_imag + i);
                    load(rotor_operand_imag, rotor_imag + i);                  

                    operand difference_operand_real
                        = spectra_operand_real - rspectra_operand_real;

                    operand difference_operand_imag
                        = spectra_operand_imag - rspectra_operand_imag;

                    operand rotated_operand_real
                        = difference_operand_real * rotor_operand_imag
                        - difference_operand_imag * rotor_operand_real;

                    operand rotated_operand_imag
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
                    = sample(0.5) * (ampl_at_zero + ampl_at_nyquist);

                compact_spectra_imag[0] 
                    = sample(0.5) * (ampl_at_zero - ampl_at_nyquist);

                sample* raw_combined_signal_real = signal_real;
                sample* raw_combined_signal_imag 
                    = signal_real + half_signal_len;


                if (Normalizing)
                {
                    fft_complex.ifft_normalized(
                        raw_combined_signal_real,
                        raw_combined_signal_imag,
                        compact_spectra_real,
                        compact_spectra_imag,
                    );
                }
                else {
                    fft_complex.ifft(
                        raw_combined_signal_real,
                        raw_combined_signal_imag,
                        compact_spectra_real,
                        compact_spectra_imag,
                    );
                }   
            
                // Interweave the combined signal;

                sample* combined_signal_real = work_real;
                sample* combined_signal_imag = work_imag;

                for (
                    std::size_t i = 0;
                    i < (half_signal_len);
                    i += n_samples_per_operand
                ) {
                    operand real;
                    operand imag;

                    load(real, raw_combined_signal_real + i);
                    load(imag, raw_combined_signal_imag + i);

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
            }
    };
}
#endif