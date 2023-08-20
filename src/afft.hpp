#ifndef GOLDENROCEKEFELLER_AFFT_HPP
#define GOLDENROCEKEFELLER_AFFT_HPP

#include <cstddef>
#include <vector>
#include <functional>
#include <algorithm>
#include <cstring>

namespace goldenrockefeller{ namespace afft{
    std::size_t IntLog2(std::size_t n) {
        if (n == 0) {
            return 0;
        }
        std::size_t res = 0;
        while (n > 0) {
            res ++;
            n = n >> 1;
        }
        return res - 1;
    }

    std::vector<std::size_t> ScrambledIndexes(std::size_t n_indexes) {
        std::vector<std::size_t> scrambled_indexes(n_indexes);
        std::size_t n_bits = IntLog2(n_indexes);

        for (
            std::size_t id = 0;
            id < n_indexes;
            id++
        ) {
            auto work = id;
            std::size_t bit_reversed_id = 0;
            for (
                std::size_t bit_id = 0;
                bit_id < n_bits;
                bit_id++
            ) {
                bit_reversed_id = bit_reversed_id << 1;
                bit_reversed_id += work & 1;
                work = work >> 1;
            }
            scrambled_indexes[id] = bit_reversed_id;
        }
        return scrambled_indexes;
    }

    template<
        std::size_t k_TRANSFORM_LEN, 
        typename SampleSpec, 
        typename OperandSpec>
    class FftComplex {
    
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
            ((k_TRANSFORM_LEN & (k_TRANSFORM_LEN-1)) == 0), 
            "The transform length must be a power of 2"
        );

        static_assert(
            ((k_N_SAMPLES_PER_OPERAND & (k_N_SAMPLES_PER_OPERAND-1)) == 0), 
            "The number of samples per operand must be a power of 2"
        );

        static_assert(
            (k_TRANSFORM_LEN >= k_N_SAMPLES_PER_OPERAND), 
            "The transform length must not be less than the operand size."
        );

        static_assert(
            (k_N_SAMPLES_PER_OPERAND>=1), 
            "The number of samples per operand must be positive"
        );

        static_assert(
            (k_TRANSFORM_LEN>=4), 
            "The transform length must 4 or greater."
        );
        public:
        
            std::size_t n_radix_2_butterflies;
            std::size_t n_radix_4_butterflies;
            bool using_final_radix_2_butterfly;

            std::vector<std::vector<std::vector<Sample>>> twiddles_real;
            std::vector<std::vector<std::vector<Sample>>> twiddles_imag;

            std::vector<std::size_t> scrambled_indexes;
            std::vector<std::size_t> scrambled_indexes_dft;

            std::vector<std::vector<Sample>> dft_real;
            std::vector<std::vector<Sample>> dft_imag;
        
            FftComplex():
                n_radix_2_butterflies(
                    IntLog2(k_TRANSFORM_LEN / k_N_SAMPLES_PER_OPERAND)
                ),
                n_radix_4_butterflies(n_radix_2_butterflies >> 1),
                using_final_radix_2_butterfly(
                    n_radix_2_butterflies != 2 * n_radix_4_butterflies
                ),
                twiddles_real(
                    Twiddles(
                        n_radix_4_butterflies,
                        using_final_radix_2_butterfly,
                        Cos,
                        1
                    )
                ),
                twiddles_imag(
                    Twiddles(
                        n_radix_4_butterflies,
                        using_final_radix_2_butterfly,
                        Sin,
                        -1
                    )
                ),
                scrambled_indexes(ScrambledIndexes(k_TRANSFORM_LEN)),
                scrambled_indexes_dft(
                    ScrambledIndexes(k_N_SAMPLES_PER_OPERAND)
                ),
                dft_real(Dft(scrambled_indexes_dft, Cos, 1)),
                dft_imag(Dft(scrambled_indexes_dft, Sin, -1))
            {
                    // Nothing else to do.
            }

            template<bool k_CALCULATING_INVERSE>
            void Process(
                Sample* transform_real, 
                Sample* transform_imag,
                Sample* signal_real, 
                Sample* signal_imag
            ) const {
                constexpr bool k_DFT_IS_FINAL_PHASE 
                    = k_TRANSFORM_LEN == k_N_SAMPLES_PER_OPERAND;

                if (k_CALCULATING_INVERSE) {
                    std::swap(signal_real, signal_imag);
                }

                Sample scale_factor = Sample(1.);

                if (k_CALCULATING_INVERSE) {    
                    std::swap(transform_real, transform_imag);
                    scale_factor = 1. / Sample(k_TRANSFORM_LEN);
                }
                
                ScrambleSignal(
                    transform_real, 
                    transform_imag,
                    signal_real, 
                    signal_imag,
                    scale_factor
                );

                std::size_t subfft_len = 1;
                std::size_t n_subfft_len = k_TRANSFORM_LEN;

                //--------------------------------------------------------------
                // SCRAMBLE SIGNAL
                //--------------------------------------------------------------

                for (
                    std::size_t new_index = 0;
                    new_index < k_TRANSFORM_LEN; 
                    new_index++
                ) {
                    auto old_index = scrambled_indexes[new_index];

                    transform_real[new_index] 
                        = scale_factor * signal_real[old_index];

                    transform_imag[new_index] 
                        = scale_factor * signal_imag[old_index];
                }

                //--------------------------------------------------------------
                // DFT PHASE
                //--------------------------------------------------------------
                
                if (k_N_SAMPLES_PER_OPERAND > 1) {
                    subfft_len *= k_N_SAMPLES_PER_OPERAND;
                    n_subfft_len /= k_N_SAMPLES_PER_OPERAND;

                    const Sample* a_real = transform_real;
                    const Sample* a_imag = transform_imag;
                    Sample* b_real = transform_real;
                    Sample* b_imag = transform_imag;

                    for (
                        std::size_t subfft_id = 0; 
                        subfft_id < n_subfft_len;
                        subfft_id++
                    ) {
                        Operand dft_operand_real;
                        Operand dft_operand_imag;
                        Operand a_operand_real = Operand(*a_real); 
                        Operand a_operand_imag = Operand(*a_imag);     

                        Operand b_operand_real = a_operand_real;
                        Operand b_operand_imag = a_operand_imag;

                        a_real++;
                        a_imag++;

                        a_operand_real = Operand(*a_real); 
                        a_operand_imag = Operand(*a_imag);

                        Load(
                            dft_real[1].data(), 
                            dft_operand_real
                        );
                        
                        b_operand_real += a_operand_real * dft_operand_real;
                        b_operand_imag += a_operand_imag * dft_operand_real;

                        a_real++;
                        a_imag++;

                        for (
                            std::size_t dft_basis_id = 2; 
                            dft_basis_id < k_N_SAMPLES_PER_OPERAND;
                            dft_basis_id++
                        ) {
                            a_operand_real = Operand(*a_real); 
                            a_operand_imag = Operand(*a_imag);
                            Load(
                                dft_real[dft_basis_id].data(), 
                                dft_operand_real
                            );
                            Load(
                                dft_imag[dft_basis_id].data(), 
                                dft_operand_imag
                            );
                            
                            b_operand_real 
                                += a_operand_real * dft_operand_real
                                - a_operand_imag * dft_operand_imag;

                            b_operand_imag 
                                += a_operand_imag * dft_operand_real
                                + a_operand_real * dft_operand_imag;
                        
                            a_real++;
                            a_imag++;
                        }

                        Store(b_imag, b_operand_real);
                        Store(b_imag, b_operand_imag);

                        b_real += k_N_SAMPLES_PER_OPERAND;
                        b_imag += k_N_SAMPLES_PER_OPERAND;
                    }
                }

                //--------------------------------------------------------------
                // RADIX-4 PHASE
                //--------------------------------------------------------------
                if (k_N_SAMPLES_PER_OPERAND == 1) {
                    subfft_len = subfft_len << 2;
                    n_subfft_len = n_subfft_len >> 2;

                    auto a0_real = transform_real;
                    auto a1_real = transform_real + 1;
                    auto a2_real = transform_real + 2;
                    auto a3_real = transform_real + 3;
                    auto a0_imag = transform_imag;
                    auto a1_imag = transform_imag + 1;
                    auto a2_imag = transform_imag + 2;
                    auto a3_imag = transform_imag + 3; 

                    Sample b0_real;
                    Sample b1_real;
                    Sample b2_real;
                    Sample b3_real;
                    Sample b0_imag;
                    Sample b1_imag;
                    Sample b2_imag;
                    Sample b3_imag;

                    for (
                        std::size_t subfft_id = 0;
                        subfft_id < n_subfft_len;
                        subfft_id++
                    ) {
                        
                        b0_real = *a0_real + *a1_real;
                        b1_real = *a0_real - *a1_real;
                        b2_real = *a2_real + *a3_real;
                        b3_real = *a2_real - *a3_real;

                        b0_imag = *a0_imag + *a1_imag;
                        b1_imag = *a0_imag - *a1_imag;
                        b2_imag = *a2_imag + *a3_imag;
                        b3_imag = *a2_imag - *a3_imag;  

                        *a0_real = b0_real + b2_real;
                        *a1_real = b1_real + b3_imag;
                        *a2_real = b0_real - b2_real;
                        *a3_real = b1_real - b3_imag;

                        *a0_imag = b0_imag + b2_imag;
                        *a1_imag = b1_imag - b3_real;
                        *a2_imag = b0_imag - b2_imag;
                        *a3_imag = b1_imag + b3_real;  
                        
                        a1_real += 4;
                        a2_real += 4;
                        a3_real += 4;
                        a0_imag += 4;
                        a1_imag += 4;
                        a2_imag += 4;
                        a3_imag += 4;
                    }                   
                }

                for (
                    std::size_t butterfly_id 
                        = size_t(k_N_SAMPLES_PER_OPERAND == 1); 
                    butterfly_id < n_radix_4_butterflies;
                    butterfly_id++
                ) {
                    auto subtwiddle_len = subfft_len;
                    subfft_len = subfft_len << 2;
                    n_subfft_len = n_subfft_len >> 2;
                    
                    auto& twiddle_real = twiddles_real[butterfly_id];
                    auto& twiddle_imag = twiddles_imag[butterfly_id];
                    
                    std::size_t two_subtwiddle_len = 2 * subtwiddle_len;
                    std::size_t three_subtwiddle_len = 3 * subtwiddle_len;

                    auto a0_real = transform_real;
                    auto a1_real = transform_real + subtwiddle_len;
                    auto a2_real = transform_real + two_subtwiddle_len;
                    auto a3_real = transform_real + three_subtwiddle_len;
                    auto a0_imag = transform_imag;
                    auto a1_imag = transform_imag + subtwiddle_len;
                    auto a2_imag = transform_imag + two_subtwiddle_len;
                    auto a3_imag = transform_imag + three_subtwiddle_len; 

                    std::size_t jump = std::size_t(subfft_len - subtwiddle_len);

                    for (
                        std::size_t subfft_id = 0;
                        subfft_id < n_subfft_len;
                        subfft_id++
                    ) {

                        auto tw1_real = twiddle_real[0].data();
                        auto tw2_real = twiddle_real[1].data();
                        auto tw3_real = twiddle_real[2].data();
                        auto tw1_imag = twiddle_imag[0].data();
                        auto tw2_imag = twiddle_imag[1].data();
                        auto tw3_imag = twiddle_imag[2].data();

                        for (
                            std::size_t i = 0; 
                            i < subtwiddle_len;
                            i += k_N_SAMPLES_PER_OPERAND
                        ) {
                            Operand a0_operand_real;
                            Operand a1_operand_real;
                            Operand a2_operand_real;
                            Operand a3_operand_real;
                            Operand a0_operand_imag;
                            Operand a1_operand_imag;
                            Operand a2_operand_imag;
                            Operand a3_operand_imag;

                            Load(a0_real, a0_operand_real);
                            Load(a1_real, a1_operand_real);
                            Load(a2_real, a2_operand_real);
                            Load(a3_real, a3_operand_real);
                            Load(a0_imag, a0_operand_imag);
                            Load(a1_imag, a1_operand_imag);
                            Load(a2_imag, a2_operand_imag);
                            Load(a3_imag, a3_operand_imag);
                            
                            {
                                Operand tw1_operand_real;
                                Operand tw2_operand_real;
                                Operand tw3_operand_real;
                                Operand tw1_operand_imag;
                                Operand tw2_operand_imag;
                                Operand tw3_operand_imag;

                                Load(tw1_real, tw1_operand_real);
                                Load(tw2_real, tw2_operand_real);
                                Load(tw3_real, tw3_operand_real);
                                Load(tw1_imag, tw1_operand_imag);
                                Load(tw2_imag, tw2_operand_imag);
                                Load(tw3_imag, tw3_operand_imag);

                                a1_operand_real 
                                    = a1_operand_real * tw1_operand_real
                                    - a1_operand_imag * tw1_operand_imag;

                                a1_operand_imag 
                                    = a1_operand_imag * tw1_operand_real
                                    + a1_operand_real * tw1_operand_imag;

                                a2_operand_real 
                                    += a2_operand_real * tw2_operand_real
                                    - a2_operand_imag * tw2_operand_imag;

                                a2_operand_imag 
                                    = a2_operand_imag * tw2_operand_real
                                    + a2_operand_real * tw2_operand_imag;

                                a3_operand_real 
                                    = a3_operand_real * tw3_operand_real
                                    - a3_operand_imag * tw3_operand_imag;

                                a3_operand_imag 
                                    = a3_operand_imag * tw3_operand_real
                                    + a3_operand_real * tw3_operand_imag;

                                tw1_real += k_N_SAMPLES_PER_OPERAND;
                                tw2_real += k_N_SAMPLES_PER_OPERAND;
                                tw3_real += k_N_SAMPLES_PER_OPERAND;
                                tw1_imag += k_N_SAMPLES_PER_OPERAND;
                                tw2_imag += k_N_SAMPLES_PER_OPERAND;
                                tw3_imag += k_N_SAMPLES_PER_OPERAND;
                            }
                            {
                                Operand b0_operand_real;
                                Operand b1_operand_real;
                                Operand b2_operand_real;
                                Operand b3_operand_real;
                                Operand b0_operand_imag;
                                Operand b1_operand_imag;
                                Operand b2_operand_imag;
                                Operand b3_operand_imag;

                                b0_operand_real 
                                    = a0_operand_real 
                                    + a1_operand_real;

                                b1_operand_real 
                                    = a0_operand_real 
                                    - a1_operand_real;
                                b2_operand_real 
                                    = a2_operand_real 
                                    + a3_operand_real;

                                b3_operand_real 
                                    = a2_operand_real 
                                    - a3_operand_real;

                                b0_operand_imag 
                                    = a0_operand_imag 
                                    + a1_operand_imag;
                                b1_operand_imag 
                                    = a0_operand_imag 
                                    - a1_operand_imag;
                                b2_operand_imag 
                                    = a2_operand_imag 
                                    + a3_operand_imag;
                                b3_operand_imag 
                                    = a2_operand_imag 
                                    - a3_operand_imag;  

                                a0_operand_real 
                                    = b0_operand_real 
                                    + b2_operand_real;
                                a1_operand_real 
                                    = b1_operand_real 
                                    + b3_operand_imag;
                                a2_operand_real 
                                    = b0_operand_real 
                                    - b2_operand_real;
                                a3_operand_real 
                                    = b1_operand_real 
                                    - b3_operand_imag;

                                a0_operand_imag 
                                    = b0_operand_imag 
                                    + b2_operand_imag;
                                a1_operand_imag 
                                    = b1_operand_imag 
                                    - b3_operand_real;
                                a2_operand_imag 
                                    = b0_operand_imag 
                                    - b2_operand_imag;
                                a3_operand_imag 
                                    = b1_operand_imag 
                                    + b3_operand_real;  
                            }

                            Store(a0_real, a0_operand_real);
                            Store(a1_real, a1_operand_real);
                            Store(a2_real, a2_operand_real);
                            Store(a3_real, a3_operand_real);
                            Store(a0_imag, a0_operand_imag);
                            Store(a1_imag, a1_operand_imag);
                            Store(a2_imag, a2_operand_imag);
                            Store(a3_imag, a3_operand_imag);

                            a0_real += k_N_SAMPLES_PER_OPERAND;
                            a1_real += k_N_SAMPLES_PER_OPERAND;
                            a2_real += k_N_SAMPLES_PER_OPERAND;
                            a3_real += k_N_SAMPLES_PER_OPERAND;
                            a0_imag += k_N_SAMPLES_PER_OPERAND;
                            a1_imag += k_N_SAMPLES_PER_OPERAND;
                            a2_imag += k_N_SAMPLES_PER_OPERAND;
                            a3_imag += k_N_SAMPLES_PER_OPERAND;
                        }
                        a0_real += jump;
                        a1_real += jump;
                        a2_real += jump;
                        a3_real += jump;
                        a0_imag += jump;
                        a1_imag += jump;
                        a2_imag += jump;
                        a3_imag += jump;
                    } 
                }

                // -------------------------------------------------------------
                // RADIX-2 PHASE
                // -------------------------------------------------------------
                if (using_final_radix_2_butterfly) {
                    auto subtwiddle_len = subfft_len;
                    n_subfft_len = n_subfft_len >> 1;
                    
                    std::size_t last_twiddle_id 
                        = std::size_t(twiddles_real.size() - 1);

                    auto& twiddle_real = twiddles_real[last_twiddle_id];
                    auto& twiddle_imag = twiddles_imag[last_twiddle_id];

                    auto a0_real = transform_real;
                    auto a1_real = transform_real + subtwiddle_len;
                    auto a0_imag = transform_imag;
                    auto a1_imag = transform_imag + subtwiddle_len;

                    for (
                        std::size_t subfft_id = 0;
                        subfft_id < n_subfft_len;
                        subfft_id++
                    ) {

                        auto tw1_real = twiddle_real[0].data();
                        auto tw1_imag = twiddle_imag[0].data();

                        for (
                            std::size_t i = 0; 
                            i < subtwiddle_len;
                            i += k_N_SAMPLES_PER_OPERAND
                        ) {
                            Operand a0_operand_real;
                            Operand a1_operand_real;
                            Operand a0_operand_imag;
                            Operand a1_operand_imag;

                            Load(a0_real, a0_operand_real);
                            Load(a1_real, a1_operand_real);
                            
                            Load(a0_imag, a0_operand_imag);
                            Load(a1_imag, a1_operand_imag);

                            {
                                Operand tw1_operand_real;
                                Operand tw1_operand_imag;

                                Load(tw1_real, tw1_operand_real);
                                Load(tw1_imag, tw1_operand_imag);

                                a1_operand_real 
                                    = a1_operand_real * tw1_operand_real
                                    - a1_operand_imag * tw1_operand_imag;

                                a1_operand_imag 
                                    = a1_operand_imag * tw1_operand_real
                                    + a1_operand_real * tw1_operand_imag;

                                tw1_real += k_N_SAMPLES_PER_OPERAND;
                                tw1_imag += k_N_SAMPLES_PER_OPERAND;
                            }
                            {
                                Operand b0_operand_real;
                                Operand b1_operand_real;
                                Operand b0_operand_imag;
                                Operand b1_operand_imag;

                                b0_operand_real 
                                    = a0_operand_real 
                                    + a1_operand_real;

                                b1_operand_real 
                                    = a0_operand_real 
                                    - a1_operand_real;

                                b0_operand_imag 
                                    = a0_operand_imag 
                                    + a1_operand_imag;
                                b1_operand_imag 
                                    = a0_operand_imag 
                                    - a1_operand_imag;

                                Store(a0_real, b0_operand_real);
                                Store(a1_real, b1_operand_real);
                                Store(a0_imag, b0_operand_imag);
                                Store(a1_imag, b1_operand_imag);
                            }
                            
                            a0_real += k_N_SAMPLES_PER_OPERAND;
                            a1_real += k_N_SAMPLES_PER_OPERAND;
                            a0_imag += k_N_SAMPLES_PER_OPERAND;
                            a1_imag += k_N_SAMPLES_PER_OPERAND;
                        }
                    } 
                }

            }

        private:

            static std::vector<std::vector<std::vector<Sample>>> Twiddles(
                size_t n_radix_4_butterflies,
                bool using_final_radix_2_butterfly,
                const std::function<Sample(Sample)>& trig_fn,
                int multiplier
            ) {
                std::size_t initial_subfft_len = k_N_SAMPLES_PER_OPERAND;

                std::vector<std::vector<std::vector<Sample>>> twiddles;

                std::size_t n_twiddle_vectors 
                    = n_radix_4_butterflies 
                    + std::size_t(using_final_radix_2_butterfly);

                twiddles.resize(n_twiddle_vectors);

                auto subtwiddle_len = initial_subfft_len;

                for (
                    std::size_t butterfly_id = 0; 
                    butterfly_id < n_radix_4_butterflies;
                    butterfly_id++
                ) {
                    auto& twiddle = twiddles[butterfly_id];
                    
                    twiddle.resize(3);
                    twiddle[0].resize(subtwiddle_len);
                    twiddle[1].resize(subtwiddle_len);
                    twiddle[2].resize(subtwiddle_len);
                
                    for (
                        std::size_t factor_id = 0; 
                        factor_id < subtwiddle_len;
                        factor_id++
                    ) {
                        twiddle[0][factor_id] 
                            = multiplier
                            * trig_fn(Pi() * factor_id / subtwiddle_len);

                        twiddle[1][factor_id] 
                            = multiplier 
                            * trig_fn(Pi() * factor_id * 0.5 / subtwiddle_len);

                        twiddle[2][factor_id] 
                            = multiplier 
                            * trig_fn(Pi() * factor_id * 1.5 / subtwiddle_len);
                    }
                    subtwiddle_len *= 4;
                }

                if (using_final_radix_2_butterfly) {
                    std::size_t last_butterfly_id = twiddles.size() - 1;
                    auto& twiddle = twiddles[last_butterfly_id];
                    twiddle.resize(1);
                    twiddle[0].resize(subtwiddle_len);
                    for (
                        std::size_t factor_id = 0; 
                        factor_id < subtwiddle_len;
                        factor_id++
                    ) {
                        twiddle[0][factor_id] 
                            = multiplier 
                            * trig_fn(Pi() * factor_id * 0.5 / subtwiddle_len);
                    }
                }

                return twiddles;
            }

            static std::vector<std::vector<Sample>> Dft(
                const std::vector<std::size_t>& the_scrambled_indexes_dft,
                const std::function<Sample(Sample)>& trig_fn,
                int multiplier
            ) {
                std::vector<std::vector<Sample>> dft;

                dft.resize(k_N_SAMPLES_PER_OPERAND);

                for (
                    std::size_t basis_id = 0; 
                    basis_id < k_N_SAMPLES_PER_OPERAND;
                    basis_id++
                ) {
                    auto& dft_basis = dft[basis_id];
                    dft_basis.resize(k_N_SAMPLES_PER_OPERAND);
                    auto scrambled_basis_index 
                        = the_scrambled_indexes_dft[basis_id];

                    for (
                        std::size_t factor_id = 0; 
                        factor_id < k_N_SAMPLES_PER_OPERAND;
                        factor_id++
                    ) {
                        dft_basis[factor_id] 
                            = multiplier 
                            * trig_fn(
                                Pi() 
                                * 2 
                                * factor_id 
                                * scrambled_basis_index 
                                / k_N_SAMPLES_PER_OPERAND); 
                    }
                }

                return dft;
            }
    };
}}

#endif