#ifndef GOLDENROCEKEFELLER_AFFT_FFT_COMPLEX_HPP
#define GOLDENROCEKEFELLER_AFFT_FFT_COMPLEX_HPP

#include <cstddef>
#include <vector>
#include <functional>
#include <algorithm>
#include <cstring>
#include <cstdint>


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

    template <typename Value>
    Value Rem2(Value x){
        return x - std::intmax_t(x/2) * 2;
    }

    std::vector<std::size_t> BitReversedIndexes(std::size_t n_indexes) {
        std::vector<std::size_t> bit_reversed_indexes(n_indexes);
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
            bit_reversed_indexes[id] = bit_reversed_id;
        }
        return bit_reversed_indexes;
    }

    template<
        typename SampleSpec, 
        typename OperandSpec
    >
    class FftComplex {
    
        using Sample = typename SampleSpec::Value;
        using Operand = typename OperandSpec::Value;

        static constexpr size_t k_N_SAMPLES_PER_OPERAND 
            = sizeof(Operand) / sizeof(Sample);

        static inline Sample Pi(){
            return SampleSpec::Pi();
        }

        // Use to determine COBRA buffer size. 2^PGFFT_BRC_THRESH should fit in 
        // cache.
        static inline std::size_t PGFFT_BRC_Q() {
            return 5;
        }

        // After 2^PGFFT_BRC_THRESH, we will use COBRA bit reversal strategy
        static inline std::size_t PGFFT_BRC_THRESH() {
            return 11;
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
            std::size_t transform_len;

            std::size_t n_radix_2_butterflies;
            std::size_t n_radix_4_butterflies;
            bool using_final_radix_2_butterfly;

            std::vector<std::vector<std::vector<Sample>>> twiddles_real;
            std::vector<std::vector<std::vector<Sample>>> twiddles_imag;

            std::size_t log_len;
            std::size_t log_reversal_len;

            std::vector<std::size_t> bit_reversed_indexes;
            std::vector<std::size_t> bit_reversed_indexes_2;
            std::vector<std::size_t> bit_reversed_indexes_dft;

            std::vector<std::vector<Sample>> dft_real;
            std::vector<std::vector<Sample>> dft_imag;
            std::vector<std::vector<Sample>> dft_real_transpose;
            std::vector<std::vector<Sample>> dft_imag_transpose;

            mutable std::vector<Sample> shuffle_work_real;
            mutable std::vector<Sample> shuffle_work_imag;
            mutable std::vector<Sample> dif_work_real;
            mutable std::vector<Sample> dif_work_imag;
        
        public:
        
            FftComplex(std::size_t transform_len):
                transform_len(CheckedTransformLen(transform_len)),
                n_radix_2_butterflies(
                    IntLog2(transform_len / k_N_SAMPLES_PER_OPERAND)
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
                log_len(IntLog2(transform_len)),
                log_reversal_len(
                    log_len * std::size_t(log_len < PGFFT_BRC_THRESH())
                    + (log_len - 2 * PGFFT_BRC_Q()) 
                    * std::size_t(log_len >= PGFFT_BRC_THRESH())
                ),
                bit_reversed_indexes(BitReversedIndexes(1 << log_reversal_len)),
                bit_reversed_indexes_2(
                    BitReversedIndexes(
                        1L << std::min(PGFFT_BRC_Q(), log_len)
                    )
                ),
                bit_reversed_indexes_dft(
                    BitReversedIndexes(k_N_SAMPLES_PER_OPERAND)
                ),
                dft_real(Dft(bit_reversed_indexes_dft, Cos, 1)),
                dft_imag(Dft(bit_reversed_indexes_dft, Sin, -1)),
                dft_real_transpose(Transpose(dft_real)),
                dft_imag_transpose(Transpose(dft_imag)),
                shuffle_work_real(transform_len),
                shuffle_work_imag(transform_len),
                dif_work_real(transform_len),
                dif_work_imag(transform_len)
            {
                    // Nothing else to do.
            }

            void ProcessDit(
                Sample* transform_real, 
                Sample* transform_imag,
                Sample* signal_real, 
                Sample* signal_imag,
                bool calculating_inverse = false,
                bool rescaling = false,
                bool processing_for_convolution = false
            ) const {
                if (calculating_inverse) {
                    std::swap(signal_real, signal_imag);
                    std::swap(transform_real, transform_imag);
                }

                std::size_t subfft_len = 1;
                std::size_t n_subffts = transform_len;

                //--------------------------------------------------------------
                // BIT-REVERESED PERMUTE SIGNAL
                //--------------------------------------------------------------

                if (processing_for_convolution == false) {
                    if (log_len < PGFFT_BRC_THRESH())
                    {
                        for (
                            std::size_t new_index = 0;
                            new_index < transform_len; 
                            new_index++
                        ) {
                            auto old_index = bit_reversed_indexes[new_index];

                            transform_real[new_index] = signal_real[old_index];
                            transform_imag[new_index] = signal_imag[old_index];
                        }
                    }
                    else {
                        // Adapted from PGFFT
                        Sample* work_real = shuffle_work_real.data();
                        Sample* work_imag = shuffle_work_imag.data();

                        const std::size_t* rev_flex = bit_reversed_indexes.data();
                        const std::size_t* rev_fixed = bit_reversed_indexes_2.data();

                        const auto A_real = signal_real;
                        const auto A_imag = signal_imag;
                        auto B_real = transform_real;
                        auto B_imag = transform_imag;

                        std::size_t q = PGFFT_BRC_Q();
                        
                        for (
                            std::size_t b = 0; 
                            b < bit_reversed_indexes.size(); 
                            b++
                        ) {
                            std::size_t b1 = rev_flex[b]; 
                            for (
                                std::size_t a = 0; 
                                a < bit_reversed_indexes_2.size(); 
                                a++
                            ) {
                                std::size_t a1 = rev_fixed[a]; 

                                Sample* T_p_real = work_real + (a1 << q);
                                Sample* T_p_imag = work_imag + (a1 << q);

                                const Sample* A_p_real 
                                    = A_real 
                                    + (a << (log_reversal_len +q)) 
                                    + (b << q);
                                
                                const Sample* A_p_imag 
                                    = A_imag 
                                    + (a << (log_reversal_len +q)) 
                                    + (b << q);
                                    
                                for (
                                    long c = 0; 
                                    c < bit_reversed_indexes_2.size(); 
                                    c+=k_N_SAMPLES_PER_OPERAND
                                ) {
                                    Operand store_real;
                                    Operand store_imag;
                                    Load(A_p_real + c, store_real);
                                    Load(A_p_imag + c, store_imag);
                                    Store(T_p_real + c, store_real);
                                    Store(T_p_imag + c, store_imag);
                                }
                            }

                            for (
                                long c = 0; 
                                c < bit_reversed_indexes_2.size(); 
                                c++
                            ) {
                                long c1 = rev_fixed[c];

                                Sample* B_p_real 
                                    = B_real 
                                    + (c1 << (log_reversal_len +q)) 
                                    + (b1 << q);
                                
                                Sample* B_p_imag 
                                    = B_imag 
                                    + (c1 << (log_reversal_len +q)) 
                                    + (b1 << q);

                                Sample* T_p_real = work_real + c;
                                Sample* T_p_imag = work_imag + c;
                                
                                for (
                                    long a1 = 0; 
                                    a1 < bit_reversed_indexes_2.size(); 
                                    a1+=1
                                ) { 
                                    B_p_real[a1] = T_p_real[a1 << q];
                                    B_p_imag[a1] = T_p_imag[a1 << q];
                                }
                            }
                        }
                    }
                }
                else {
                    std::memcpy(
                        transform_real, 
                        signal_real, 
                        sizeof(Sample) * transform_len);
                    std::memcpy(
                        transform_imag, 
                        signal_imag, 
                        sizeof(Sample) * transform_len);
                }
 
                //--------------------------------------------------------------
                // DFT PHASE
                //--------------------------------------------------------------
                
                if (k_N_SAMPLES_PER_OPERAND > 1) {
                    subfft_len *= k_N_SAMPLES_PER_OPERAND;
                    n_subffts /= k_N_SAMPLES_PER_OPERAND;

                    const Sample* a_real = transform_real;
                    const Sample* a_imag = transform_imag;
                    Sample* b_real = transform_real;
                    Sample* b_imag = transform_imag;

                    for (
                        std::size_t subfft_id = 0; 
                        subfft_id < n_subffts;
                        subfft_id++
                    ) {
                        auto a_real_1 = b_real;
                        auto a_imag_1 = b_imag;
                        auto a_real_2 = b_real + 1;
                        auto a_imag_2 = b_imag + 1;

                        Operand dft_operand_real_1;
                        Operand dft_operand_imag_1;
                        Operand dft_operand_real_2;
                        Operand dft_operand_imag_2;
                        Operand a_operand_real_1 = Operand(*a_real_1); 
                        Operand a_operand_imag_1 = Operand(*a_imag_1);   
                        Operand a_operand_real_2 = Operand(*a_real_2); 
                        Operand a_operand_imag_2 = Operand(*a_imag_2);

                        Operand b_operand_real_1 = a_operand_real_1;
                        Operand b_operand_imag_1 = a_operand_imag_1;

                        Load(
                            dft_real[1].data(), 
                            dft_operand_real_2
                        );
                        
                        Operand b_operand_real_2 
                            = a_operand_real_2 * dft_operand_real_2;
                        
                        Operand b_operand_imag_2 
                            = a_operand_imag_2 * dft_operand_real_2;

                        a_real_1 += 2;
                        a_imag_1 += 2;
                        a_real_2 += 2;
                        a_imag_2 += 2;

                        for (
                            std::size_t dft_basis_id = 2; 
                            dft_basis_id < k_N_SAMPLES_PER_OPERAND;
                            dft_basis_id += 2
                        ) {
                            Operand a_operand_real_1 = Operand(*a_real_1); 
                            Operand a_operand_imag_1 = Operand(*a_imag_1);   
                            Operand a_operand_real_2 = Operand(*a_real_2); 
                            Operand a_operand_imag_2 = Operand(*a_imag_2);

                            Load(
                                dft_real[dft_basis_id].data(), 
                                dft_operand_real_1
                            );
                            Load(
                                dft_imag[dft_basis_id].data(), 
                                dft_operand_imag_1
                            );

                            Load(
                                dft_real[dft_basis_id+1].data(), 
                                dft_operand_real_2
                            );
                            Load(
                                dft_imag[dft_basis_id+1].data(), 
                                dft_operand_imag_2
                            );
                            
                            b_operand_real_1 
                                += a_operand_real_1 * dft_operand_real_1
                                - a_operand_imag_1 * dft_operand_imag_1;

                            b_operand_imag_1 
                                += a_operand_imag_1 * dft_operand_real_1
                                + a_operand_real_1 * dft_operand_imag_1;

                            b_operand_real_2 
                                += a_operand_real_2 * dft_operand_real_2
                                - a_operand_imag_2 * dft_operand_imag_2;

                            b_operand_imag_2 
                                += a_operand_imag_2 * dft_operand_real_2
                                + a_operand_real_2 * dft_operand_imag_2;
                        
                            a_real_1 += 2;
                            a_imag_1 += 2;
                            a_real_2 += 2;
                            a_imag_2 += 2;
                        }

                        Store(b_real, b_operand_real_1 + b_operand_real_2);
                        Store(b_imag, b_operand_imag_1 + b_operand_imag_2);

                        b_real += k_N_SAMPLES_PER_OPERAND;
                        b_imag += k_N_SAMPLES_PER_OPERAND;
                    }
                }
                // --------------------------------------------------------------
                // RADIX-4 PHASE
                // --------------------------------------------------------------

                if (k_N_SAMPLES_PER_OPERAND == 1) {
                    subfft_len = subfft_len << 2;
                    n_subffts = n_subffts >> 2;

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
                        subfft_id < n_subffts;
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
                        
                        a0_real += 4;
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
                    n_subffts = n_subffts >> 2;
                    
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

                    auto tw1_real_start = twiddle_real[0].data();
                    auto tw2_real_start = twiddle_real[1].data();
                    auto tw3_real_start = twiddle_real[2].data();
                    auto tw1_imag_start = twiddle_imag[0].data();
                    auto tw2_imag_start = twiddle_imag[1].data();
                    auto tw3_imag_start = twiddle_imag[2].data();

                    for (
                        std::size_t subfft_id = 0;
                        subfft_id < n_subffts;
                        subfft_id++
                    ) {

                        auto tw1_real = tw1_real_start;
                        auto tw2_real = tw2_real_start;
                        auto tw3_real = tw3_real_start;
                        auto tw1_imag = tw1_imag_start;
                        auto tw2_imag = tw2_imag_start;
                        auto tw3_imag = tw3_imag_start;

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
                                Operand store1;
                                Operand store2;
                                Operand store3;

                                Load(tw1_real, tw1_operand_real);
                                Load(tw2_real, tw2_operand_real);
                                Load(tw3_real, tw3_operand_real);
                                Load(tw1_imag, tw1_operand_imag);
                                Load(tw2_imag, tw2_operand_imag);
                                Load(tw3_imag, tw3_operand_imag);

                                store1 
                                    = a1_operand_real * tw1_operand_real
                                    - a1_operand_imag * tw1_operand_imag;

                                a1_operand_imag 
                                    = a1_operand_imag * tw1_operand_real
                                    + a1_operand_real * tw1_operand_imag;

                                store2 
                                    = a2_operand_real * tw2_operand_real
                                    - a2_operand_imag * tw2_operand_imag;

                                a2_operand_imag 
                                    = a2_operand_imag * tw2_operand_real
                                    + a2_operand_real * tw2_operand_imag;

                                store3 
                                    = a3_operand_real * tw3_operand_real
                                    - a3_operand_imag * tw3_operand_imag;

                                a3_operand_imag 
                                    = a3_operand_imag * tw3_operand_real
                                    + a3_operand_real * tw3_operand_imag;

                                a1_operand_real = store1;
                                a2_operand_real = store2;
                                a3_operand_real = store3;

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
                    
                    std::size_t last_twiddle_id 
                        = std::size_t(twiddles_real.size() - 1);

                    auto& twiddle_real = twiddles_real[last_twiddle_id];
                    auto& twiddle_imag = twiddles_imag[last_twiddle_id];

                    auto a0_real = transform_real;
                    auto a1_real = transform_real + subtwiddle_len;
                    auto a0_imag = transform_imag;
                    auto a1_imag = transform_imag + subtwiddle_len;

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
                            Operand store;

                            Load(tw1_real, tw1_operand_real);
                            Load(tw1_imag, tw1_operand_imag);

                            store 
                                = a1_operand_real * tw1_operand_real
                                - a1_operand_imag * tw1_operand_imag;

                            a1_operand_imag 
                                = a1_operand_imag * tw1_operand_real
                                + a1_operand_real * tw1_operand_imag;

                            a1_operand_real = store;

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

                // -------------------------------------------------------------
                // Rescaling
                // -------------------------------------------------------------

                if (rescaling) {
                    Operand scale_factor(Sample(1.) /  Sample(transform_len));

                    auto t_real = transform_real;
                    auto t_imag = transform_imag;
                    
                    for (
                        std::size_t i = 0;
                        i < transform_len; 
                        i += k_N_SAMPLES_PER_OPERAND
                    ) {
                        Operand t_operand_real;
                        Operand t_operand_imag;

                        Load(t_real, t_operand_real);
                        Load(t_imag, t_operand_imag);

                        t_operand_real *= scale_factor;
                        t_operand_imag *= scale_factor;

                        Store(t_real, t_operand_real);
                        Store(t_imag, t_operand_imag);

                        t_real += k_N_SAMPLES_PER_OPERAND;
                        t_imag += k_N_SAMPLES_PER_OPERAND;
                    }
                }
            }

            void ProcessDif(
                Sample* transform_real, 
                Sample* transform_imag,
                Sample* signal_real, 
                Sample* signal_imag,
                bool calculating_inverse = false,
                bool rescaling = false,
                bool processing_for_convolution = false
            ) const {
                // TODO Actually implement Dif for fast convolution.
                // ProcessDit(
                //     transform_real, 
                //     transform_imag,
                //     signal_real, 
                //     signal_imag,
                //     calculating_inverse,
                //     rescaling,
                //     processing_for_convolution
                // );

                std::memcpy(
                    transform_real, 
                    signal_real, 
                    sizeof(Sample) * transform_len);
                std::memcpy(
                    transform_imag, 
                    signal_imag, 
                    sizeof(Sample) * transform_len);

                if (calculating_inverse) {
                    std::swap(signal_real, signal_imag);
                    std::swap(transform_real, transform_imag);
                }

                // Copy signal and move it to transger

                std::size_t subfft_len = transform_len;
                std::size_t n_subffts = 1;
                std::size_t twiddle_id = std::size_t(twiddles_real.size() - 1);

                // -------------------------------------------------------------
                // RADIX-2 PHASE
                // -------------------------------------------------------------
                if (using_final_radix_2_butterfly) {
                    auto subtwiddle_len = subfft_len >> 1;

                    auto& twiddle_real = twiddles_real[twiddle_id];
                    auto& twiddle_imag = twiddles_imag[twiddle_id];

                    auto a0_real = transform_real;
                    auto a1_real = transform_real + subtwiddle_len;
                    auto a0_imag = transform_imag;
                    auto a1_imag = transform_imag + subtwiddle_len;

                    auto tw1_real = twiddle_real[0].data();
                    auto tw1_imag = twiddle_imag[0].data();

                    for (
                        std::size_t i = 0; 
                        i < subtwiddle_len;
                        i += k_N_SAMPLES_PER_OPERAND
                    ) {
                        
                        Operand a1_operand_real;
                        Operand a1_operand_imag;
                        Operand b1_operand_real;
                        Operand b1_operand_imag;

                        Load(a1_real, a1_operand_real);
                        Load(a1_imag, a1_operand_imag);
                        {
                            Operand a0_operand_real;
                            Operand a0_operand_imag;
                            Operand b0_operand_real;
                            Operand b0_operand_imag;

                            Load(a0_real, a0_operand_real);
                            Load(a0_imag, a0_operand_imag);

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
                            Store(a0_imag, b0_operand_imag);
                        }
                        {
                            Operand tw1_operand_real;
                            Operand tw1_operand_imag;
                            Operand store;

                            Load(tw1_real, tw1_operand_real);
                            Load(tw1_imag, tw1_operand_imag);

                            store 
                                = b1_operand_real * tw1_operand_real
                                - b1_operand_imag * tw1_operand_imag;

                            b1_operand_imag 
                                = b1_operand_imag * tw1_operand_real
                                + b1_operand_real * tw1_operand_imag;

                            b1_operand_real = store;

                            Store(a1_real, b1_operand_real);
                            Store(a1_imag, b1_operand_imag);

                            tw1_real += k_N_SAMPLES_PER_OPERAND;
                            tw1_imag += k_N_SAMPLES_PER_OPERAND;
                        }
                        
                        a0_real += k_N_SAMPLES_PER_OPERAND;
                        a1_real += k_N_SAMPLES_PER_OPERAND;
                        a0_imag += k_N_SAMPLES_PER_OPERAND;
                        a1_imag += k_N_SAMPLES_PER_OPERAND;
                    }       
                    subfft_len = subfft_len >> 1;
                    n_subffts = n_subffts << 1;
                    twiddle_id -= 1;
                }

                // --------------------------------------------------------------
                // RADIX-4 PHASE
                // --------------------------------------------------------------
                for (
                    std::size_t butterfly_id 
                        = size_t(k_N_SAMPLES_PER_OPERAND == 1); 
                    butterfly_id < n_radix_4_butterflies;
                    butterfly_id++
                ) {
                    auto subtwiddle_len = subfft_len >> 2;
                    
                    auto& twiddle_real = twiddles_real[twiddle_id];
                    auto& twiddle_imag = twiddles_imag[twiddle_id];
                    
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

                    auto tw1_real_start = twiddle_real[0].data();
                    auto tw2_real_start = twiddle_real[1].data();
                    auto tw3_real_start = twiddle_real[2].data();
                    auto tw1_imag_start = twiddle_imag[0].data();
                    auto tw2_imag_start = twiddle_imag[1].data();
                    auto tw3_imag_start = twiddle_imag[2].data();
                    for (
                        std::size_t subfft_id = 0;
                        subfft_id < n_subffts;
                        subfft_id++
                    ) {

                        auto tw1_real = tw1_real_start;
                        auto tw2_real = tw2_real_start;
                        auto tw3_real = tw3_real_start;
                        auto tw1_imag = tw1_imag_start;
                        auto tw2_imag = tw2_imag_start;
                        auto tw3_imag = tw3_imag_start;

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
                                    + a2_operand_real;
                                b1_operand_real 
                                    = a1_operand_real 
                                    + a3_operand_real;
                                b2_operand_real 
                                    = a0_operand_real 
                                    - a2_operand_real;
                                b3_operand_real 
                                    = a1_operand_imag 
                                    - a3_operand_imag;

                                b0_operand_imag 
                                    = a0_operand_imag 
                                    + a2_operand_imag;
                                b1_operand_imag 
                                    = a1_operand_imag 
                                    + a3_operand_imag;
                                b2_operand_imag 
                                    = a0_operand_imag 
                                    - a2_operand_imag;
                                b3_operand_imag 
                                    = a3_operand_real 
                                    - a1_operand_real;  

                                a0_operand_real 
                                    = b0_operand_real 
                                    + b1_operand_real;

                                a1_operand_real 
                                    = b0_operand_real 
                                    - b1_operand_real;

                                a2_operand_real 
                                    = b2_operand_real 
                                    + b3_operand_real;

                                a3_operand_real 
                                    = b2_operand_real 
                                    - b3_operand_real;

                                a0_operand_imag 
                                    = b0_operand_imag 
                                    + b1_operand_imag;

                                a1_operand_imag 
                                    = b0_operand_imag 
                                    - b1_operand_imag;

                                a2_operand_imag 
                                    = b2_operand_imag 
                                    + b3_operand_imag;

                                a3_operand_imag 
                                    = b2_operand_imag 
                                    - b3_operand_imag;  

                            }
                            {
                                Operand tw1_operand_real;
                                Operand tw2_operand_real;
                                Operand tw3_operand_real;
                                Operand tw1_operand_imag;
                                Operand tw2_operand_imag;
                                Operand tw3_operand_imag;
                                Operand store1;
                                Operand store2;
                                Operand store3;

                                Load(tw1_real, tw1_operand_real);
                                Load(tw2_real, tw2_operand_real);
                                Load(tw3_real, tw3_operand_real);
                                Load(tw1_imag, tw1_operand_imag);
                                Load(tw2_imag, tw2_operand_imag);
                                Load(tw3_imag, tw3_operand_imag);

                                store1 
                                    = a1_operand_real * tw1_operand_real
                                    - a1_operand_imag * tw1_operand_imag;

                                a1_operand_imag 
                                    = a1_operand_imag * tw1_operand_real
                                    + a1_operand_real * tw1_operand_imag;

                                store2 
                                    = a2_operand_real * tw2_operand_real
                                    - a2_operand_imag * tw2_operand_imag;

                                a2_operand_imag 
                                    = a2_operand_imag * tw2_operand_real
                                    + a2_operand_real * tw2_operand_imag;

                                store3 
                                    = a3_operand_real * tw3_operand_real
                                    - a3_operand_imag * tw3_operand_imag;

                                a3_operand_imag 
                                    = a3_operand_imag * tw3_operand_real
                                    + a3_operand_real * tw3_operand_imag;

                                a1_operand_real = store1;
                                a2_operand_real = store2;
                                a3_operand_real = store3;

                                tw1_real += k_N_SAMPLES_PER_OPERAND;
                                tw2_real += k_N_SAMPLES_PER_OPERAND;
                                tw3_real += k_N_SAMPLES_PER_OPERAND;
                                tw1_imag += k_N_SAMPLES_PER_OPERAND;
                                tw2_imag += k_N_SAMPLES_PER_OPERAND;
                                tw3_imag += k_N_SAMPLES_PER_OPERAND;
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
                    
                    subfft_len = subfft_len >> 2;
                    n_subffts = n_subffts << 2;           
                    twiddle_id -= 1;     
                }

                if (k_N_SAMPLES_PER_OPERAND == 1) {

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
                        subfft_id < n_subffts;
                        subfft_id++
                    ) {
                        
                        b0_real = *a0_real + *a2_real;
                        b1_real = *a1_real + *a3_real;
                        b2_real = *a0_real - *a2_real;
                        b3_real = *a1_imag - *a3_imag;

                        b0_imag = *a0_imag + *a2_imag;
                        b1_imag = *a1_imag + *a3_imag;
                        b2_imag = *a0_imag - *a2_imag;
                        b3_imag = *a3_real - *a1_real;  

                        *a0_real = b0_real + b1_real;
                        *a1_real = b0_real - b1_real;
                        *a2_real = b2_real + b3_real;
                        *a3_real = b2_real - b3_real;

                        *a0_imag = b0_imag + b1_imag;
                        *a1_imag = b0_imag - b1_imag;
                        *a2_imag = b2_imag + b3_imag;
                        *a3_imag = b2_imag - b3_imag;  
                        
                        a0_real += 4;
                        a1_real += 4;
                        a2_real += 4;
                        a3_real += 4;
                        a0_imag += 4;
                        a1_imag += 4;
                        a2_imag += 4;
                        a3_imag += 4;
                    }        
                    
                    subfft_len = subfft_len >> 2;
                    n_subffts = n_subffts << 2;
                    twiddle_id -= 1;           
                }
                //--------------------------------------------------------------
                // DFT PHASE
                //--------------------------------------------------------------
                
                if (k_N_SAMPLES_PER_OPERAND > 1) {
                    size_t half_dft_len = k_N_SAMPLES_PER_OPERAND >> 1;

                    const Sample* a_real = transform_real;
                    const Sample* a_imag = transform_imag;
                    Sample* b_real = transform_real;
                    Sample* b_imag = transform_imag;

                    for (
                        std::size_t subfft_id = 0; 
                        subfft_id < n_subffts;
                        subfft_id++
                    ) {
                        auto a_real_1 = b_real;
                        auto a_imag_1 = b_imag;
                        auto a_real_2 = b_real + half_dft_len;
                        auto a_imag_2 = b_imag + half_dft_len;

                        Operand dft_operand_real_1;
                        Operand dft_operand_imag_1;
                        Operand dft_operand_real_2;
                        Operand dft_operand_imag_2;
                        Operand a_operand_real_1 = Operand(*a_real_1); 
                        Operand a_operand_imag_1 = Operand(*a_imag_1);   
                        Operand a_operand_real_2 = Operand(*a_real_2); 
                        Operand a_operand_imag_2 = Operand(*a_imag_2);

                        Operand b_operand_real_1 = a_operand_real_1;
                        Operand b_operand_imag_1 = a_operand_imag_1;

                        Load(
                            dft_real_transpose[half_dft_len].data(), 
                            dft_operand_real_2
                        );
                        
                        Operand b_operand_real_2 
                            = a_operand_real_2 * dft_operand_real_2;
                        
                        Operand b_operand_imag_2 
                            = a_operand_imag_2 * dft_operand_real_2;

                        a_real_1 += 1;
                        a_imag_1 += 1;
                        a_real_2 += 1;
                        a_imag_2 += 1;

                        for (
                            std::size_t dft_basis_id = 1; 
                            dft_basis_id < half_dft_len;
                            dft_basis_id += 1
                        ) {
                            Operand a_operand_real_1 = Operand(*a_real_1); 
                            Operand a_operand_imag_1 = Operand(*a_imag_1);   
                            Operand a_operand_real_2 = Operand(*a_real_2); 
                            Operand a_operand_imag_2 = Operand(*a_imag_2);

                            Load(
                                dft_real_transpose[dft_basis_id].data(), 
                                dft_operand_real_1
                            );
                            Load(
                                dft_imag_transpose[dft_basis_id].data(), 
                                dft_operand_imag_1
                            );

                            Load(
                                dft_real_transpose[dft_basis_id + half_dft_len].data(), 
                                dft_operand_real_2
                            );
                            Load(
                                dft_imag_transpose[dft_basis_id + half_dft_len].data(), 
                                dft_operand_imag_2
                            );
                            
                            b_operand_real_1 
                                += a_operand_real_1 * dft_operand_real_1
                                - a_operand_imag_1 * dft_operand_imag_1;

                            b_operand_imag_1 
                                += a_operand_imag_1 * dft_operand_real_1
                                + a_operand_real_1 * dft_operand_imag_1;

                            b_operand_real_2 
                                += a_operand_real_2 * dft_operand_real_2
                                - a_operand_imag_2 * dft_operand_imag_2;

                            b_operand_imag_2 
                                += a_operand_imag_2 * dft_operand_real_2
                                + a_operand_real_2 * dft_operand_imag_2;
                        
                            a_real_1 += 1;
                            a_imag_1 += 1;
                            a_real_2 += 1;
                            a_imag_2 += 1;
                        }

                        Store(b_real, b_operand_real_1 + b_operand_real_2);
                        Store(b_imag, b_operand_imag_1 + b_operand_imag_2);

                        b_real += k_N_SAMPLES_PER_OPERAND;
                        b_imag += k_N_SAMPLES_PER_OPERAND;
                    }
                }

                //--------------------------------------------------------------
                // BIT-REVERESED PERMUTE SIGNAL
                //--------------------------------------------------------------

                if (processing_for_convolution == false) {
                    if (log_len < PGFFT_BRC_THRESH())
                    {
                        Sample* work_real = shuffle_work_real.data();
                        Sample* work_imag = shuffle_work_imag.data();

                        std::memcpy(
                            work_real, 
                            transform_real, 
                            sizeof(Sample) * transform_len);
                        std::memcpy(
                            work_imag, 
                            transform_imag, 
                            sizeof(Sample) * transform_len);

                        for (
                            std::size_t new_index = 0;
                            new_index < transform_len; 
                            new_index++
                        ) {
                            auto old_index = bit_reversed_indexes[new_index];

                            transform_real[new_index] 
                                = shuffle_work_real[old_index];

                            transform_imag[new_index] 
                                = shuffle_work_imag[old_index];
                        }
                    }
                    else {
                        // Adapted from PGFFT

                        std::memcpy(
                            dif_work_real.data(), 
                            transform_real, 
                            sizeof(Sample) * transform_len);
                        std::memcpy(
                            dif_work_imag.data(), 
                            transform_imag, 
                            sizeof(Sample) * transform_len);

                        Sample* work_real = shuffle_work_real.data();
                        Sample* work_imag = shuffle_work_imag.data();

                        const std::size_t* rev_flex = bit_reversed_indexes.data();
                        const std::size_t* rev_fixed = bit_reversed_indexes_2.data();

                        const auto A_real = dif_work_real.data();
                        const auto A_imag = dif_work_imag.data();
                        auto B_real = transform_real;
                        auto B_imag = transform_imag;

                        std::size_t q = PGFFT_BRC_Q();
                        
                        for (
                            std::size_t b = 0; 
                            b < bit_reversed_indexes.size(); 
                            b++
                        ) {
                            std::size_t b1 = rev_flex[b]; 
                            for (
                                std::size_t a = 0; 
                                a < bit_reversed_indexes_2.size(); 
                                a++
                            ) {
                                std::size_t a1 = rev_fixed[a]; 

                                Sample* T_p_real = work_real + (a1 << q);
                                Sample* T_p_imag = work_imag + (a1 << q);

                                const Sample* A_p_real 
                                    = A_real 
                                    + (a << (log_reversal_len +q)) 
                                    + (b << q);
                                
                                const Sample* A_p_imag 
                                    = A_imag 
                                    + (a << (log_reversal_len +q)) 
                                    + (b << q);
                                    
                                for (
                                    long c = 0; 
                                    c < bit_reversed_indexes_2.size(); 
                                    c+=k_N_SAMPLES_PER_OPERAND
                                ) {
                                    Operand store_real;
                                    Operand store_imag;
                                    Load(A_p_real + c, store_real);
                                    Load(A_p_imag + c, store_imag);
                                    Store(T_p_real + c, store_real);
                                    Store(T_p_imag + c, store_imag);
                                }
                            }

                            for (
                                long c = 0; 
                                c < bit_reversed_indexes_2.size(); 
                                c++
                            ) {
                                long c1 = rev_fixed[c];

                                Sample* B_p_real 
                                    = B_real 
                                    + (c1 << (log_reversal_len +q)) 
                                    + (b1 << q);
                                
                                Sample* B_p_imag 
                                    = B_imag 
                                    + (c1 << (log_reversal_len +q)) 
                                    + (b1 << q);

                                Sample* T_p_real = work_real + c;
                                Sample* T_p_imag = work_imag + c;
                                
                                for (
                                    long a1 = 0; 
                                    a1 < bit_reversed_indexes_2.size(); 
                                    a1+=1
                                ) { 
                                    B_p_real[a1] = T_p_real[a1 << q];
                                    B_p_imag[a1] = T_p_imag[a1 << q];
                                }
                            }
                        }
                    }
                }

                if (rescaling) {
                    Operand scale_factor(Sample(1.) /  Sample(transform_len));

                    auto t_real = transform_real;
                    auto t_imag = transform_imag;
                    
                    for (
                        std::size_t i = 0;
                        i < transform_len; 
                        i += k_N_SAMPLES_PER_OPERAND
                    ) {
                        Operand t_operand_real;
                        Operand t_operand_imag;

                        Load(t_real, t_operand_real);
                        Load(t_imag, t_operand_imag);

                        t_operand_real *= scale_factor;
                        t_operand_imag *= scale_factor;

                        Store(t_real, t_operand_real);
                        Store(t_imag, t_operand_imag);

                        t_real += k_N_SAMPLES_PER_OPERAND;
                        t_imag += k_N_SAMPLES_PER_OPERAND;
                    }
                }
            }

        private:

            static std::vector<std::vector<Sample>> Transpose(
                const std::vector<std::vector<Sample>> matrix
            ) {
                std::vector<std::vector<Sample>> transpose;
                auto size0 = matrix.size();
                if (size0 == 0) {
                    return transpose;
                }
                auto size1 = matrix[0].size();

                transpose.resize(size1);
                for (auto& v : transpose) {
                    v.resize(size0);
                }

                for (std::size_t i = 0; i < size0; i++) {
                    for (std::size_t j = 0; j < size1; j++) {
                        transpose[j][i] = matrix[i][j];
                    }
                }
                return transpose;
            }

            static std::size_t CheckedTransformLen(std::size_t transform_len) {

                if ((transform_len & (transform_len-1)) != 0) {
                    throw std::invalid_argument(
                        "The transform length must be a power of 2"
                    );
                }

                if (transform_len < k_N_SAMPLES_PER_OPERAND) {
                    throw std::invalid_argument(
                        "The transform length must not be "
                        "less than the operand size."
                    );
                }

                if (transform_len < 4) {
                    throw std::invalid_argument(
                        "The transform length must 4 or greater."
                    );
                }

                return transform_len;
            }

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
                            * trig_fn(
                                Pi() 
                                * Rem2(Sample(factor_id) / subtwiddle_len)
                            );

                        twiddle[1][factor_id] 
                            = multiplier 
                            * trig_fn(
                                Pi() 
                                * Rem2(Sample(factor_id) * 0.5 / subtwiddle_len)
                            );

                        twiddle[2][factor_id] 
                            = multiplier 
                            * trig_fn(
                                Pi() 
                                * Rem2(Sample(factor_id) * 1.5 / subtwiddle_len)
                            );
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
                            * trig_fn(
                                Pi() 
                                * Rem2(Sample(factor_id) / subtwiddle_len)
                            );
                    }
                }

                return twiddles;
            }

            static std::vector<std::vector<Sample>> Dft(
                const std::vector<std::size_t>& the_bit_reversed_indexes_dft,
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
                    auto bit_reversed_basis_index 
                        = the_bit_reversed_indexes_dft[basis_id];

                    for (
                        std::size_t factor_id = 0; 
                        factor_id < k_N_SAMPLES_PER_OPERAND;
                        factor_id++
                    ) {
                        dft_basis[factor_id] 
                            = multiplier 
                            * trig_fn(
                                Pi() 
                                * Rem2(
                                    Sample(factor_id)
                                    * 2 
                                    * bit_reversed_basis_index 
                                    / k_N_SAMPLES_PER_OPERAND
                                )
                            ); 
                    }
                }

                return dft;
            }
    };
}}

#endif