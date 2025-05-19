#ifndef GOLDENROCEKEFELLER_AFFT_FFT_COMPLEX_HPP
#define GOLDENROCEKEFELLER_AFFT_FFT_COMPLEX_HPP

#include <cstddef>
#include <vector>
#include <functional>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include "xsimd/xsimd.hpp"


namespace afft{
    std::size_t int_log_2(std::size_t n) {
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
    Value rem_2(Value x){
        return x - std::intmax_t(x/2) * 2;
    }

    std::vector<std::size_t> bit_reversed_indexes(std::size_t n_indexes) {
        std::vector<std::size_t> bit_reversed_indexes_(n_indexes);
        std::size_t n_bits = int_log_2(n_indexes);

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
            bit_reversed_indexes_[id] = bit_reversed_id;
        }
        return bit_reversed_indexes_;
    }

    template<
        typename SampleSpec, 
        typename OperandSpec
    >
    class FftComplex {
    
        using Sample = typename SampleSpec::Value;
        using Operand = typename OperandSpec::Value;
        using Vector = typename std::vector<Sample, xsimd::aligned_allocator<Sample, 128>>;

        static constexpr size_t k_N_SAMPLES_PER_OPERAND 
            = sizeof(Operand) / sizeof(Sample);

        static inline Sample pi(){
            return SampleSpec::pi();
        }

        // Use to determine COBRA buffer size. 2^pgfft_brc_thresh should fit in 
        // cache.
        static inline std::size_t pgfft_brc_q() {
            return 7;
        }

        // After 2^pgfft_brc_thresh, we will use COBRA bit reversal strategy
        static inline std::size_t pgfft_brc_thresh() {
            return 15;
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

        static inline Operand fma(
            const Operand& x, 
            const Operand& y, 
            const Operand& z
        ) {
            return OperandSpec::fma(x, y, z);
        }

        static inline Operand fms(
            const Operand& x, 
            const Operand& y, 
            const Operand& z
        ) {
            return OperandSpec::fms(x, y, z);
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
            std::size_t transform_len_;

            std::size_t n_radix_2_butterflies_;
            std::size_t n_radix_4_butterflies_;
            bool using_final_radix_2_butterfly_;

            std::vector<std::vector<Vector>> twiddles_real_;
            std::vector<std::vector<Vector>> twiddles_imag_;

            std::size_t log_len_;
            std::size_t log_reversal_len_;

            std::vector<std::size_t> bit_reversed_indexes_;
            std::vector<std::size_t> bit_reversed_indexes_2_;
            std::vector<std::size_t> bit_reversed_indexes_dft_;

            std::vector<Vector> dft_real_;
            std::vector<Vector> dft_imag_;
            std::vector<Vector> dft_real_transpose_;
            std::vector<Vector> dft_imag_transpose_;

            mutable Vector shuffle_work_real_;
            mutable Vector shuffle_work_imag_;
            mutable Vector dif_work_real_;
            mutable Vector dif_work_imag_;
        
        public:
            FftComplex() : 
                transform_len_(0)
            {}
        
            explicit FftComplex(std::size_t transform_len):
                transform_len_(checked_transform_len(transform_len)),
                n_radix_2_butterflies_(
                    int_log_2(transform_len / k_N_SAMPLES_PER_OPERAND)
                ),
                n_radix_4_butterflies_(n_radix_2_butterflies_ >> 1),
                using_final_radix_2_butterfly_(
                    n_radix_2_butterflies_ != 2 * n_radix_4_butterflies_
                ),
                twiddles_real_(
                    Twiddles(
                        n_radix_4_butterflies_,
                        using_final_radix_2_butterfly_,
                        cos,
                        1
                    )
                ),
                twiddles_imag_(
                    Twiddles(
                        n_radix_4_butterflies_,
                        using_final_radix_2_butterfly_,
                        sin,
                        -1
                    )
                ),
                log_len_(int_log_2(transform_len)),
                log_reversal_len_(
                    log_len_ * std::size_t(log_len_ < pgfft_brc_thresh())
                    + (log_len_ - 2 * pgfft_brc_q()) 
                    * std::size_t(log_len_ >= pgfft_brc_thresh())
                ),
                bit_reversed_indexes_(bit_reversed_indexes(1 << log_reversal_len_)),
                bit_reversed_indexes_2_(
                    bit_reversed_indexes(
                        1L << std::min(pgfft_brc_q(), log_len_)
                    )
                ),
                bit_reversed_indexes_dft_(
                    bit_reversed_indexes(k_N_SAMPLES_PER_OPERAND)
                ),
                dft_real_(dft(bit_reversed_indexes_dft_, cos, 1)),
                dft_imag_(dft(bit_reversed_indexes_dft_, sin, -1)),
                dft_real_transpose_(transpose(dft_real_)),
                dft_imag_transpose_(transpose(dft_imag_)),
                shuffle_work_real_(transform_len),
                shuffle_work_imag_(transform_len),
                dif_work_real_(transform_len),
                dif_work_imag_(transform_len)
            {}

            void process_dit(
                Sample* transform_real, 
                Sample* transform_imag,
                Sample* signal_real, 
                Sample* signal_imag,
                bool calculating_inverse = false,
                bool rescaling = false,
                bool processing_for_convolution = false
            ) const {
                if (transform_len_ == 0) {
                    return;
                }

                if (calculating_inverse) {
                    std::swap(signal_real, signal_imag);
                    std::swap(transform_real, transform_imag);
                }

                std::size_t subfft_len = 1;
                std::size_t n_subffts = transform_len_;
                std::size_t twiddle_id = 0;

                //--------------------------------------------------------------
                // BIT-REVERESED PERMUTE SIGNAL
                //--------------------------------------------------------------

                if (processing_for_convolution == false) {
                    if (log_len_ < pgfft_brc_thresh())
                    {
                        for (
                            std::size_t new_index = 0;
                            new_index < transform_len_; 
                            new_index++
                        ) {
                            auto old_index_0 = bit_reversed_indexes_[new_index];

                            transform_real[new_index] = signal_real[old_index_0];
                            transform_imag[new_index] = signal_imag[old_index_0];
                        }
                    }
                    else {
                        // Adapted from PGFFT
                        Sample* work_real = shuffle_work_real_.data();
                        Sample* work_imag = shuffle_work_imag_.data();

                        const std::size_t* rev_flex = bit_reversed_indexes_.data();
                        const std::size_t* rev_fixed = bit_reversed_indexes_2_.data();

                        const auto A_real = signal_real;
                        const auto A_imag = signal_imag;
                        auto B_real = transform_real;
                        auto B_imag = transform_imag;

                        std::size_t q = pgfft_brc_q();
                        
                        for (
                            std::size_t b = 0; 
                            b < bit_reversed_indexes_.size(); 
                            b++
                        ) {
                            std::size_t b1 = rev_flex[b]; 
                            for (
                                std::size_t a = 0; 
                                a < bit_reversed_indexes_2_.size(); 
                                a++
                            ) {
                                std::size_t a1 = rev_fixed[a]; 

                                Sample* T_p_real = work_real + (a1 << q);
                                Sample* T_p_imag = work_imag + (a1 << q);

                                const Sample* A_p_real 
                                    = A_real 
                                    + (a << (log_reversal_len_ +q)) 
                                    + (b << q);
                                
                                const Sample* A_p_imag 
                                    = A_imag 
                                    + (a << (log_reversal_len_ +q)) 
                                    + (b << q);
                                    
                                for (
                                    long c = 0; 
                                    c < bit_reversed_indexes_2_.size(); 
                                    c+=k_N_SAMPLES_PER_OPERAND
                                ) {
                                    Operand store_real;
                                    Operand store_imag;
                                    load(A_p_real + c, store_real);
                                    load(A_p_imag + c, store_imag);
                                    store(T_p_real + c, store_real);
                                    store(T_p_imag + c, store_imag);
                                }
                            }

                            for (
                                long c = 0; 
                                c < bit_reversed_indexes_2_.size(); 
                                c++
                            ) {
                                long c1 = rev_fixed[c];

                                Sample* B_p_real 
                                    = B_real 
                                    + (c1 << (log_reversal_len_ +q)) 
                                    + (b1 << q);
                                
                                Sample* B_p_imag 
                                    = B_imag 
                                    + (c1 << (log_reversal_len_ +q)) 
                                    + (b1 << q);

                                Sample* T_p_real = work_real + c;
                                Sample* T_p_imag = work_imag + c;
                                
                                for (
                                    long a1 = 0; 
                                    a1 < bit_reversed_indexes_2_.size(); 
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
                        sizeof(Sample) * transform_len_);
                    std::memcpy(
                        transform_imag, 
                        signal_imag, 
                        sizeof(Sample) * transform_len_);
                }

                //--------------------------------------------------------------
                // DFT PHASE
                //--------------------------------------------------------------
                
                if (k_N_SAMPLES_PER_OPERAND > 1 && subfft_len == 1) {
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

                        load(
                            dft_real_[1].data(), 
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

                            load(
                                dft_real_[dft_basis_id].data(), 
                                dft_operand_real_1
                            );
                            load(
                                dft_imag_[dft_basis_id].data(), 
                                dft_operand_imag_1
                            );

                            load(
                                dft_real_[dft_basis_id+1].data(), 
                                dft_operand_real_2
                            );
                            load(
                                dft_imag_[dft_basis_id+1].data(), 
                                dft_operand_imag_2
                            );
                            
                            b_operand_real_1 
                                = fms(
                                    a_operand_real_1,
                                    dft_operand_real_1,
                                    fms(
                                        a_operand_imag_1,
                                        dft_operand_imag_1,
                                        b_operand_real_1
                                    )
                                );

                            b_operand_imag_1 
                                = fma(
                                    a_operand_imag_1,
                                    dft_operand_real_1,
                                    fma(
                                        a_operand_real_1,
                                        dft_operand_imag_1,
                                        b_operand_imag_1
                                    )
                                );

                            b_operand_real_2 
                                = fms(
                                    a_operand_real_2,
                                    dft_operand_real_2,
                                    fms(
                                        a_operand_imag_2,
                                        dft_operand_imag_2,
                                        b_operand_real_2
                                    )
                                );

                            b_operand_imag_2 
                                = fma(
                                    a_operand_imag_2,
                                    dft_operand_real_2,
                                    fma(
                                        a_operand_real_2,
                                        dft_operand_imag_2,
                                        b_operand_imag_2
                                    )
                                );
                        
                            a_real_1 += 2;
                            a_imag_1 += 2;
                            a_real_2 += 2;
                            a_imag_2 += 2;
                        }

                        store(b_real, b_operand_real_1 + b_operand_real_2);
                        store(b_imag, b_operand_imag_1 + b_operand_imag_2);

                        b_real += k_N_SAMPLES_PER_OPERAND;
                        b_imag += k_N_SAMPLES_PER_OPERAND;
                    }
                }
                // --------------------------------------------------------------
                // RADIX-4 PHASE
                // --------------------------------------------------------------

                if (subfft_len == 1) {
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
                    twiddle_id += 1;              
                }

                while ((n_subffts >> 2) > 0) {
                    auto subtwiddle_len = subfft_len;
                    subfft_len = subfft_len << 2;
                    n_subffts = n_subffts >> 2;
                    
                    auto& twiddle_real = twiddles_real_[twiddle_id];
                    auto& twiddle_imag = twiddles_imag_[twiddle_id];
                    
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

                            load(a0_real, a0_operand_real);
                            load(a1_real, a1_operand_real);
                            load(a2_real, a2_operand_real);
                            load(a3_real, a3_operand_real);
                            load(a0_imag, a0_operand_imag);
                            load(a1_imag, a1_operand_imag);
                            load(a2_imag, a2_operand_imag);
                            load(a3_imag, a3_operand_imag);
                            
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

                                load(tw1_real, tw1_operand_real);
                                load(tw2_real, tw2_operand_real);
                                load(tw3_real, tw3_operand_real);
                                load(tw1_imag, tw1_operand_imag);
                                load(tw2_imag, tw2_operand_imag);
                                load(tw3_imag, tw3_operand_imag);

                                store1 
                                    = fms(
                                        a1_operand_real,
                                        tw1_operand_real,
                                        a1_operand_imag * tw1_operand_imag
                                    );

                                a1_operand_imag 
                                    = fma(
                                        a1_operand_imag,
                                        tw1_operand_real,
                                        a1_operand_real * tw1_operand_imag
                                    );

                                store2 
                                    = fms(
                                        a2_operand_real,
                                        tw2_operand_real,
                                        a2_operand_imag * tw2_operand_imag
                                    );

                                a2_operand_imag 
                                    = fma(
                                        a2_operand_imag,
                                        tw2_operand_real,
                                        a2_operand_real * tw2_operand_imag
                                    );

                                store3 
                                    = fms(
                                        a3_operand_real,
                                        tw3_operand_real,
                                        a3_operand_imag * tw3_operand_imag
                                    );

                                a3_operand_imag 
                                    = fma(
                                        a3_operand_imag,
                                        tw3_operand_real,
                                        a3_operand_real * tw3_operand_imag
                                    );

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

                            store(a0_real, a0_operand_real);
                            store(a1_real, a1_operand_real);
                            store(a2_real, a2_operand_real);
                            store(a3_real, a3_operand_real);
                            store(a0_imag, a0_operand_imag);
                            store(a1_imag, a1_operand_imag);
                            store(a2_imag, a2_operand_imag);
                            store(a3_imag, a3_operand_imag);

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
                    twiddle_id += 1;                  
                }

                // -------------------------------------------------------------
                // RADIX-2 PHASE
                // -------------------------------------------------------------
                if (using_final_radix_2_butterfly_) {
                    auto subtwiddle_len = subfft_len;
                    
                    std::size_t last_twiddle_id 
                        = std::size_t(twiddles_real_.size() - 1);

                    auto& twiddle_real = twiddles_real_[last_twiddle_id];
                    auto& twiddle_imag = twiddles_imag_[last_twiddle_id];

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

                        load(a0_real, a0_operand_real);
                        load(a1_real, a1_operand_real);
                        
                        load(a0_imag, a0_operand_imag);
                        load(a1_imag, a1_operand_imag);

                        {
                            Operand tw1_operand_real;
                            Operand tw1_operand_imag;
                            Operand store;

                            load(tw1_real, tw1_operand_real);
                            load(tw1_imag, tw1_operand_imag);

                            store 
                                = fms(a1_operand_real, tw1_operand_real,
                                a1_operand_imag * tw1_operand_imag);

                            a1_operand_imag 
                                = fma(a1_operand_imag, tw1_operand_real,
                                a1_operand_real * tw1_operand_imag);

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

                            store(a0_real, b0_operand_real);
                            store(a1_real, b1_operand_real);
                            store(a0_imag, b0_operand_imag);
                            store(a1_imag, b1_operand_imag);
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
                    Operand scale_factor(Sample(1.) /  Sample(transform_len_));

                    auto t_real = transform_real;
                    auto t_imag = transform_imag;
                    
                    for (
                        std::size_t i = 0;
                        i < transform_len_; 
                        i += k_N_SAMPLES_PER_OPERAND
                    ) {
                        Operand t_operand_real;
                        Operand t_operand_imag;

                        load(t_real, t_operand_real);
                        load(t_imag, t_operand_imag);

                        t_operand_real = t_operand_real * scale_factor;
                        t_operand_imag = t_operand_imag * scale_factor;

                        store(t_real, t_operand_real);
                        store(t_imag, t_operand_imag);

                        t_real += k_N_SAMPLES_PER_OPERAND;
                        t_imag += k_N_SAMPLES_PER_OPERAND;
                    }
                }
            }

            void process_dif(
                Sample* transform_real, 
                Sample* transform_imag,
                Sample* signal_real, 
                Sample* signal_imag,
                bool calculating_inverse = false,
                bool rescaling = false,
                bool processing_for_convolution = false
            ) const {
                if (transform_len_ == 0) {
                    return;
                }

                std::memcpy(
                    transform_real, 
                    signal_real, 
                    sizeof(Sample) * transform_len_);
                std::memcpy(
                    transform_imag, 
                    signal_imag, 
                    sizeof(Sample) * transform_len_);

                if (calculating_inverse) {
                    std::swap(signal_real, signal_imag);
                    std::swap(transform_real, transform_imag);
                }

                // Copy signal and move it to transger

                std::size_t subfft_len = transform_len_;
                std::size_t n_subffts = 1;
                std::size_t twiddle_id = std::size_t(twiddles_real_.size() - 1);

                // -------------------------------------------------------------
                // RADIX-2 PHASE
                // -------------------------------------------------------------
                // TODO unroll x2
                if (using_final_radix_2_butterfly_) {
                    auto subtwiddle_len = subfft_len >> 1;

                    auto& twiddle_real = twiddles_real_[twiddle_id];
                    auto& twiddle_imag = twiddles_imag_[twiddle_id];

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

                        load(a1_real, a1_operand_real);
                        load(a1_imag, a1_operand_imag);
                        {
                            Operand a0_operand_real;
                            Operand a0_operand_imag;
                            Operand b0_operand_real;
                            Operand b0_operand_imag;

                            load(a0_real, a0_operand_real);
                            load(a0_imag, a0_operand_imag);

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

                            store(a0_real, b0_operand_real);
                            store(a0_imag, b0_operand_imag);
                        }
                        {
                            Operand tw1_operand_real;
                            Operand tw1_operand_imag;
                            Operand tmp;

                            load(tw1_real, tw1_operand_real);
                            load(tw1_imag, tw1_operand_imag);
                            
                            tmp 
                                = fms(b1_operand_real, tw1_operand_real,
                                b1_operand_imag * tw1_operand_imag);

                            b1_operand_imag 
                                = fma(b1_operand_imag, tw1_operand_real,
                                b1_operand_real * tw1_operand_imag);

                            b1_operand_real = tmp;

                            store(a1_real, b1_operand_real);
                            store(a1_imag, b1_operand_imag);

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
                    butterfly_id < n_radix_4_butterflies_;
                    butterfly_id++
                ) {
                    auto subtwiddle_len = subfft_len >> 2;
                    
                    auto& twiddle_real = twiddles_real_[twiddle_id];
                    auto& twiddle_imag = twiddles_imag_[twiddle_id];
                    
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

                            load(a0_real, a0_operand_real);
                            load(a1_real, a1_operand_real);
                            load(a2_real, a2_operand_real);
                            load(a3_real, a3_operand_real);
                            load(a0_imag, a0_operand_imag);
                            load(a1_imag, a1_operand_imag);
                            load(a2_imag, a2_operand_imag);
                            load(a3_imag, a3_operand_imag);
                            
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

                                load(tw1_real, tw1_operand_real);
                                load(tw2_real, tw2_operand_real);
                                load(tw3_real, tw3_operand_real);
                                load(tw1_imag, tw1_operand_imag);
                                load(tw2_imag, tw2_operand_imag);
                                load(tw3_imag, tw3_operand_imag);
                                // TODO FMA
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

                            store(a0_real, a0_operand_real);
                            store(a1_real, a1_operand_real);
                            store(a2_real, a2_operand_real);
                            store(a3_real, a3_operand_real);
                            store(a0_imag, a0_operand_imag);
                            store(a1_imag, a1_operand_imag);
                            store(a2_imag, a2_operand_imag);
                            store(a3_imag, a3_operand_imag);

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

                        load(
                            dft_real_transpose_[half_dft_len].data(), 
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

                            load(
                                dft_real_transpose_[dft_basis_id].data(), 
                                dft_operand_real_1
                            );
                            load(
                                dft_imag_transpose_[dft_basis_id].data(), 
                                dft_operand_imag_1
                            );

                            load(
                                dft_real_transpose_[dft_basis_id + half_dft_len].data(), 
                                dft_operand_real_2
                            );
                            load(
                                dft_imag_transpose_[dft_basis_id + half_dft_len].data(), 
                                dft_operand_imag_2
                            );
                            // TODO FMA
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

                        store(b_real, b_operand_real_1 + b_operand_real_2);
                        store(b_imag, b_operand_imag_1 + b_operand_imag_2);

                        b_real += k_N_SAMPLES_PER_OPERAND;
                        b_imag += k_N_SAMPLES_PER_OPERAND;
                    }
                }

                //--------------------------------------------------------------
                // BIT-REVERESED PERMUTE SIGNAL
                //--------------------------------------------------------------

                if (processing_for_convolution == false) {
                    if (log_len_ < pgfft_brc_thresh())
                    {
                        Sample* work_real = shuffle_work_real_.data();
                        Sample* work_imag = shuffle_work_imag_.data();

                        std::memcpy(
                            work_real, 
                            transform_real, 
                            sizeof(Sample) * transform_len_);
                        std::memcpy(
                            work_imag, 
                            transform_imag, 
                            sizeof(Sample) * transform_len_);

                        for (
                            std::size_t new_index = 0;
                            new_index < transform_len_; 
                            new_index++
                        ) {
                            auto old_index = bit_reversed_indexes_[new_index];

                            transform_real[new_index] 
                                = shuffle_work_real_[old_index];

                            transform_imag[new_index] 
                                = shuffle_work_imag_[old_index];
                        }
                    }
                    else {
                        // Adapted from PGFFT

                        std::memcpy(
                            dif_work_real_.data(), 
                            transform_real, 
                            sizeof(Sample) * transform_len_);
                        std::memcpy(
                            dif_work_imag_.data(), 
                            transform_imag, 
                            sizeof(Sample) * transform_len_);

                        Sample* work_real = shuffle_work_real_.data();
                        Sample* work_imag = shuffle_work_imag_.data();

                        const std::size_t* rev_flex = bit_reversed_indexes_.data();
                        const std::size_t* rev_fixed = bit_reversed_indexes_2_.data();

                        const auto A_real = dif_work_real_.data();
                        const auto A_imag = dif_work_imag_.data();
                        auto B_real = transform_real;
                        auto B_imag = transform_imag;

                        std::size_t q = pgfft_brc_q();
                        
                        for (
                            std::size_t b = 0; 
                            b < bit_reversed_indexes_.size(); 
                            b++
                        ) {
                            std::size_t b1 = rev_flex[b]; 
                            for (
                                std::size_t a = 0; 
                                a < bit_reversed_indexes_2_.size(); 
                                a++
                            ) {
                                std::size_t a1 = rev_fixed[a]; 

                                Sample* T_p_real = work_real + (a1 << q);
                                Sample* T_p_imag = work_imag + (a1 << q);

                                const Sample* A_p_real 
                                    = A_real 
                                    + (a << (log_reversal_len_ +q)) 
                                    + (b << q);
                                
                                const Sample* A_p_imag 
                                    = A_imag 
                                    + (a << (log_reversal_len_ +q)) 
                                    + (b << q);
                                    
                                for (
                                    long c = 0; 
                                    c < bit_reversed_indexes_2_.size(); 
                                    c+=k_N_SAMPLES_PER_OPERAND
                                ) {
                                    Operand store_real;
                                    Operand store_imag;
                                    load(A_p_real + c, store_real);
                                    load(A_p_imag + c, store_imag);
                                    store(T_p_real + c, store_real);
                                    store(T_p_imag + c, store_imag);
                                }
                            }

                            for (
                                long c = 0; 
                                c < bit_reversed_indexes_2_.size(); 
                                c++
                            ) {
                                long c1 = rev_fixed[c];

                                Sample* B_p_real 
                                    = B_real 
                                    + (c1 << (log_reversal_len_ +q)) 
                                    + (b1 << q);
                                
                                Sample* B_p_imag 
                                    = B_imag 
                                    + (c1 << (log_reversal_len_ +q)) 
                                    + (b1 << q);

                                Sample* T_p_real = work_real + c;
                                Sample* T_p_imag = work_imag + c;
                                
                                for (
                                    long a1 = 0; 
                                    a1 < bit_reversed_indexes_2_.size(); 
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
                    Operand scale_factor(Sample(1.) /  Sample(transform_len_));

                    auto t_real = transform_real;
                    auto t_imag = transform_imag;
                    
                    for (
                        std::size_t i = 0;
                        i < transform_len_; 
                        i += k_N_SAMPLES_PER_OPERAND
                    ) {
                        Operand t_operand_real;
                        Operand t_operand_imag;

                        load(t_real, t_operand_real);
                        load(t_imag, t_operand_imag);

                        t_operand_real *= scale_factor;
                        t_operand_imag *= scale_factor;

                        store(t_real, t_operand_real);
                        store(t_imag, t_operand_imag);

                        t_real += k_N_SAMPLES_PER_OPERAND;
                        t_imag += k_N_SAMPLES_PER_OPERAND;
                    }
                }
            }

        private:

            static std::vector<Vector> transpose(
                const std::vector<Vector> matrix
            ) {
                std::vector<Vector> transpose;
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

            static std::size_t checked_transform_len(std::size_t transform_len) {

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

            static std::vector<std::vector<Vector>> Twiddles(
                size_t n_radix_4_butterflies,
                bool using_final_radix_2_butterfly,
                const std::function<Sample(Sample)>& trig_fn,
                int multiplier
            ) {
                std::size_t initial_subfft_len = k_N_SAMPLES_PER_OPERAND;

                std::vector<std::vector<Vector>> twiddles;

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
                                pi() 
                                * rem_2(Sample(factor_id) / subtwiddle_len)
                            );

                        twiddle[1][factor_id] 
                            = multiplier 
                            * trig_fn(
                                pi() 
                                * rem_2(Sample(factor_id) * 0.5 / subtwiddle_len)
                            );

                        twiddle[2][factor_id] 
                            = multiplier 
                            * trig_fn(
                                pi() 
                                * rem_2(Sample(factor_id) * 1.5 / subtwiddle_len)
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
                                pi() 
                                * rem_2(Sample(factor_id) / subtwiddle_len)
                            );
                    }
                }

                return twiddles;
            }

            static std::vector<Vector> dft(
                const std::vector<std::size_t>& the_bit_reversed_indexes_dft,
                const std::function<Sample(Sample)>& trig_fn,
                int multiplier
            ) {
                std::vector<Vector> dft;

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
                                pi() 
                                * rem_2(
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
}

#endif