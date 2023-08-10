#ifndef GOLDENROCEKEFELLER_AFFT_HPP
#define GOLDENROCEKEFELLER_AFFT_HPP

#include <cstddef>
#include <vector>
#include <functional>

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
        std::size_t index_adder = n_indexes >> 1;
        std::vector<std::size_t> scrambled_indexes = {0};
        auto n_scrambled_indexes_steps = IntLog2(n_indexes);

        if (n_scrambled_indexes_steps == 0) {
            return scrambled_indexes;
        }
        
        scrambled_indexes.push_back(index_adder);

        for (
            std::size_t i = 0; 
            i < (n_scrambled_indexes_steps-1);
            i++
        ) {
            index_adder = index_adder >> 1;
            auto scrambled_indexes_next = scrambled_indexes;
            for (auto index : scrambled_indexes) {
                scrambled_indexes_next.push_back(index + index_adder);
            }
            scrambled_indexes = scrambled_indexes_next;
        }

        return scrambled_indexes;
    }


    template<
        std::size_t k_TRANSFORM_LEN, 
        typename SampleSpec, 
        typename OperandSpec>
    class Fft {
    
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

        static_assert(
            ((k_TRANSFORM_LEN & (k_TRANSFORM_LEN-1)) == 0), 
            "The transform length must be a power of 2"
        );

        static_assert(
            ((k_N_SAMPLES_PER_OPERAND & (k_N_SAMPLES_PER_OPERAND-1)) == 0), 
            "The number of samples per operand must be a power of 2"
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

        
            Fft():
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