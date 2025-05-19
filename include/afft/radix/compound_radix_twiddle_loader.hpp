#ifndef AFFT_COMPOUND_RADIX_TWIDDLE_LOADER_HPP
#define AFFT_COMPOUND_RADIX_TWIDDLE_LOADER_HPP

#include <cstddef>

namespace afft{
    template<typename Spec, std::size_t StageFactor, std::size_t StageId>
    struct CompoundRadixTwiddleLoader{
        static inline void load_compound_radix_tw4(
            typename Spec::operand *init_tw_real_b_op,
            typename Spec::operand *init_tw_imag_b_op,
            typename Spec::operand *init_tw_real_c_op,
            typename Spec::operand *init_tw_imag_c_op,
            typename Spec::operand *init_tw_real_d_op,
            typename Spec::operand *init_tw_imag_d_op,
            const typename Spec::sample* tw_real, 
            const typename Spec::sample* tw_imag
        ) {
            using operand = typename Spec::operand;
            constexpr std::size_t n_samples_per_operand = Spec::n_samples_per_operand;
            Spec::load(init_tw_real_b_op[StageId], tw_real + ((3 * StageId + 0) * n_samples_per_operand));
            Spec::load(init_tw_imag_b_op[StageId], tw_imag + ((3 * StageId + 0) * n_samples_per_operand));
            Spec::load(init_tw_real_c_op[StageId], tw_real + ((3 * StageId + 1) * n_samples_per_operand));
            Spec::load(init_tw_imag_c_op[StageId], tw_imag + ((3 * StageId + 1) * n_samples_per_operand));
            Spec::load(init_tw_real_d_op[StageId], tw_real + ((3 * StageId + 2) * n_samples_per_operand));
            Spec::load(init_tw_imag_d_op[StageId], tw_imag + ((3 * StageId + 2) * n_samples_per_operand));

            CompoundRadixTwiddleLoader<Spec, StageFactor/4, StageId + 1>::load_compound_radix_tw4(
                init_tw_real_b_op,
                init_tw_imag_b_op,
                init_tw_real_c_op,
                init_tw_imag_c_op,
                init_tw_real_d_op,
                init_tw_imag_d_op,
                tw_real, 
                tw_imag
            ); 
        }

        static inline void load_compound_radix_tw2(
            typename Spec::operand *init_tw_real_b_op,
            typename Spec::operand *init_tw_imag_b_op,
            const typename Spec::sample* tw_real, 
            const typename Spec::sample* tw_imag
        ) {
            using operand = typename Spec::operand;
            constexpr std::size_t n_samples_per_operand = Spec::n_samples_per_operand;
            Spec::load(init_tw_real_b_op[StageId], tw_real + (StageId * n_samples_per_operand));
            Spec::load(init_tw_imag_b_op[StageId], tw_imag + (StageId * n_samples_per_operand));

            CompoundRadixTwiddleLoader<Spec, StageFactor/2, StageId + 1>::load_compound_radix_tw2(
                init_tw_real_b_op,
                init_tw_imag_b_op,
                tw_real, 
                tw_imag
            ); 
        }
    };
    

    template<typename Spec, std::size_t StageId>
    struct CompoundRadixTwiddleLoader<Spec, 1, StageId>{
        static inline void load_compound_radix_tw4(
            typename Spec::operand *init_tw_real_b_op,
            typename Spec::operand *init_tw_imag_b_op,
            typename Spec::operand *init_tw_real_c_op,
            typename Spec::operand *init_tw_imag_c_op,
            typename Spec::operand *init_tw_real_d_op,
            typename Spec::operand *init_tw_imag_d_op,
            const typename Spec::sample* tw_real, 
            const typename Spec::sample* tw_imag
        ) {
            // Do Nothing
        }

        static inline void load_compound_radix_tw2(
            typename Spec::operand *init_tw_real_b_op,
            typename Spec::operand *init_tw_imag_b_op,
            const typename Spec::sample* tw_real, 
            const typename Spec::sample* tw_imag
        ) {
            // Do Nothing
        }
    };

    template<typename Spec, std::size_t StageId>
    struct CompoundRadixTwiddleLoader<Spec, 2, StageId> {
        static inline void load_compound_radix_tw4(
            typename Spec::operand *init_tw_real_b_op,
            typename Spec::operand *init_tw_imag_b_op,
            typename Spec::operand *init_tw_real_c_op,
            typename Spec::operand *init_tw_imag_c_op,
            typename Spec::operand *init_tw_real_d_op,
            typename Spec::operand *init_tw_imag_d_op,
            const typename Spec::sample* tw_real, 
            const typename Spec::sample* tw_imag
        ) {
            using operand = typename Spec::operand;
            constexpr std::size_t n_samples_per_operand = Spec::n_samples_per_operand;
            Spec::load(init_tw_real_c_op[StageId], tw_real + (3 * StageId * n_samples_per_operand));
            Spec::load(init_tw_imag_c_op[StageId], tw_imag + (3 * StageId * n_samples_per_operand));
            Spec::load(init_tw_real_d_op[StageId], tw_real + ((3 * StageId + 1) * n_samples_per_operand));
            Spec::load(init_tw_imag_d_op[StageId], tw_imag + ((3 * StageId + 1) * n_samples_per_operand));
        }

        static inline void load_compound_radix_tw2(
            typename Spec::operand *init_tw_real_b_op,
            typename Spec::operand *init_tw_imag_b_op,
            const typename Spec::sample* tw_real, 
            const typename Spec::sample* tw_imag
        ) {
            using operand = typename Spec::operand;
            constexpr std::size_t n_samples_per_operand = Spec::n_samples_per_operand;
            Spec::load(init_tw_real_b_op[StageId], tw_real + (StageId * n_samples_per_operand));
            Spec::load(init_tw_imag_b_op[StageId], tw_imag + (StageId * n_samples_per_operand));

            CompoundRadixTwiddleLoader<Spec, 1, StageId + 1>::load_compound_radix_tw2(
                init_tw_real_b_op,
                init_tw_imag_b_op,
                tw_real, 
                tw_imag
            ); 
        }
    };    
}

#endif