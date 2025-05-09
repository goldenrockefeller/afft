#ifndef AFFT_LOAD_EDGE_TW_HPP
#define AFFT_LOAD_EDGE_TW_HPP

#include <cstddef>

namespace afft{
    template<typename Spec, std::size_t StageFactor, std::size_t StageId>
    inline void load_edge_tw4(
        operand* init_tw_real_b_op,
        operand* init_tw_imag_b_op,
        operand* init_tw_real_c_op,
        operand* init_tw_imag_c_op,
        operand* init_tw_real_d_op,
        operand* init_tw_imag_d_op,
        const typename Spec::sample* tw_real, 
        const typename Spec::sample* tw_imag
    ) {
        using operand = typename Spec::operand;
        constexpr std::size_t n_samples_per_operand = Spec::n_samples_per_operand;
        Spec::load(init_tw_real_b_op[StageId], tw_real[(3 * StageId + 0) * n_samples_per_operand]);
        Spec::load(init_tw_imag_b_op[StageId], tw_imag[(3 * StageId + 0) * n_samples_per_operand]);
        Spec::load(init_tw_real_c_op[StageId], tw_real[(3 * StageId + 1) * n_samples_per_operand]);
        Spec::load(init_tw_imag_c_op[StageId], tw_imag[(3 * StageId + 1) * n_samples_per_operand]);
        Spec::load(init_tw_real_d_op[StageId], tw_real[(3 * StageId + 2) * n_samples_per_operand]);
        Spec::load(init_tw_imag_d_op[StageId], tw_imag[(3 * StageId + 2) * n_samples_per_operand]);

        load_edge_tw4<Spec, StageFactor/4, StageId + 1>(
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

    template<typename Spec, std::size_t StageId>
    inline void load_edge_tw4<Spec, 1, StageId>(
        operand* init_tw_real_b_op,
        operand* init_tw_imag_b_op,
        operand* init_tw_real_c_op,
        operand* init_tw_imag_c_op,
        operand* init_tw_real_d_op,
        operand* init_tw_imag_d_op,
        const typename Spec::sample* tw_real, 
        const typename Spec::sample* tw_imag
    ) {
        // Do Nothing
    }

    template<typename Spec, std::size_t StageId>
    inline void load_edge_tw4<Spec, 2, StageId>(
        operand* init_tw_real_b_op,
        operand* init_tw_imag_b_op,
        operand* init_tw_real_c_op,
        operand* init_tw_imag_c_op,
        operand* init_tw_real_d_op,
        operand* init_tw_imag_d_op,
        const typename Spec::sample* tw_real, 
        const typename Spec::sample* tw_imag
    ) {
        using operand = typename Spec::operand;
        constexpr std::size_t n_samples_per_operand = Spec::n_samples_per_operand;
        Spec::load(init_tw_real_b_op[StageId], tw_real[3 * StageId * n_samples_per_operand]);
        Spec::load(init_tw_imag_b_op[StageId], tw_imag[3 * StageId * n_samples_per_operand]);
    }


    template<typename Spec, std::size_t StageFactor, std::size_t StageId>
    inline void load_edge_tw2(
        operand* init_tw_real_b_op,
        operand* init_tw_imag_b_op,
        const typename Spec::sample* tw_real, 
        const typename Spec::sample* tw_imag
    ) {
        using operand = typename Spec::operand;
        constexpr std::size_t n_samples_per_operand = Spec::n_samples_per_operand;
        Spec::load(init_tw_real_b_op[StageId], tw_real[StageId * n_samples_per_operand]);
        Spec::load(init_tw_imag_b_op[StageId], tw_imag[StageId * n_samples_per_operand]);

        load_edge_tw2<Spec, StageFactor/2, StageId + 1>(
            init_tw_real_b_op,
            init_tw_imag_b_op,
            tw_real, 
            tw_imag
        ); 
    }

    template<typename Spec, std::size_t StageId>
    inline void load_edge_tw2<Spec, 1, StageId>(
        operand* init_tw_real_b_op,
        operand* init_tw_imag_b_op,
        const typename Spec::sample* tw_real, 
        const typename Spec::sample* tw_imag
    ) {
        // Do Nothing
    }s
}

#endif