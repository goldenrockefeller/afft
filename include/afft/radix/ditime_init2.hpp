#ifndef AFFT_DITIME_INIT2_HPP
#define AFFT_DITIME_INIT2_HPP

#include <cstddef>
#include "afft/radix/load_edge_tw.hpp"

namespace afft{
    template<typename Spec, std::size_t StageFactor, std::size_t StageId>
    inline void compute_ditime_init2_substage(
        operand& alpha_real_a_op,
        operand& alpha_real_b_op,
        operand& alpha_imag_a_op,
        operand& alpha_imag_b_op,
        operand& beta_real_a_op,
        operand& beta_real_b_op,
        operand& beta_imag_a_op,
        operand& beta_imag_b_op,
        const operand* init_tw_real_b_op,
        const operand* init_tw_imag_b_op
    ) {
        Spec::deinterleave(beta_real_a_op, beta_real_b_op, alpha_real_a_op, alpha_real_b_op);
        Spec::deinterleave(beta_imag_a_op, beta_imag_b_op, alpha_imag_a_op, alpha_imag_b_op);

        beta_real_b_op = init_tw_imag_b_op[StageId] * alpha_imag_b_op;
        beta_imag_b_op = init_tw_imag_b_op[StageId] * alpha_real_b_op;

        alpha_real_b_op *= init_tw_real_b_op[StageId];
        alpha_imag_b_op *= init_tw_real_b_op[StageId];

        alpha_real_b_op -= beta_real_b_op;
        alpha_imag_b_op += beta_imag_b_op;

        alpha_real_a_op = beta_real_a_op + beta_real_b_op;
        alpha_real_b_op = beta_real_a_op - beta_real_b_op;
        alpha_imag_a_op = beta_imag_a_op + beta_imag_b_op;
        alpha_imag_b_op = beta_imag_a_op - beta_imag_b_op;
        

        compute_ditime_init2_substage<Spec, StageFactor/2, StageId+1>(
            alpha_real_a_op,
            alpha_real_b_op,
            alpha_imag_a_op,
            alpha_imag_b_op,
            beta_real_a_op,
            beta_real_b_op,
            beta_imag_a_op,
            beta_imag_b_op,
            init_tw_real_b_op,
            init_tw_imag_b_op
        );
    }

    template<typename Spec, std::size_t StageId>
    inline void compute_ditime_init2_substage<Spec, 1, StageId>(
        operand& alpha_real_a_op,
        operand& alpha_real_b_op,
        operand& alpha_imag_a_op,
        operand& alpha_imag_b_op,
        operand& beta_real_a_op,
        operand& beta_real_b_op,
        operand& beta_imag_a_op,
        operand& beta_imag_b_op,
        const operand* init_tw_real_b_op,
        const operand* init_tw_imag_b_op,
    ) {
        // Do nothing
    }

    template<typename Spec, bool Rescaling>
    inline void do_ditime_init2_stage(
        typename Spec::sample* out_real, 
        typename Spec::sample* out_imag, 
        const typename Spec::sample* in_real, 
        const typename Spec::sample* in_imag, 
        const typename Spec::sample* tw_real, 
        const typename Spec::sample* tw_imag, 
        typename Spec::sample scaling_factor
    ) {
        using operand = typename Spec::operand;
        constexpr std::size_t n_samples_per_operand = Spec::n_samples_per_operand;

        // DECLARE
        operand alpha_real_a_op;
        operand alpha_real_b_op;

        operand alpha_imag_a_op;
        operand alpha_imag_b_op;

        operand beta_real_a_op;
        operand beta_real_b_op;

        operand beta_imag_a_op;
        operand beta_imag_b_op;

        // 5 twiddle stages should be enough for an operand size of 256 samples!
        operand init_tw_real_b_op[5];
        operand init_tw_imag_b_op[5];

        load_edge_tw2<Spec, n_samples_per_operand, 0>(
            init_tw_real_b_op,
            init_tw_imag_b_op,
            tw_real, 
            tw_imag
        );

        // OFFSET
        auto in_real_a = in_real;
        auto in_imag_a = in_imag;
        auto in_real_b = in_real + n_samples_per_operand;
        auto in_imag_b = in_imag + n_samples_per_operand;

        auto out_real_a = out_real;
        auto out_imag_a = out_imag;
        auto out_real_b = out_real + n_samples_per_operand;
        auto out_imag_b = out_imag + n_samples_per_operand;
        
        //LOAD
        Spec::load(alpha_real_a_op, in_real_a);
        Spec::load(alpha_imag_a_op, in_imag_a);
        Spec::load(alpha_real_b_op, in_real_b);
        Spec::load(alpha_imag_b_op, in_imag_b);

        
        // COMPUTE  
        if (Rescaling){
            alpha_real_a_op *= scaling_factor;
            alpha_imag_a_op *= scaling_factor;
            alpha_real_b_op *= scaling_factor;
            alpha_imag_b_op *= scaling_factor;
        }
        if (n_samples_per_operand != 1) {
            Spec::deinterleave(beta_real_a_op, beta_real_b_op, alpha_real_a_op, alpha_real_b_op);
            Spec::deinterleave(beta_imag_a_op, beta_imag_b_op, alpha_imag_a_op, alpha_imag_b_op);
        }

        alpha_real_a_op = beta_real_a_op + beta_real_b_op;
        alpha_real_b_op = beta_real_a_op - beta_real_b_op;
        alpha_imag_a_op = beta_imag_a_op + beta_imag_b_op;
        alpha_imag_b_op = beta_imag_a_op - beta_imag_b_op;
        
        compute_ditime_init2_substage<Spec, n_samples_per_operand, 0>(
            alpha_real_a_op,
            alpha_real_b_op,
            alpha_imag_a_op,
            alpha_imag_b_op,
            beta_real_a_op,
            beta_real_b_op,
            beta_imag_a_op,
            beta_imag_b_op,
            init_tw_real_b_op,
            init_tw_imag_b_op
        );

        // STORE
        Spec::store(out_real_a, alpha_real_a_op);
        Spec::store(out_imag_a, alpha_imag_a_op);
        Spec::store(out_real_b, alpha_real_b_op);
        Spec::store(out_imag_b, alpha_imag_b_op);
    }
}

#endif