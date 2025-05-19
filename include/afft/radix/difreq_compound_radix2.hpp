#ifndef AFFT_DIFREQ_COMPOUND_RADIX2_HPP
#define AFFT_DIFREQ_COMPOUND_RADIX2_HPP

#include <cstddef>
#include "afft/radix/compound_radix_twiddle_loader.hpp"

namespace afft
{
    template <typename Spec, std::size_t StageFactor, std::size_t StageId>
    struct ComputeDifreqCompoundRadix2Substage {
        static inline void eval(
            typename Spec::operand &alpha_real_a_op,
            typename Spec::operand &alpha_real_b_op,
            typename Spec::operand &alpha_imag_a_op,
            typename Spec::operand &alpha_imag_b_op,
            typename Spec::operand &beta_real_a_op,
            typename Spec::operand &beta_real_b_op,
            typename Spec::operand &beta_imag_a_op,
            typename Spec::operand &beta_imag_b_op,
            const typename Spec::operand *tw_real_b_op,
            const typename Spec::operand *tw_imag_b_op)
        {

            ComputeDifreqCompoundRadix2Substage<Spec, StageFactor / 2, StageId + 1>::eval(
                alpha_real_a_op,
                alpha_real_b_op,
                alpha_imag_a_op,
                alpha_imag_b_op,
                beta_real_a_op,
                beta_real_b_op,
                beta_imag_a_op,
                beta_imag_b_op,
                tw_real_b_op,
                tw_imag_b_op);

            beta_real_a_op = alpha_real_a_op + alpha_real_b_op;
            beta_real_b_op = alpha_real_a_op - alpha_real_b_op;
            beta_imag_a_op = alpha_imag_a_op + alpha_imag_b_op;
            beta_imag_b_op = alpha_imag_a_op - alpha_imag_b_op;

            alpha_real_b_op = tw_imag_b_op[StageId] * beta_imag_b_op;
            alpha_imag_b_op = tw_imag_b_op[StageId] * beta_real_b_op;

            beta_real_b_op *= tw_real_b_op[StageId];
            beta_imag_b_op *= tw_real_b_op[StageId];

            beta_real_b_op -= alpha_real_b_op;
            beta_imag_b_op += alpha_imag_b_op;

            Spec::interleave2(
                alpha_real_a_op, alpha_imag_a_op, 
                alpha_real_b_op, alpha_imag_b_op,
                beta_real_a_op, beta_imag_a_op, 
                beta_real_b_op, beta_imag_b_op);
        }
    };

    template <typename Spec, std::size_t StageId>
    struct ComputeDifreqCompoundRadix2Substage<Spec, 1, StageId> {
        static inline void eval(
            typename Spec::operand &alpha_real_a_op,
            typename Spec::operand &alpha_real_b_op,
            typename Spec::operand &alpha_imag_a_op,
            typename Spec::operand &alpha_imag_b_op,
            typename Spec::operand &beta_real_a_op,
            typename Spec::operand &beta_real_b_op,
            typename Spec::operand &beta_imag_a_op,
            typename Spec::operand &beta_imag_b_op,
            const typename Spec::operand *tw_real_b_op,
            const typename Spec::operand *tw_imag_b_op)
        {
            // Do nothing
        }
    };

    template <typename Spec, bool Rescaling>
    inline void do_difreq_compound_radix2_stage(
        typename Spec::sample *out_real,
        typename Spec::sample *out_imag,
        const typename Spec::sample *in_real,
        const typename Spec::sample *in_imag,
        const typename Spec::sample *tw_real,
        const typename Spec::sample *tw_imag,
        const typename Spec::sample &scaling_factor)
    {
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

        // 10 twiddle stages should be enough for an operand size of 256 samples!
        operand tw_real_b_op[10];
        operand tw_imag_b_op[10];

        CompoundRadixTwiddleLoader<Spec, n_samples_per_operand, 0>::load_compound_radix_tw2(
            tw_real_b_op,
            tw_imag_b_op,
            tw_real,
            tw_imag);

        // OFFSET
        auto in_real_a = in_real;
        auto in_imag_a = in_imag;
        auto in_real_b = in_real + n_samples_per_operand;
        auto in_imag_b = in_imag + n_samples_per_operand;

        auto out_real_a = out_real;
        auto out_imag_a = out_imag;
        auto out_real_b = out_real + n_samples_per_operand;
        auto out_imag_b = out_imag + n_samples_per_operand;

        // LOAD
        Spec::load(alpha_real_a_op, in_real_a);
        Spec::load(alpha_imag_a_op, in_imag_a);
        Spec::load(alpha_real_b_op, in_real_b);
        Spec::load(alpha_imag_b_op, in_imag_b);

        // COMPUTE
        ComputeDifreqCompoundRadix2Substage<Spec, n_samples_per_operand, 0>::eval(
            alpha_real_a_op,
            alpha_real_b_op,
            alpha_imag_a_op,
            alpha_imag_b_op,
            beta_real_a_op,
            beta_real_b_op,
            beta_imag_a_op,
            beta_imag_b_op,
            tw_real_b_op,
            tw_imag_b_op);

        if (Rescaling)
        {
            alpha_real_a_op *= scaling_factor;
            alpha_imag_a_op *= scaling_factor;
            alpha_real_b_op *= scaling_factor;
            alpha_imag_b_op *= scaling_factor;
        }
    
        beta_real_a_op = alpha_real_a_op + alpha_real_b_op;
        beta_real_b_op = alpha_real_a_op - alpha_real_b_op;
        beta_imag_a_op = alpha_imag_a_op + alpha_imag_b_op;
        beta_imag_b_op = alpha_imag_a_op - alpha_imag_b_op;

        if (n_samples_per_operand == 1)
        {
            Spec::store(out_real_a, beta_real_a_op);
            Spec::store(out_imag_a, beta_imag_a_op);
            Spec::store(out_real_b, beta_real_b_op);
            Spec::store(out_imag_b, beta_imag_b_op);
            return;
        }

        Spec::interleave2(
            alpha_real_a_op, alpha_imag_a_op, 
            alpha_real_b_op, alpha_imag_b_op,
            beta_real_a_op, beta_imag_a_op, 
            beta_real_b_op, beta_imag_b_op);
        
        // STORE
        Spec::store(out_real_a, alpha_real_a_op);
        Spec::store(out_imag_a, alpha_imag_a_op);
        Spec::store(out_real_b, alpha_real_b_op);
        Spec::store(out_imag_b, alpha_imag_b_op);
    }
}

#endif