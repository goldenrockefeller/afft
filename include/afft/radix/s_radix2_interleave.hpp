#ifndef AFFT_S_RADIX2_INTERLEAVE_HPP
#define AFFT_S_RADIX2_INTERLEAVE_HPP

#include <cstddef>

namespace afft
{
    template <typename Spec, bool Rescaling, bool HasTwiddles, std::size_t LogInterleaveFactor>
    inline void do_s_radix2_stage_interleave(
        typename Spec::sample *out_real,
        typename Spec::sample *out_imag,
        const typename Spec::sample *in_real,
        const typename Spec::sample *in_imag,
        const typename Spec::sample *tw_real_b_0,
        const typename Spec::sample *tw_imag_b_0,
        std::size_t subfft_id_start,
        std::size_t subfft_id_end,
        std::size_t n_samples,
        const typename Spec::sample &scaling_factor
    )
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

        operand scaling_factor_op(scaling_factor);

        operand tw_real_b_op;
        operand tw_imag_b_op;

        if (HasTwiddles)
        {
            Spec::load(tw_real_b_op, tw_real_b_0);
            Spec::load(tw_imag_b_op, tw_imag_b_0);
        }

        auto box_size = Spec::n_samples_per_operand * 2;

        std::size_t in_a_offset = subfft_id_start * Spec::n_samples_per_operand;
        std::size_t in_b_offset = subfft_id_start * Spec::n_samples_per_operand + n_samples / 2;

        std::size_t out_a_offset = subfft_id_start * box_size;
        std::size_t out_b_offset = subfft_id_start * box_size + Spec::n_samples_per_operand;

        auto in_real_a = in_real + in_a_offset;
        auto in_imag_a = in_imag + in_a_offset;
        auto in_real_b = in_real + in_b_offset;
        auto in_imag_b = in_imag + in_b_offset;

        auto out_real_a = out_real + out_a_offset;
        auto out_imag_a = out_imag + out_a_offset;
        auto out_real_b = out_real + out_b_offset;
        auto out_imag_b = out_imag + out_b_offset;


        for (
            std::size_t subfft_id = subfft_id_start;
            subfft_id < subfft_id_end;
            subfft_id++)
        {
            // LOAD
            Spec::load(alpha_real_a_op, in_real_a);
            Spec::load(alpha_imag_a_op, in_imag_a);
            Spec::load(alpha_real_b_op, in_real_b);
            Spec::load(alpha_imag_b_op, in_imag_b);
            

            // COMPUTE

            if (Rescaling) {
                alpha_real_b_op *= scaling_factor_op;
                alpha_imag_b_op *= scaling_factor_op;
                alpha_real_b_op *= scaling_factor_op;
                alpha_imag_b_op *= scaling_factor_op;
            }

            if (HasTwiddles)
            {
                beta_real_b_op = tw_imag_b_op * alpha_imag_b_op;
                beta_imag_b_op = tw_imag_b_op * alpha_real_b_op;

                alpha_real_b_op *= tw_real_b_op;
                alpha_imag_b_op *= tw_real_b_op;

                alpha_real_b_op -= beta_real_b_op;
                alpha_imag_b_op += beta_imag_b_op;
            
            }

            beta_real_a_op = alpha_real_a_op + alpha_real_b_op;
            beta_real_b_op = alpha_real_a_op - alpha_real_b_op;
            beta_imag_a_op = alpha_imag_a_op + alpha_imag_b_op;
            beta_imag_b_op = alpha_imag_a_op - alpha_imag_b_op;

            // STORE

            Spec::template interleave2<LogInterleaveFactor>(
                alpha_real_a_op, 
                alpha_real_b_op, 
                beta_real_a_op, 
                beta_real_b_op
            );

            Spec::template interleave2<LogInterleaveFactor>(
                alpha_imag_a_op, 
                alpha_imag_b_op, 
                beta_imag_a_op, 
                beta_imag_b_op
            );

            Spec::store(out_real_a, alpha_real_a_op);
            Spec::store(out_imag_a, alpha_imag_a_op);
            Spec::store(out_real_b, alpha_real_b_op);
            Spec::store(out_imag_b, alpha_imag_b_op);

            in_real_a += Spec::n_samples_per_operand;
            in_imag_a += Spec::n_samples_per_operand;
            in_real_b += Spec::n_samples_per_operand;
            in_imag_b += Spec::n_samples_per_operand;

            out_real_a += box_size;
            out_imag_a += box_size;
            out_real_b += box_size;
            out_imag_b += box_size;
        }
    }
}

#endif