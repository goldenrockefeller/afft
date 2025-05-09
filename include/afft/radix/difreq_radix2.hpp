#ifndef AFFT_DIFREQ_RADIX2_HPP
#define AFFT_DIFREQ_RADIX2_HPP

#include <cstddef>

namespace afft{
    template<typename Spec>
    inline void do_difreq_radix2_stage(
        typename Spec::sample* out_real,
        typename Spec::sample* out_imag,
        const typename Spec::sample* in_real, 
        const typename Spec::sample* in_imag, 
        const typename Spec::sample* tw_real_b, 
        const typename Spec::sample* tw_imag_b, 
        std::size_t subtwiddle_len,
        std::size_t subtwiddle_start,
        std::size_t subtwiddle_end
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
        
        operand tw_real_b_op;
        operand tw_imag_b_op;

        std::size_t a_offset = subtwiddle_start;
        std::size_t b_offset = a_offset + subtwiddle_len;

        // OFFSET
        auto in_real_a = in_real + a_offset;
        auto in_imag_a = in_imag + a_offset;
        auto in_real_b = in_real + b_offset;
        auto in_imag_b = in_imag + b_offset;
        auto out_real_a = out_real + a_offset;
        auto out_imag_a = out_imag + a_offset;
        auto out_real_b = out_real + b_offset;
        auto out_imag_b = out_imag + b_offset;
        auto tw_real_b = tw_real_b + subtwiddle_start;
        auto tw_imag_b = tw_imag_b + subtwiddle_start;

        for (
            std::size_t i = subtwiddle_start; 
            i < subtwiddle_end;
            i += n_samples_per_operand
        ) {
            
            //LOAD
            Spec::load(alpha_real_a_op, in_real_a);
            Spec::load(alpha_imag_a_op, in_imag_a);
            Spec::load(alpha_real_b_op, in_real_b);
            Spec::load(alpha_imag_b_op, in_imag_b);

            Spec::load(tw_real_b_op, tw_real_b);
            Spec::load(tw_imag_b_op, tw_imag_b);

            // COMPUTE 
            
            beta_real_a_op = alpha_real_a_op + alpha_real_b_op;
            beta_real_b_op = alpha_real_a_op - alpha_real_b_op;
            beta_imag_a_op = alpha_imag_a_op + alpha_imag_b_op;
            beta_imag_b_op = alpha_imag_a_op - alpha_imag_b_op;

            alpha_real_b_op = tw_imag_b_op * beta_imag_b_op;
            alpha_imag_b_op = tw_imag_b_op * beta_real_b_op;

            beta_real_b_op *= tw_real_b_op;
            beta_imag_b_op *= tw_real_b_op;

            beta_real_b_op -= alpha_real_b_op;
            beta_imag_b_op += alpha_imag_b_op;

            // STORE
            Spec::store(out_real_a, beta_real_a_op);
            Spec::store(out_imag_a, beta_imag_a_op);
            Spec::store(out_real_b, beta_real_b_op);
            Spec::store(out_imag_b, beta_imag_b_op);

            // UPDATE OFFSET
            in_real_a += n_samples_per_operand;
            in_imag_a += n_samples_per_operand;
            in_real_b += n_samples_per_operand;
            in_imag_b += n_samples_per_operand;
            out_real_a += n_samples_per_operand;
            out_imag_a += n_samples_per_operand;
            out_real_b += n_samples_per_operand;
            out_imag_b += n_samples_per_operand;
            tw_real_b += n_samples_per_operand;
            tw_imag_b += n_samples_per_operand;
        }
    }
}

#endif