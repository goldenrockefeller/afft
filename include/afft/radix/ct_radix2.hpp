#ifndef AFFT_CT_RADIX2_HPP
#define AFFT_CT_RADIX2_HPP

#include <cstddef>

namespace afft{
    template<typename Spec>
    inline void do_ct_radix2_stage(
        typename Spec::sample* out_real,
        typename Spec::sample* out_imag,
        const typename Spec::sample* twiddles,
        std::size_t subtwiddle_len,
        std::size_t subtwiddle_start,
        std::size_t subtwiddle_end,
        std::ptrdiff_t output_offset = 0
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
        auto out_real_a = out_real + a_offset;
        auto out_imag_a = out_imag + a_offset;
        auto out_real_b = out_real + b_offset;
        auto out_imag_b = out_imag + b_offset;
        auto tw_real_b = twiddles + 2 * subtwiddle_start;
        auto tw_imag_b = tw_real_b + n_samples_per_operand;

        const std::size_t data_stride = n_samples_per_operand;
        const std::size_t twiddle_stride = 2*n_samples_per_operand;

        for (
            std::size_t i = subtwiddle_start; 
            i < subtwiddle_end;
            i += data_stride
        ) {
            
            //LOAD
            Spec::load(alpha_real_a_op, out_real_a);
            Spec::load(alpha_imag_a_op, out_imag_a);
            Spec::load(alpha_real_b_op, out_real_b);
            Spec::load(alpha_imag_b_op, out_imag_b);

            Spec::load(tw_real_b_op, tw_real_b);
            Spec::load(tw_imag_b_op, tw_imag_b);

            // COMPUTE 
            beta_real_b_op = tw_imag_b_op * alpha_imag_b_op;
            beta_imag_b_op = tw_imag_b_op * alpha_real_b_op;

            alpha_real_b_op *= tw_real_b_op;
            alpha_imag_b_op *= tw_real_b_op;

            alpha_real_b_op -= beta_real_b_op;
            alpha_imag_b_op += beta_imag_b_op;
            
            beta_real_a_op = alpha_real_a_op + alpha_real_b_op;
            beta_real_b_op = alpha_real_a_op - alpha_real_b_op;
            beta_imag_a_op = alpha_imag_a_op + alpha_imag_b_op;
            beta_imag_b_op = alpha_imag_a_op - alpha_imag_b_op;

            // STORE
            Spec::store(out_real_a, beta_real_a_op);
            Spec::store(out_imag_a, beta_imag_a_op);
            Spec::store(out_real_b, beta_real_b_op);
            Spec::store(out_imag_b, beta_imag_b_op);

            // UPDATE OFFSET
            out_real_a += data_stride;
            out_imag_a += data_stride;
            out_real_b += data_stride;
            out_imag_b += data_stride;
            tw_real_b += twiddle_stride;
            tw_imag_b += twiddle_stride;
        }
    }
}

#endif