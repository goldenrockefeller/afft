#ifndef AFFT_DITIME_RADIX4_HPP
#define AFFT_DITIME_RADIX4_HPP

#include <cstddef>

namespace afft{
    template<typename Spec>
    inline void do_ditime_radix4_stage(
        typename Spec::sample* out_real, 
        typename Spec::sample* out_imag, 
        const typename Spec::sample* in_real, 
        const typename Spec::sample* in_imag, 
        const typename Spec::sample* tw_real_b, 
        const typename Spec::sample* tw_imag_b, 
        const typename Spec::sample* tw_real_c, 
        const typename Spec::sample* tw_imag_c, 
        const typename Spec::sample* tw_real_d, 
        const typename Spec::sample* tw_imag_d,
        std::size_t subfft_id_start,
        std::size_t subfft_id_end,
        std::size_t subtwiddle_len,
        std::size_t subtwiddle_start,
        std::size_t subtwiddle_end
    ) {
        using operand = typename Spec::operand;
        constexpr std::size_t n_samples_per_operand = Spec::n_samples_per_operand;
        const std::size_t subfft_len = 4 * subtwiddle_len;

        // DECLARE
        operand alpha_real_a_op;
        operand alpha_real_b_op;
        operand alpha_imag_a_op;
        operand alpha_imag_b_op;
        operand beta_real_a_op;
        operand beta_real_b_op;
        operand beta_imag_a_op;
        operand beta_imag_b_op;

        operand alpha_real_c_op;
        operand alpha_real_d_op;
        operand alpha_imag_c_op;
        operand alpha_imag_d_op;
        operand beta_real_c_op;
        operand beta_real_d_op;
        operand beta_imag_c_op;
        operand beta_imag_d_op;
        
        operand tw_real_b_op;
        operand tw_imag_b_op;
        operand tw_real_c_op;
        operand tw_imag_c_op;
        operand tw_real_d_op;
        operand tw_imag_d_op;

        for (
            std::size_t subfft_id = subfft_id_start;
            subfft_id < subfft_id_end;
            subfft_id++
        ) {
            std::size_t a_offset = subfft_id * subfft_len + subtwiddle_start;
            std::size_t b_offset = a_offset + subtwiddle_len;
            std::size_t c_offset = a_offset + 2 * subtwiddle_len;
            std::size_t d_offset = a_offset + 3 * subtwiddle_len;

            // OFFSET
            auto in_real_a = in_real + a_offset;
            auto in_imag_a = in_imag + a_offset;
            auto in_real_b = in_real + b_offset;
            auto in_imag_b = in_imag + b_offset;
            auto in_real_c = in_real + c_offset;
            auto in_imag_c = in_imag + c_offset;
            auto in_real_d = in_real + d_offset;
            auto in_imag_d = in_imag + d_offset;

            auto out_real_a = out_real + a_offset;
            auto out_imag_a = out_imag + a_offset;
            auto out_real_b = out_real + b_offset;
            auto out_imag_b = out_imag + b_offset;
            auto out_real_c = out_real + c_offset;
            auto out_imag_c = out_imag + c_offset;
            auto out_real_d = out_real + d_offset;
            auto out_imag_d = out_imag + d_offset;

            auto tw_real_b = tw_real_b + subtwiddle_start;
            auto tw_imag_b = tw_imag_b + subtwiddle_start;
            auto tw_real_c = tw_real_c + subtwiddle_start;
            auto tw_imag_c = tw_imag_c + subtwiddle_start;
            auto tw_real_d = tw_real_d + subtwiddle_start;
            auto tw_imag_d = tw_imag_d + subtwiddle_start;

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
                Spec::load(alpha_real_c_op, in_real_c);
                Spec::load(alpha_imag_c_op, in_imag_c);
                Spec::load(alpha_real_d_op, in_real_d);
                Spec::load(alpha_imag_d_op, in_imag_d);

                Spec::load(tw_real_b_op, tw_real_b);
                Spec::load(tw_imag_b_op, tw_imag_b);
                Spec::load(tw_real_c_op, tw_real_c);
                Spec::load(tw_imag_c_op, tw_imag_c);
                Spec::load(tw_real_d_op, tw_real_d);
                Spec::load(tw_imag_d_op, tw_imag_d);

                // COMPUTE 
                
                beta_real_b_op = tw_imag_b_op * alpha_imag_b_op;
                beta_real_c_op = tw_imag_c_op * alpha_imag_c_op;
                beta_real_d_op = tw_imag_d_op * alpha_imag_d_op;

                beta_imag_b_op = tw_imag_b_op * alpha_real_b_op;
                beta_imag_c_op = tw_imag_c_op * alpha_real_c_op;
                beta_imag_d_op = tw_imag_d_op * alpha_real_d_op;

                alpha_real_b_op *= tw_real_b_op;
                alpha_imag_b_op *= tw_real_b_op;
                alpha_real_c_op *= tw_real_c_op;
                alpha_imag_c_op *= tw_real_c_op;
                alpha_real_d_op *= tw_real_d_op;
                alpha_imag_d_op *= tw_real_d_op;

                alpha_real_b_op -= beta_real_b_op;
                alpha_real_c_op -= beta_real_c_op;
                alpha_real_d_op -= beta_real_d_op;
                alpha_imag_b_op += beta_imag_b_op;
                alpha_imag_c_op += beta_imag_c_op;
                alpha_imag_d_op += beta_imag_d_op;    
                
                beta_real_a_op = alpha_real_a_op + alpha_real_b_op;
                beta_real_b_op = alpha_real_a_op - alpha_real_b_op;
                beta_real_c_op = alpha_real_c_op + alpha_real_d_op;
                beta_real_d_op = alpha_real_c_op - alpha_real_d_op;

                beta_imag_a_op = alpha_imag_a_op + alpha_imag_b_op;
                beta_imag_b_op = alpha_imag_a_op - alpha_imag_b_op;
                beta_imag_c_op = alpha_imag_c_op + alpha_imag_d_op;
                beta_imag_d_op = alpha_imag_c_op - alpha_imag_d_op;  

                alpha_real_a_op = beta_real_a_op + beta_real_c_op;
                alpha_real_b_op = beta_real_b_op + beta_imag_d_op;
                alpha_real_c_op = beta_real_a_op - beta_real_c_op;
                alpha_real_d_op = beta_real_b_op - beta_imag_d_op;

                alpha_imag_a_op = beta_imag_a_op + beta_imag_c_op;
                alpha_imag_b_op = beta_imag_b_op - beta_real_d_op;
                alpha_imag_c_op = beta_imag_a_op - beta_imag_c_op;
                alpha_imag_d_op = beta_imag_b_op + beta_real_d_op; 

                // STORE

                Spec::store(out_real_a, alpha_real_a_op);
                Spec::store(out_imag_a, alpha_imag_a_op);
                Spec::store(out_real_b, alpha_real_b_op);
                Spec::store(out_imag_b, alpha_imag_b_op);
                Spec::store(out_real_c, alpha_real_c_op);
                Spec::store(out_imag_c, alpha_imag_c_op);
                Spec::store(out_real_d, alpha_real_d_op);
                Spec::store(out_imag_d, alpha_imag_d_op);

                // UPDATE OFFSET
                in_real_a += n_samples_per_operand;
                in_imag_a += n_samples_per_operand;
                in_real_b += n_samples_per_operand;
                in_imag_b += n_samples_per_operand;
                in_real_c += n_samples_per_operand;
                in_imag_c += n_samples_per_operand;
                in_real_d += n_samples_per_operand;
                in_imag_d += n_samples_per_operand;

                out_real_a += n_samples_per_operand;
                out_imag_a += n_samples_per_operand;
                out_real_b += n_samples_per_operand;
                out_imag_b += n_samples_per_operand;
                out_real_c += n_samples_per_operand;
                out_imag_c += n_samples_per_operand;
                out_real_d += n_samples_per_operand;
                out_imag_d += n_samples_per_operand;

                tw_real_b += n_samples_per_operand;
                tw_imag_b += n_samples_per_operand;
                tw_real_c += n_samples_per_operand;
                tw_imag_c += n_samples_per_operand;
                tw_real_d += n_samples_per_operand;
                tw_imag_d += n_samples_per_operand;
            }
        }
    }
}

#endif