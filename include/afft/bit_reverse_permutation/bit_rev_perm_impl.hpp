#ifndef AFFT_BIT_REV_PERM_IMPL_HPP
#define AFFT_BIT_REV_PERM_IMPL_HPP

#include <cstddef>
#include <vector>
#include "afft/bit_reverse_permutation/bit_rev_perm_plan.hpp"

namespace afft
{
    template<typename Spec>
    class BitRevPermImpl {
        BitRevPermPlan plan_;

    public:
        explicit BitRevPermImpl(std::size_t n_indexes) : plan_(n_indexes, Spec::n_samples_per_operand) {}

        const BitRevPermPlan& plan() const {
            return plan_;
        }

        static void eval(
            typename Spec::sample* out_real, 
            typename Spec::sample* out_imag,
            const typename Spec::sample* in_real, 
            const typename Spec::sample* in_imag,
            const BitRevPermPlan& plan
        ) {
            using operand = typename Spec::operand;
            constexpr std::size_t n_samples_per_operand = Spec::n_samples_per_operand;
            BitRevPermPlanType plan_type = plan.type();
            const std::vector<std::size_t>& plan_indexes = plan.plan_indexes();
            const std::vector<std::size_t>& off_diagonal_streak_lens = plan.off_diagonal_streak_lens();
            auto plan_indexes_data = plan_indexes.data();
            
            std::size_t transpose_id = 0;
    
            switch (plan_type){
            case BitRevPermPlanType::n_indexes_equals_base_size_sqr:
                Spec::transpose_diagonal(out_real, out_imag, in_real, in_imag, plan_indexes_data);
                return;
    
            case BitRevPermPlanType::n_indexes_equals_2_base_size_sqr:
                Spec::transpose_diagonal(out_real, out_imag, in_real, in_imag, plan_indexes_data);
                Spec::transpose_diagonal(out_real, out_imag, in_real, in_imag, plan_indexes_data + n_samples_per_operand);
                return;
    
            case BitRevPermPlanType::n_indexes_equals_4_base_size_sqr:
                Spec::transpose_diagonal(out_real, out_imag, in_real, in_imag, plan_indexes_data);
                Spec::transpose_off_diagonal(out_real, out_imag, in_real, in_imag, plan_indexes_data + n_samples_per_operand);
                Spec::transpose_diagonal(out_real, out_imag, in_real, in_imag, plan_indexes_data + 3 * n_samples_per_operand);
                return;
    
            case BitRevPermPlanType::n_indexes_equals_8_base_size_sqr:
                Spec::transpose_diagonal(out_real, out_imag, in_real, in_imag, plan_indexes_data);
                Spec::transpose_diagonal(out_real, out_imag, in_real, in_imag, plan_indexes_data + n_samples_per_operand);
                Spec::transpose_off_diagonal(out_real, out_imag, in_real, in_imag, plan_indexes_data + 2 * n_samples_per_operand);
                Spec::transpose_off_diagonal(out_real, out_imag, in_real, in_imag, plan_indexes_data + 4 * n_samples_per_operand);
                Spec::transpose_diagonal(out_real, out_imag, in_real, in_imag, plan_indexes_data + 6 * n_samples_per_operand);
                Spec::transpose_diagonal(out_real, out_imag, in_real, in_imag, plan_indexes_data + 7 * n_samples_per_operand);
                return;
    
            case BitRevPermPlanType::indexes_mat_is_large_square:
                Spec::transpose_diagonal(out_real, out_imag, in_real, in_imag, plan_indexes_data);
                Spec::transpose_off_diagonal(out_real, out_imag, in_real, in_imag, plan_indexes_data + n_samples_per_operand);
                Spec::transpose_diagonal(out_real, out_imag, in_real, in_imag, plan_indexes_data + 3 * n_samples_per_operand);
                transpose_id = 4;
    
                for (std::size_t streak_len : off_diagonal_streak_lens) {
                    for (std::size_t off_diagonal_id = 0; off_diagonal_id < streak_len; ++off_diagonal_id) {
                        Spec::transpose_off_diagonal(out_real, out_imag, in_real, in_imag, plan_indexes_data + transpose_id * n_samples_per_operand);
                        transpose_id += 2;
                    }
    
                Spec::transpose_diagonal(out_real, out_imag, in_real, in_imag, plan_indexes_data + transpose_id * n_samples_per_operand);
                Spec::transpose_off_diagonal(out_real, out_imag, in_real, in_imag, plan_indexes_data + (transpose_id + 1) * n_samples_per_operand);
                Spec::transpose_diagonal(out_real, out_imag, in_real, in_imag, plan_indexes_data + (transpose_id + 3) * n_samples_per_operand);
    
                    transpose_id += 4;
                }
                return;
    
            case BitRevPermPlanType::indexes_mat_is_large_nonsquare:
                Spec::transpose_diagonal(out_real, out_imag, in_real, in_imag, plan_indexes_data);
                Spec::transpose_diagonal(out_real, out_imag, in_real, in_imag, plan_indexes_data + n_samples_per_operand);
                Spec::transpose_off_diagonal(out_real, out_imag, in_real, in_imag, plan_indexes_data + 2 * n_samples_per_operand);
                Spec::transpose_off_diagonal(out_real, out_imag, in_real, in_imag, plan_indexes_data + 4 * n_samples_per_operand);
                Spec::transpose_diagonal(out_real, out_imag, in_real, in_imag, plan_indexes_data + 6 * n_samples_per_operand);
                Spec::transpose_diagonal(out_real, out_imag, in_real, in_imag, plan_indexes_data + 7 * n_samples_per_operand);
    
                transpose_id = 8;
    
                for (std::size_t streak_len : off_diagonal_streak_lens) {
                    for (std::size_t off_diagonal_id = 0; off_diagonal_id < streak_len; ++off_diagonal_id) {
                        Spec::transpose_off_diagonal(out_real, out_imag, in_real, in_imag, plan_indexes_data + transpose_id * n_samples_per_operand);
                        Spec::transpose_off_diagonal(out_real, out_imag, in_real, in_imag, plan_indexes_data + (transpose_id + 2) * n_samples_per_operand);
                        transpose_id += 4;
                    }
    
                    Spec::transpose_diagonal(out_real, out_imag, in_real, in_imag, plan_indexes_data + transpose_id * n_samples_per_operand);
                    Spec::transpose_diagonal(out_real, out_imag, in_real, in_imag, plan_indexes_data + (transpose_id + 1) * n_samples_per_operand);
                    Spec::transpose_off_diagonal(out_real, out_imag, in_real, in_imag, plan_indexes_data + (transpose_id + 2) * n_samples_per_operand);
                    Spec::transpose_off_diagonal(out_real, out_imag, in_real, in_imag, plan_indexes_data + (transpose_id + 4) * n_samples_per_operand);
                    Spec::transpose_diagonal(out_real, out_imag, in_real, in_imag, plan_indexes_data + (transpose_id + 6) * n_samples_per_operand);
                    Spec::transpose_diagonal(out_real, out_imag, in_real, in_imag, plan_indexes_data + (transpose_id + 7) * n_samples_per_operand);
    
                    transpose_id += 8;
                }
                return;
            }
        }

        void eval(
            typename Spec::sample* out_real, 
            typename Spec::sample* out_imag,
            const typename Spec::sample* in_real, 
            const typename Spec::sample* in_imag
        ) const {
            eval (
                out_real,
                out_imag,
                in_real,
                in_imag,
                plan_
            );
        }
    };
}

#endif