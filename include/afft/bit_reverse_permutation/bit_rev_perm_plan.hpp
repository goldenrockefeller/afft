#ifndef AFFT_BIT_REV_PERM_PLAN_HPP
#define AFFT_BIT_REV_PERM_PLAN_HPP

#include <cstddef>
#include <vector>
#include <algorithm>
#include "afft/bit_reverse_permutation/bit_rev_perm_plan_type.hpp"
#include "afft/bit_reverse_permutation/plan_indexes_manipulation.hpp"
#include "afft/common_math.hpp"

namespace afft
{
    class BitRevPermPlan
    {        
        std::size_t log_n_sample_per_operand_;
        BitRevPermPlanType type_;
        std::vector<std::size_t> plan_indexes_;
        std::vector<std::size_t> off_diagonal_streak_lens_;

    public:
    std::size_t log_n_sample_per_operand() const {
            return log_n_sample_per_operand_;
        }

        BitRevPermPlanType type() const {
            return type_;
        }

        const std::vector<std::size_t>& plan_indexes() const {
            return plan_indexes_;
        }

        const std::vector<std::size_t>& off_diagonal_streak_lens() const {
            return off_diagonal_streak_lens_;
        }

        BitRevPermPlan(std::size_t n_indexes, std::size_t max_n_samples_per_operand)
        {
            using common_math::int_log_2;
            namespace pim = plan_indexes_manipulation;
            std::size_t n_bits = int_log_2(n_indexes);
            bool indexes_mat_is_large_square = n_bits % 2 == 0;

            log_n_sample_per_operand_ = 
                std::min(
                    int_log_2(max_n_samples_per_operand),
                    int_log_2(n_indexes) / 2 
                );

            std::size_t n_samples_per_operand = 1 << log_n_sample_per_operand_;
            
            if (n_indexes == n_samples_per_operand * n_samples_per_operand)
            {
                type_ = BitRevPermPlanType::n_indexes_equals_base_size_sqr;
            }
            else if (n_indexes == 2 * n_samples_per_operand * n_samples_per_operand)
            {
                type_ = BitRevPermPlanType::n_indexes_equals_2_base_size_sqr;
            }
            else if (n_indexes == 4 * n_samples_per_operand * n_samples_per_operand)
            {
                type_ = BitRevPermPlanType::n_indexes_equals_4_base_size_sqr;
            }
            else if (n_indexes == 8 * n_samples_per_operand * n_samples_per_operand)
            {
                type_ = BitRevPermPlanType::n_indexes_equals_8_base_size_sqr;
            }
            else if (indexes_mat_is_large_square)
            {
                type_ = BitRevPermPlanType::indexes_mat_is_large_square;
            }
            else
            {
                type_ = BitRevPermPlanType::indexes_mat_is_large_nonsquare;
            }
            
            std::vector<std::vector<std::vector<std::size_t>>> indexes_as_mats_ = pim::indexes_as_mats(n_indexes);
            std::vector<std::vector<std::vector<std::vector<std::size_t>>>> pre_plans_indexes_;

            for (auto &indexes_as_mat_ : indexes_as_mats_)
            {
                indexes_as_mat_ = pim::bit_rev_permute_rows(indexes_as_mat_);
                pre_plans_indexes_.push_back(pim::plan_transpose_diagonal_indexes(indexes_as_mat_, n_samples_per_operand));
            }
            
            off_diagonal_streak_lens_ = pim::off_diagonal_streak_lens(pre_plans_indexes_[0]);
            std::size_t transpose_pair_id = 0;
            
            // Compress diagonal
            for (auto &pre_plan : pre_plans_indexes_)
            {
                for (std::size_t j = 0; j < n_samples_per_operand; ++j)
                {
                    plan_indexes_.push_back(pre_plan[transpose_pair_id][0][j]);
                }
            }
            
            ++transpose_pair_id;

            if (n_indexes == n_samples_per_operand * n_samples_per_operand || n_indexes == 2 * n_samples_per_operand * n_samples_per_operand)
            {
                off_diagonal_streak_lens_ = {};
                return;
            }
            
            for (auto &pre_plan : pre_plans_indexes_)
            {
                for (std::size_t j = 0; j < n_samples_per_operand; ++j)
                {
                    plan_indexes_.push_back(pre_plan[transpose_pair_id][0][j]);
                }
                for (std::size_t j = 0; j < n_samples_per_operand; ++j)
                {
                    plan_indexes_.push_back(pre_plan[transpose_pair_id][1][j]);
                }
            }
            
            ++transpose_pair_id;

            for (auto &pre_plan : pre_plans_indexes_)
            {
                for (std::size_t j = 0; j < n_samples_per_operand; ++j)
                {
                    plan_indexes_.push_back(pre_plan[transpose_pair_id][0][j]);
                }
            }
            
            ++transpose_pair_id;

            // Process off-diagonal streaks
            for (auto streak_len : off_diagonal_streak_lens_)
            {
                for (std::size_t off_diagonal_id = 0; off_diagonal_id < streak_len; ++off_diagonal_id)
                {
                    for (auto &pre_plan : pre_plans_indexes_)
                    {
                        for (std::size_t j = 0; j < n_samples_per_operand; ++j)
                        {
                            plan_indexes_.push_back(pre_plan[transpose_pair_id][0][j]);
                        }
                        for (std::size_t j = 0; j < n_samples_per_operand; ++j)
                        {
                            plan_indexes_.push_back(pre_plan[transpose_pair_id][1][j]);
                        }
                    }

                    ++transpose_pair_id;
                }

                for (auto &pre_plan : pre_plans_indexes_)
                {
                    for (std::size_t j = 0; j < n_samples_per_operand; ++j)
                    {
                        plan_indexes_.push_back(pre_plan[transpose_pair_id][0][j]);
                    }
                }
                ++transpose_pair_id;

                for (auto &pre_plan : pre_plans_indexes_)
                {
                    for (std::size_t j = 0; j < n_samples_per_operand; ++j)
                    {
                        plan_indexes_.push_back(pre_plan[transpose_pair_id][0][j]);
                    }
                    for (std::size_t j = 0; j < n_samples_per_operand; ++j)
                    {
                        plan_indexes_.push_back(pre_plan[transpose_pair_id][1][j]);
                    }
                }
                ++transpose_pair_id;

                for (auto &pre_plan : pre_plans_indexes_)
                {
                    for (std::size_t j = 0; j < n_samples_per_operand; ++j)
                    {
                        plan_indexes_.push_back(pre_plan[transpose_pair_id][0][j]);
                    }
                }
                ++transpose_pair_id;
            }
        }
    };
}

#endif
