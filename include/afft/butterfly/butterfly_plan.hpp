#ifndef AFFT_BUTTERFLY_PLAN_HPP
#define AFFT_BUTTERFLY_PLAN_HPP

#include <vector>
#include <cstddef>

#include <algorithm>

#include "afft/radix/radix_stage/radix_stage.hpp"
#include "afft/radix/radix_stage/radix_type.hpp"
#include "afft/butterfly/twiddles.hpp"
#include "afft/bit_reverse_permutation/plan_indexes_manipulation.hpp"

namespace afft
{
    template <typename Spec, class Allocator = std::allocator<typename Spec::sample>>
    class ButterflyPlan
    {
        static_assert(Spec::n_samples_per_operand == 1, "Butterfly must use sample spec");

        std::size_t log_n_samples_per_operand_;
        std::size_t n_s_radix_stages_;
        std::size_t n_samples_;
        std::vector<RadixStage<Spec>> radix_stages_;
        std::vector<typename Spec::sample, Allocator> twiddles_;
        std::vector<std::size_t> out_indexes;
        std::vector<std::size_t> in_indexes;
        std::vector<std::size_t> inout_indexes;
        typename Spec::sample scaling_factor_;

    public:
        const std::size_t &log_n_samples_per_operand() const
        {
            return log_n_samples_per_operand_;
        }

        const std::size_t &n_s_radix_stages() const
        {
            return n_s_radix_stages_;
        }

        const std::size_t &n_samples() const
        {
            return n_samples_;
        }

        // Getter for radix_stages_
        const std::vector<RadixStage<Spec>> &radix_stages() const
        {
            return radix_stages_;
        }

        const typename Spec::sample &scaling_factor() const
        {
            return scaling_factor_;
        }

        ButterflyPlan(std::size_t n_samples, std::size_t max_n_samples_per_operand, std::size_t prefetch_lookahead, std::size_t min_partition_len) : n_samples_(n_samples)
        {
            namespace tw = afft::twiddles;
            namespace cm = afft::common_math;
            namespace pim = afft::plan_indexes_manipulation;
            using sample = typename Spec::sample;

            scaling_factor_ = sample(1) / sample(n_samples);
            n_s_radix_stages_ = 0;

            if (n_samples <= 1)
            {
                return;
            }

            log_n_samples_per_operand_ =
                std::min(
                    cm::int_log_2(max_n_samples_per_operand),
                    std::size_t(cm::int_log_2(n_samples / 2)));

            std::size_t n_samples_per_operand = 1 << log_n_samples_per_operand_;

            twiddles_.resize(2 * n_samples + 2 * n_samples_per_operand * 6 * (n_samples_per_operand + 1));
            std::size_t tw_pos = 0;

            std::size_t indexes_len;
            if (n_samples == 2 * n_samples_per_operand) {
                indexes_len = n_samples / n_samples_per_operand / 2;
            } 
            else {
                indexes_len = n_samples / n_samples_per_operand / 4;
            }

            for (std::size_t i = 0; i < indexes_len; i++) {
                inout_indexes.push_back(i);
            }

            auto pair_indexes = pim::ordered_bit_rev_indexes(indexes_len);
            in_indexes = pair_indexes.first;
            out_indexes = pair_indexes.second;

            for (std::size_t i = 0; i < prefetch_lookahead; i++) {
                inout_indexes.push_back(indexes_len);
                in_indexes.push_back(indexes_len);
                out_indexes.push_back(indexes_len);
            }

            std::size_t subtwiddle_len = 1;
            std::size_t log_subtwiddle_len = 0;

            std::size_t input_id = 0; // Input, Output, Buffer or Input, Buffer, Output 
            std::size_t output_id = 0; // Input, Output, Buffer or Input, Buffer, Output
            
            if (n_samples == 2 * n_samples_per_operand) {
                // Stockham x2 Interleave

                while (subtwiddle_len < n_samples_per_operand) {
                    auto stage_twiddles = tw::s_radix2_twiddles<Spec, Allocator>(subtwiddle_len, n_samples_per_operand);

                    RadixStage<Spec> radix_stage_;
                    radix_stage_.type = RadixType::s_radix2;
                    auto &params = radix_stage_.params.s_r2;
                    params.twiddles = twiddles_.data() + tw_pos;
                    params.out_indexes = inout_indexes.data();
                    params.in_indexes = inout_indexes.data();
                    params.subfft_id_start = 0;
                    params.subfft_id_end = n_samples / n_samples_per_operand / 2;
                    params.log_subtwiddle_len = log_subtwiddle_len;
                    params.input_id = input_id;
                    params.output_id = output_id;

                    for (auto twiddle : stage_twiddles) {
                        twiddles_[tw_pos] = twiddle;
                        tw_pos++;
                    }

                    input_id = output_id;
                    output_id ^= 1;

                    radix_stages_.push_back(radix_stage_);
                    subtwiddle_len *= 2;
                    log_subtwiddle_len += 1;
                    n_s_radix_stages_ += 1;

                }
                // Stockham x2 Permute
                auto stage_twiddles = tw::s_radix2_twiddles<Spec, Allocator>(subtwiddle_len, n_samples_per_operand);

                RadixStage<Spec> radix_stage_;
                radix_stage_.type = RadixType::s_radix2;
                auto &params = radix_stage_.params.s_r2;
                params.twiddles = twiddles_.data() + tw_pos;
                params.out_indexes = out_indexes.data();
                params.in_indexes = in_indexes.data();
                params.subfft_id_start = 0;
                params.subfft_id_end = n_samples / n_samples_per_operand / 2;
                params.log_subtwiddle_len = log_subtwiddle_len;
                params.input_id = input_id;
                params.output_id = output_id;

                for (auto twiddle : stage_twiddles) {
                    twiddles_[tw_pos] = twiddle;
                    tw_pos++;
                }

                radix_stages_.push_back(radix_stage_);
                subtwiddle_len *= 2;
                log_subtwiddle_len += 1;
                n_s_radix_stages_ += 1;

                if (output_id == 0) { // the last output should not be the buffer
                    for (auto &stage : radix_stages_) {
                        if (stage.type ==  RadixType::s_radix4) {
                            auto &params = stage.params.s_r4;
                            params.input_id ^= 1;
                            params.output_id ^= 1;
                        }

                        if (stage.type ==  RadixType::s_radix2) {
                            auto &params = stage.params.s_r2;
                            params.input_id ^= 1;
                            params.output_id ^= 1;
                        }
                    }
                }

                return;
            }


            // Stockham x4 Interleave
            while (subtwiddle_len * 4 <= n_samples_per_operand) {
                auto stage_twiddles = tw::s_radix4_twiddles<Spec, Allocator>(subtwiddle_len, n_samples_per_operand);

                RadixStage<Spec> radix_stage_;
                radix_stage_.type = RadixType::s_radix4;
                auto &params = radix_stage_.params.s_r4;
                params.twiddles = twiddles_.data() + tw_pos;
                params.out_indexes = inout_indexes.data();
                params.in_indexes = inout_indexes.data();
                params.subfft_id_start = 0;
                params.subfft_id_end = n_samples / n_samples_per_operand / 4;
                params.log_subtwiddle_len = log_subtwiddle_len;
                params.input_id = input_id;
                params.output_id = output_id;

                for (auto twiddle : stage_twiddles) {
                    twiddles_[tw_pos] = twiddle;
                    tw_pos++;
                }

                // input_id = output_id;
                // output_id ^= 1;

                radix_stages_.push_back(radix_stage_);
                subtwiddle_len *= 4;
                log_subtwiddle_len += 2;
                n_s_radix_stages_ += 1;
            }

            // Stockham x2 Interleave
            if (subtwiddle_len < n_samples_per_operand) {

                auto stage_twiddles = tw::s_radix2_twiddles<Spec, Allocator>(subtwiddle_len, n_samples_per_operand);

                
                RadixStage<Spec> radix_stage_;
                radix_stage_.type = RadixType::s_radix2;
                auto &params = radix_stage_.params.s_r2;
                params.twiddles = twiddles_.data() + tw_pos;
                params.out_indexes = inout_indexes.data();
                params.in_indexes = inout_indexes.data();
                params.subfft_id_start = 0;
                params.subfft_id_end = n_samples / n_samples_per_operand / 2;
                params.log_subtwiddle_len = log_subtwiddle_len;
                params.input_id = input_id;
                params.output_id = output_id;

                for (auto twiddle : stage_twiddles) {
                    twiddles_[tw_pos] = twiddle;
                    tw_pos++;
                }

                input_id = output_id;
                output_id ^= 1;

                radix_stages_.push_back(radix_stage_);
                subtwiddle_len *= 2;
                log_subtwiddle_len += 1;
                n_s_radix_stages_ += 1;
            }

            // Stockham x4 Permute
            auto stage_twiddles = tw::s_radix4_twiddles<Spec, Allocator>(subtwiddle_len, n_samples_per_operand);

            RadixStage<Spec> radix_stage_;
            radix_stage_.type = RadixType::s_radix4;
            auto &params = radix_stage_.params.s_r4;
            params.twiddles = twiddles_.data() + tw_pos;
            params.out_indexes = out_indexes.data();
            params.in_indexes = in_indexes.data();
            params.subfft_id_start = 0;
            params.subfft_id_end = n_samples / n_samples_per_operand / 4;
            params.log_subtwiddle_len = log_subtwiddle_len;

            params.input_id = input_id;
            params.output_id = output_id;

            for (auto twiddle : stage_twiddles) {
                twiddles_[tw_pos] = twiddle;
                tw_pos++;
            }

            // radix_stages_.push_back(radix_stage_);
            // subtwiddle_len *= 4;
            // log_subtwiddle_len += 2;
            // n_s_radix_stages_ += 1;

            if (output_id == 0) { // the last output should not be the buffer
                for (auto &stage : radix_stages_) {
                    if (stage.type ==  RadixType::s_radix4) {
                        auto &params = stage.params.s_r4;
                        params.input_id ^= 1;
                        params.output_id ^= 1;
                    }

                    if (stage.type ==  RadixType::s_radix2) {
                        auto &params = stage.params.s_r2;
                        params.input_id ^= 1;
                        params.output_id ^= 1;
                    }
                }
            }
            

            // Cooley Tukey
            bool is_first_ct_radix_stage = true;
            while (subtwiddle_len < n_samples)
            {

                if (2 * subtwiddle_len == n_samples)
                {
                    auto stage_twiddles = tw::ct_radix2_twiddles<Spec, Allocator>(subtwiddle_len, n_samples_per_operand);

                    RadixStage<Spec> radix_stage_;
                    radix_stage_.type = RadixType::ct_radix2;
                    auto &params = radix_stage_.params.ct_r2; 
                    params.twiddles = twiddles_.data() + tw_pos;
                    params.subtwiddle_len = subtwiddle_len;
                    params.subtwiddle_start = 0;
                    params.subtwiddle_end = subtwiddle_len;

                    for (auto twiddle : stage_twiddles) {
                        twiddles_[tw_pos] = twiddle;
                        tw_pos++;
                    }

                    radix_stages_.push_back(radix_stage_);
                    subtwiddle_len *= 2;
                    log_subtwiddle_len += 1;
                    break;
                }
                else
                {
                    auto stage_twiddles = tw::ct_radix4_twiddles<Spec, Allocator>(subtwiddle_len, n_samples_per_operand);

                    RadixStage<Spec> radix_stage_;
                    radix_stage_.type = RadixType::ct_radix4;
                    auto &params = radix_stage_.params.ct_r4; 
                    params.twiddles = twiddles_.data() + tw_pos;
                    params.subfft_id_start = 0;
                    params.subfft_id_end = n_samples / subtwiddle_len / 4;
                    params.subtwiddle_len = subtwiddle_len;
                    params.subtwiddle_start = 0;
                    params.subtwiddle_end = subtwiddle_len;
                    
                    for (auto twiddle : stage_twiddles) {
                        twiddles_[tw_pos] = twiddle;
                        tw_pos++;
                    }

                    radix_stages_.push_back(radix_stage_);
                    subtwiddle_len *= 4;
                    log_subtwiddle_len += 2;
                }
            }

            // This version of the constructor turns an iterative butterfly plan into a hybrid/recursive plan
            std::vector<std::size_t> open_radix_stage_ids;
            std::vector<std::size_t> open_radix_stage_partition_counts;
            std::vector<std::size_t> open_radix_stage_partition_ids;

            if (n_samples < min_partition_len) {
                return;
            }

            auto initial_radix_stages = radix_stages_;
            radix_stages_.clear();
            auto reversed_radix_stages_ = radix_stages_;

            // Add the Stockham stages back in 
            for (std::size_t stage_id = 0; stage_id < n_s_radix_stages_; stage_id++) {
                radix_stages_.push_back(initial_radix_stages[stage_id]);
            }

            if (radix_stages_.size() == initial_radix_stages.size()) {
                return;
            }

            open_radix_stage_ids.push_back(initial_radix_stages.size() - 1);
            open_radix_stage_partition_counts.push_back(1);
            open_radix_stage_partition_ids.push_back(0);


            while (open_radix_stage_ids.size() > 0)
            {
                auto radix_stage_id = open_radix_stage_ids.back();
                open_radix_stage_ids.pop_back();

                auto radix_stage_partition_count = open_radix_stage_partition_counts.back();
                open_radix_stage_partition_counts.pop_back();

                auto radix_stage_partition_id = open_radix_stage_partition_ids.back();
                open_radix_stage_partition_ids.pop_back();

                std::size_t sub_stage_partition_count;
                auto radix_stage = initial_radix_stages[radix_stage_id];

                auto radix_stage_partition = radix_stage;

                if (initial_radix_stages[radix_stage_id].type == RadixType::s_radix4) {
                    continue; // Don't partition Stockham Stages
                }

                if (initial_radix_stages[radix_stage_id].type == RadixType::s_radix2) {
                    continue; // Don't partition Stockham Stages
                }

                if (radix_stage.type == RadixType::ct_radix4)
                {
                    sub_stage_partition_count = 4;
                    radix_stage_partition.params.ct_r4.subfft_id_start = radix_stage.params.ct_r4.subfft_id_end * radix_stage_partition_id / radix_stage_partition_count;
                    radix_stage_partition.params.ct_r4.subfft_id_end = radix_stage.params.ct_r4.subfft_id_end * (radix_stage_partition_id + 1) / radix_stage_partition_count;
                }
                else if (radix_stage.type == RadixType::ct_radix2) {
                    sub_stage_partition_count = 2;
                }

                reversed_radix_stages_.push_back(radix_stage_partition);

                if (n_samples / sub_stage_partition_count / radix_stage_partition_count > min_partition_len)
                {
                    for (std::size_t i = 0; i < sub_stage_partition_count; i++)
                    {
                        open_radix_stage_ids.push_back(radix_stage_id - 1);
                        open_radix_stage_partition_counts.push_back(radix_stage_partition_count * sub_stage_partition_count);
                        open_radix_stage_partition_ids.push_back(sub_stage_partition_count * radix_stage_partition_id + i);
                    }
                }

                else
                {
                    for (; radix_stage_id-- > 0;)
                    {
                        if (initial_radix_stages[radix_stage_id].type == RadixType::s_radix4) {
                            break; // Don't re-add Stockham Stages
                        }

                        if (initial_radix_stages[radix_stage_id].type == RadixType::s_radix2) {
                            break; // Don't re-add Stockham Stages
                        }

                        radix_stage = initial_radix_stages[radix_stage_id];
                        radix_stage_partition = radix_stage;
                        radix_stage_partition.params.ct_r4.subfft_id_start = radix_stage.params.ct_r4.subfft_id_end * radix_stage_partition_id / radix_stage_partition_count;
                        radix_stage_partition.params.ct_r4.subfft_id_end = radix_stage.params.ct_r4.subfft_id_end * (radix_stage_partition_id + 1) / radix_stage_partition_count;
                        reversed_radix_stages_.push_back(radix_stage_partition);
                    }
                }
            }

            radix_stages_.insert(
                radix_stages_.end(),
                reversed_radix_stages_.rbegin(),
                reversed_radix_stages_.rend()
            );
        }
    };
}

#endif