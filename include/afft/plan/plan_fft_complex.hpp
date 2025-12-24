#ifndef AFFT_PLAN_FFT_COMPLEX_HPP
#define AFFT_PLAN_FFT_COMPLEX_HPP

#include <vector>
#include <cstddef>
#include <memory>
#include <unordered_map>
#include <algorithm>
#include <utility>

#include "afft/stage/stage.hpp"
#include "afft/stage/stage_type.hpp"
#include "afft/plan/twiddles.hpp"
#include "afft/plan/bit_reverse_permute.hpp"
#include "afft/common_math.hpp"
#include "afft/stage/stage_params/log_interleave_permute.hpp"

namespace afft
{
    namespace plan {
        template <typename Sample>
        std::vector<Stage<Sample>> complex_fft_plan(std::size_t n_samples, std::size_t n_samples_per_operand, std::size_t min_partition_len) 
        {
            namespace cm = afft::common_math;

            std::vector<Stage<Sample>> plan_;

            Sample scaling_factor = Sample(1) / n_samples;

            std::size_t n_s_stages = 0;

            if (n_samples <= 1)
            {
                return {};
            }

            std::size_t subtwiddle_len = 1;
            std::size_t log_subtwiddle_len = 0;

            if (n_samples == 2 * n_samples_per_operand) {
                // Stockham x2 Interleave

                while (subtwiddle_len < n_samples_per_operand) {
                    Stage<Sample> stage;

                    if (subtwiddle_len == 1) {
                        stage.type = StageType::s_radix2_init;
                    }
                    else{
                        stage.type = StageType::s_radix2;
                    }

                    auto &params = stage.params.s_r;
                    params.subfft_id_start = 0;
                    params.subfft_id_end = n_samples / n_samples_per_operand / 2;
                    params.subtwiddle_len = subtwiddle_len;
                    params.n_samples = n_samples;
                    params.scaling_factor = scaling_factor;

                    if (2 * subtwiddle_len < n_samples_per_operand) {
                        params.log_interleave_permute = as_log_interleave_permute(log_subtwiddle_len, false);
                    }
                    else {
                        params.log_interleave_permute = as_log_interleave_permute(log_subtwiddle_len, true);
                    }

                    plan_.push_back(stage);
                    subtwiddle_len *= 2;
                    log_subtwiddle_len += 1;
                    n_s_stages += 1;

                }

                if (n_samples_per_operand == 1)
                {
                    // Stockham x2 Permute
                    Stage<Sample> stage;

                    if (subtwiddle_len == 1) {
                        stage.type = StageType::s_radix2_init;
                    }
                    else{
                        stage.type = StageType::s_radix2;
                    }

                    auto &params = stage.params.s_r;
                    params.subfft_id_start = 0;
                    params.subfft_id_end = n_samples / n_samples_per_operand / 2;
                    params.subtwiddle_len = subtwiddle_len;
                    params.n_samples = n_samples;
                    params.scaling_factor = scaling_factor;
                    params.log_interleave_permute = as_log_interleave_permute(log_subtwiddle_len, true);

                    plan_.push_back(stage);
                    subtwiddle_len *= 2;
                    log_subtwiddle_len += 1;
                    n_s_stages += 1;
                }
            }

            else{
                // Stockham x4 Interleave
                while (subtwiddle_len * 4 <= n_samples_per_operand) {
                    Stage<Sample> stage;

                    if (subtwiddle_len == 1) {
                        stage.type = StageType::s_radix4_init;
                    }
                    else{
                        stage.type = StageType::s_radix4;
                    }

                    auto &params = stage.params.s_r;
                    params.subfft_id_start = 0;
                    params.subfft_id_end = n_samples / n_samples_per_operand / 4;
                    params.subtwiddle_len = subtwiddle_len;
                    params.n_samples = n_samples;
                    params.scaling_factor = scaling_factor;

                    if (4 * subtwiddle_len < n_samples_per_operand) {
                        params.log_interleave_permute = as_log_interleave_permute(log_subtwiddle_len, false);
                    }
                    else {
                        params.log_interleave_permute = as_log_interleave_permute(log_subtwiddle_len, true);
                    }

                    plan_.push_back(stage);
                    subtwiddle_len *= 4;
                    log_subtwiddle_len += 2;
                    n_s_stages += 1;
                }

                // Stockham x2 Interleave
                if (subtwiddle_len < n_samples_per_operand) {
                    
                    Stage<Sample> stage;

                    if (subtwiddle_len == 1) {
                        stage.type = StageType::s_radix2_init;
                    }
                    else{
                        stage.type = StageType::s_radix2;
                    }

                    auto &params = stage.params.s_r;
                    params.subfft_id_start = 0;
                    params.subfft_id_end = n_samples / n_samples_per_operand / 2;
                    params.subtwiddle_len = subtwiddle_len;
                    params.n_samples = n_samples;
                    params.scaling_factor = scaling_factor;

                    if (2 * subtwiddle_len < n_samples_per_operand) {
                        params.log_interleave_permute = as_log_interleave_permute(log_subtwiddle_len, false);
                    }
                    else {
                        params.log_interleave_permute = as_log_interleave_permute(log_subtwiddle_len, true);
                    }

                    plan_.push_back(stage);
                    subtwiddle_len *= 2;
                    log_subtwiddle_len += 1;
                    n_s_stages += 1;
                }

                if (n_samples_per_operand == 1) {
                    // Stockham x4 Permute

                    Stage<Sample> stage;

                    if (subtwiddle_len == 1) {
                        stage.type = StageType::s_radix4_init;
                    }
                    else{
                        stage.type = StageType::s_radix4;
                    }

                    auto &params = stage.params.s_r;
                    params.log_interleave_permute = as_log_interleave_permute(log_subtwiddle_len, true);
                    params.subfft_id_start = 0;
                    params.subfft_id_end = n_samples / n_samples_per_operand / 4;
                    params.subtwiddle_len = subtwiddle_len;
                    params.n_samples = n_samples;
                    params.scaling_factor = scaling_factor;
                    plan_.push_back(stage);
                    subtwiddle_len *= 4;
                    log_subtwiddle_len += 2;
                    n_s_stages += 1;
                }
            }    

            // Cooley Tukey
            while (subtwiddle_len < n_samples)
            {

                if (2 * subtwiddle_len == n_samples)
                {

                    Stage<Sample> stage;
                    stage.type = StageType::ct_radix2;
                    auto &params = stage.params.ct_r2; 
                    params.subtwiddle_len = subtwiddle_len;
                    params.subtwiddle_start = 0;
                    params.subtwiddle_end = subtwiddle_len;

                    plan_.push_back(stage);
                    subtwiddle_len *= 2;
                    log_subtwiddle_len += 1;
                    break;
                }
                else
                {
                    Stage<Sample> stage;
                    stage.type = StageType::ct_radix4;
                    auto &params = stage.params.ct_r4; 
                    params.subfft_id_start = 0;
                    params.subfft_id_end = n_samples / subtwiddle_len / 4;
                    params.subtwiddle_len = subtwiddle_len;
                    params.subtwiddle_start = 0;
                    params.subtwiddle_end = subtwiddle_len;

                    plan_.push_back(stage);
                    subtwiddle_len *= 4;
                    log_subtwiddle_len += 2;
                }
            }

            // This version of the constructor turns an iterative butterfly plan_ into a hybrid/recursive plan_
            std::vector<std::size_t> open_stage_ids;
            std::vector<std::size_t> open_stage_partition_counts;
            std::vector<std::size_t> open_stage_partition_ids;

            if (n_samples < min_partition_len) {
                return plan_;
            }

            auto initial_plan = plan_;
            plan_.clear();
            auto reversed_plan = plan_;

            // Add the Stockham stages back in 
            for (std::size_t stage_id = 0; stage_id < initial_plan.size(); stage_id++) {
                auto stage = initial_plan[stage_id];
                if (stage.type != StageType::ct_radix4 && stage.type != StageType::ct_radix2) {
                    plan_.push_back(stage);
                }
            }

            if (plan_.size() == initial_plan.size()) {
                return plan_;
            }

            open_stage_ids.push_back(initial_plan.size() - 1);
            open_stage_partition_counts.push_back(1);
            open_stage_partition_ids.push_back(0);

            while (open_stage_ids.size() > 0)
            {
                auto stage_id = open_stage_ids.back();
                open_stage_ids.pop_back();

                auto stage_partition_count = open_stage_partition_counts.back();
                open_stage_partition_counts.pop_back();

                auto stage_partition_id = open_stage_partition_ids.back();
                open_stage_partition_ids.pop_back();

                std::size_t sub_stage_partition_count;
                auto stage = initial_plan[stage_id];

                auto stage_partition = stage;

                if (stage.type == StageType::ct_radix4)
                {
                    sub_stage_partition_count = 4;
                    stage_partition.params.ct_r4.subfft_id_start = stage.params.ct_r4.subfft_id_end * stage_partition_id / stage_partition_count;
                    stage_partition.params.ct_r4.subfft_id_end = stage.params.ct_r4.subfft_id_end * (stage_partition_id + 1) / stage_partition_count;
                }
                else if (stage.type == StageType::ct_radix2) 
                {
                    sub_stage_partition_count = 2;
                } 
                else
                {
                    continue; // Don't partition Stockham Stages
                }
                
                reversed_plan.push_back(stage_partition);

                if (n_samples / sub_stage_partition_count / stage_partition_count > min_partition_len)
                {
                    for (std::size_t i = 0; i < sub_stage_partition_count; i++)
                    {
                        open_stage_ids.push_back(stage_id - 1);
                        open_stage_partition_counts.push_back(stage_partition_count * sub_stage_partition_count);
                        open_stage_partition_ids.push_back(sub_stage_partition_count * stage_partition_id + i);
                    }
                }

                else // Don't partition small stages.
                {
                    for (; stage_id-- > 0;)
                    {
                        
                        stage = initial_plan[stage_id];

                        if (stage.type != StageType::ct_radix4 && stage.type != StageType::ct_radix2) {
                            break; // Only re-add partitionable Cooley Tukey stages
                        }

                        stage_partition = stage;
                        stage_partition.params.ct_r4.subfft_id_start = stage.params.ct_r4.subfft_id_end * stage_partition_id / stage_partition_count;
                        stage_partition.params.ct_r4.subfft_id_end = stage.params.ct_r4.subfft_id_end * (stage_partition_id + 1) / stage_partition_count;
                        reversed_plan.push_back(stage_partition);
                    }
                }
            }

            plan_.insert(
                plan_.end(),
                reversed_plan.rbegin(),
                reversed_plan.rend()
            );

            return plan_;
        }

        template <typename Spec, class Allocator = std::allocator<typename Spec::sample>>
        std::pair<std::vector<typename Spec::sample, Allocator>, std::unordered_map<std::size_t, const typename Spec::sample*>> twiddles(const std::vector<Stage<typename Spec::sample>>& plan, std::size_t n_samples, std::size_t n_samples_per_operand) {
            std::unordered_map<std::size_t, std::vector<typename Spec::sample, Allocator>> twiddles_map;

            namespace cm = afft::common_math;

            if (n_samples <= 1)
            {
                return { {}, {} };
            }

            namespace tw = afft::twiddles;

            for (const auto &stage : plan) {
                std::size_t subtwiddle_len = 0;

                switch (stage.type) {
                case StageType::ct_radix4:
                    subtwiddle_len = stage.params.ct_r4.subtwiddle_len;
                    break;
                case StageType::ct_radix2:
                    subtwiddle_len = stage.params.ct_r2.subtwiddle_len;
                    break;
                case StageType::s_radix4:
                case StageType::s_radix4_init:
                case StageType::s_radix4_init_rescale:
                case StageType::s_radix2:
                case StageType::s_radix2_init:
                case StageType::s_radix2_init_rescale:
                    subtwiddle_len = stage.params.s_r.subtwiddle_len;
                    break;
                default:
                    continue; // skip unsupported
                }

                // If we already have a twiddle for this subtwiddle_len, skip creation
                if (twiddles_map.find(subtwiddle_len) != twiddles_map.end()) {
                    continue;
                }

                // Create twiddles for this subtwiddle_len based on stage type
                switch (stage.type) {
                case StageType::ct_radix4:
                    twiddles_map.emplace(subtwiddle_len, tw::ct_radix4_twiddles<Spec, Allocator>(subtwiddle_len, n_samples_per_operand));
                    break;
                case StageType::ct_radix2:
                    twiddles_map.emplace(subtwiddle_len, tw::ct_radix2_twiddles<Spec, Allocator>(subtwiddle_len, n_samples_per_operand));
                    break;
                case StageType::s_radix4:
                case StageType::s_radix4_init:
                case StageType::s_radix4_init_rescale:
                    twiddles_map.emplace(subtwiddle_len, tw::s_radix4_twiddles<Spec, Allocator>(subtwiddle_len, n_samples_per_operand));
                    break;
                case StageType::s_radix2:
                case StageType::s_radix2_init:
                case StageType::s_radix2_init_rescale:
                    twiddles_map.emplace(subtwiddle_len, tw::s_radix2_twiddles<Spec, Allocator>(subtwiddle_len, n_samples_per_operand));
                    break;
                default:
                    break;
                }
            }

            // Now concatenate into master vector and create pointer map
            std::vector<typename Spec::sample, Allocator> master_twiddles;
            std::unordered_map<std::size_t, const typename Spec::sample*> pointer_map;

            // Pre-calculate total size to avoid reallocation invalidating pointers
            std::size_t total_size = 0;
            for (const auto& pair : twiddles_map) {
                total_size += pair.second.size();
            }
            master_twiddles.reserve(total_size);

            for (const auto& pair : twiddles_map) {
                auto subtwiddle_len = pair.first;
                const auto& vec = pair.second;
                pointer_map[subtwiddle_len] = master_twiddles.data() + master_twiddles.size();
                master_twiddles.insert(master_twiddles.end(), vec.begin(), vec.end());
            }

            return { std::move(master_twiddles), std::move(pointer_map) };

        }

        template <typename Sample>
        void set_twiddle_pointers(std::vector<Stage<Sample>>& plan, const std::unordered_map<std::size_t, const Sample*>& twiddle_map) {
            for (auto& stage : plan) {
                std::size_t subtwiddle_len = 0;

                switch (stage.type) {
                case StageType::ct_radix4:
                    subtwiddle_len = stage.params.ct_r4.subtwiddle_len;
                    stage.params.ct_r4.twiddles = twiddle_map.at(subtwiddle_len);
                    break;
                case StageType::ct_radix2:
                    subtwiddle_len = stage.params.ct_r2.subtwiddle_len;
                    stage.params.ct_r2.twiddles = twiddle_map.at(subtwiddle_len);
                    break;
                case StageType::s_radix4:
                case StageType::s_radix4_init:
                case StageType::s_radix4_init_rescale:
                case StageType::s_radix2:
                case StageType::s_radix2_init:
                case StageType::s_radix2_init_rescale:
                    subtwiddle_len = stage.params.s_r.subtwiddle_len;
                    stage.params.s_r.twiddles = twiddle_map.at(subtwiddle_len);
                    break;
                default:
                    break;
                }
            }
        }

        template <typename Sample>
        std::pair<std::vector<std::size_t>, std::vector<std::size_t>> bit_reverse_indexes(const std::vector<Stage<Sample>>& plan, std::size_t n_samples, std::size_t n_samples_per_operand) {
            namespace brp = afft::bit_reverse_permute;
            std::size_t radix = 2; // default

            // Find the radix from the s_radix stage that is permuting
            for (const auto& stage : plan) {
                if (stage.type == StageType::s_radix2 || stage.type == StageType::s_radix2_init || stage.type == StageType::s_radix2_init_rescale) {
                    if (stage.params.s_r.log_interleave_permute >= afft::LogInterleavePermute::n0Permuting) {
                        radix = 2;
                        break;
                    }
                } else if (stage.type == StageType::s_radix4 || stage.type == StageType::s_radix4_init || stage.type == StageType::s_radix4_init_rescale) {
                    if (stage.params.s_r.log_interleave_permute >= afft::LogInterleavePermute::n0Permuting) {
                        radix = 4;
                        break;
                    }
                }
            }

            if (n_samples_per_operand == 1)
                return brp::bit_rev_indexes_for_1(n_samples, radix);  // Renamed from pim::bit_rev_indexes_for_1
            else
                return brp::bit_rev_indexes_for_op(n_samples, radix, n_samples_per_operand);  // Renamed from pim::bit_rev_indexes_for_op
        }

        template <typename Sample>
        void set_bit_reverse_pointers(std::vector<Stage<Sample>>& stages, const std::vector<std::size_t>& in_permute_indexes, const std::vector<std::size_t>& out_permute_indexes) {
            for (auto& stage : stages) {
                bool is_s_radix = (stage.type == StageType::s_radix2 || stage.type == StageType::s_radix2_init || stage.type == StageType::s_radix2_init_rescale ||
                                   stage.type == StageType::s_radix4 || stage.type == StageType::s_radix4_init || stage.type == StageType::s_radix4_init_rescale);
                if (is_s_radix){
                    if (stage.params.s_r.log_interleave_permute >= afft::LogInterleavePermute::n0Permuting) {
                        stage.params.s_r.in_permute_indexes = in_permute_indexes.data();
                        stage.params.s_r.out_permute_indexes = out_permute_indexes.data();
                        break; // Assuming only one permuting stage
                    }
                }
            }
        }

        template <typename Sample>
        void set_s_stage_data_ids_for_complex_fft(
            std::vector<Stage<Sample>>& plan,
            std::size_t in_real_id,
            std::size_t in_imag_id,
            std::size_t out_real_id,
            std::size_t out_imag_id,
            std::size_t buf_real_id,
            std::size_t buf_imag_id
        ) {
            // Collect all s_radix stages
            std::vector<Stage<Sample>*> s_stages;
            for (auto& stage : plan) {
                bool is_s_radix = (stage.type == StageType::s_radix2 ||
                                   stage.type == StageType::s_radix2_init ||
                                   stage.type == StageType::s_radix2_init_rescale ||
                                   stage.type == StageType::s_radix4 ||
                                   stage.type == StageType::s_radix4_init ||
                                   stage.type == StageType::s_radix4_init_rescale);
                if (is_s_radix) {
                    s_stages.push_back(&stage);
                }
            }


            if (s_stages.empty()) return;

            s_stages[0]->params.s_r.in_real_id = in_real_id;
            s_stages[0]->params.s_r.in_imag_id = in_imag_id;

            if (s_stages.size() == 1) {
                s_stages[0]->params.s_r.out_real_id = out_real_id;
                s_stages[0]->params.s_r.out_imag_id = out_imag_id;
                return;
            }

            // Set the last stage's input to be the output of the second-to-last stage
            s_stages[s_stages.size() - 1]->params.s_r.out_real_id = out_real_id;
            s_stages[s_stages.size() - 1]->params.s_r.out_imag_id = out_imag_id;
            s_stages[s_stages.size() - 1]->params.s_r.in_real_id = buf_real_id;
            s_stages[s_stages.size() - 1]->params.s_r.in_imag_id = buf_imag_id;

            // input and output ids for each stage excluding the first and last in reverse order
            for (std::size_t i = s_stages.size() - 2; i > 0; i--) {
                s_stages[i]->params.s_r.in_real_id = s_stages[i + 1]->params.s_r.out_real_id;
                s_stages[i]->params.s_r.in_imag_id = s_stages[i + 1]->params.s_r.out_imag_id;
                s_stages[i]->params.s_r.out_real_id = s_stages[i + 1]->params.s_r.in_real_id;
                s_stages[i]->params.s_r.out_imag_id = s_stages[i + 1]->params.s_r.in_imag_id; 
            }

            s_stages[0]->params.s_r.out_real_id = s_stages[1]->params.s_r.in_real_id;
            s_stages[0]->params.s_r.out_imag_id = s_stages[1]->params.s_r.in_imag_id;
        }

        template <typename Sample>
        void set_ct_stage_data_ids_for_complex_fft(
            std::vector<Stage<Sample>>& plan,
            std::size_t inout_real_id,
            std::size_t inout_imag_id
        ) {
            for (auto& stage : plan) {
                if (stage.type == StageType::ct_radix4 || stage.type == StageType::ct_radix2) {
                    stage.params.ct_r4.inout_real_id = inout_real_id;
                    stage.params.ct_r4.inout_imag_id = inout_imag_id;
                }
            }
        }

        template <typename Sample>
        void set_data_ids_for_complex_fft(
            std::vector<Stage<Sample>>& plan,
            std::size_t in_real_id,
            std::size_t in_imag_id,
            std::size_t out_real_id,
            std::size_t out_imag_id,
            std::size_t buf_real_id,
            std::size_t buf_imag_id
        ) {
            set_s_stage_data_ids_for_complex_fft(plan, in_real_id, in_imag_id, out_real_id, out_imag_id, buf_real_id, buf_imag_id);
            set_ct_stage_data_ids_for_complex_fft(plan, out_real_id, out_imag_id);
        }

        template <typename Sample>
        void replace_init_stages_with_rescale(std::vector<Stage<Sample>>& plan, const Sample& scaling_factor)
        {
            for (auto& stage : plan) {
                if (stage.type == StageType::s_radix4_init) {
                    stage.type = StageType::s_radix4_init_rescale;
                } else if (stage.type == StageType::s_radix2_init) {
                    stage.type = StageType::s_radix2_init_rescale;
                }
            }
        }
    }
}

#endif