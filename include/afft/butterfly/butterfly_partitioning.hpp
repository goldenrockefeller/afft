#ifndef AFFT_BUTTERFLY_PARTITIONING_HPP
#define AFFT_BUTTERFLY_PARTITIONING_HPP

#include <vector>
#include <cstddef>

#include <utility>

#include "afft/radix/radix_stage/radix_stage.hpp"

namespace afft
{
    namespace butterfly_partitioning{
                template <typename Spec>
        std::vector<RadixStage<Spec>> sequential_partition(
            const std::vector<RadixStage<Spec>> &initial_stages
        ) {
            auto last_radix_stage = initial_stages.back();
            std::vector<RadixStage<Spec>> ret_;
            std::size_t n_partitions;
            if (last_radix_stage.type == RadixType::ct_radix2) {
            }
            if (last_radix_stage.type == RadixType::ct_radix4) {
                auto &params = last_radix_stage.params.ct_r4;
                n_partitions = params.subfft_id_end - params.subfft_id_start;
            }

            for (std::size_t partition_id = 0; partition_id < n_partitions; partition_id++) {
                for (auto radix_stage : initial_stages) {
                    auto &params = radix_stage.params.ct_r4;
                    auto n_subffts = params.subfft_id_end - params.subfft_id_start;
                    auto n_subffts_per_cut = n_subffts / n_partitions;
                    params.subfft_id_start = params.subfft_id_start + partition_id * n_subffts_per_cut;
                    params.subfft_id_end = params.subfft_id_start + n_subffts_per_cut;
                    ret_.push_back(radix_stage);
                }
            } 
            return ret_;
        }

        template <typename Spec>
        std::vector<RadixStage<Spec>> strided_partition(
            const std::vector<RadixStage<Spec>> &initial_stages,
            std::size_t stride, 
            std::size_t n_samples_per_operand
        ) {
            std::vector<RadixStage<Spec>> ret_;

            std::size_t n_partitions_per_stride = stride / n_samples_per_operand;

            for (std::size_t partition_id = 0; partition_id < n_partitions_per_stride; partition_id++) {
                std::size_t radix_stage_id = 0;
                for (auto radix_stage : initial_stages) {
                    if (radix_stage_id == 0) {
                        if (radix_stage.type == RadixType::ct_radix2) {
                            auto &params = radix_stage.params.ct_r2;
                            params.subtwiddle_start = partition_id * n_samples_per_operand;
                            params.input_stride_mul = stride / n_samples_per_operand;
                            ret_.push_back(radix_stage);
                        }
                        if (radix_stage.type == RadixType::ct_radix4) {
                            auto &params = radix_stage.params.ct_r4;
                            params.subtwiddle_start = partition_id * n_samples_per_operand;
                            params.input_stride_mul = stride / n_samples_per_operand;
                            ret_.push_back(radix_stage);
                        }
                    }
                    else if (radix_stage_id == initial_stages.size()-1) {
                        
                        if (radix_stage.type == RadixType::ct_radix2) {
                            auto &params = radix_stage.params.ct_r2;
                            params.subtwiddle_start = partition_id * n_samples_per_operand;
                            params.input_stride_mul = stride / n_samples_per_operand;
                            ret_.push_back(radix_stage);
                        }
                        if (radix_stage.type == RadixType::ct_radix4) {
                            auto &params = radix_stage.params.ct_r4;
                            params.subtwiddle_start = partition_id * n_samples_per_operand;
                            params.input_stride_mul = stride / n_samples_per_operand;
                            ret_.push_back(radix_stage);
                        }
                    }
                    else {
                        if (radix_stage.type == RadixType::ct_radix2) {
                            auto &params = radix_stage.params.ct_r2;
                            params.subtwiddle_start = 0;
                            params.subtwiddle_end = params.subtwiddle_len / stride;
                            ret_.push_back(radix_stage);
                        }
                        if (radix_stage.type == RadixType::ct_radix4) {
                            auto &params = radix_stage.params.ct_r4;
                            params.subtwiddle_start = 0;
                            params.subtwiddle_end = params.subtwiddle_len / stride;
                            params.subtwiddle_len = params.subtwiddle_len / stride;
                            ret_.push_back(radix_stage);
                        }
                    }
                    radix_stage_id ++;
                }
            }

            return ret_;
        }

        template <typename Spec>
        std::pair<std::vector<RadixStage<Spec>>, std::vector<RadixStage<Spec>>> partition(
            const std::vector<RadixStage<Spec>> &initial_stages,
            std::size_t n_samples_per_operand,
            std::size_t n_stride_paritions
        ) {
            std::size_t collection_len = 1024;
            std::size_t n_partitions_per_collection = collection_len / n_samples_per_operand;
            
            // Get stride
            auto last_radix_stage = initial_stages.back();
            std::size_t subtwiddle_len;
            std::size_t subfft_len;
            std::size_t stage_size;
            if (last_radix_stage.type == RadixType::ct_radix2) {
                auto &params = last_radix_stage.params.ct_r2;
                subtwiddle_len = params.subtwiddle_len;
                subfft_len = 2 * params.subtwiddle_len;
                stage_size = subfft_len;
            }
            if (last_radix_stage.type == RadixType::ct_radix4) {
                auto &params = last_radix_stage.params.ct_r4;
                subtwiddle_len = params.subtwiddle_len;
                subfft_len = 4 * params.subtwiddle_len;
                stage_size = subfft_len * (params.subfft_id_end - params.subfft_id_start);
            }

            auto stride = stage_size / n_partitions_per_collection;

            if (stride <  n_samples_per_operand) {
                // Stages are too small to partition
                return {{}, initial_stages};
            }

            if (stride > subtwiddle_len) {
                //Stage are too large to partition by strides
                return {sequential_partition<Spec>(initial_stages), {}};
            }

            // Last stage is right size to partition by strides

            std::vector<RadixStage<Spec>> non_partitioned_stages;
            std::vector<RadixStage<Spec>> strided_partitioned_stages;

            for (auto &radix_stage : initial_stages) {
                if (radix_stage.type == RadixType::ct_radix2) {
                    auto &params = radix_stage.params.ct_r2;
                    subtwiddle_len = params.subtwiddle_len;
                }
                if (radix_stage.type == RadixType::ct_radix4) {
                    auto &params = radix_stage.params.ct_r4;
                    subtwiddle_len = params.subtwiddle_len;
                }
                if (stride > subtwiddle_len ){//|| n_samples_per_operand * n_stride_paritions > subtwiddle_len) {
                    non_partitioned_stages.push_back(radix_stage);
                } else {
                    strided_partitioned_stages.push_back(radix_stage);
                }
            }

            if (strided_partitioned_stages.size() < 3) {
                return {non_partitioned_stages, strided_partitioned_stages};
            }

            return {
                non_partitioned_stages, 
                strided_partition<Spec>(
                    strided_partitioned_stages, 
                    stride, 
                    n_samples_per_operand
                )
            };
        }
    }
}

#endif