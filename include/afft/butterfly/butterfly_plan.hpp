#ifndef AFFT_BUTTERFLY_PLAN_HPP
#define AFFT_BUTTERFLY_PLAN_HPP

#include <vector>
#include <cstddef>

#include <algorithm>

#include "afft/radix/radix_stage.hpp"
#include "afft/radix/radix_type.hpp"
#include "afft/butterfly/twiddles.hpp"
#include "afft/log_n_samples_per_operand.hpp"

namespace afft
{
    template <typename Spec, class Allocator = std::allocator<typename Spec::sample>>
    class ButterflyPlan
    {
        static_assert(Spec::n_samples_per_operand == 1, "Butterfly must use sample spec");

        LogNSamplesPerOperand log_n_samples_per_operand_;
        std::vector<RadixStage<Spec>> radix_stages_;
        std::vector<typename Spec::sample, Allocator> compound_radix_twiddles_real_;
        std::vector<typename Spec::sample, Allocator> compound_radix_twiddles_imag_;
        std::vector<std::vector<std::vector<typename Spec::sample, Allocator>>> twiddles_real_;
        std::vector<std::vector<std::vector<typename Spec::sample, Allocator>>> twiddles_imag_;
        typename Spec::sample scaling_factor_;

    public:
        const LogNSamplesPerOperand &log_n_samples_per_operand() const
        {
            return log_n_samples_per_operand_;
        }

        // Getter for radix_stages_
        const std::vector<RadixStage<Spec>> &radix_stages() const
        {
            return radix_stages_;
        }

        // Getter for compound_radix_twiddles_real_
        const std::vector<typename Spec::sample, Allocator> &compound_radix_twiddles_real() const
        {
            return compound_radix_twiddles_real_;
        }

        // Getter for compound_radix_twiddles_imag_
        const std::vector<typename Spec::sample, Allocator> &compound_radix_twiddles_imag() const
        {
            return compound_radix_twiddles_imag_;
        }

        // Getter for twiddles_real_
        const std::vector<std::vector<std::vector<typename Spec::sample, Allocator>>> &twiddles_real() const
        {
            return twiddles_real_;
        }

        // Getter for twiddles_imag_
        const std::vector<std::vector<std::vector<typename Spec::sample, Allocator>>> &twiddles_imag() const
        {
            return twiddles_imag_;
        }

        const typename Spec::sample &scaling_factor() const
        {
            return scaling_factor_;
        }

        ButterflyPlan(std::size_t n_samples, std::size_t max_n_samples_per_operand)
        {
            namespace tw = afft::twiddles;
            namespace cm = afft::common_math;
            using sample = typename Spec::sample;

            scaling_factor_ = sample(1) / sample(n_samples);

            if (n_samples <= 1)
            {
                return;
            }

            auto log_n_samples_per_operand_as_size_t =
                std::min(
                    cm::int_log_2(max_n_samples_per_operand),
                    std::size_t(cm::int_log_2(n_samples / 2)));

            std::size_t n_samples_per_operand = 1 << log_n_samples_per_operand_as_size_t;

            log_n_samples_per_operand_ =
                as_log_n_samples_per_operand(
                    log_n_samples_per_operand_as_size_t);

            if (n_samples == n_samples_per_operand * 2)
            {
                compound_radix_twiddles_real_ = tw::compound_radix2_twiddles_real<Spec, Allocator>(n_samples_per_operand);
                compound_radix_twiddles_imag_ = tw::compound_radix2_twiddles_imag<Spec, Allocator>(n_samples_per_operand);

                RadixStage<Spec> compound_radix_stage_;
                compound_radix_stage_.type = RadixType::compound_radix2;
                compound_radix_stage_.tw_real_b = compound_radix_twiddles_real_.data();
                compound_radix_stage_.tw_imag_b = compound_radix_twiddles_imag_.data();
                radix_stages_.push_back(compound_radix_stage_);
                return;
            }

            // else n_samples >= n_samples_per_operand * 4
            compound_radix_twiddles_real_ = tw::compound_radix4_twiddles_real<Spec, Allocator>(n_samples_per_operand);
            compound_radix_twiddles_imag_ = tw::compound_radix4_twiddles_imag<Spec, Allocator>(n_samples_per_operand);

            RadixStage<Spec> compound_radix_stage_;
            compound_radix_stage_.type = RadixType::compound_radix4;
            compound_radix_stage_.tw_real_b = compound_radix_twiddles_real_.data();
            compound_radix_stage_.tw_imag_b = compound_radix_twiddles_imag_.data();
            compound_radix_stage_.subfft_id_start = 0;
            compound_radix_stage_.subfft_id_end = n_samples / n_samples_per_operand / 4;
            radix_stages_.push_back(compound_radix_stage_);

            auto subtwiddle_len = n_samples_per_operand * 4;

            while (subtwiddle_len < n_samples)
            {

                if (2 * subtwiddle_len == n_samples)
                {
                    RadixStage<Spec> radix_stage_;
                    auto stage_twiddles_real = tw::radix2_twiddles_real<Spec, Allocator>(subtwiddle_len);
                    auto stage_twiddles_imag = tw::radix2_twiddles_imag<Spec, Allocator>(subtwiddle_len);
                    twiddles_real_.push_back(stage_twiddles_real);
                    twiddles_imag_.push_back(stage_twiddles_imag);

                    radix_stage_.type = RadixType::radix2;
                    radix_stage_.tw_real_b = twiddles_real_.back()[0].data();
                    radix_stage_.tw_imag_b = twiddles_imag_.back()[0].data();
                    radix_stage_.subtwiddle_len = subtwiddle_len;
                    radix_stage_.subtwiddle_start = 0;
                    radix_stage_.subtwiddle_end = subtwiddle_len;

                    radix_stages_.push_back(radix_stage_);
                    break;
                }
                else
                {
                    RadixStage<Spec> radix_stage_;
                    auto stage_twiddles_real = tw::radix4_twiddles_real<Spec, Allocator>(subtwiddle_len);
                    auto stage_twiddles_imag = tw::radix4_twiddles_imag<Spec, Allocator>(subtwiddle_len);
                    twiddles_real_.push_back(stage_twiddles_real);
                    twiddles_imag_.push_back(stage_twiddles_imag);

                    radix_stage_.type = RadixType::radix4;
                    radix_stage_.tw_real_b = twiddles_real_.back()[0].data();
                    radix_stage_.tw_imag_b = twiddles_imag_.back()[0].data();
                    radix_stage_.tw_real_c = twiddles_real_.back()[1].data();
                    radix_stage_.tw_imag_c = twiddles_imag_.back()[1].data();
                    radix_stage_.tw_real_d = twiddles_real_.back()[2].data();
                    radix_stage_.tw_imag_d = twiddles_imag_.back()[2].data();
                    radix_stage_.subfft_id_start = 0;
                    radix_stage_.subfft_id_end = n_samples / subtwiddle_len / 4;
                    radix_stage_.subtwiddle_len = subtwiddle_len;
                    radix_stage_.subtwiddle_start = 0;
                    radix_stage_.subtwiddle_end = subtwiddle_len;

                    radix_stages_.push_back(radix_stage_);
                    subtwiddle_len *= 4;
                }
            }

            // Carry Radix Stages is for a opening (potentially) out-of-place radix stage, mostly
            // for Decimation-In-Frequency (DIF), as the carry radix stage is the first radix stage for DIF, and the last radix stage for DIT.
            // Most other radix stages are executed in-place. The first radix stage for DIT will always be a Compound Radix Stage.
            auto &carry_radix_stage = radix_stages_.back();

            if (carry_radix_stage.type == RadixType::radix4)
            {
                carry_radix_stage.type = RadixType::carry_radix4;
            }
            else if (carry_radix_stage.type == RadixType::compound_radix4)
            {
                carry_radix_stage.type = RadixType::carry_compound_radix4;
            }
        }

        ButterflyPlan(std::size_t n_samples, std::size_t max_n_samples_per_operand, std::size_t min_partition_len)
            : ButterflyPlan(n_samples, max_n_samples_per_operand)
        {
            // This version of the constructor turns an iterative butterfly plan into a hybrid/recursive plan
            std::vector<std::size_t> open_radix_stage_ids;
            std::vector<std::size_t> open_radix_stage_partition_counts;
            std::vector<std::size_t> open_radix_stage_partition_ids;

            auto iterative_radix_stages = radix_stages_;
            radix_stages_.clear();

            open_radix_stage_ids.push_back(iterative_radix_stages.size() - 1);
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
                auto radix_stage = iterative_radix_stages[radix_stage_id];

                auto radix_stage_partition = radix_stage;
                radix_stage_partition.subfft_id_start = radix_stage.subfft_id_end * radix_stage_partition_id / radix_stage_partition_count;
                radix_stage_partition.subfft_id_end = radix_stage.subfft_id_end * (radix_stage_partition_id + 1) / radix_stage_partition_count;
                radix_stages_.push_back(radix_stage_partition);

                switch (radix_stage.type)
                {
                case RadixType::radix4:
                case RadixType::carry_radix4:
                case RadixType::compound_radix4:
                case RadixType::carry_compound_radix4:
                    sub_stage_partition_count = 4;
                    break;

                case RadixType::radix2:
                case RadixType::compound_radix2:
                    sub_stage_partition_count = 2;
                    break;
                }

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
                        radix_stage = iterative_radix_stages[radix_stage_id];
                        radix_stage_partition = radix_stage;
                        radix_stage_partition.subfft_id_start = radix_stage.subfft_id_end * radix_stage_partition_id / radix_stage_partition_count;
                        radix_stage_partition.subfft_id_end = radix_stage.subfft_id_end * (radix_stage_partition_id + 1) / radix_stage_partition_count;
                        radix_stages_.push_back(radix_stage_partition);
                    }
                }
            }
            std::reverse(radix_stages_.begin(), radix_stages_.end());
        }
    };
}

#endif