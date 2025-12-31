#ifndef AFFT_PLAN_REAL_FFT_HPP
#define AFFT_PLAN_REAL_FFT_HPP

#include <vector>
#include <cstddef>
#include <memory>
#include <unordered_map>
#include <algorithm>
#include <utility>

#include "afft/stage/stage.hpp"
#include "afft/stage/stage_type.hpp"
#include "afft/plan/plan_complex_fft.hpp"

namespace afft
{
    namespace plan {
        template <typename Sample>
        std::vector<Stage<Sample>> real_fft_plan(
            std::size_t signal_len,
            std::size_t n_samples_per_operand,
            std::size_t min_partition_len,
            bool using_hermitian_packed_form = false)
        {
            std::vector<Stage<Sample>> plan;
            if (signal_len == 0)
            {
                return plan;
            }

            const std::size_t spectra_len = signal_len >> 1;

            Stage<Sample> deinterleave{};
            deinterleave.type = StageType::deinterleave;
            deinterleave.params.deinterleave.id_start = 0;
            deinterleave.params.deinterleave.id_end = spectra_len;
            plan.push_back(deinterleave);

            auto complex_plan = plan::complex_fft_plan<Sample>(
                spectra_len,
                n_samples_per_operand,
                min_partition_len);
            plan.insert(plan.end(), complex_plan.begin(), complex_plan.end());

            Stage<Sample> conj_stage{};
            conj_stage.type = StageType::conj_reverse;
            conj_stage.params.conj_rev.spectra_len = spectra_len;
            conj_stage.params.conj_rev.id_start = 0;
            conj_stage.params.conj_rev.id_end = spectra_len >> 1;
            plan.push_back(conj_stage);

            Stage<Sample> rotor_stage{};
            rotor_stage.type = StageType::apply_rfft_rotor;
            rotor_stage.params.apply_rfft_rotor.id_start = 0;
            rotor_stage.params.apply_rfft_rotor.id_end = spectra_len;
            rotor_stage.params.apply_rfft_rotor.spectra_len = spectra_len;
            rotor_stage.params.apply_rfft_rotor.using_hermitian_packed_form = using_hermitian_packed_form;
            plan.push_back(rotor_stage);

            return plan;
        }

        template <typename Sample>
        std::vector<Stage<Sample>> inv_real_fft_plan(std::size_t signal_len, std::size_t n_samples_per_operand, std::size_t min_partition_len, bool using_hermitian_packed_form = false)
        {
            std::vector<Stage<Sample>> plan;
            if (signal_len == 0)
            {
                return plan;
            }

            const std::size_t spectra_len = signal_len >> 1;

            Stage<Sample> conj_stage{};
            conj_stage.type = StageType::conj_reverse;
            conj_stage.params.conj_rev.spectra_len = spectra_len;
            conj_stage.params.conj_rev.id_start = 0;
            conj_stage.params.conj_rev.id_end = spectra_len >> 1;
            plan.push_back(conj_stage);

            Stage<Sample> rotor_stage{};
            rotor_stage.type = StageType::apply_inv_rfft_rotor;
            rotor_stage.params.apply_rfft_rotor.id_start = 0;
            rotor_stage.params.apply_rfft_rotor.id_end = spectra_len;
            rotor_stage.params.apply_rfft_rotor.spectra_len = spectra_len;
            rotor_stage.params.apply_rfft_rotor.using_hermitian_packed_form = using_hermitian_packed_form;
            plan.push_back(rotor_stage);

            auto complex_plan = plan::complex_fft_plan<Sample>(
                spectra_len,
                n_samples_per_operand,
                min_partition_len);
            plan.insert(plan.end(), complex_plan.begin(), complex_plan.end());

            Stage<Sample> interleave_stage{};
            interleave_stage.type = StageType::interleave;
            interleave_stage.params.interleave.id_start = 0;
            interleave_stage.params.interleave.id_end = spectra_len;
            plan.push_back(interleave_stage);

            return plan;
        }

        template <typename Spec, class Allocator = std::allocator<typename Spec::sample>>
        std::pair<std::vector<typename Spec::sample, Allocator>, std::vector<typename Spec::sample, Allocator>> rfft_rotor(std::size_t signal_len) {
            
            std::size_t spectra_len = signal_len >> 1;
            std::vector<typename Spec::sample, Allocator> rotor_real(spectra_len);
            std::vector<typename Spec::sample, Allocator> rotor_imag(spectra_len);
            for (
                std::size_t i = 0;
                i < spectra_len ;
                i ++
            ) {
                using sample = typename Spec::sample;
                const sample angle = Spec::pi() * sample(i) / sample(spectra_len);

                rotor_real[i] = Spec::cos(angle);
                rotor_imag[i] = -Spec::sin(angle);
            }

            return {std::move(rotor_real), std::move(rotor_imag)};
        }

        template <typename Sample>
        void set_data_ids_for_real_fft(
            std::vector<Stage<Sample>>& plan,
            std::size_t signal_id,
            std::size_t spectra_real_id,
            std::size_t spectra_imag_id,
            std::size_t buf_a_real_id,
            std::size_t buf_a_imag_id,
            std::size_t buf_b_real_id,
            std::size_t buf_b_imag_id
        ) {
            // If signal is not treated as const, it (and its offset) can be reused as buf_b
            if (plan.empty())
            {
                return;
            }

            for (auto &stage : plan)
            {
                switch (stage.type)
                {
                case StageType::deinterleave:
                {
                    auto &params = stage.params.deinterleave;
                    params.out_real_id = buf_a_real_id;
                    params.out_imag_id = buf_a_imag_id;
                    params.in_id = signal_id;
                    break;
                }
                // complex fft -> buf_b
                case StageType::conj_reverse:
                {
                    auto &params = stage.params.conj_rev;
                    params.out_real_id = buf_a_real_id;
                    params.out_imag_id = buf_a_imag_id;
                    params.spectra_real_id = buf_b_real_id;
                    params.spectra_imag_id = buf_b_imag_id;
                    break;
                }
                case StageType::apply_rfft_rotor:
                {
                    auto &params = stage.params.apply_rfft_rotor;
                    params.spectra_real_id = buf_b_real_id;
                    params.spectra_imag_id = buf_b_imag_id;
                    params.reversed_real_id = buf_a_real_id;
                    params.reversed_imag_id = buf_a_imag_id;
                    params.out_real_id = spectra_real_id;
                    params.out_imag_id = spectra_imag_id;
                    break;
                }
                default:
                    break;
                }
            }

            set_data_ids_for_complex_fft(
                plan,
                buf_b_real_id,
                buf_b_imag_id,
                buf_a_real_id,
                buf_a_imag_id,
                spectra_real_id,
                spectra_imag_id);
        }

        template <typename Sample>
        void set_data_ids_for_real_ifft(
            std::vector<Stage<Sample>>& plan,
            std::size_t signal_id,
            std::size_t signal_offset_id,
            std::size_t spectra_real_id,
            std::size_t spectra_imag_id,
            std::size_t buf_a_real_id,
            std::size_t buf_a_imag_id,
            std::size_t buf_b_real_id,
            std::size_t buf_b_imag_id
        ) {
            // If spectra is not treated as const, it can be reused as buf_b
            if (plan.empty())
            {
                return;
            }

            for (auto &stage : plan)
            {
                switch (stage.type)
                {
                case StageType::conj_reverse:
                {
                    auto &params = stage.params.conj_rev;
                    params.out_real_id = buf_a_real_id;
                    params.out_imag_id = buf_a_imag_id;
                    params.spectra_real_id = spectra_real_id;
                    params.spectra_imag_id = spectra_imag_id;
                    break;
                }
                case StageType::apply_inv_rfft_rotor:
                {
                    auto &params = stage.params.apply_rfft_rotor;
                    params.spectra_real_id = spectra_real_id;
                    params.spectra_imag_id = spectra_imag_id;
                    params.reversed_real_id = buf_a_real_id;
                    params.reversed_imag_id = buf_a_imag_id;
                    params.out_real_id = buf_a_real_id;
                    params.out_imag_id = buf_a_imag_id;
                    break;
                }
                case StageType::interleave:
                {
                    auto &params = stage.params.interleave;
                    params.out_id = signal_id;
                    params.in_real_id = buf_b_real_id;
                    params.in_imag_id = buf_b_imag_id;
                    break;
                }

                default:
                    break;
                }
            }

            set_data_ids_for_complex_fft(
                plan,
                buf_b_imag_id,
                buf_b_real_id,
                buf_a_imag_id,
                buf_a_real_id,
                signal_id,
                signal_offset_id);
        }

        template <typename Sample>
        std::vector<Stage<Sample>> real_conv_plan(
            std::size_t signal_len,
            std::size_t n_samples_per_operand,
            std::size_t min_partition_len,
            std::size_t out_id,
            std::size_t out_offset_id,
            std::size_t in_a_id,
            std::size_t in_b_id,
            std::size_t spectra_a_real_id,
            std::size_t spectra_a_imag_id,
            std::size_t spectra_b_real_id,
            std::size_t spectra_b_imag_id,
            std::size_t buf_a_real_id,
            std::size_t buf_a_imag_id,
            std::size_t buf_b_real_id,
            std::size_t buf_b_imag_id
        )
        {
            std::vector<Stage<Sample>> plan;
            if (signal_len == 0)
            {
                return plan;
            }

            const std::size_t spectra_len = (signal_len >> 1);

            auto forward_plan_a = real_fft_plan<Sample>(
                signal_len,
                n_samples_per_operand,
                min_partition_len,
                true);
            auto forward_plan_b = forward_plan_a;

            set_data_ids_for_real_fft<Sample>(
                forward_plan_a,
                in_a_id,
                spectra_a_real_id,
                spectra_a_imag_id,
                buf_a_real_id,
                buf_a_imag_id,
                buf_b_real_id,
                buf_b_imag_id
            );

            set_data_ids_for_real_fft<Sample>(
                forward_plan_b,
                in_b_id,
                spectra_b_real_id,
                spectra_b_imag_id,
                buf_a_real_id,
                buf_a_imag_id,
                buf_b_real_id,
                buf_b_imag_id
            );

            plan.insert(plan.end(), forward_plan_a.begin(), forward_plan_a.end());
            plan.insert(plan.end(), forward_plan_b.begin(), forward_plan_b.end());

            Stage<Sample> multiply_stage{};
            multiply_stage.type = StageType::complex_multiply;
            auto& multiply_params = multiply_stage.params.complex_multiply;
            multiply_params.id_start = 0;
            multiply_params.id_end = spectra_len;
            multiply_params.using_hermitian_packed_form = true;
            multiply_params.out_real_id = spectra_a_real_id;
            multiply_params.out_imag_id = spectra_a_imag_id;
            multiply_params.in_real_a_id = spectra_a_real_id;
            multiply_params.in_imag_a_id = spectra_a_imag_id;
            multiply_params.in_real_b_id = spectra_b_real_id;
            multiply_params.in_imag_b_id = spectra_b_imag_id;
            plan.push_back(multiply_stage);

            auto inverse_plan = inv_real_fft_plan<Sample>(
                signal_len,
                n_samples_per_operand,
                min_partition_len,
                true);
            
            plan::replace_init_stages_with_rescale(inverse_plan, Sample(2.) / Sample(signal_len));

            set_data_ids_for_real_ifft<Sample>(
                inverse_plan,
                out_id,
                out_offset_id,
                spectra_a_real_id,
                spectra_a_imag_id,
                buf_a_real_id,
                buf_a_imag_id,
                buf_b_real_id,
                buf_b_imag_id
            );

            plan.insert(plan.end(), inverse_plan.begin(), inverse_plan.end());

            return plan;
        }
    }
}

#endif