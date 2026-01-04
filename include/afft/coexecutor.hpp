#ifndef AFFT_COEXECUTOR_HPP
#define AFFT_COEXECUTOR_HPP

#include <vector>
#include <cstddef>
#include <algorithm>
#include <stdexcept>

#include "afft/stage/stage.hpp"
#include "afft/stage/stage_type.hpp"
#include "afft/spec/bounded_spec.hpp"
#include "afft/operations/s_radix2.hpp"
#include "afft/operations/s_radix4.hpp"
#include "afft/operations/ct_radix2.hpp"
#include "afft/operations/ct_radix4.hpp"
#include "afft/operations/conj_reverse.hpp"
#include "afft/operations/deinterleave.hpp"
#include "afft/operations/interleave.hpp"
#include "afft/operations/apply_rfft_rotor.hpp"
#include "afft/operations/apply_inv_rfft_rotor.hpp"
#include "afft/operations/complex_multiply.hpp"

namespace afft
{
    namespace execute
    {
        template <typename BoundedSpec>
        inline void execute_stage(
            typename BoundedSpec::sample **data,
            const Stage<typename BoundedSpec::sample> &stage,
            const typename BoundedSpec::sample *rfft_rotor_real,
            const typename BoundedSpec::sample *rfft_rotor_imag,
            const typename BoundedSpec::sample *twiddles,
            const std::size_t *out_permute_indexes,
            const std::size_t *in_permute_indexes)
        {
            using sample = typename BoundedSpec::sample;

            switch (stage.type)
            {
            case StageType::ct_radix4:
                {
                    auto &params = stage.params.ct_r4;
                    do_ct_radix4_stage<BoundedSpec>(
                        data[params.inout_real_id],
                        data[params.inout_imag_id],
                        twiddles + params.twiddles_offset,
                        params.subfft_id_start,
                        params.subfft_id_end,
                        params.subtwiddle_len,
                        params.subtwiddle_start,
                        params.subtwiddle_end);
                }
                break;

            case StageType::ct_radix2:
                {
                    auto &params = stage.params.ct_r2;

                    do_ct_radix2_stage<BoundedSpec>(
                        data[params.inout_real_id],
                        data[params.inout_imag_id],
                        twiddles + params.twiddles_offset,
                        params.subtwiddle_len,
                        params.subtwiddle_start,
                        params.subtwiddle_end);
                }
                break;
            case StageType::s_radix4:
                {
                    auto &params = stage.params.s_r;
                    do_s_radix4_stage<BoundedSpec, false, true>(
                        data[params.out_real_id],
                        data[params.out_imag_id],
                        data[params.in_real_id],
                        data[params.in_imag_id],
                        twiddles + params.twiddles_offset,
                        out_permute_indexes,
                        in_permute_indexes,
                        params.subfft_id_start,
                        params.subfft_id_end,
                        params.n_samples,
                        sample(1),
                        params.log_interleave_permute
                    );

                }
                break;
            case StageType::s_radix2:
                {
                    auto &params = stage.params.s_r;
                    do_s_radix2_stage<BoundedSpec, false, true>(
                        data[params.out_real_id],
                        data[params.out_imag_id],
                        data[params.in_real_id],
                        data[params.in_imag_id],
                        twiddles + params.twiddles_offset,
                        out_permute_indexes,
                        in_permute_indexes,
                        params.subfft_id_start,
                        params.subfft_id_end,
                        params.n_samples,
                        sample(1),
                        params.log_interleave_permute
                    );
                }
                break;
            case StageType::s_radix4_init:
                {
                    auto &params = stage.params.s_r;
                    do_s_radix4_stage<BoundedSpec, false, false>(
                        data[params.out_real_id],
                        data[params.out_imag_id],
                        data[params.in_real_id],
                        data[params.in_imag_id],
                        twiddles + params.twiddles_offset,
                        out_permute_indexes,
                        in_permute_indexes,
                        params.subfft_id_start,
                        params.subfft_id_end,
                        params.n_samples,
                        sample(1),
                        params.log_interleave_permute
                    );
                }
                break;
            case StageType::s_radix2_init:
                {
                    auto &params = stage.params.s_r;
                    do_s_radix2_stage<BoundedSpec, false, false>(
                        data[params.out_real_id],
                        data[params.out_imag_id],
                        data[params.in_real_id],
                        data[params.in_imag_id],
                        twiddles + params.twiddles_offset,
                        out_permute_indexes,
                        in_permute_indexes,
                        params.subfft_id_start,
                        params.subfft_id_end,
                        params.n_samples,
                        sample(1),
                        params.log_interleave_permute
                    );
                }
                break;
            case StageType::s_radix4_init_rescale:
                {
                    auto &params = stage.params.s_r;
                    do_s_radix4_stage<BoundedSpec, true, false>(
                        data[params.out_real_id],
                        data[params.out_imag_id],
                        data[params.in_real_id],
                        data[params.in_imag_id],
                        twiddles + params.twiddles_offset,
                        out_permute_indexes,
                        in_permute_indexes,
                        params.subfft_id_start,
                        params.subfft_id_end,
                        params.n_samples,
                        params.scaling_factor,
                        params.log_interleave_permute
                        );
                }
                break;
            case StageType::s_radix2_init_rescale:
                {
                    auto &params = stage.params.s_r;
                    do_s_radix2_stage<BoundedSpec, true, false>(
                        data[params.out_real_id],
                        data[params.out_imag_id],
                        data[params.in_real_id],
                        data[params.in_imag_id],
                        twiddles + params.twiddles_offset,
                        out_permute_indexes,
                        in_permute_indexes,
                        params.subfft_id_start,
                        params.subfft_id_end,
                        params.n_samples,
                        params.scaling_factor,
                        params.log_interleave_permute);
                }
                break;
            case StageType::conj_reverse:
                {
                    auto &params = stage.params.conj_rev;
                    conj_reverse(
                        data[params.out_real_id],
                        data[params.out_imag_id],
                        data[params.spectra_real_id],
                        data[params.spectra_imag_id],
                        params.spectra_len,
                        params.id_start,
                        params.id_end);
                }
                break;
            case StageType::deinterleave:
                {
                    auto &params = stage.params.deinterleave;
                    afft::deinterleave(
                        data[params.out_real_id],
                        data[params.out_imag_id],
                        data[params.in_id],
                        params.id_start,
                        params.id_end);
                }
                break;
            case StageType::interleave:
                {
                    auto &params = stage.params.interleave;
                    afft::interleave(
                        data[params.out_id],
                        data[params.in_real_id],
                        data[params.in_imag_id],
                        params.id_start,
                        params.id_end);
                }
                break;
            case StageType::apply_inv_rfft_rotor:
                {
                    auto &params = stage.params.apply_rfft_rotor;
                    apply_inv_rfft_rotor<BoundedSpec>(
                        data[params.out_real_id],
                        data[params.out_imag_id],
                        data[params.spectra_real_id],
                        data[params.spectra_imag_id],
                        data[params.reversed_real_id],
                        data[params.reversed_imag_id],
                        rfft_rotor_real,
                        rfft_rotor_imag,
                        params.id_start,
                        params.id_end,
                        params.spectra_len,
                        params.using_hermitian_packed_form);
                }
                break;
            case StageType::apply_rfft_rotor:
                {
                    auto &params = stage.params.apply_rfft_rotor;
                    apply_rfft_rotor<BoundedSpec>(
                        data[params.out_real_id],
                        data[params.out_imag_id],
                        data[params.spectra_real_id],
                        data[params.spectra_imag_id],
                        data[params.reversed_real_id],
                        data[params.reversed_imag_id],
                        rfft_rotor_real,
                        rfft_rotor_imag,
                        params.id_start,
                        params.id_end,
                        params.spectra_len,
                        params.using_hermitian_packed_form);
                }
                break;
            case StageType::complex_multiply:
                {
                    auto &params = stage.params.complex_multiply;
                    complex_multiply<BoundedSpec>(
                        data[params.out_real_id],
                        data[params.out_imag_id],
                        data[params.in_real_a_id],
                        data[params.in_imag_a_id],
                        data[params.in_real_b_id],
                        data[params.in_imag_b_id],
                        params.id_start,
                        params.id_end,
                        params.using_hermitian_packed_form);
                }
                break;
            default:
                break;
            }
        }

        

        template <typename Spec>
        class Coexecutor
        {
        public:
            using sample = typename Spec::sample;

                        Coexecutor(const std::vector<Stage<sample>> &plan, std::size_t log_n_samples_per_operand)
                                : plan_(plan),
                                    log_n_samples_per_operand_(log_n_samples_per_operand),
                                    operand_width_(std::size_t(1) << log_n_samples_per_operand),
                  stage_index_(0),
                  stage_offset_(0),
                  global_offset_(0)
            {
                precompute_lengths();
            }

            void reset()
            {
                stage_index_ = 0;
                stage_offset_ = 0;
                global_offset_ = 0;
            }

            std::size_t stage_count() const
            {
                return plan_.size();
            }

            std::size_t stage_length(std::size_t stage_index) const
            {
                if (stage_index >= stage_lengths_.size())
                {
                    return 0;
                }
                return stage_lengths_[stage_index];
            }

            std::size_t total_length() const
            {
                return total_length_;
            }

            std::size_t global_position() const
            {
                return global_offset_;
            }

            std::size_t current_stage_index() const
            {
                return stage_index_;
            }

            std::size_t stage_position() const
            {
                return stage_offset_;
            }

            std::size_t steps_remaining() const
            {
                return total_length_ > global_offset_ ? total_length_ - global_offset_ : 0;
            }

            bool finished() const
            {
                // prefer global progress as authoritative; keep original check for safety
                return global_offset_ >= total_length_ || stage_index_ >= plan_.size();
            }

            std::size_t process(
                sample **data,
                const sample *rfft_rotor_real,
                const sample *rfft_rotor_imag,
                const sample *twiddles,
                const std::size_t *out_permute_indexes,
                const std::size_t *in_permute_indexes,
                std::size_t n_steps)
            {
                if (n_steps == 0)
                {
                    return 0;
                }

                if (finished())
                {
                    // Diagnostic: print internal state to help track inconsistency
                    std::cerr << "[Coexecutor] process called but finished() == true\n"
                              << "  plan_size=" << plan_.size()
                              << " stage_index=" << stage_index_
                              << " stage_offset=" << stage_offset_
                              << " global_offset=" << global_offset_
                              << " total_length=" << total_length_
                              << " n_steps=" << n_steps << std::endl;
                    return 0;
                }

                switch (log_n_samples_per_operand_)
                {
                case 0:
                    return process_impl<typename BoundedSpec<Spec, 0>::spec>(
                        data, rfft_rotor_real, rfft_rotor_imag, twiddles, out_permute_indexes, in_permute_indexes, n_steps);
                case 1:
                    return process_impl<typename BoundedSpec<Spec, 1>::spec>(
                        data, rfft_rotor_real, rfft_rotor_imag, twiddles, out_permute_indexes, in_permute_indexes, n_steps);
                case 2:
                    return process_impl<typename BoundedSpec<Spec, 2>::spec>(
                        data, rfft_rotor_real, rfft_rotor_imag, twiddles, out_permute_indexes, in_permute_indexes, n_steps);
                case 3:
                    return process_impl<typename BoundedSpec<Spec, 3>::spec>(
                        data, rfft_rotor_real, rfft_rotor_imag, twiddles, out_permute_indexes, in_permute_indexes, n_steps);
                case 4:
                    return process_impl<typename BoundedSpec<Spec, 4>::spec>(
                        data, rfft_rotor_real, rfft_rotor_imag, twiddles, out_permute_indexes, in_permute_indexes, n_steps);
                case 5:
                    return process_impl<typename BoundedSpec<Spec, 5>::spec>(
                        data, rfft_rotor_real, rfft_rotor_imag, twiddles, out_permute_indexes, in_permute_indexes, n_steps);
                case 6:
                    return process_impl<typename BoundedSpec<Spec, 6>::spec>(
                        data, rfft_rotor_real, rfft_rotor_imag, twiddles, out_permute_indexes, in_permute_indexes, n_steps);
                case 7:
                    return process_impl<typename BoundedSpec<Spec, 7>::spec>(
                        data, rfft_rotor_real, rfft_rotor_imag, twiddles, out_permute_indexes, in_permute_indexes, n_steps);
                case 8:
                    return process_impl<typename BoundedSpec<Spec, 8>::spec>(
                        data, rfft_rotor_real, rfft_rotor_imag, twiddles, out_permute_indexes, in_permute_indexes, n_steps);
                default:
                    throw std::invalid_argument("Unsupported operand log for Coexecutor");
                }
            }

        private:
            using StageTypeAlias = Stage<sample>;

            void precompute_lengths()
            {
                stage_lengths_.reserve(plan_.size());
                total_length_ = 0;
                for (const auto &stage : plan_)
                {
                    const std::size_t len = compute_stage_length(stage);
                    stage_lengths_.push_back(len);
                    total_length_ += len;
                }
            }

            std::size_t iterations_for_range(std::size_t begin, std::size_t end, std::size_t stride) const
            {
                if (end <= begin || stride == 0)
                {
                    return 0;
                }
                const std::size_t distance = end - begin;
                return (distance + stride - 1) / stride;
            }

            std::size_t compute_stage_length(const StageTypeAlias &stage) const
            {
                switch (stage.type)
                {
                case StageType::ct_radix4:
                {
                    const auto &params = stage.params.ct_r4;
                    const std::size_t inner = iterations_for_range(params.subtwiddle_start, params.subtwiddle_end, operand_width_);
                    const std::size_t outer = params.subfft_id_end > params.subfft_id_start
                                                  ? params.subfft_id_end - params.subfft_id_start
                                                  : 0;
                    return inner * outer;
                }
                case StageType::ct_radix2:
                {
                    const auto &params = stage.params.ct_r2;
                    return iterations_for_range(params.subtwiddle_start, params.subtwiddle_end, operand_width_);
                }
                case StageType::s_radix4:
                case StageType::s_radix4_init:
                case StageType::s_radix4_init_rescale:
                case StageType::s_radix2:
                case StageType::s_radix2_init:
                case StageType::s_radix2_init_rescale:
                {
                    const auto &params = stage.params.s_r;
                    return params.subfft_id_end > params.subfft_id_start
                               ? params.subfft_id_end - params.subfft_id_start
                               : 0;
                }
                case StageType::conj_reverse:
                {
                    const auto &params = stage.params.conj_rev;
                    return iterations_for_range(params.id_start, params.id_end, 1);
                }
                case StageType::deinterleave:
                {
                    const auto &params = stage.params.deinterleave;
                    return iterations_for_range(params.id_start, params.id_end, 1);
                }
                case StageType::interleave:
                {
                    const auto &params = stage.params.interleave;
                    return iterations_for_range(params.id_start, params.id_end, 1);
                }
                case StageType::apply_inv_rfft_rotor:
                case StageType::apply_rfft_rotor:
                {
                    const auto &params = stage.params.apply_rfft_rotor;
                    return iterations_for_range(params.id_start, params.id_end, operand_width_);
                }
                case StageType::complex_multiply:
                {
                    const auto &params = stage.params.complex_multiply;
                    return iterations_for_range(params.id_start, params.id_end, operand_width_);
                }
                default:
                    return 0;
                }
            }

            template <typename BoundedSpec>
            std::size_t process_impl(
                typename BoundedSpec::sample **data,
                const sample *rfft_rotor_real,
                const sample *rfft_rotor_imag,
                const sample *twiddles,
                const std::size_t *out_permute_indexes,
                const std::size_t *in_permute_indexes,
                std::size_t n_steps)
            {
                std::size_t steps_processed = 0;

                while (steps_processed < n_steps && stage_index_ < plan_.size())
                {
                    const std::size_t stage_len = stage_lengths_[stage_index_];
                    if (stage_len == 0)
                    {
                        stage_index_++;
                        stage_offset_ = 0;
                        continue;
                    }

                    const std::size_t remaining_in_stage = stage_len - stage_offset_;
                    const std::size_t chunk = std::min(n_steps - steps_processed, remaining_in_stage);

                    execute_stage_chunk<BoundedSpec>(
                        plan_[stage_index_],
                        stage_offset_,
                        stage_offset_ + chunk,
                        data,
                        rfft_rotor_real,
                        rfft_rotor_imag,
                        twiddles,
                        out_permute_indexes,
                        in_permute_indexes);

                    stage_offset_ += chunk;
                    steps_processed += chunk;
                    global_offset_ += chunk;

                    if (stage_offset_ == stage_len)
                    {
                        stage_index_++;
                        stage_offset_ = 0;
                    }
                }

                return steps_processed;
            }

            template <typename BoundedSpec>
            void execute_stage_chunk(
                const StageTypeAlias &stage,
                std::size_t iter_begin,
                std::size_t iter_end,
                typename BoundedSpec::sample **data,
                const sample *rfft_rotor_real,
                const sample *rfft_rotor_imag,
                const sample *twiddles,
                const std::size_t *out_permute_indexes,
                const std::size_t *in_permute_indexes) const
            {
                if (iter_end <= iter_begin)
                {
                    return;
                }

                switch (stage.type)
                {
                case StageType::ct_radix4:
                    run_ct_radix4_chunk<BoundedSpec>(stage, iter_begin, iter_end, data, rfft_rotor_real, rfft_rotor_imag, twiddles, out_permute_indexes, in_permute_indexes);
                    break;
                case StageType::ct_radix2:
                    run_ct_radix2_chunk<BoundedSpec>(stage, iter_begin, iter_end - iter_begin, data, rfft_rotor_real, rfft_rotor_imag, twiddles, out_permute_indexes, in_permute_indexes);
                    break;
                case StageType::s_radix4:
                case StageType::s_radix4_init:
                case StageType::s_radix4_init_rescale:
                case StageType::s_radix2:
                case StageType::s_radix2_init:
                case StageType::s_radix2_init_rescale:
                    run_s_stage_chunk<BoundedSpec>(stage, iter_begin, iter_end - iter_begin, data, rfft_rotor_real, rfft_rotor_imag, twiddles, out_permute_indexes, in_permute_indexes);
                    break;
                case StageType::conj_reverse:
                    run_conj_reverse_chunk<BoundedSpec>(stage, iter_begin, iter_end - iter_begin, data, rfft_rotor_real, rfft_rotor_imag, twiddles, out_permute_indexes, in_permute_indexes);
                    break;
                case StageType::deinterleave:
                    run_deinterleave_chunk<BoundedSpec>(stage, iter_begin, iter_end - iter_begin, data, rfft_rotor_real, rfft_rotor_imag, twiddles, out_permute_indexes, in_permute_indexes);
                    break;
                case StageType::interleave:
                    run_interleave_chunk<BoundedSpec>(stage, iter_begin, iter_end - iter_begin, data, rfft_rotor_real, rfft_rotor_imag, twiddles, out_permute_indexes, in_permute_indexes);
                    break;
                case StageType::apply_inv_rfft_rotor:
                case StageType::apply_rfft_rotor:
                    run_apply_rotor_chunk<BoundedSpec>(stage, iter_begin, iter_end - iter_begin, data, rfft_rotor_real, rfft_rotor_imag, twiddles, out_permute_indexes, in_permute_indexes);
                    break;
                case StageType::complex_multiply:
                    run_complex_multiply_chunk<BoundedSpec>(stage, iter_begin, iter_end - iter_begin, data, rfft_rotor_real, rfft_rotor_imag, twiddles, out_permute_indexes, in_permute_indexes);
                    break;
                default:
                    break;
                }
            }

            template <typename BoundedSpec>
            void run_ct_radix2_chunk(
                const StageTypeAlias &stage,
                std::size_t iter_begin,
                std::size_t iter_count,
                typename BoundedSpec::sample **data,
                const sample *rfft_rotor_real,
                const sample *rfft_rotor_imag,
                const sample *twiddles,
                const std::size_t *out_permute_indexes,
                const std::size_t *in_permute_indexes) const
            {
                if (iter_count == 0)
                {
                    return;
                }

                StageTypeAlias partial = stage;
                auto &params = partial.params.ct_r2;
                const std::size_t start = params.subtwiddle_start + iter_begin * operand_width_;
                params.subtwiddle_start = start;
                params.subtwiddle_end = start + iter_count * operand_width_;
                execute_stage<BoundedSpec>(data, partial, rfft_rotor_real, rfft_rotor_imag, twiddles, out_permute_indexes, in_permute_indexes);
            }

            template <typename BoundedSpec>
            void run_ct_radix4_chunk(
                const StageTypeAlias &stage,
                std::size_t iter_begin,
                std::size_t iter_end,
                typename BoundedSpec::sample **data,
                const sample *rfft_rotor_real,
                const sample *rfft_rotor_imag,
                const sample *twiddles,
                const std::size_t *out_permute_indexes,
                const std::size_t *in_permute_indexes) const
            {
                const auto &orig = stage.params.ct_r4;
                const std::size_t inner = iterations_for_range(orig.subtwiddle_start, orig.subtwiddle_end, operand_width_);
                const std::size_t outer = orig.subfft_id_end > orig.subfft_id_start ? orig.subfft_id_end - orig.subfft_id_start : 0;

                if (inner == 0 || outer == 0)
                {
                    return;
                }

                std::size_t processed = iter_begin;
                std::size_t remaining = iter_end - iter_begin;

                while (remaining > 0)
                {
                    const std::size_t outer_idx = processed / inner;
                    if (outer_idx >= outer)
                    {
                        break;
                    }
                    const std::size_t inner_offset = processed % inner;
                    const std::size_t available_outer = outer - outer_idx;

                    if (inner_offset != 0)
                    {
                        const std::size_t chunk_iters = std::min(inner - inner_offset, remaining);
                        StageTypeAlias partial = stage;
                        auto &params = partial.params.ct_r4;
                        params.subfft_id_start = orig.subfft_id_start + outer_idx;
                        params.subfft_id_end = params.subfft_id_start + 1;
                        params.subtwiddle_start = orig.subtwiddle_start + inner_offset * operand_width_;
                        params.subtwiddle_end = params.subtwiddle_start + chunk_iters * operand_width_;
                        execute_stage<BoundedSpec>(data, partial, rfft_rotor_real, rfft_rotor_imag, twiddles, out_permute_indexes, in_permute_indexes);
                        processed += chunk_iters;
                        remaining -= chunk_iters;
                        continue;
                    }

                    const std::size_t full_outer = std::min(remaining / inner, available_outer);
                    if (full_outer > 0)
                    {
                        StageTypeAlias partial = stage;
                        auto &params = partial.params.ct_r4;
                        params.subfft_id_start = orig.subfft_id_start + outer_idx;
                        params.subfft_id_end = params.subfft_id_start + full_outer;
                        params.subtwiddle_start = orig.subtwiddle_start;
                        params.subtwiddle_end = orig.subtwiddle_end;
                        execute_stage<BoundedSpec>(data, partial, rfft_rotor_real, rfft_rotor_imag, twiddles, out_permute_indexes, in_permute_indexes);
                        const std::size_t advanced = full_outer * inner;
                        processed += advanced;
                        remaining -= advanced;
                        continue;
                    }

                    const std::size_t chunk_iters = std::min(inner, remaining);
                    StageTypeAlias partial = stage;
                    auto &params = partial.params.ct_r4;
                    params.subfft_id_start = orig.subfft_id_start + outer_idx;
                    params.subfft_id_end = params.subfft_id_start + 1;
                    params.subtwiddle_start = orig.subtwiddle_start;
                    params.subtwiddle_end = params.subtwiddle_start + chunk_iters * operand_width_;
                    execute_stage<BoundedSpec>(data, partial, rfft_rotor_real, rfft_rotor_imag, twiddles, out_permute_indexes, in_permute_indexes);
                    processed += chunk_iters;
                    remaining -= chunk_iters;
                }
            }

            template <typename BoundedSpec>
            void run_s_stage_chunk(
                const StageTypeAlias &stage,
                std::size_t iter_begin,
                std::size_t iter_count,
                typename BoundedSpec::sample **data,
                const sample *rfft_rotor_real,
                const sample *rfft_rotor_imag,
                const sample *twiddles,
                const std::size_t *out_permute_indexes,
                const std::size_t *in_permute_indexes) const
            {
                if (iter_count == 0)
                {
                    return;
                }

                StageTypeAlias partial = stage;
                auto &params = partial.params.s_r;
                const std::size_t start = params.subfft_id_start + iter_begin;
                params.subfft_id_start = start;
                params.subfft_id_end = start + iter_count;
                execute_stage<BoundedSpec>(data, partial, rfft_rotor_real, rfft_rotor_imag, twiddles, out_permute_indexes, in_permute_indexes);
            }

            template <typename BoundedSpec>
            void run_conj_reverse_chunk(
                const StageTypeAlias &stage,
                std::size_t iter_begin,
                std::size_t iter_count,
                typename BoundedSpec::sample **data,
                const sample *rfft_rotor_real,
                const sample *rfft_rotor_imag,
                const sample *twiddles,
                const std::size_t *out_permute_indexes,
                const std::size_t *in_permute_indexes) const
            {
                if (iter_count == 0)
                {
                    return;
                }

                StageTypeAlias partial = stage;
                auto &params = partial.params.conj_rev;
                const std::size_t start = params.id_start + iter_begin;
                params.id_start = start;
                params.id_end = start + iter_count;
                execute_stage<BoundedSpec>(data, partial, rfft_rotor_real, rfft_rotor_imag, twiddles, out_permute_indexes, in_permute_indexes);
            }

            template <typename BoundedSpec>
            void run_deinterleave_chunk(
                const StageTypeAlias &stage,
                std::size_t iter_begin,
                std::size_t iter_count,
                typename BoundedSpec::sample **data,
                const sample *rfft_rotor_real,
                const sample *rfft_rotor_imag,
                const sample *twiddles,
                const std::size_t *out_permute_indexes,
                const std::size_t *in_permute_indexes) const
            {
                if (iter_count == 0)
                {
                    return;
                }

                StageTypeAlias partial = stage;
                auto &params = partial.params.deinterleave;
                const std::size_t start = params.id_start + iter_begin;
                params.id_start = start;
                params.id_end = start + iter_count;
                execute_stage<BoundedSpec>(data, partial, rfft_rotor_real, rfft_rotor_imag, twiddles, out_permute_indexes, in_permute_indexes);
            }

            template <typename BoundedSpec>
            void run_interleave_chunk(
                const StageTypeAlias &stage,
                std::size_t iter_begin,
                std::size_t iter_count,
                typename BoundedSpec::sample **data,
                const sample *rfft_rotor_real,
                const sample *rfft_rotor_imag,
                const sample *twiddles,
                const std::size_t *out_permute_indexes,
                const std::size_t *in_permute_indexes) const
            {
                if (iter_count == 0)
                {
                    return;
                }

                StageTypeAlias partial = stage;
                auto &params = partial.params.interleave;
                const std::size_t start = params.id_start + iter_begin;
                params.id_start = start;
                params.id_end = start + iter_count;
                execute_stage<BoundedSpec>(data, partial, rfft_rotor_real, rfft_rotor_imag, twiddles, out_permute_indexes, in_permute_indexes);
            }

            template <typename BoundedSpec>
            void run_apply_rotor_chunk(
                const StageTypeAlias &stage,
                std::size_t iter_begin,
                std::size_t iter_count,
                typename BoundedSpec::sample **data,
                const sample *rfft_rotor_real,
                const sample *rfft_rotor_imag,
                const sample *twiddles,
                const std::size_t *out_permute_indexes,
                const std::size_t *in_permute_indexes) const
            {
                if (iter_count == 0)
                {
                    return;
                }

                StageTypeAlias partial = stage;
                auto &params = partial.params.apply_rfft_rotor;
                const std::size_t start = params.id_start + iter_begin * operand_width_;
                params.id_start = start;
                params.id_end = start + iter_count * operand_width_;
                execute_stage<BoundedSpec>(data, partial, rfft_rotor_real, rfft_rotor_imag, twiddles, out_permute_indexes, in_permute_indexes);
            }

            template <typename BoundedSpec>
            void run_complex_multiply_chunk(
                const StageTypeAlias &stage,
                std::size_t iter_begin,
                std::size_t iter_count,
                typename BoundedSpec::sample **data,
                const sample *rfft_rotor_real,
                const sample *rfft_rotor_imag,
                const sample *twiddles,
                const std::size_t *out_permute_indexes,
                const std::size_t *in_permute_indexes) const
            {
                if (iter_count == 0)
                {
                    return;
                }

                StageTypeAlias partial = stage;
                auto &params = partial.params.complex_multiply;
                const std::size_t start = params.id_start + iter_begin * operand_width_;
                params.id_start = start;
                params.id_end = start + iter_count * operand_width_;
                execute_stage<BoundedSpec>(data, partial, rfft_rotor_real, rfft_rotor_imag, twiddles, out_permute_indexes, in_permute_indexes);
            }

            std::vector<Stage<sample>> plan_;
            std::vector<std::size_t> stage_lengths_;
            std::size_t total_length_ = 0;
            std::size_t log_n_samples_per_operand_ = 0;
            std::size_t operand_width_ = 1;
            std::size_t stage_index_ = 0;
            std::size_t stage_offset_ = 0;
            std::size_t global_offset_ = 0;
        };
    }
}

#endif
