#ifndef AFFT_REAL_TAIL_CONV_HPP
#define AFFT_REAL_TAIL_CONV_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "afft/co_real_conv_cached.hpp"
#include "afft/real_fft.hpp"

namespace afft
{
    namespace streaming_detail
    {
        template <typename Sample>
        inline void circular_add(
            Sample *dest,
            std::size_t dest_len,
            const Sample *src,
            std::size_t src_len,
            std::size_t write_pos)
        {
            if (dest_len == 0 || src_len == 0)
            {
                return;
            }

            const std::size_t cut_one = std::min(src_len, dest_len - write_pos);
            for (std::size_t i = 0; i < cut_one; ++i)
            {
                dest[write_pos + i] += src[i];
            }

            const std::size_t cut_two = src_len - cut_one;
            for (std::size_t i = 0; i < cut_two; ++i)
            {
                dest[i] += src[cut_one + i];
            }
        }

        inline void wrap_index(std::size_t &value, std::size_t modulo)
        {
            if (modulo == 0)
            {
                return;
            }

            while (value >= modulo)
            {
                value -= modulo;
            }
        }
    }

    template <typename Spec, class Allocator = std::allocator<typename Spec::sample>>
    class RealTailConv
    {
    private:
        enum class CommitPhase
        {
            Idle,
            CopyInput,
            ZeroPad,
            Convolution,
            OutputAdd
        };

    public:
        using sample = typename Spec::sample;
        using allocator_type = Allocator;
        using sample_vector = std::vector<sample, allocator_type>;

        explicit RealTailConv(sample_vector padded_tail_impulse)
                : impulse_response_(padded_tail_impulse),
                    tail_input_buffer_(checked_len(impulse_response_.size()), sample(0)),
                    tail_conv_buffer_(impulse_response_.size(), sample(0)),
                    tail_len_(impulse_response_.size() / 2),
                    tail_conv_len_(impulse_response_.size()),
                    fft(tail_conv_len_),
                    co_real_conv_(tail_conv_len_),
                    staging_input_id_start_(0),
                    staging_output_id_start_(tail_len_),
                    commit_input_id_start_(0),
                    commit_output_id_start_(tail_len_),
                    staging_len_(0),
                    commit_len_(0),
                    commit_offset_(0),
                    committing_(false),
                    staging_(false),
                    commit_phase_(CommitPhase::Idle)
        {
            if ((impulse_response_.size() & (impulse_response_.size() - 1)) != 0)
            {
                throw std::invalid_argument("Tail impulse response length must be a power of two");
            }
            if (impulse_response_.size() < 2)
            {
                throw std::invalid_argument("Tail impulse response must be at least length 2");
            }

            const std::size_t spectra_len = tail_conv_len_ >> 1;
            cached_spectra_real_.assign(spectra_len + 1, sample(0));
            cached_spectra_imag_.assign(spectra_len + 1, sample(0));

            fft.fft(cached_spectra_real_.data(), cached_spectra_imag_.data(), impulse_response_.data());
            cached_spectra_imag_[0] = cached_spectra_real_[spectra_len];
        }

        std::size_t tail_len() const
        {
            return tail_len_;
        }

        bool staging() const
        {
            return staging_;
        }

        bool committing() const
        {
            return committing_;
        }

        std::size_t staging_len() const
        {
            return staging_len_;
        }

        void seek_staging(std::size_t n_steps, std::size_t max_input_id, std::size_t max_output_id)
        {
            staging_input_id_start_ += n_steps;
            staging_output_id_start_ += n_steps;
            streaming_detail::wrap_index(staging_input_id_start_, max_input_id);
            streaming_detail::wrap_index(staging_output_id_start_, max_output_id);
        }

        void advance_commit(
            std::size_t n_steps,
            sample *input_buffer,
            sample *output_buffer,
            std::size_t max_input_id,
            std::size_t max_output_id)
        {
            if (!committing_)
            {
                return;
            }

            commit_offset_ += n_steps;
            progress_commit(input_buffer, output_buffer, max_input_id, max_output_id);
        }

        void advance_stage(
            std::size_t n_steps,
            sample *input_buffer,
            sample *output_buffer,
            std::size_t max_input_id,
            std::size_t max_output_id)
        {
            staging_ = true;
            staging_len_ += n_steps;

            if (staging_len_ >= (tail_len_ / 2))
            {
                if (committing_) {
                    throw std::invalid_argument("Commit should have completed");
                }
                begin_commit(max_input_id, max_output_id);
                progress_commit(input_buffer, output_buffer, max_input_id, max_output_id);
            }
        }

    private:
        static std::size_t checked_len(std::size_t len)
        {
            if (len < 2 || (len & (len - 1)) != 0)
            {
                throw std::invalid_argument("Tail convolution length must be a power of two and >= 2");
            }
            return len;
        }

        void begin_commit(std::size_t max_input_id, std::size_t max_output_id)
        {
            if (committing_)
            {
                return;
            }

            commit_input_id_start_ = staging_input_id_start_;
            commit_output_id_start_ = staging_output_id_start_;
            seek_staging(staging_len_, max_input_id, max_output_id);

            commit_len_ = staging_len_;
            commit_offset_ = staging_len_;
            staging_len_ = 0;

            committing_ = true;
            staging_ = false;

            commit_phase_ = CommitPhase::CopyInput;
            input_copy_progress_ = 0;
            zero_pad_progress_ = 0;
            output_add_progress_ = 0;
            conv_steps_total_ = co_real_conv_.total_steps();
            conv_steps_done_ = 0;
            work_progress_ = 0;
            total_work_units_ = commit_len_ + (tail_conv_len_ - commit_len_) + conv_steps_total_ + tail_conv_len_;
            if (total_work_units_ == 0)
            {
                total_work_units_ = 1;
            }

            commit_input_read_pos_ = commit_input_id_start_;
            commit_output_write_pos_ = commit_output_id_start_;
            co_real_conv_.reset();

        }

        void progress_commit(
            sample *input_buffer,
            sample *output_buffer,
            std::size_t max_input_id,
            std::size_t max_output_id)
        {

            
            if (!committing_ || total_work_units_ == 0 || max_input_id == 0 || max_output_id == 0)
            {
                return;
            }

            if (commit_len_ > tail_input_buffer_.size())
            {
                throw std::runtime_error("Tail commit length exceeds buffer capacity");
            }

            if (input_buffer == nullptr || output_buffer == nullptr)
            {
                throw std::invalid_argument("Tail commit buffers must be valid");
            }

            const std::size_t half_tail = tail_len_ / 2;
            if (commit_offset_ <= half_tail)
            {
                return;
            }

            double target_fraction = (2.0 * static_cast<double>(commit_offset_) / static_cast<double>(tail_len_)) - 1.0;
            if (target_fraction < 0.0)
            {
                target_fraction = 0.0;
            }
            else if (target_fraction > 1.0)
            {
                target_fraction = 1.0;
            }

            std::size_t target_work = static_cast<std::size_t>(std::ceil(target_fraction * static_cast<double>(total_work_units_)));
            if (target_work > total_work_units_)
            {
                target_work = total_work_units_;
            }

            while (committing_ && work_progress_ < target_work)
            {
                switch (commit_phase_)
                {
                case CommitPhase::CopyInput:

                    if (input_copy_progress_ >= commit_len_)
                    {
                        commit_phase_ = CommitPhase::ZeroPad;
                        break;
                    }

                    tail_input_buffer_[input_copy_progress_] = input_buffer[commit_input_read_pos_];
                    ++input_copy_progress_;
                    ++work_progress_;
                    ++commit_input_read_pos_;
                    if (commit_input_read_pos_ >= max_input_id)
                    {
                        commit_input_read_pos_ = 0;
                    }
                    break;

                case CommitPhase::ZeroPad:
                    {
                        const std::size_t zero_len = tail_conv_len_ > commit_len_ ? tail_conv_len_ - commit_len_ : 0;
                        if (zero_pad_progress_ >= zero_len)
                        {
                            commit_phase_ = CommitPhase::Convolution;
                            break;
                        }

                        tail_input_buffer_[commit_len_ + zero_pad_progress_] = sample(0);
                        ++zero_pad_progress_;
                        ++work_progress_;
                    }
                    break;

                case CommitPhase::Convolution:
                    if (conv_steps_done_ >= conv_steps_total_)
                    {
                        commit_phase_ = CommitPhase::OutputAdd;
                        break;
                    }
                    {
                        const std::size_t remaining_conv = conv_steps_total_ - conv_steps_done_;
                        const std::size_t remaining_work = target_work - work_progress_;
                        if (remaining_work == 0)
                        {
                            return;
                        }

                        const std::size_t request_steps = std::min(remaining_conv, remaining_work);

                        if (tail_input_buffer_.size() < tail_conv_len_ ||
                            tail_conv_buffer_.size() < tail_conv_len_ ||
                            impulse_response_.size() != tail_conv_len_)
                        {
                            throw std::runtime_error("RealTailConv: buffer size mismatch before convolution");
                        }

                        // run incremental convolution
                        const std::size_t executed = co_real_conv_.process(
                            tail_conv_buffer_.data(),
                            tail_input_buffer_.data(),
                            cached_spectra_real_.data(),
                            cached_spectra_imag_.data(),
                            request_steps);

                        conv_steps_done_ += executed;
                        work_progress_ += executed;
                    }
                    break;

                case CommitPhase::OutputAdd:
                    if (output_add_progress_ >= tail_conv_len_)
                    {
                        finish_commit();
                        break;
                    }

                    output_buffer[commit_output_write_pos_] += tail_conv_buffer_[output_add_progress_];
                    ++output_add_progress_;
                    ++work_progress_;
                    ++commit_output_write_pos_;
                    if (commit_output_write_pos_ >= max_output_id)
                    {
                        commit_output_write_pos_ = 0;
                    }

                    if (output_add_progress_ >= tail_conv_len_)
                    {
                        finish_commit();
                    }
                    break;

                case CommitPhase::Idle:
                default:
                    return;
                }
            }

        }

        void finish_commit()
        {
            
            committing_ = false;
            commit_len_ = 0;
            commit_offset_ = 0;
            work_progress_ = total_work_units_;
            total_work_units_ = 0;
            conv_steps_total_ = 0;
            conv_steps_done_ = 0;
            input_copy_progress_ = 0;
            zero_pad_progress_ = 0;
            output_add_progress_ = 0;
            commit_phase_ = CommitPhase::Idle;
        }

        sample_vector impulse_response_;
        sample_vector tail_input_buffer_;
        sample_vector tail_conv_buffer_;
        std::size_t tail_len_;
        std::size_t tail_conv_len_;
        CoRealConvCached<Spec, Allocator> co_real_conv_;
        RealFft<Spec, Allocator> fft;
        sample_vector cached_spectra_real_;
        sample_vector cached_spectra_imag_;
        std::size_t staging_input_id_start_;
        std::size_t staging_output_id_start_;
        std::size_t commit_input_id_start_;
        std::size_t commit_output_id_start_;
        std::size_t staging_len_;
        std::size_t commit_len_;
        std::size_t commit_offset_;
        bool committing_;
        bool staging_;
        CommitPhase commit_phase_;
        std::size_t commit_input_read_pos_ = 0;
        std::size_t commit_output_write_pos_ = 0;
        std::size_t input_copy_progress_ = 0;
        std::size_t zero_pad_progress_ = 0;
        std::size_t output_add_progress_ = 0;
        std::size_t conv_steps_total_ = 0;
        std::size_t conv_steps_done_ = 0;
        std::size_t work_progress_ = 0;
        std::size_t total_work_units_ = 0;
    };
}

#endif
