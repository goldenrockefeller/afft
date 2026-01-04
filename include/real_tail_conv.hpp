#ifndef AFFT_REAL_TAIL_CONV_HPP
#define AFFT_REAL_TAIL_CONV_HPP

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

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

    template <typename Spec>
    class RealTailConv
    {
    public:
        using sample = typename Spec::sample;

        explicit RealTailConv(std::vector<sample> padded_tail_impulse)
            : impulse_response_(std::move(padded_tail_impulse)),
              real_fft_(checked_len(impulse_response_.size())),
              tail_input_buffer_(impulse_response_.size(), sample(0)),
              tail_conv_buffer_(impulse_response_.size(), sample(0)),
              tail_len_(impulse_response_.size() / 2),
              tail_conv_len_(impulse_response_.size()),
              staging_input_id_start_(0),
              staging_output_id_start_(tail_len_),
              commit_input_id_start_(0),
              commit_output_id_start_(tail_len_),
              staging_len_(0),
              commit_len_(0),
              commit_offset_(0),
              committing_(false),
              staging_(false)
        {
            if ((impulse_response_.size() & (impulse_response_.size() - 1)) != 0)
            {
                throw std::invalid_argument("Tail impulse response length must be a power of two");
            }
            if (impulse_response_.size() < 2)
            {
                throw std::invalid_argument("Tail impulse response must be at least length 2");
            }
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
            do_commit(input_buffer, output_buffer, max_input_id, max_output_id);
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
                begin_commit(max_input_id, max_output_id);
                do_commit(input_buffer, output_buffer, max_input_id, max_output_id);
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
        }

        void do_commit(
            sample *input_buffer,
            sample *output_buffer,
            std::size_t max_input_id,
            std::size_t max_output_id)
        {
            if (!committing_ || commit_offset_ < tail_len_)
            {
                return;
            }

            if (commit_len_ > tail_input_buffer_.size())
            {
                throw std::runtime_error("Tail commit length exceeds buffer capacity");
            }

            std::fill(tail_input_buffer_.begin(), tail_input_buffer_.end(), sample(0));

            std::size_t remaining = commit_len_;
            std::size_t dest_offset = 0;
            std::size_t read_pos = commit_input_id_start_;

            while (remaining > 0)
            {
                const std::size_t contiguous = std::min(remaining, max_input_id - read_pos);
                std::copy_n(
                    input_buffer + read_pos,
                    contiguous,
                    tail_input_buffer_.data() + dest_offset);

                remaining -= contiguous;
                dest_offset += contiguous;
                read_pos += contiguous;
                if (read_pos >= max_input_id)
                {
                    read_pos = 0;
                }
            }

            real_fft_.conv(
                tail_conv_buffer_.data(),
                tail_input_buffer_.data(),
                impulse_response_.data());

            streaming_detail::circular_add(
                output_buffer,
                max_output_id,
                tail_conv_buffer_.data(),
                tail_conv_len_,
                commit_output_id_start_);

            committing_ = false;
            commit_len_ = 0;
            commit_offset_ = 0;
        }

        std::vector<sample> impulse_response_;
        RealFft<Spec> real_fft_;
        std::vector<sample> tail_input_buffer_;
        std::vector<sample> tail_conv_buffer_;
        std::size_t tail_len_;
        std::size_t tail_conv_len_;
        std::size_t staging_input_id_start_;
        std::size_t staging_output_id_start_;
        std::size_t commit_input_id_start_;
        std::size_t commit_output_id_start_;
        std::size_t staging_len_;
        std::size_t commit_len_;
        std::size_t commit_offset_;
        bool committing_;
        bool staging_;
    };
}

#endif
