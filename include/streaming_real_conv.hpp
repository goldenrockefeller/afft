#ifndef AFFT_STREAMING_REAL_CONV_HPP
#define AFFT_STREAMING_REAL_CONV_HPP

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "afft/real_fft.hpp"
#include "real_tail_conv.hpp"

namespace afft
{
    template <typename Spec>
    class StreamingRealConv
    {
    public:
        using sample = typename Spec::sample;

        StreamingRealConv(const sample *impulse_response, std::size_t impulse_len)
                        : convolution_len_(compute_convolution_len(impulse_len)),
                            input_buffer_len_(4 * convolution_len_),
              output_buffer_len_(4 * convolution_len_),
              head_input_id_(0),
              head_output_id_(0),
              input_buffer_(input_buffer_len_, sample(0)),
              output_buffer_(output_buffer_len_, sample(0)),
              head_input_buffer_(2 * convolution_len_, sample(0)),
              head_conv_buffer_(2 * convolution_len_, sample(0))
        {
            if (impulse_response == nullptr || impulse_len == 0)
            {
                throw std::invalid_argument("Impulse response must not be empty");
            }

            build_head_partitions(impulse_response, impulse_len);
            build_tail_convs(impulse_response, impulse_len);
        }

        std::size_t convolution_len() const
        {
            return convolution_len_;
        }

        std::size_t max_safe_subslice_len() const
        {
            std::size_t max_safe = convolution_len_;
            for (const auto &tail : tail_convs_)
            {
                if (tail.staging())
                {
                    const std::size_t available = tail.tail_len() - tail.staging_len();
                    max_safe = std::min(max_safe, available);
                }
            }
            return max_safe;
        }

        void process_slice(const sample *slice, std::size_t slice_len, sample *output)
        {
            if (slice_len == 0)
            {
                return;
            }
            if (output == nullptr)
            {
                throw std::invalid_argument("Output buffer must be valid");
            }

            std::size_t consumed = 0;
            while (consumed < slice_len)
            {
                const std::size_t remaining = slice_len - consumed;
                const std::size_t subslice_len = std::min(remaining, max_safe_subslice_len());
                if (subslice_len == 0)
                {
                    throw std::runtime_error("Unable to determine a safe subslice length");
                }
                process_subslice(slice + consumed, subslice_len, output + consumed);
                consumed += subslice_len;
            }
        }

    private:
        struct HeadPartition
        {
            std::size_t head_len;
            std::size_t padded_len;
            RealFft<Spec> fft;
            std::vector<sample> impulse;

            HeadPartition(std::size_t head_len_value, const sample *impulse_src, std::size_t impulse_len)
                : head_len(head_len_value),
                  padded_len(head_len_value * 2),
                  fft(padded_len),
                  impulse(padded_len, sample(0))
            {
                const std::size_t copy_len = std::min(head_len, impulse_len);
                std::copy_n(impulse_src, copy_len, impulse.begin());
            }
        };

        static std::size_t next_power_two(std::size_t value)
        {
            if (value == 0)
            {
                return 1;
            }

            std::size_t power = 1;
            while (power < value)
            {
                power <<= 1;
            }
            return power;
        }

        static std::size_t compute_convolution_len(std::size_t impulse_len)
        {
            const std::size_t next_pow_two = next_power_two(impulse_len);
            return std::max<std::size_t>(2, next_pow_two);
        }

        HeadPartition &partition_for_len(std::size_t padded_len)
        {
            for (auto &partition : head_partitions_)
            {
                if (partition.padded_len == padded_len)
                {
                    return partition;
                }
            }
            throw std::invalid_argument("Requested head length is not available");
        }

        void build_head_partitions(const sample *impulse_response, std::size_t impulse_len)
        {
            std::size_t head_len = 2;
            while (head_len <= convolution_len_)
            {
                head_partitions_.emplace_back(head_len, impulse_response, impulse_len);
                head_len <<= 1;
            }
        }

        void build_tail_convs(const sample *impulse_response, std::size_t impulse_len)
        {
            std::size_t tail_len = 2;
            while ((tail_len * 2) <= convolution_len_)
            {
                const std::size_t padded_len = tail_len * 2;
                std::vector<sample> padded_response(padded_len, sample(0));

                const std::size_t impulse_end = std::min(tail_len * 2, impulse_len);
                if (impulse_end > tail_len)
                {
                    const std::size_t impulse_section_len = impulse_end - tail_len;
                    std::copy_n(impulse_response + tail_len, impulse_section_len, padded_response.begin());
                }

                tail_convs_.emplace_back(std::move(padded_response));
                tail_len <<= 1;
            }
        }

        void process_subslice(const sample *subslice, std::size_t slice_len, sample *output)
        {
            const std::size_t first_cut = std::min(slice_len, input_buffer_len_ - head_input_id_);
            std::copy_n(subslice, first_cut, input_buffer_.data() + head_input_id_);
            const std::size_t second_cut = slice_len - first_cut;
            if (second_cut > 0)
            {
                std::copy_n(subslice + first_cut, second_cut, input_buffer_.data());
            }

            std::size_t head_len = 2;
            std::size_t seeked_tail_len = 0;
            for (auto &tail : tail_convs_)
            {
                if (!tail.staging())
                {
                    if (slice_len >= (tail.tail_len() / 2))
                    {
                        seeked_tail_len = tail.tail_len();
                        head_len = 2 * tail.tail_len();
                        tail.seek_staging(slice_len, input_buffer_len_, output_buffer_len_);
                    }
                    else
                    {
                        break;
                    }
                }
                else
                {
                    break;
                }
            }

            const std::size_t padded_len = head_len * 2;
            HeadPartition &partition = partition_for_len(padded_len);
            std::fill(head_input_buffer_.begin(), head_input_buffer_.begin() + padded_len, sample(0));
            std::copy_n(subslice, slice_len, head_input_buffer_.begin());
            partition.fft.conv(head_conv_buffer_.data(), head_input_buffer_.data(), partition.impulse.data());
            streaming_detail::circular_add(
                output_buffer_.data(),
                output_buffer_len_,
                head_conv_buffer_.data(),
                padded_len,
                head_output_id_);

            for (auto &tail : tail_convs_)
            {
                if (tail.committing())
                {
                    tail.advance_commit(slice_len, input_buffer_.data(), output_buffer_.data(), input_buffer_len_, output_buffer_len_);
                }
                if (tail.tail_len() > seeked_tail_len)
                {
                    tail.advance_stage(slice_len, input_buffer_.data(), output_buffer_.data(), input_buffer_len_, output_buffer_len_);
                }
            }

            const std::size_t out_cut_one = std::min(slice_len, output_buffer_len_ - head_output_id_);
            std::copy_n(output_buffer_.data() + head_output_id_, out_cut_one, output);
            const std::size_t out_cut_two = slice_len - out_cut_one;
            if (out_cut_two > 0)
            {
                std::copy_n(output_buffer_.data(), out_cut_two, output + out_cut_one);
            }

            std::fill_n(output_buffer_.data() + head_output_id_, out_cut_one, sample(0));
            if (out_cut_two > 0)
            {
                std::fill_n(output_buffer_.data(), out_cut_two, sample(0));
            }

            head_input_id_ += slice_len;
            if (head_input_id_ >= input_buffer_len_)
            {
                head_input_id_ -= input_buffer_len_;
            }

            head_output_id_ += slice_len;
            if (head_output_id_ >= output_buffer_len_)
            {
                head_output_id_ -= output_buffer_len_;
            }
        }

        std::size_t convolution_len_;
        std::size_t input_buffer_len_;
        std::size_t output_buffer_len_;
        std::size_t head_input_id_;
        std::size_t head_output_id_;
        std::vector<sample> input_buffer_;
        std::vector<sample> output_buffer_;
        std::vector<sample> head_input_buffer_;
        std::vector<sample> head_conv_buffer_;
        std::vector<HeadPartition> head_partitions_;
        std::vector<RealTailConv<Spec>> tail_convs_;
    };
}

#endif
