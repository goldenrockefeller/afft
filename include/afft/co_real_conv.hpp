#ifndef AFFT_CO_REAL_CONV_HPP
#define AFFT_CO_REAL_CONV_HPP

#include <array>
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "afft/common_math.hpp"
#include "afft/execute.hpp"
#include "afft/coexecutor.hpp"
#include "afft/plan/plan_real_fft.hpp"
#include "afft/spec/bounded_spec.hpp"

namespace afft
{
    template <typename Spec, class Allocator = std::allocator<typename Spec::sample>>
    class CoRealConv
    {
    public:
        using sample = typename Spec::sample;
        using sample_spec = typename BoundedSpec<Spec, 0>::spec;

                explicit CoRealConv(std::size_t signal_len)
                        : signal_len_(checked_signal_len(signal_len)),
              spectra_len_(signal_len_ >> 1),
              log_n_samples_per_operand_(compute_operand_log(spectra_len_)),
              n_samples_per_operand_(std::size_t(1) << log_n_samples_per_operand_),
              conv_plan_(make_conv_plan(signal_len_, n_samples_per_operand_)),
                            coexecutor_(conv_plan_, log_n_samples_per_operand_),
                            is_small_case_(signal_len_ == 2),
                            small_case_progress_(0)
        {
            initialize_workspace();
        }

        std::size_t process(sample *signal_out, const sample *signal_a, const sample *signal_b, std::size_t n_steps)
        {
            if (n_steps == 0)
            {
                return 0;
            }

            if (signal_out == nullptr || signal_a == nullptr || signal_b == nullptr)
            {
                throw std::invalid_argument("CoRealConv requires non-null buffers");
            }

            if (is_small_case_)
            {
                if (small_case_progress_ > 0)
                {
                    return 0;
                }

                const sample a0 = signal_a[0];
                const sample a1 = signal_a[1];
                const sample b0 = signal_b[0];
                const sample b1 = signal_b[1];

                signal_out[0] = a0 * b0 + a1 * b1;
                signal_out[1] = a0 * b1 + a1 * b0;

                small_case_progress_ = 1;
                return 1;
            }

            data_[ConvolutionIds::conv_out] = signal_out;
            data_[ConvolutionIds::conv_out_offset] = signal_out + spectra_len_;
            data_[ConvolutionIds::conv_in_a] = const_cast<sample *>(signal_a);
            data_[ConvolutionIds::conv_in_b] = const_cast<sample *>(signal_b);

            data_[ConvolutionIds::conv_spectra_a_real] = conv_spectra_a_real_.data();
            data_[ConvolutionIds::conv_spectra_a_imag] = conv_spectra_a_imag_.data();
            data_[ConvolutionIds::conv_spectra_b_real] = conv_spectra_b_real_.data();
            data_[ConvolutionIds::conv_spectra_b_imag] = conv_spectra_b_imag_.data();

            sample *buf_a_real = buf_a_.data();
            sample *buf_b_real = buf_b_.data();
            data_[ConvolutionIds::conv_buf_a_real] = buf_a_real;
            data_[ConvolutionIds::conv_buf_a_imag] = buf_a_real + spectra_len_;
            data_[ConvolutionIds::conv_buf_b_real] = buf_b_real;
            data_[ConvolutionIds::conv_buf_b_imag] = buf_b_real + spectra_len_;

            const sample *twiddles_ptr = twiddles_.empty() ? nullptr : twiddles_.data();
            const std::size_t *out_perm_ptr = out_permute_.empty() ? nullptr : out_permute_.data();
            const std::size_t *in_perm_ptr = in_permute_.empty() ? nullptr : in_permute_.data();

            return coexecutor_.process(
                data_.data(),
                rotor_real_.data(),
                rotor_imag_.data(),
                twiddles_ptr,
                out_perm_ptr,
                in_perm_ptr,
                n_steps);
        }

        void reset()
        {
            if (is_small_case_)
            {
                small_case_progress_ = 0;
            }
            else
            {
                coexecutor_.reset();
            }
        }

        std::size_t total_steps() const
        {
            return is_small_case_ ? 1 : coexecutor_.total_length();
        }

        std::size_t steps_remaining() const
        {
            if (is_small_case_)
            {
                return small_case_progress_ == 0 ? 1 : 0;
            }
            return coexecutor_.steps_remaining();
        }

        bool finished() const
        {
            return is_small_case_ ? (small_case_progress_ > 0) : coexecutor_.finished();
        }

    private:
        enum ConvolutionIds : std::size_t
        {
            conv_out = 0,
            conv_out_offset,
            conv_in_a,
            conv_in_b,
            conv_spectra_a_real,
            conv_spectra_a_imag,
            conv_spectra_b_real,
            conv_spectra_b_imag,
            conv_buf_a_real,
            conv_buf_a_imag,
            conv_buf_b_real,
            conv_buf_b_imag
        };

        static constexpr std::size_t data_slot_count = ConvolutionIds::conv_buf_b_imag + 1;

        static std::size_t checked_signal_len(std::size_t len)
        {
            if (len < 2 || (len & (len - 1)) != 0)
            {
                throw std::invalid_argument("CoRealConv length must be a power of two and >= 2");
            }
            return len;
        }

        static std::size_t compute_operand_log(std::size_t spectra_len)
        {
            namespace cm = afft::common_math;
            if (spectra_len <= 1)
            {
                return 0;
            }

            const std::size_t max_operand_log = cm::int_log_2(Spec::n_samples_per_operand);
            const std::size_t spectra_bound_arg = std::max<std::size_t>(std::size_t(1), spectra_len >> 1);
            const std::size_t spectra_bound_log = spectra_bound_arg <= 1 ? 0 : cm::int_log_2(spectra_bound_arg);
            return std::min(max_operand_log, spectra_bound_log);
        }

        static std::vector<Stage<sample>> make_conv_plan(std::size_t signal_len, std::size_t n_samples_per_operand)
        {
            return plan::real_conv_plan<sample>(
                signal_len,
                n_samples_per_operand,
                Spec::min_partition_len,
                ConvolutionIds::conv_out,
                ConvolutionIds::conv_out_offset,
                ConvolutionIds::conv_in_a,
                ConvolutionIds::conv_in_b,
                ConvolutionIds::conv_spectra_a_real,
                ConvolutionIds::conv_spectra_a_imag,
                ConvolutionIds::conv_spectra_b_real,
                ConvolutionIds::conv_spectra_b_imag,
                ConvolutionIds::conv_buf_a_real,
                ConvolutionIds::conv_buf_a_imag,
                ConvolutionIds::conv_buf_b_real,
                ConvolutionIds::conv_buf_b_imag);
        }

        void initialize_workspace()
        {
            twiddles_ = plan::twiddles<sample_spec, Allocator>(conv_plan_, n_samples_per_operand_);
            auto permute = plan::bit_reverse_indexes(conv_plan_, spectra_len_, n_samples_per_operand_);
            in_permute_ = std::move(permute.first);
            out_permute_ = std::move(permute.second);

            auto rotor = plan::rfft_rotor<sample_spec, Allocator>(signal_len_);
            rotor_real_ = std::move(rotor.first);
            rotor_imag_ = std::move(rotor.second);

            buf_a_.resize(spectra_len_ * 2);
            buf_b_.resize(spectra_len_ * 2);
            conv_spectra_a_real_.resize(spectra_len_);
            conv_spectra_a_imag_.resize(spectra_len_);
            conv_spectra_b_real_.resize(spectra_len_);
            conv_spectra_b_imag_.resize(spectra_len_);
        }

        std::size_t signal_len_ = 0;
        std::size_t spectra_len_ = 0;
        std::size_t log_n_samples_per_operand_ = 0;
        std::size_t n_samples_per_operand_ = 0;

        std::vector<Stage<sample>> conv_plan_;
        execute::Coexecutor<Spec> coexecutor_;
        bool is_small_case_ = false;
        std::size_t small_case_progress_ = 0;

        std::vector<sample, Allocator> twiddles_;
        std::vector<std::size_t> in_permute_;
        std::vector<std::size_t> out_permute_;
        std::vector<sample, Allocator> rotor_real_;
        std::vector<sample, Allocator> rotor_imag_;

        std::vector<sample, Allocator> buf_a_;
        std::vector<sample, Allocator> buf_b_;
        std::vector<sample, Allocator> conv_spectra_a_real_;
        std::vector<sample, Allocator> conv_spectra_a_imag_;
        std::vector<sample, Allocator> conv_spectra_b_real_;
        std::vector<sample, Allocator> conv_spectra_b_imag_;

        std::array<sample *, data_slot_count> data_{};
    };
}

#endif
