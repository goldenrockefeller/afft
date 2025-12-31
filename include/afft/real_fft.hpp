#ifndef AFFT_REAL_FFT_HPP
#define AFFT_REAL_FFT_HPP

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "afft/common_math.hpp"
#include "afft/execute.hpp"
#include "afft/plan/plan_real_fft.hpp"
#include "afft/spec/bounded_spec.hpp"

namespace afft
{
    template <typename Spec, class Allocator = std::allocator<typename Spec::sample>>
    class RealFft
    {
    public:
        using sample = typename Spec::sample;
        using sample_spec = typename BoundedSpec<Spec, 0>::spec;

        explicit RealFft(std::size_t signal_len)
            : signal_len_(checked_signal_len(signal_len)),
              spectra_len_(signal_len_ >> 1),
              unpacked_spectra_len_((signal_len_ >> 1) + 1),
              log_n_samples_per_operand_(compute_operand_log(spectra_len_)),
              n_samples_per_operand_(std::size_t(1) << log_n_samples_per_operand_),
              scaling_factor_(sample(1) / sample(signal_len_))
        {
            build_fft_plan();            
            build_ifft_plan();
            build_conv_plan();

            twiddles_ = plan::twiddles<sample_spec, Allocator>(fft_plan_, n_samples_per_operand_);
            auto permute = plan::bit_reverse_indexes(fft_plan_, spectra_len_, n_samples_per_operand_);
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

        std::size_t signal_len() const
        {
            return signal_len_;
        }

        std::size_t spectra_len() const
        {
            return spectra_len_;
        }

        std::size_t unpacked_spectra_len() const
        {
            return unpacked_spectra_len_;
        }

        std::size_t log_n_samples_per_operand() const
        {
            return log_n_samples_per_operand_;
        }

        void fft(sample *spectra_real, sample *spectra_imag, const sample *signal) const
        {
            execute_fft_plan(spectra_real, spectra_imag, signal, false);
        }

        void fft_normalized(sample *spectra_real, sample *spectra_imag, const sample *signal) const
        {
            execute_fft_plan(spectra_real, spectra_imag, signal, true);
        }

        void ifft(sample *signal, const sample *spectra_real, const sample *spectra_imag) const
        {
            execute_ifft_plan(signal, spectra_real, spectra_imag, false);
        }

        void ifft_normalized(sample *signal, const sample *spectra_real, const sample *spectra_imag) const
        {
            execute_ifft_plan(signal, spectra_real, spectra_imag, true);
        }

        void conv(sample *signal_out, const sample *signal_a, const sample *signal_b) const
        {
            if (signal_len_ == 2) {
                const sample a0 = signal_a[0];
                const sample a1 = signal_a[1];
                const sample b0 = signal_b[0];
                const sample b1 = signal_b[1];

                signal_out[0] = a0 * b0 + a1 * b1;
                signal_out[1] = a0 * b1 + a1 * b0;
                return;
            }
            execute_conv_plan(signal_out, signal_a, signal_b);
        }

    private:
        enum ForwardIds : std::size_t
        {
            forward_signal = 0,
            forward_spectra_real,
            forward_spectra_imag,
            forward_buf_a_real,
            forward_buf_a_imag,
            forward_buf_b_real,
            forward_buf_b_imag
        };

        enum InverseIds : std::size_t
        {
            inverse_signal = 0,
            inverse_signal_offset,
            inverse_spectra_real,
            inverse_spectra_imag,
            inverse_buf_a_real,
            inverse_buf_a_imag,
            inverse_buf_b_real,
            inverse_buf_b_imag
        };

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

        static std::size_t checked_signal_len(std::size_t len)
        {
            if (len < 2 || (len & (len - 1)) != 0)
            {
                throw std::invalid_argument("RealFft length must be a power of two and >= 2");
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

        void build_fft_plan()
        {
            fft_plan_ = plan::real_fft_plan<sample>(signal_len_, n_samples_per_operand_, Spec::min_partition_len);
            plan::set_data_ids_for_real_fft(
                fft_plan_,
                ForwardIds::forward_signal,
                ForwardIds::forward_spectra_real,
                ForwardIds::forward_spectra_imag,
                ForwardIds::forward_buf_a_real,
                ForwardIds::forward_buf_a_imag,
                ForwardIds::forward_buf_b_real,
                ForwardIds::forward_buf_b_imag);

            fft_scaled_plan_ = fft_plan_;
            plan::replace_init_stages_with_rescale(fft_scaled_plan_, scaling_factor_);
        }

        void build_ifft_plan()
        {
            auto ifft_plan_base = plan::inv_real_fft_plan<sample>(signal_len_, n_samples_per_operand_, Spec::min_partition_len);
            plan::set_data_ids_for_real_ifft(
                ifft_plan_base,
                InverseIds::inverse_signal,
                InverseIds::inverse_signal_offset,
                InverseIds::inverse_spectra_real,
                InverseIds::inverse_spectra_imag,
                InverseIds::inverse_buf_a_real,
                InverseIds::inverse_buf_a_imag,
                InverseIds::inverse_buf_b_real,
                InverseIds::inverse_buf_b_imag);

            ifft_plan_ = ifft_plan_base;
            ifft_scaled_plan_ = ifft_plan_base;
            plan::replace_init_stages_with_rescale(ifft_plan_, sample(2.));
            plan::replace_init_stages_with_rescale(ifft_scaled_plan_, sample(2.) * scaling_factor_);           
        }

        void build_conv_plan()
        {
            conv_plan_ = plan::real_conv_plan<sample>(
                signal_len_,
                n_samples_per_operand_,
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

        void execute_fft_plan(sample *spectra_real, sample *spectra_imag, const sample *signal, bool normalized) const
        {
            // Special-case N=2: compute DC and Nyquist directly to avoid plan edge-cases
            if (signal_len_ == 2) {
                sample x0 = signal[0];
                sample x1 = signal[1];
                sample factor = normalized ? sample(0.5) : sample(1);

                // unpacked_spectra_len_ == 2 for N=2
                spectra_real[0] = (x0 + x1) * factor;
                spectra_imag[0] = sample(0);
                spectra_real[1] = (x0 - x1) * factor;
                spectra_imag[1] = sample(0);
                return;
            }

            sample *data[7];
            sample *buf_a_real = buf_a_.data();
            sample *buf_b_real = buf_b_.data();

            // Assign stage buffers in the order expected by plan_real_fft helpers.
            data[ForwardIds::forward_signal] = const_cast<sample *>(signal);
            data[ForwardIds::forward_spectra_real] = spectra_real;
            data[ForwardIds::forward_spectra_imag] = spectra_imag;
            data[ForwardIds::forward_buf_a_real] = buf_a_real;
            data[ForwardIds::forward_buf_a_imag] = buf_a_real + spectra_len_;
            data[ForwardIds::forward_buf_b_real] = buf_b_real;
            data[ForwardIds::forward_buf_b_imag] = buf_b_real + spectra_len_;

            const auto &plan_ref = normalized ? fft_scaled_plan_ : fft_plan_;
            const auto *twiddles = twiddles_.empty() ? nullptr : twiddles_.data();
            const auto *out_perm = out_permute_.empty() ? nullptr : out_permute_.data();
            const auto *in_perm = in_permute_.empty() ? nullptr : in_permute_.data();

            execute::eval<Spec>(
                data,
                plan_ref,
                rotor_real_.data(),
                rotor_imag_.data(),
                twiddles,
                out_perm,
                in_perm,
                log_n_samples_per_operand_);
        }

        void execute_ifft_plan(sample *signal, const sample *spectra_real, const sample *spectra_imag, bool normalized) const
        {
            // Special-case N=2: inverse can be computed directly
            if (signal_len_ == 2) {
                sample X0 = spectra_real[0];
                sample X1 = spectra_real[1];

                if (normalized) {
                    // 
                    signal[0] = (X0 + X1) * sample(0.5);
                    signal[1] = (X0 - X1) * sample(0.5);
                } else {
                    // 
                    signal[0] = (X0 + X1);
                    signal[1] = (X0 - X1);
                }
                return;
            }

            sample *data[8];
            sample *buf_a_real = buf_a_.data();
            sample *buf_b_real = buf_b_.data();

            data[InverseIds::inverse_signal] = signal;
            data[InverseIds::inverse_signal_offset] = signal + spectra_len_;
            data[InverseIds::inverse_spectra_real] = const_cast<sample *>(spectra_real);
            data[InverseIds::inverse_spectra_imag] = const_cast<sample *>(spectra_imag);
            data[InverseIds::inverse_buf_a_real] = buf_a_real;
            data[InverseIds::inverse_buf_a_imag] = buf_a_real + spectra_len_;
            data[InverseIds::inverse_buf_b_real] = buf_b_real;
            data[InverseIds::inverse_buf_b_imag] = buf_b_real + spectra_len_;

            const auto &plan_ref = normalized ? ifft_scaled_plan_ : ifft_plan_;
            const auto *twiddles = twiddles_.empty() ? nullptr : twiddles_.data();
            const auto *out_perm = out_permute_.empty() ? nullptr : out_permute_.data();
            const auto *in_perm = in_permute_.empty() ? nullptr : in_permute_.data();

            execute::eval<Spec>(
                data,
                plan_ref,
                rotor_real_.data(),
                rotor_imag_.data(),
                twiddles,
                out_perm,
                in_perm,
                log_n_samples_per_operand_);
        }
        void execute_conv_plan(sample *signal_out, const sample *signal_a, const sample *signal_b) const
        {
            sample *data[ConvolutionIds::conv_buf_b_imag + 1];

            data[ConvolutionIds::conv_out] = signal_out;
            data[ConvolutionIds::conv_out_offset] = signal_out + spectra_len_;
            data[ConvolutionIds::conv_in_a] = const_cast<sample *>(signal_a);
            data[ConvolutionIds::conv_in_b] = const_cast<sample *>(signal_b);

            data[ConvolutionIds::conv_spectra_a_real] = conv_spectra_a_real_.data();
            data[ConvolutionIds::conv_spectra_a_imag] = conv_spectra_a_imag_.data();
            data[ConvolutionIds::conv_spectra_b_real] = conv_spectra_b_real_.data();
            data[ConvolutionIds::conv_spectra_b_imag] = conv_spectra_b_imag_.data();

            sample *buf_a_real = buf_a_.data();
            sample *buf_a_imag = buf_a_real + spectra_len_;
            sample *buf_b_real = buf_b_.data();
            sample *buf_b_imag = buf_b_real + spectra_len_;

            data[ConvolutionIds::conv_buf_a_real] = buf_a_real;
            data[ConvolutionIds::conv_buf_a_imag] = buf_a_imag;
            data[ConvolutionIds::conv_buf_b_real] = buf_b_real;
            data[ConvolutionIds::conv_buf_b_imag] = buf_b_imag;

            const auto *twiddles = twiddles_.empty() ? nullptr : twiddles_.data();
            const auto *out_perm = out_permute_.empty() ? nullptr : out_permute_.data();
            const auto *in_perm = in_permute_.empty() ? nullptr : in_permute_.data();

            execute::eval<Spec>(
                data,
                conv_plan_,
                rotor_real_.data(),
                rotor_imag_.data(),
                twiddles,
                out_perm,
                in_perm,
                log_n_samples_per_operand_);
        }

    private:
        std::size_t signal_len_;
        std::size_t spectra_len_;
        std::size_t unpacked_spectra_len_;
        std::size_t log_n_samples_per_operand_;
        std::size_t n_samples_per_operand_;
        sample scaling_factor_;

        std::vector<Stage<sample>> fft_plan_;
        std::vector<Stage<sample>> fft_scaled_plan_;
        std::vector<Stage<sample>> ifft_plan_;
        std::vector<Stage<sample>> ifft_scaled_plan_;
        std::vector<Stage<sample>> conv_plan_;

        std::vector<sample, Allocator> twiddles_;
        std::vector<std::size_t> in_permute_;
        std::vector<std::size_t> out_permute_;

        std::vector<sample, Allocator> rotor_real_;
        std::vector<sample, Allocator> rotor_imag_;

        mutable std::vector<sample, Allocator> buf_a_;
        mutable std::vector<sample, Allocator> buf_b_;
        mutable std::vector<sample, Allocator> conv_spectra_a_real_;
        mutable std::vector<sample, Allocator> conv_spectra_a_imag_;
        mutable std::vector<sample, Allocator> conv_spectra_b_real_;
        mutable std::vector<sample, Allocator> conv_spectra_b_imag_;
    };
}

#endif
