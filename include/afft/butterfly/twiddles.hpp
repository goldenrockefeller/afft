
#ifndef AFFT_TWIDDLES_HPP
#define AFFT_TWIDDLES_HPP

#include <vector>
#include <cstddef>

namespace afft
{
    namespace twiddles
    {
        template <typename Vector>
        Vector repeat(const Vector& arr, std::size_t times) {
            Vector new_vec(arr.size() * times);

            for (std::size_t i = 0; i < times; ++i) {
                for (std::size_t j = 0; j < arr.size(); ++j) {
                    new_vec[i * arr.size() + j] = arr[j];
                }
            }

            return new_vec;
        }

        template <typename Spec, class Allocator = std::allocator<typename Spec::sample>>
        std::vector<typename Spec::sample, Allocator> subtwiddle_real(std::size_t subtwiddle_len, typename Spec::sample freq) {
            using sample = typename Spec::sample;
            std::vector<sample, Allocator> result(subtwiddle_len);

            for (std::size_t i = 0; i < subtwiddle_len; ++i) {
                sample x = i * freq / typename Spec::sample(subtwiddle_len);
                result[i] = Spec::cos( Spec::pi() * x);
            }

            return result;
        }

        template <typename Spec, class Allocator = std::allocator<typename Spec::sample>>
        std::vector<typename Spec::sample, Allocator> subtwiddle_imag(std::size_t subtwiddle_len, typename Spec::sample freq) {
            using sample = typename Spec::sample;
            std::vector<sample, Allocator> result(subtwiddle_len);

            for (std::size_t i = 0; i < subtwiddle_len; ++i) {
                sample x = i * freq / typename Spec::sample(subtwiddle_len);
                result[i] = -Spec::sin(Spec::pi() * x);
            }

            return result;
        }


        template <typename Spec, class Allocator = std::allocator<typename Spec::sample>>
        std::vector<std::vector<typename Spec::sample, Allocator>> ct_radix2_twiddles_real(std::size_t subtwiddle_len){
            std::vector<std::vector<typename Spec::sample, Allocator>> res_;

            res_.push_back(subtwiddle_real<Spec, Allocator>(subtwiddle_len, typename Spec::sample(1)));

            return res_;
        }

        template <typename Spec, class Allocator = std::allocator<typename Spec::sample>>
        std::vector<std::vector<typename Spec::sample, Allocator>> ct_radix2_twiddles_imag(std::size_t subtwiddle_len){
            std::vector<std::vector<typename Spec::sample, Allocator>> res_;

            res_.push_back(subtwiddle_imag<Spec, Allocator>(subtwiddle_len, typename Spec::sample(1)));

            return res_;
        }

        template <typename Spec, class Allocator = std::allocator<typename Spec::sample>>
        std::vector<typename Spec::sample, Allocator> ct_radix2_twiddles(std::size_t subtwiddle_len, std::size_t n_samples_per_operand) {
            std::vector<typename Spec::sample, Allocator> res_;

            auto tw_real = ct_radix2_twiddles_real<Spec, Allocator>(subtwiddle_len);
            auto tw_imag = ct_radix2_twiddles_imag<Spec, Allocator>(subtwiddle_len);

            auto tw_len = tw_real[0].size();

            for (std::size_t i = 0; i < tw_len; i+= n_samples_per_operand) {
                for (std::size_t j = 0; j < n_samples_per_operand; j++) {
                    res_.push_back(tw_real[0][i+j]);
                }

                for (std::size_t j = 0; j < n_samples_per_operand; j++) {
                    res_.push_back(tw_imag[0][i+j]);
                }
            }

            return res_;
        }

        template <typename Spec, class Allocator = std::allocator<typename Spec::sample>>
        std::vector<std::vector<typename Spec::sample, Allocator>> ct_radix4_twiddles_real(std::size_t subtwiddle_len){
            std::vector<std::vector<typename Spec::sample, Allocator>> res_;

            res_.push_back(subtwiddle_real<Spec, Allocator>(subtwiddle_len, typename Spec::sample(1)));
            res_.push_back(subtwiddle_real<Spec, Allocator>(subtwiddle_len, typename Spec::sample(0.5)));
            res_.push_back(subtwiddle_real<Spec, Allocator>(subtwiddle_len, typename Spec::sample(1.5)));

            return res_;
        }
        

        template <typename Spec, class Allocator = std::allocator<typename Spec::sample>>
        std::vector<std::vector<typename Spec::sample, Allocator>> ct_radix4_twiddles_imag(std::size_t subtwiddle_len){
            std::vector<std::vector<typename Spec::sample, Allocator>> res_;

            res_.push_back(subtwiddle_imag<Spec, Allocator>(subtwiddle_len, typename Spec::sample(1)));
            res_.push_back(subtwiddle_imag<Spec, Allocator>(subtwiddle_len, typename Spec::sample(0.5)));
            res_.push_back(subtwiddle_imag<Spec, Allocator>(subtwiddle_len, typename Spec::sample(1.5)));

            return res_;
        }

        template <typename Spec, class Allocator = std::allocator<typename Spec::sample>>
        std::vector<typename Spec::sample, Allocator> ct_radix4_twiddles(std::size_t subtwiddle_len, std::size_t n_samples_per_operand) {
            std::vector<typename Spec::sample, Allocator> res_;

            auto tw_real = ct_radix4_twiddles_real<Spec, Allocator>(subtwiddle_len);
            auto tw_imag = ct_radix4_twiddles_imag<Spec, Allocator>(subtwiddle_len);

            auto tw_len = tw_real[0].size();

            for (std::size_t i = 0; i < tw_len; i+= n_samples_per_operand) {
                for (std::size_t j = 0; j < n_samples_per_operand; j++) {
                    res_.push_back(tw_real[0][i+j]);
                }

                for (std::size_t j = 0; j < n_samples_per_operand; j++) {
                    res_.push_back(tw_imag[0][i+j]);
                }

                for (std::size_t j = 0; j < n_samples_per_operand; j++) {
                    res_.push_back(tw_real[1][i+j]);
                }

                for (std::size_t j = 0; j < n_samples_per_operand; j++) {
                    res_.push_back(tw_imag[1][i+j]);
                }

                for (std::size_t j = 0; j < n_samples_per_operand; j++) {
                    res_.push_back(tw_real[2][i+j]);
                }

                for (std::size_t j = 0; j < n_samples_per_operand; j++) {
                    res_.push_back(tw_imag[2][i+j]);
                }
            }

            return res_;
        }

        template <typename Spec, class Allocator = std::allocator<typename Spec::sample>>
        std::vector<std::vector<typename Spec::sample, Allocator>> s_radix2_twiddles_real(std::size_t subtwiddle_len, std::size_t n_samples_per_operand){
            std::vector<std::vector<typename Spec::sample, Allocator>> res_;

            res_.push_back(
                repeat(
                    subtwiddle_real<Spec, Allocator>(subtwiddle_len, typename Spec::sample(1)),
                    n_samples_per_operand / subtwiddle_len
                )
            );

            return res_;
        }

        template <typename Spec, class Allocator = std::allocator<typename Spec::sample>>
        std::vector<std::vector<typename Spec::sample, Allocator>> s_radix2_twiddles_imag(std::size_t subtwiddle_len, std::size_t n_samples_per_operand){
            std::vector<std::vector<typename Spec::sample, Allocator>> res_;

            res_.push_back(
                repeat(
                    subtwiddle_imag<Spec, Allocator>(subtwiddle_len, typename Spec::sample(1)),
                    n_samples_per_operand / subtwiddle_len
                )
            );

            return res_;
        }

        template <typename Spec, class Allocator = std::allocator<typename Spec::sample>>
        std::vector<typename Spec::sample, Allocator> s_radix2_twiddles(std::size_t subtwiddle_len, std::size_t n_samples_per_operand) {
            std::vector<typename Spec::sample, Allocator> res_;

            auto tw_real = s_radix2_twiddles_real<Spec, Allocator>(subtwiddle_len, n_samples_per_operand);
            auto tw_imag = s_radix2_twiddles_imag<Spec, Allocator>(subtwiddle_len, n_samples_per_operand);

            auto tw_len = tw_real[0].size();

            for (std::size_t i = 0; i < tw_len; i+= n_samples_per_operand) {
                for (std::size_t j = 0; j < n_samples_per_operand; j++) {
                    res_.push_back(tw_real[0][i+j]);
                }

                for (std::size_t j = 0; j < n_samples_per_operand; j++) {
                    res_.push_back(tw_imag[0][i+j]);
                }
            }


            return res_;
        }


        template <typename Spec, class Allocator = std::allocator<typename Spec::sample>>
        std::vector<std::vector<typename Spec::sample, Allocator>> s_radix4_twiddles_real(std::size_t subtwiddle_len, std::size_t n_samples_per_operand){
            std::vector<std::vector<typename Spec::sample, Allocator>> res_;

            res_.push_back(
                repeat(
                    subtwiddle_real<Spec, Allocator>(subtwiddle_len, typename Spec::sample(1)),
                    n_samples_per_operand / subtwiddle_len
                )
            );

            res_.push_back(
                repeat(
                    subtwiddle_real<Spec, Allocator>(subtwiddle_len, typename Spec::sample(0.5)),
                    n_samples_per_operand / subtwiddle_len
                )
            );

            res_.push_back(
                repeat(
                    subtwiddle_real<Spec, Allocator>(subtwiddle_len, typename Spec::sample(1.5)),
                    n_samples_per_operand / subtwiddle_len
                )
            );

            return res_;
        }

        template <typename Spec, class Allocator = std::allocator<typename Spec::sample>>
        std::vector<std::vector<typename Spec::sample, Allocator>> s_radix4_twiddles_imag(std::size_t subtwiddle_len, std::size_t n_samples_per_operand){
            std::vector<std::vector<typename Spec::sample, Allocator>> res_;

            res_.push_back(
                repeat(
                    subtwiddle_imag<Spec, Allocator>(subtwiddle_len, typename Spec::sample(1)),
                    n_samples_per_operand / subtwiddle_len
                )
            );

            res_.push_back(
                repeat(
                    subtwiddle_imag<Spec, Allocator>(subtwiddle_len, typename Spec::sample(0.5)),
                    n_samples_per_operand / subtwiddle_len
                )
            );

            res_.push_back(
                repeat(
                    subtwiddle_imag<Spec, Allocator>(subtwiddle_len, typename Spec::sample(1.5)),
                    n_samples_per_operand / subtwiddle_len
                )
            );

            return res_;
        }

        template <typename Spec, class Allocator = std::allocator<typename Spec::sample>>
        std::vector<typename Spec::sample, Allocator> s_radix4_twiddles(std::size_t subtwiddle_len, std::size_t n_samples_per_operand) {
            std::vector<typename Spec::sample, Allocator> res_;

            auto tw_real = s_radix4_twiddles_real<Spec, Allocator>(subtwiddle_len, n_samples_per_operand);
            auto tw_imag = s_radix4_twiddles_imag<Spec, Allocator>(subtwiddle_len, n_samples_per_operand);

            auto tw_len = tw_real[0].size();

            for (std::size_t i = 0; i < tw_len; i+= n_samples_per_operand) {
                for (std::size_t j = 0; j < n_samples_per_operand; j++) {
                    res_.push_back(tw_real[0][i+j]);
                }

                for (std::size_t j = 0; j < n_samples_per_operand; j++) {
                    res_.push_back(tw_imag[0][i+j]);
                }

                for (std::size_t j = 0; j < n_samples_per_operand; j++) {
                    res_.push_back(tw_real[1][i+j]);
                }

                for (std::size_t j = 0; j < n_samples_per_operand; j++) {
                    res_.push_back(tw_imag[1][i+j]);
                }

                for (std::size_t j = 0; j < n_samples_per_operand; j++) {
                    res_.push_back(tw_real[2][i+j]);
                }

                for (std::size_t j = 0; j < n_samples_per_operand; j++) {
                    res_.push_back(tw_imag[2][i+j]);
                }
            }

            return res_;
        }
    }
}

#endif