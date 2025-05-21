
#ifndef AFFT_TWIDDLES_HPP
#define AFFT_TWIDDLES_HPP

#include <vector>
#include <cstddef>

#include "afft/radix/radix_stage.hpp"
#include "afft/radix/radix_type.hpp"
#include "afft/butterfly/twiddles.hpp"

namespace afft
{
    namespace twiddles
    {
        template <typename Vector>
        Vector extend(const Vector& arr, std::size_t times) {
            Vector new_vec(arr.size() * times);

            for (std::size_t i = 0; i < arr.size(); ++i) {
                for (std::size_t j = 0; j < times; ++j) {
                    new_vec[times * i + j] = arr[i];
                }
            }
            return new_vec;
        }

        template <typename Vector>
        Vector concatenate(const Vector& a, const Vector& b) {
            Vector result;
            result.reserve(a.size() + b.size()); // Reserve memory to improve performance
            result.insert(result.end(), a.begin(), a.end());
            result.insert(result.end(), b.begin(), b.end());
            return result;
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
        std::vector<typename Spec::sample, Allocator> compound_radix2_twiddles_real(std::size_t n_samples_per_operand) {
            std::size_t subtwiddle_len = 1;
            std::size_t extend_len = n_samples_per_operand;
            std::vector<typename Spec::sample, Allocator> res_;

            subtwiddle_len = subtwiddle_len << 1;
            extend_len = extend_len >> 1;

            while (extend_len > 0) {
                auto subtwiddle_ = subtwiddle_real<Spec, Allocator>(subtwiddle_len, typename Spec::sample(1));
                subtwiddle_ = extend(subtwiddle_, extend_len);
                res_ = concatenate(res_, subtwiddle_);
                subtwiddle_len = subtwiddle_len << 1;
                extend_len = extend_len >> 1;
            }

            return res_;
        }

        template <typename Spec, class Allocator = std::allocator<typename Spec::sample>>
        std::vector<typename Spec::sample, Allocator> compound_radix2_twiddles_imag(std::size_t n_samples_per_operand) {
            std::size_t subtwiddle_len = 1;
            std::size_t extend_len = n_samples_per_operand;
            std::vector<typename Spec::sample, Allocator> res_;

            subtwiddle_len = subtwiddle_len << 1;
            extend_len = extend_len >> 1;

            while (extend_len > 0) {
                auto subtwiddle_ = subtwiddle_imag<Spec, Allocator>(subtwiddle_len, typename Spec::sample(1));
                subtwiddle_ = extend(subtwiddle_, extend_len);
                res_ = concatenate(res_, subtwiddle_);
                subtwiddle_len = subtwiddle_len << 1;
                extend_len = extend_len >> 1;
            }

            return res_;
        }
        
        template <typename Spec, class Allocator = std::allocator<typename Spec::sample>>
        std::vector<typename Spec::sample, Allocator> compound_radix4_twiddles_real(std::size_t n_samples_per_operand) {
            std::size_t subtwiddle_len = 1;
            std::size_t extend_len = n_samples_per_operand;
            std::vector<typename Spec::sample, Allocator> res_;

            subtwiddle_len = subtwiddle_len << 2;
            extend_len = extend_len >> 2;

            while (extend_len > 0) {
                auto subtwiddle_ = subtwiddle_real<Spec, Allocator>(subtwiddle_len, typename Spec::sample(1));
                subtwiddle_ = extend(subtwiddle_, extend_len);
                res_ = concatenate(res_, subtwiddle_);

                subtwiddle_ = subtwiddle_real<Spec, Allocator>(subtwiddle_len, typename Spec::sample(0.5));
                subtwiddle_ = extend(subtwiddle_, extend_len);
                res_ = concatenate(res_, subtwiddle_);

                subtwiddle_ = subtwiddle_real<Spec, Allocator>(subtwiddle_len, typename Spec::sample(1.5));
                subtwiddle_ = extend(subtwiddle_, extend_len);
                res_ = concatenate(res_, subtwiddle_);

                subtwiddle_len = subtwiddle_len << 2;
                extend_len = extend_len >> 2;
            }

            if (subtwiddle_len < (4 * n_samples_per_operand)) {
                auto subtwiddle_ = subtwiddle_real<Spec, Allocator>(subtwiddle_len, typename Spec::sample(1));
                res_ = concatenate(res_, subtwiddle_);
            }

            return res_;
        } 

        template <typename Spec, class Allocator = std::allocator<typename Spec::sample>>
        std::vector<typename Spec::sample, Allocator> compound_radix4_twiddles_imag(std::size_t n_samples_per_operand) {
            std::size_t subtwiddle_len = 1;
            std::size_t extend_len = n_samples_per_operand;
            std::vector<typename Spec::sample, Allocator> res_;

            subtwiddle_len = subtwiddle_len << 2;
            extend_len = extend_len >> 2;

            while (extend_len > 0) {
                auto subtwiddle_ = subtwiddle_imag<Spec, Allocator>(subtwiddle_len, typename Spec::sample(1));
                subtwiddle_ = extend(subtwiddle_, extend_len);
                res_ = concatenate(res_, subtwiddle_);

                subtwiddle_ = subtwiddle_imag<Spec, Allocator>(subtwiddle_len, typename Spec::sample(0.5));
                subtwiddle_ = extend(subtwiddle_, extend_len);
                res_ = concatenate(res_, subtwiddle_);

                subtwiddle_ = subtwiddle_imag<Spec, Allocator>(subtwiddle_len, typename Spec::sample(1.5));
                subtwiddle_ = extend(subtwiddle_, extend_len);
                res_ = concatenate(res_, subtwiddle_);

                subtwiddle_len = subtwiddle_len << 2;
                extend_len = extend_len >> 2;
            }

            if (subtwiddle_len < (4 * n_samples_per_operand)) {
                auto subtwiddle_ = subtwiddle_imag<Spec, Allocator>(subtwiddle_len, typename Spec::sample(1));
                res_ = concatenate(res_, subtwiddle_);
            }

            return res_;
        } 

        template <typename Spec, class Allocator = std::allocator<typename Spec::sample>>
        std::vector<std::vector<typename Spec::sample, Allocator>> radix2_twiddles_real(std::size_t subtwiddle_len){
            std::vector<std::vector<typename Spec::sample, Allocator>> res_;

            res_.push_back(subtwiddle_real<Spec, Allocator>(subtwiddle_len, typename Spec::sample(1)));

            return res_;
        }

        template <typename Spec, class Allocator = std::allocator<typename Spec::sample>>
        std::vector<std::vector<typename Spec::sample, Allocator>> radix2_twiddles_imag(std::size_t subtwiddle_len){
            std::vector<std::vector<typename Spec::sample, Allocator>> res_;

            res_.push_back(subtwiddle_imag<Spec, Allocator>(subtwiddle_len, typename Spec::sample(1)));

            return res_;
        }

        template <typename Spec, class Allocator = std::allocator<typename Spec::sample>>
        std::vector<std::vector<typename Spec::sample, Allocator>> radix4_twiddles_real(std::size_t subtwiddle_len){
            std::vector<std::vector<typename Spec::sample, Allocator>> res_;

            res_.push_back(subtwiddle_real<Spec, Allocator>(subtwiddle_len, typename Spec::sample(1)));
            res_.push_back(subtwiddle_real<Spec, Allocator>(subtwiddle_len, typename Spec::sample(0.5)));
            res_.push_back(subtwiddle_real<Spec, Allocator>(subtwiddle_len, typename Spec::sample(1.5)));

            return res_;
        }

        template <typename Spec, class Allocator = std::allocator<typename Spec::sample>>
        std::vector<std::vector<typename Spec::sample, Allocator>> radix4_twiddles_imag(std::size_t subtwiddle_len){
            std::vector<std::vector<typename Spec::sample, Allocator>> res_;

            res_.push_back(subtwiddle_imag<Spec, Allocator>(subtwiddle_len, typename Spec::sample(1)));
            res_.push_back(subtwiddle_imag<Spec, Allocator>(subtwiddle_len, typename Spec::sample(0.5)));
            res_.push_back(subtwiddle_imag<Spec, Allocator>(subtwiddle_len, typename Spec::sample(1.5)));

            return res_;
        }
    }
}

#endif