#ifndef AFFT_ITERATIVE_BUTTERFLY_PLAN_HPP
#define AFFT_ITERATIVE_BUTTERFLY_PLAN_HPP

#include <vector>
#include <cstddef>

#include "afft/radix/radix_stage.hpp"
#include "afft/radix/radix_type.hpp"
#include "afft/butterfly/twiddles.hpp"

namespace afft
{
    template <typename Spec, class Allocator = std::allocator<typename Spec::sample>>
    class IterativeButteflyPlan
    {
        std::size_t log_n_sample_per_operand_;
        std::vector<RadixStage> radix_stages_;
        std::vector<Spec::sample, Allocator> edge_twiddles_real_;
        std::vector<Spec::sample, Allocator> edge_twiddles_imag_;
        std::vector<std::vector<std::vector<Spec::sample, Allocator>>> twiddles_real_;
        std::vector<std::vector<std::vector<Spec::sample, Allocator>>> twiddles_imag_;
        namespace tw = afft::twiddles;

    public:
        explicit IterativeButteflyPlan(std::size_t transform_len, std::size_t max_n_samples_per_operand)
        {
            if (transform_len <= 1) {
                return;
            }

            log_n_sample_per_operand_ =
                std::min(
                    int_log_2(max_n_samples_per_operand),
                    std::size_t(int_log_2(transform_len) - 1));

            std::size_t n_samples_per_operand = 1 << log_n_sample_per_operand_;

            if (transform_len == n_samples_per_operand * 2) {
                edge_twiddles_real_ = tw::edge2_twiddles_real<Spec, Allocator>(n_samples_per_operand);
                edge_twiddles_imag_ = tw::edge2_twiddles_imag<Spec, Allocator>(n_samples_per_operand);

                RadixStage edge_radix_stage_;
                edge_radix_stage_.type = RadixType::edge2;
                edge_radix_stage_.tw_real_b = edge_twiddles_real_.data(); 
                edge_radix_stage_.tw_imag_b = edge_twiddles_imag_.data(); 
                radix_stages_.append(edge_radix_stage_);
                return;
            }

            // else transform_len >= n_samples_per_operand * 4
            edge_twiddles_real_ = tw::edge4_twiddles_real<Spec, Allocator>(n_samples_per_operand);
            edge_twiddles_imag_ = tw::edge4_twiddles_imag<Spec, Allocator>(n_samples_per_operand);

            RadixStage edge_radix_stage_;
            edge_radix_stage_.type = RadixType::edge2;
            edge_radix_stage_.tw_real_b = edge_twiddles_real_.data(); 
            edge_radix_stage_.tw_imag_b = edge_twiddles_imag_.data(); 
            edge_radix_stage_.subfft_id_start = 0;
            edge_radix_stage_.subfft_id_end = transform_len / n_samples_per_operand / 4;
            radix_stages_.append(edge_radix_stage_);

            subtwiddle_len = n_samples_per_operand * 4;
            
            while (subtwiddle_len < transform_len) {
                
                if (2 * subtwiddle_len == transform_len) {
                    RadixStage radix_stage_;
                    auto stage_twiddles_real = tw::radix2_twiddles_real<Spec, Allocator>(subtwiddle_len);
                    auto stage_twiddles_imag = tw::radix2_twiddles_imag<Spec, Allocator>(subtwiddle_len);
                    twiddles_real_.append(stage_twiddles_real);
                    twiddles_imag_.append(stage_twiddles_imag);


                    radix_stage_.type = RadixType::radix2;
                    radix_stage_.tw_real_b = twiddles_real_.back()[0].data(); 
                    radix_stage_.tw_imag_b = twiddles_imag_.back()[0].data(); 
                    radix_stage_.subtwiddle_len = subtwiddle_len;
                    radix_stage_.subtwiddle_start = 0;
                    radix_stage_.subtwiddle_end = subtwiddle_len;

                    radix_stages_.append(radix_stage_);
                    break;
                }
                else{
                    RadixStage radix_stage_;
                    auto stage_twiddles_real = tw::radix4_twiddles_real<Spec, Allocator>(subtwiddle_len);
                    auto stage_twiddles_imag = tw::radix4_twiddles_imag<Spec, Allocator>(subtwiddle_len);
                    twiddles_real_.append(stage_twiddles_real);
                    twiddles_imag_.append(stage_twiddles_imag);

                    radix_stage_.type = RadixType::radix4;
                    radix_stage_.tw_real_b = twiddles_real_.back()[0].data(); 
                    radix_stage_.tw_imag_b = twiddles_imag_.back()[0].data(); 
                    radix_stage_.tw_real_c = twiddles_real_.back()[1].data(); 
                    radix_stage_.tw_imag_c = twiddles_imag_.back()[1].data(); 
                    radix_stage_.tw_real_d = twiddles_real_.back()[2].data(); 
                    radix_stage_.tw_imag_d = twiddles_imag_.back()[2].data(); 
                    radix_stage_.subfft_id_start = 0;
                    radix_stage_.subfft_id_end = transform_len / subtwiddle_len / 4;
                    radix_stage_.subtwiddle_len = subtwiddle_len;
                    radix_stage_.subtwiddle_start = 0;
                    radix_stage_.subtwiddle_end = subtwiddle_len;

                    radix_stages_.append(radix_stage_);
                    subtwiddle_len *= 4;
                }
            }
        }
    };
}

#endif