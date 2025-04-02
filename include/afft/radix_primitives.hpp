
#include "afft/radix_primitives/radix4_ditime_regular_core.hpp"
#include "afft/radix_primitives/radix4_difreq_regular_core.hpp"

template<typename Sample, typename OperandSpec>
inline void base_radix_2_fma(Sample* real, Sample* imag, const Sample* tw_real, const Sample* tw_imag, std::size_t transform_len) {
    using Operand = typename OperandSpec::Value;
    std::size_t n_samples_per_operand = sizeof(Operand) / sizeof(Sample);
    std::size_t n_radixes = transform_len / 2 / n_samples_per_operand;
    auto two_n_samples_per_operand = 2 * n_samples_per_operand;
    std::size_t a_offset = 0;
    std::size_t b_offset = n_samples_per_operand;
    std::size_t tw_offset = 0;
    
    // DECLARE
    Operand in_real_a_op;
    Operand in_real_b_op;
    Operand in_imag_a_op;
    Operand in_imag_b_op;
    Operand out_real_a_op;
    Operand out_real_b_op;
    Operand out_imag_a_op;
    Operand out_imag_b_op;
    Operand tw_real_b_op;
    Operand tw_imag_b_op;
    Operand two(2);

    // OFFSET
    auto real_a = real + a_offset;
    auto imag_a = imag + a_offset;
    auto real_b = real + b_offset;
    auto imag_b = imag + b_offset;
    auto tw_real_b = tw_real + tw_offset;
    auto tw_imag_b = tw_imag + tw_offset;
    
    for(std::size_t radix_id = 0; radix_id < n_radixes; radix_id++) {

        //LOAD
        OperandSpec::load(real_a, in_real_a_op);
        OperandSpec::load(imag_a, in_imag_a_op);
        OperandSpec::load(real_b, in_real_b_op);
        OperandSpec::load(imag_b, in_imag_b_op);
        OperandSpec::load(tw_real_b, tw_real_b_op);
        OperandSpec::load(tw_imag_b, tw_imag_b_op);

        // COMPUTE
        out_real_a_op = OperandSpec::fnma(tw_imag_b_op, in_imag_b_op, OperandSpec::fma(tw_real_b_op, in_real_b_op, in_real_a_op));
        out_imag_a_op = OperandSpec::fma(tw_imag_b_op, in_real_b_op, OperandSpec::fma(tw_real_b_op, in_imag_b_op, in_imag_a_op));
        out_real_b_op = OperandSpec::fms(two, in_real_a_op, out_real_a_op);
        out_imag_b_op = OperandSpec::fms(two, in_imag_a_op, out_imag_a_op);

        // STORE
        OperandSpec::store(real_a, out_real_a_op);
        OperandSpec::store(imag_a, out_imag_a_op);
        OperandSpec::store(real_b, out_real_b_op);
        OperandSpec::store(imag_b, out_imag_b_op);

        // UPDATE OFFSET
        real_a += two_n_samples_per_operand;
        imag_a += two_n_samples_per_operand;
        real_b += two_n_samples_per_operand;
        imag_b += two_n_samples_per_operand;
        tw_real_b += n_samples_per_operand;
        tw_imag_b += n_samples_per_operand;
    }
}

template<typename Sample, typename OperandSpec>
inline void base_radix_2(Sample* real, Sample* imag, const Sample* tw_real, const Sample* tw_imag, std::size_t transform_len) {
    using Operand = typename OperandSpec::Value;
    std::size_t n_samples_per_operand = sizeof(Operand) / sizeof(Sample);
    std::size_t n_radixes = transform_len / 2 / n_samples_per_operand;
    auto two_n_samples_per_operand = 2 * n_samples_per_operand;
    std::size_t a_offset = 0;
    std::size_t b_offset = n_samples_per_operand;
    std::size_t tw_offset = 0;
    Operand two(2);

    // DECLARE
    Operand inout_real_a_op;
    Operand inout_real_b_op;
    Operand inout_imag_a_op;
    Operand inout_imag_b_op;
    Operand tw_real_b_op;
    Operand tw_imag_b_op;
    Operand post_tw_real_b_op;
    Operand post_tw_imag_b_op;

    for(std::size_t radix_id = 0; radix_id < n_radixes; radix_id++) {

        // OFFSET
        auto real_a = real + a_offset;
        auto imag_a = imag + a_offset;
        auto real_b = real + b_offset;
        auto imag_b = imag + b_offset;
        auto tw_real_b = tw_real + tw_offset;
        auto tw_imag_b = tw_imag + tw_offset;

        //LOAD
        OperandSpec::load(real_a, inout_real_a_op);
        OperandSpec::load(imag_a, inout_imag_a_op);
        OperandSpec::load(real_b, inout_real_b_op);
        OperandSpec::load(imag_b, inout_imag_b_op);
        OperandSpec::load(tw_real_b, tw_real_b_op);
        OperandSpec::load(tw_imag_b, tw_imag_b_op);

        // COMPUTE
        post_tw_real_b_op = tw_real_b_op * inout_real_b_op - tw_imag_b_op * inout_imag_b_op;
        post_tw_imag_b_op = tw_real_b_op * inout_imag_b_op + tw_imag_b_op * inout_real_b_op;

        inout_real_b_op = inout_real_a_op;
        inout_imag_b_op = inout_imag_a_op;

        inout_real_a_op += post_tw_real_b_op;
        inout_imag_a_op += post_tw_imag_b_op;
        inout_real_b_op -= post_tw_real_b_op;
        inout_imag_b_op -= post_tw_imag_b_op;

        // STORE
        OperandSpec::store(real_a, inout_real_a_op);
        OperandSpec::store(imag_a, inout_imag_a_op);
        OperandSpec::store(real_b, inout_real_b_op);
        OperandSpec::store(imag_b, inout_imag_b_op);

        // UPDATE OFFSET
        a_offset += two_n_samples_per_operand;
        b_offset += two_n_samples_per_operand;
        tw_offset += n_samples_per_operand;
    }
}

template<typename Sample, typename OperandSpec>
inline void base_radix_4_fma(Sample* real, Sample* imag, const Sample* tw_real, const Sample* tw_imag, std::size_t transform_len) {
    using Operand = typename OperandSpec::Value;
    std::size_t n_samples_per_operand = sizeof(Operand) / sizeof(Sample);
    std::size_t n_radixes = transform_len / 4 / n_samples_per_operand;
    auto four_n_samples_per_operand = 4 * n_samples_per_operand;
    std::size_t a_offset = 0;
    std::size_t b_offset = n_samples_per_operand;
    std::size_t c_offset = 2 * n_samples_per_operand;
    std::size_t d_offset = 3 * n_samples_per_operand;
    std::size_t tw_b_offset = 0;
    std::size_t tw_d_offset = n_samples_per_operand;
    std::size_t tw_bb_offset = 2 * n_samples_per_operand;
    std::size_t tw_dd_offset = 3 * n_samples_per_operand;
    
    // DECLARE
    Operand two(2);
    Operand inout_real_a_op;
    Operand inout_real_b_op;
    Operand inout_imag_a_op;
    Operand inout_imag_b_op;
    Operand inter_real_a_op;
    Operand inter_real_b_op;
    Operand inter_imag_a_op;
    Operand inter_imag_b_op;
    Operand tw_real_b_op;
    Operand tw_imag_b_op;

    Operand inout_real_c_op;
    Operand inout_real_d_op;
    Operand inout_imag_c_op;
    Operand inout_imag_d_op;
    Operand inter_real_c_op;
    Operand inter_real_d_op;
    Operand inter_imag_c_op;
    Operand inter_imag_d_op;
    Operand tw_real_d_op;
    Operand tw_imag_d_op;

    Operand tw_real_bb_op;
    Operand tw_imag_bb_op;
    Operand tw_real_dd_op;
    Operand tw_imag_dd_op;

    
    for(std::size_t radix_id = 0; radix_id < n_radixes; radix_id++) {
        // OFFSET
        auto real_a = real + a_offset;
        auto imag_a = imag + a_offset;
        auto real_b = real + b_offset;
        auto imag_b = imag + b_offset;

        auto real_c = real + c_offset;
        auto imag_c = imag + c_offset;
        auto real_d = real + d_offset;
        auto imag_d = imag + d_offset;

        auto tw_real_b = tw_real + tw_b_offset;
        auto tw_imag_b = tw_imag + tw_b_offset;

        auto tw_real_d = tw_real + tw_d_offset;
        auto tw_imag_d = tw_imag + tw_d_offset;

        auto tw_real_bb = tw_real + tw_bb_offset;
        auto tw_imag_bb = tw_imag + tw_bb_offset;

        auto tw_real_dd = tw_real + tw_dd_offset;
        auto tw_imag_dd = tw_imag + tw_dd_offset;

        //LOAD
        OperandSpec::load(real_a, inout_real_a_op);
        OperandSpec::load(imag_a, inout_imag_a_op);
        OperandSpec::load(real_b, inout_real_b_op);
        OperandSpec::load(imag_b, inout_imag_b_op);
        OperandSpec::load(real_c, inout_real_c_op);
        OperandSpec::load(imag_c, inout_imag_c_op);
        OperandSpec::load(real_d, inout_real_d_op);
        OperandSpec::load(imag_d, inout_imag_d_op);

        OperandSpec::load(tw_real_b, tw_real_b_op);
        OperandSpec::load(tw_imag_b, tw_imag_b_op);
        OperandSpec::load(tw_real_d, tw_real_d_op);
        OperandSpec::load(tw_imag_d, tw_imag_d_op);
        OperandSpec::load(tw_real_bb, tw_real_bb_op);
        OperandSpec::load(tw_imag_bb, tw_imag_bb_op);
        OperandSpec::load(tw_real_dd, tw_real_dd_op);
        OperandSpec::load(tw_imag_dd, tw_imag_dd_op);

        // COMPUTE
        inter_real_a_op = OperandSpec::fnma(tw_imag_b_op, inout_imag_b_op, OperandSpec::fma(tw_real_b_op, inout_real_b_op, inout_real_a_op));
        inter_imag_a_op = OperandSpec::fma(tw_imag_b_op, inout_real_b_op, OperandSpec::fma(tw_real_b_op,inout_imag_b_op, inout_imag_a_op));
        inter_real_b_op = OperandSpec::fms(two, inout_real_a_op, inter_real_a_op);
        inter_imag_b_op = OperandSpec::fms(two, inout_imag_a_op, inter_imag_a_op);

        inter_real_c_op = OperandSpec::fnma(tw_imag_b_op, inout_imag_d_op, OperandSpec::fma(tw_real_b_op, inout_real_d_op, inout_real_c_op));
        inter_imag_c_op = OperandSpec::fma(tw_imag_b_op, inout_real_d_op, OperandSpec::fma(tw_real_b_op,inout_imag_d_op, inout_imag_c_op));
        inter_real_d_op = OperandSpec::fms(two, inout_real_c_op, inter_real_c_op);
        inter_imag_d_op = OperandSpec::fms(two, inout_imag_c_op, inter_imag_c_op);

        inout_real_a_op = OperandSpec::fnma(tw_imag_bb_op, inter_imag_c_op, OperandSpec::fma(tw_real_bb_op, inter_real_c_op, inter_real_a_op));
        inout_imag_a_op = OperandSpec::fma(tw_imag_bb_op, inter_real_c_op, OperandSpec::fma(tw_real_bb_op,inter_imag_c_op, inter_imag_a_op));
        inout_real_c_op = OperandSpec::fms(two, inter_real_a_op, inout_real_a_op);
        inout_imag_c_op = OperandSpec::fms(two, inter_imag_a_op, inout_imag_a_op);

        inout_real_b_op = OperandSpec::fnma(tw_imag_dd_op, inter_imag_d_op, OperandSpec::fma(tw_real_dd_op, inter_real_d_op, inter_real_b_op));
        inout_imag_b_op = OperandSpec::fma(tw_imag_dd_op, inter_real_d_op, OperandSpec::fma(tw_real_dd_op,inter_imag_d_op, inter_imag_b_op));
        inout_real_d_op = OperandSpec::fms(two, inter_real_b_op, inout_real_b_op);
        inout_imag_d_op = OperandSpec::fms(two, inter_imag_b_op, inout_imag_b_op);

        // STORE
        OperandSpec::store(real_a, inout_real_a_op);
        OperandSpec::store(imag_a, inout_imag_a_op);
        OperandSpec::store(real_b, inout_real_b_op);
        OperandSpec::store(imag_b, inout_imag_b_op);
        OperandSpec::store(real_c, inout_real_c_op);
        OperandSpec::store(imag_c, inout_imag_c_op);
        OperandSpec::store(real_d, inout_real_d_op);
        OperandSpec::store(imag_d, inout_imag_d_op);

        // UPDATE OFFSET
        a_offset += four_n_samples_per_operand;
        b_offset += four_n_samples_per_operand;
        c_offset += four_n_samples_per_operand;
        d_offset += four_n_samples_per_operand;
        tw_b_offset += four_n_samples_per_operand;
        tw_d_offset += four_n_samples_per_operand;
        tw_bb_offset += four_n_samples_per_operand;
        tw_dd_offset += four_n_samples_per_operand;
    }
}

template<typename Sample, typename OperandSpec>
inline void base_radix_4(Sample* real, Sample* imag, const Sample* tw_real, const Sample* tw_imag, std::size_t transform_len) {

    using Operand = typename OperandSpec::Value;
    std::size_t n_samples_per_operand = sizeof(Operand) / sizeof(Sample);
    std::size_t n_radixes = transform_len / 4 / n_samples_per_operand;
    auto four_n_samples_per_operand = 4 * n_samples_per_operand;
    auto six_n_samples_per_operand = 6 * n_samples_per_operand;
    std::size_t a_offset = 0;
    std::size_t b_offset = n_samples_per_operand;
    std::size_t c_offset = 2 * n_samples_per_operand;
    std::size_t d_offset = 3 * n_samples_per_operand;
    std::size_t tw_b_offset = 0;
    std::size_t tw_c_offset = n_samples_per_operand;
    std::size_t tw_d_offset = 2 * n_samples_per_operand;
    
    // // DECLARE
    Operand in_real_a_op;
    Operand in_real_b_op;
    Operand in_imag_a_op;
    Operand in_imag_b_op;
    Operand out_real_a_op;
    Operand out_real_b_op;
    Operand out_imag_a_op;
    Operand out_imag_b_op;

    Operand in_real_c_op;
    Operand in_real_d_op;
    Operand in_imag_c_op;
    Operand in_imag_d_op;
    Operand out_real_c_op;
    Operand out_real_d_op;
    Operand out_imag_c_op;
    Operand out_imag_d_op;
    
    Operand tw_real_b_op;
    Operand tw_imag_b_op;
    Operand tw_real_c_op;
    Operand tw_imag_c_op;
    Operand tw_real_d_op;
    Operand tw_imag_d_op;

    Operand post_tw_real_b_op;
    Operand post_tw_imag_b_op;
    Operand post_tw_real_c_op;
    Operand post_tw_imag_c_op;
    Operand post_tw_real_d_op;
    Operand post_tw_imag_d_op;

    for(std::size_t radix_id = 0; radix_id < n_radixes; radix_id++) {
        // OFFSET
        auto real_a = real + a_offset;
        auto imag_a = imag + a_offset;
        auto real_b = real + b_offset;
        auto imag_b = imag + b_offset;

        auto real_c = real + c_offset;
        auto imag_c = imag + c_offset;
        auto real_d = real + d_offset;
        auto imag_d = imag + d_offset;

        auto tw_real_b = tw_real + tw_b_offset;
        auto tw_imag_b = tw_imag + tw_b_offset;

        auto tw_real_c = tw_real + tw_c_offset;
        auto tw_imag_c = tw_imag + tw_c_offset;

        auto tw_real_d = tw_real + tw_d_offset;
        auto tw_imag_d = tw_imag + tw_d_offset;

        //LOAD
        OperandSpec::load(real_a, in_real_a_op);
        OperandSpec::load(imag_a, in_imag_a_op);
        OperandSpec::load(real_b, in_real_b_op);
        OperandSpec::load(imag_b, in_imag_b_op);
        OperandSpec::load(real_c, in_real_c_op);
        OperandSpec::load(imag_c, in_imag_c_op);
        OperandSpec::load(real_d, in_real_d_op);
        OperandSpec::load(imag_d, in_imag_d_op);

        OperandSpec::load(tw_real_b, tw_real_b_op);
        OperandSpec::load(tw_imag_b, tw_imag_b_op);
        OperandSpec::load(tw_real_c, tw_real_c_op);
        OperandSpec::load(tw_imag_c, tw_imag_c_op);
        OperandSpec::load(tw_real_d, tw_real_d_op);
        OperandSpec::load(tw_imag_d, tw_imag_d_op);

        // COMPUTE
        post_tw_real_b_op = tw_real_b_op * in_real_b_op - tw_imag_b_op * in_imag_b_op;
        post_tw_imag_b_op = tw_real_b_op * in_imag_b_op + tw_imag_b_op * in_real_b_op;
        post_tw_real_c_op = tw_real_c_op * in_real_c_op - tw_imag_c_op * in_imag_c_op;
        post_tw_imag_c_op = tw_real_c_op * in_imag_c_op + tw_imag_c_op * in_real_c_op;
        post_tw_real_d_op = tw_real_d_op * in_real_d_op - tw_imag_d_op * in_imag_d_op;
        post_tw_imag_d_op = tw_real_d_op * in_imag_d_op + tw_imag_d_op * in_real_d_op;

        // post_tw_real_b_op =  in_real_b_op;
        // post_tw_imag_b_op =  in_imag_b_op;
        // post_tw_real_c_op =  in_real_c_op;
        // post_tw_imag_c_op =  in_imag_c_op;
        // post_tw_real_d_op =  in_real_d_op;
        // post_tw_imag_d_op =  in_imag_d_op;

        in_real_b_op = in_real_a_op;
        in_imag_b_op = in_imag_a_op;
        in_real_c_op = post_tw_real_c_op;
        in_imag_c_op = post_tw_imag_c_op;
        in_real_d_op = post_tw_real_c_op;
        in_imag_d_op = post_tw_imag_c_op;

        in_real_a_op += post_tw_real_b_op;
        in_imag_a_op += post_tw_imag_b_op;
        in_real_b_op -= post_tw_real_b_op;
        in_imag_b_op -= post_tw_imag_b_op;

        in_real_c_op += post_tw_real_d_op;
        in_imag_c_op += post_tw_imag_d_op;
        in_real_d_op -= post_tw_real_d_op;
        in_imag_d_op -= post_tw_imag_d_op;

        out_real_a_op = in_real_a_op + in_real_c_op;
        out_real_b_op = in_real_b_op + in_imag_d_op;
        out_real_c_op = in_real_a_op - in_real_c_op;
        out_real_d_op = in_real_b_op - in_imag_d_op;
        
        out_imag_a_op = in_imag_a_op + in_imag_c_op;
        out_imag_b_op = in_imag_b_op - in_real_d_op;
        out_imag_c_op = in_imag_a_op - in_imag_c_op;
        out_imag_d_op = in_imag_b_op + in_real_d_op; 

        // STORE
        OperandSpec::store(real_a, out_real_a_op);
        OperandSpec::store(imag_a, out_imag_a_op);
        OperandSpec::store(real_b, out_real_b_op);
        OperandSpec::store(imag_b, out_imag_b_op);
        OperandSpec::store(real_c, out_real_c_op);
        OperandSpec::store(imag_c, out_imag_c_op);
        OperandSpec::store(real_d, out_real_d_op);
        OperandSpec::store(imag_d, out_imag_d_op);

        // UPDATE OFFSET
        a_offset += four_n_samples_per_operand;
        b_offset += four_n_samples_per_operand;
        c_offset += four_n_samples_per_operand;
        d_offset += four_n_samples_per_operand;
        tw_b_offset += six_n_samples_per_operand;
        tw_c_offset += six_n_samples_per_operand;
        tw_d_offset += six_n_samples_per_operand;
    }
}

template<typename Sample, typename OperandSpec>
inline void base_radix_8_fma(Sample* real, Sample* imag, const Sample* tw_real, const Sample* tw_imag, std::size_t transform_len) {
    using Operand = typename OperandSpec::Value;
    std::size_t n_samples_per_operand = sizeof(Operand) / sizeof(Sample);
    std::size_t n_radixes = transform_len / 8 / n_samples_per_operand;
    auto eight_n_samples_per_operand = 8 * n_samples_per_operand;
    auto six_n_samples_per_operand = 6 * n_samples_per_operand;
    std::size_t a_offset = 0 * n_samples_per_operand;
    std::size_t b_offset = 1 * n_samples_per_operand;
    std::size_t c_offset = 2 * n_samples_per_operand;
    std::size_t d_offset = 3 * n_samples_per_operand;
    std::size_t e_offset = 4 * n_samples_per_operand;
    std::size_t f_offset = 5 * n_samples_per_operand;
    std::size_t g_offset = 6 * n_samples_per_operand;
    std::size_t h_offset = 7 * n_samples_per_operand;

    std::size_t tw_a0_offset = 0 * n_samples_per_operand;
    std::size_t tw_a1_offset = 1 * n_samples_per_operand;
    std::size_t tw_a2_offset = 2 * n_samples_per_operand;
    std::size_t tw_b0_offset = 3 * n_samples_per_operand;
    std::size_t tw_b1_offset = 4 * n_samples_per_operand;
    std::size_t tw_b2_offset = 5 * n_samples_per_operand;
    std::size_t tw_c0_offset = 6 * n_samples_per_operand;
    std::size_t tw_c1_offset = 7 * n_samples_per_operand;
    std::size_t tw_c2_offset = 8 * n_samples_per_operand;
    std::size_t tw_d0_offset = 9 * n_samples_per_operand;
    std::size_t tw_d1_offset = 10 * n_samples_per_operand;
    std::size_t tw_d2_offset = 11 * n_samples_per_operand;

    // DECLARE
    Operand two(2);

    Operand in_real_a_op;
    Operand out_real_a_op;
    Operand in_imag_a_op;
    Operand out_imag_a_op;
    Operand in_real_b_op;
    Operand out_real_b_op;
    Operand in_imag_b_op;
    Operand out_imag_b_op;
    Operand in_real_c_op;
    Operand out_real_c_op;
    Operand in_imag_c_op;
    Operand out_imag_c_op;
    Operand in_real_d_op;
    Operand out_real_d_op;
    Operand in_imag_d_op;
    Operand out_imag_d_op;
    Operand in_real_e_op;
    Operand out_real_e_op;
    Operand in_imag_e_op;
    Operand out_imag_e_op;
    Operand in_real_f_op;
    Operand out_real_f_op;
    Operand in_imag_f_op;
    Operand out_imag_f_op;
    Operand in_real_g_op;
    Operand out_real_g_op;
    Operand in_imag_g_op;
    Operand out_imag_g_op;
    Operand in_real_h_op;
    Operand out_real_h_op;
    Operand in_imag_h_op;
    Operand out_imag_h_op;

    Operand tw_imag_a0_op;
    Operand tw_imag_a1_op;
    Operand tw_imag_a2_op;
    Operand tw_imag_b0_op;
    Operand tw_imag_b1_op;
    Operand tw_imag_b2_op;
    Operand tw_imag_c0_op;
    Operand tw_imag_c1_op;
    Operand tw_imag_c2_op;
    Operand tw_imag_d0_op;
    Operand tw_imag_d1_op;
    Operand tw_imag_d2_op;
    Operand tw_real_a0_op;
    Operand tw_real_a1_op;
    Operand tw_real_a2_op;
    Operand tw_real_b0_op;
    Operand tw_real_b1_op;
    Operand tw_real_b2_op;
    Operand tw_real_c0_op;
    Operand tw_real_c1_op;
    Operand tw_real_c2_op;
    Operand tw_real_d0_op;
    Operand tw_real_d1_op;
    Operand tw_real_d2_op;
    
    for(std::size_t radix_id = 0; radix_id < n_radixes; radix_id++) {
        // OFFSET
        auto imag_a = imag + a_offset;
        auto imag_b = imag + b_offset;
        auto imag_c = imag + c_offset;
        auto imag_d = imag + d_offset;
        auto imag_e = imag + e_offset;
        auto imag_f = imag + f_offset;
        auto imag_g = imag + g_offset;
        auto imag_h = imag + h_offset;
        auto real_a = real + a_offset;
        auto real_b = real + b_offset;
        auto real_c = real + c_offset;
        auto real_d = real + d_offset;
        auto real_e = real + e_offset;
        auto real_f = real + f_offset;
        auto real_g = real + g_offset;
        auto real_h = real + h_offset;

        auto tw_imag_a0 = tw_real + tw_a0_offset;
        auto tw_imag_a1 = tw_real + tw_a1_offset;
        auto tw_imag_a2 = tw_real + tw_a2_offset;
        auto tw_imag_b0 = tw_real + tw_b0_offset;
        auto tw_imag_b1 = tw_real + tw_b1_offset;
        auto tw_imag_b2 = tw_real + tw_b2_offset;
        auto tw_imag_c0 = tw_real + tw_c0_offset;
        auto tw_imag_c1 = tw_real + tw_c1_offset;
        auto tw_imag_c2 = tw_real + tw_c2_offset;
        auto tw_imag_d0 = tw_real + tw_d0_offset;
        auto tw_imag_d1 = tw_real + tw_d1_offset;
        auto tw_imag_d2 = tw_real + tw_d2_offset;
        auto tw_real_a0 = tw_real + tw_a0_offset;
        auto tw_real_a1 = tw_real + tw_a1_offset;
        auto tw_real_a2 = tw_real + tw_a2_offset;
        auto tw_real_b0 = tw_real + tw_b0_offset;
        auto tw_real_b1 = tw_real + tw_b1_offset;
        auto tw_real_b2 = tw_real + tw_b2_offset;
        auto tw_real_c0 = tw_real + tw_c0_offset;
        auto tw_real_c1 = tw_real + tw_c1_offset;
        auto tw_real_c2 = tw_real + tw_c2_offset;
        auto tw_real_d0 = tw_real + tw_d0_offset;
        auto tw_real_d1 = tw_real + tw_d1_offset;
        auto tw_real_d2 = tw_real + tw_d2_offset;

        //LOAD
        OperandSpec::load(imag_a, in_imag_a_op);
        OperandSpec::load(imag_b, in_imag_b_op);
        OperandSpec::load(imag_c, in_imag_c_op);
        OperandSpec::load(imag_d, in_imag_d_op);
        OperandSpec::load(imag_e, in_imag_e_op);
        OperandSpec::load(imag_f, in_imag_f_op);
        OperandSpec::load(imag_g, in_imag_g_op);
        OperandSpec::load(imag_h, in_imag_h_op);
        OperandSpec::load(real_a, in_real_a_op);
        OperandSpec::load(real_b, in_real_b_op);
        OperandSpec::load(real_c, in_real_c_op);
        OperandSpec::load(real_d, in_real_d_op);
        OperandSpec::load(real_e, in_real_e_op);
        OperandSpec::load(real_f, in_real_f_op);
        OperandSpec::load(real_g, in_real_g_op);
        OperandSpec::load(real_h, in_real_h_op);

        OperandSpec::load(tw_imag_a0, tw_imag_a0_op);
        OperandSpec::load(tw_imag_a1, tw_imag_a1_op);
        OperandSpec::load(tw_imag_a2, tw_imag_a2_op);
        OperandSpec::load(tw_imag_b0, tw_imag_b0_op);
        OperandSpec::load(tw_imag_b1, tw_imag_b1_op);
        OperandSpec::load(tw_imag_b2, tw_imag_b2_op);
        OperandSpec::load(tw_imag_c0, tw_imag_c0_op);
        OperandSpec::load(tw_imag_c1, tw_imag_c1_op);
        OperandSpec::load(tw_imag_c2, tw_imag_c2_op);
        OperandSpec::load(tw_imag_d0, tw_imag_d0_op);
        OperandSpec::load(tw_imag_d1, tw_imag_d1_op);
        OperandSpec::load(tw_imag_d2, tw_imag_d2_op);
        OperandSpec::load(tw_real_a0, tw_real_a0_op);
        OperandSpec::load(tw_real_a1, tw_real_a1_op);
        OperandSpec::load(tw_real_a2, tw_real_a2_op);
        OperandSpec::load(tw_real_b0, tw_real_b0_op);
        OperandSpec::load(tw_real_b1, tw_real_b1_op);
        OperandSpec::load(tw_real_b2, tw_real_b2_op);
        OperandSpec::load(tw_real_c0, tw_real_c0_op);
        OperandSpec::load(tw_real_c1, tw_real_c1_op);
        OperandSpec::load(tw_real_c2, tw_real_c2_op);
        OperandSpec::load(tw_real_d0, tw_real_d0_op);
        OperandSpec::load(tw_real_d1, tw_real_d1_op);
        OperandSpec::load(tw_real_d2, tw_real_d2_op);

        // COMPUTE
        out_real_a_op = OperandSpec::fnma(tw_imag_a0_op, in_imag_b_op, OperandSpec::fma(tw_real_a0_op, in_real_b_op, in_real_a_op));
        out_imag_a_op = OperandSpec::fma(tw_imag_a0_op, in_real_b_op, OperandSpec::fma(tw_real_a0_op,in_imag_b_op, in_imag_a_op));
        out_real_b_op = OperandSpec::fms(two, in_real_a_op, out_real_a_op);
        out_imag_b_op = OperandSpec::fms(two, in_imag_a_op, out_imag_a_op);

        out_real_c_op = OperandSpec::fnma(tw_imag_b0_op, in_imag_d_op, OperandSpec::fma(tw_real_b0_op, in_real_d_op, in_real_c_op));
        out_imag_c_op = OperandSpec::fma(tw_imag_b0_op, in_real_d_op, OperandSpec::fma(tw_real_b0_op,in_imag_d_op, in_imag_c_op));
        out_real_d_op = OperandSpec::fms(two, in_real_c_op, out_real_c_op);
        out_imag_d_op = OperandSpec::fms(two, in_imag_c_op, out_imag_c_op);

        out_real_e_op = OperandSpec::fnma(tw_imag_c0_op, in_imag_f_op, OperandSpec::fma(tw_real_c0_op, in_real_f_op, in_real_e_op));
        out_imag_e_op = OperandSpec::fma(tw_imag_c0_op, in_real_f_op, OperandSpec::fma(tw_real_c0_op,in_imag_f_op, in_imag_e_op));
        out_real_f_op = OperandSpec::fms(two, in_real_e_op, out_real_e_op);
        out_imag_f_op = OperandSpec::fms(two, in_imag_e_op, out_imag_e_op);

        out_real_g_op = OperandSpec::fnma(tw_imag_d0_op, in_imag_h_op, OperandSpec::fma(tw_real_d0_op, in_real_h_op, in_real_g_op));
        out_imag_g_op = OperandSpec::fma(tw_imag_d0_op, in_real_h_op, OperandSpec::fma(tw_real_d0_op,in_imag_h_op, in_imag_g_op));
        out_real_h_op = OperandSpec::fms(two, in_real_g_op, out_real_g_op);
        out_imag_h_op = OperandSpec::fms(two, in_imag_g_op, out_imag_g_op);

        in_real_a_op = OperandSpec::fnma(tw_imag_a1_op, out_imag_c_op, OperandSpec::fma(tw_real_a1_op, out_real_c_op, out_real_a_op));
        in_imag_a_op = OperandSpec::fma(tw_imag_a1_op, out_real_c_op, OperandSpec::fma(tw_real_a1_op,out_imag_c_op, out_imag_a_op));
        in_real_c_op = OperandSpec::fms(two, out_real_a_op, in_real_a_op);
        in_imag_c_op = OperandSpec::fms(two, out_imag_a_op, in_imag_a_op);

        in_real_e_op = OperandSpec::fnma(tw_imag_b1_op, out_imag_g_op, OperandSpec::fma(tw_real_b1_op, out_real_g_op, out_real_e_op));
        in_imag_e_op = OperandSpec::fma(tw_imag_b1_op, out_real_g_op, OperandSpec::fma(tw_real_b1_op,out_imag_g_op, out_imag_e_op));
        in_real_g_op = OperandSpec::fms(two, out_real_e_op, in_real_e_op);
        in_imag_g_op = OperandSpec::fms(two, out_imag_e_op, in_imag_e_op);

        in_real_b_op = OperandSpec::fnma(tw_imag_c1_op, out_imag_d_op, OperandSpec::fma(tw_real_c1_op, out_real_d_op, out_real_b_op));
        in_imag_b_op = OperandSpec::fma(tw_imag_c1_op, out_real_d_op, OperandSpec::fma(tw_real_c1_op,out_imag_d_op, out_imag_b_op));
        in_real_d_op = OperandSpec::fms(two, out_real_b_op, in_real_b_op);
        in_imag_d_op = OperandSpec::fms(two, out_imag_b_op, in_imag_b_op);

        in_real_f_op = OperandSpec::fnma(tw_imag_d1_op, out_imag_h_op, OperandSpec::fma(tw_real_d1_op, out_real_h_op, out_real_f_op));
        in_imag_f_op = OperandSpec::fma(tw_imag_d1_op, out_real_h_op, OperandSpec::fma(tw_real_d1_op,out_imag_h_op, out_imag_f_op));
        in_real_h_op = OperandSpec::fms(two, out_real_f_op, in_real_f_op);
        in_imag_h_op = OperandSpec::fms(two, out_imag_f_op, in_imag_f_op);

        out_real_a_op = OperandSpec::fnma(tw_imag_a2_op, in_imag_e_op, OperandSpec::fma(tw_real_a2_op, in_real_e_op, in_real_a_op));
        out_imag_a_op = OperandSpec::fma(tw_imag_a2_op, in_real_e_op, OperandSpec::fma(tw_real_a2_op,in_imag_e_op, in_imag_a_op));
        out_real_e_op = OperandSpec::fms(two, in_real_a_op, out_real_a_op);
        out_imag_e_op = OperandSpec::fms(two, in_imag_a_op, out_imag_a_op);

        out_real_b_op = OperandSpec::fnma(tw_imag_b2_op, in_imag_f_op, OperandSpec::fma(tw_real_b2_op, in_real_f_op, in_real_b_op));
        out_imag_b_op = OperandSpec::fma(tw_imag_b2_op, in_real_f_op, OperandSpec::fma(tw_real_b2_op,in_imag_f_op, in_imag_b_op));
        out_real_f_op = OperandSpec::fms(two, in_real_b_op, out_real_b_op);
        out_imag_f_op = OperandSpec::fms(two, in_imag_b_op, out_imag_b_op);

        out_real_c_op = OperandSpec::fnma(tw_imag_c2_op, in_imag_g_op, OperandSpec::fma(tw_real_c2_op, in_real_g_op, in_real_c_op));
        out_imag_c_op = OperandSpec::fma(tw_imag_c2_op, in_real_g_op, OperandSpec::fma(tw_real_c2_op,in_imag_g_op, in_imag_c_op));
        out_real_g_op = OperandSpec::fms(two, in_real_c_op, out_real_c_op);
        out_imag_g_op = OperandSpec::fms(two, in_imag_c_op, out_imag_c_op);

        out_real_d_op = OperandSpec::fnma(tw_imag_d2_op, in_imag_h_op, OperandSpec::fma(tw_real_d2_op, in_real_h_op, in_real_d_op));
        out_imag_d_op = OperandSpec::fma(tw_imag_d2_op, in_real_h_op, OperandSpec::fma(tw_real_d2_op,in_imag_h_op, in_imag_d_op));
        out_real_h_op = OperandSpec::fms(two, in_real_d_op, out_real_d_op);
        out_imag_h_op = OperandSpec::fms(two, in_imag_d_op, out_imag_d_op);

        // STORE
        OperandSpec::store(imag_a, out_imag_a_op);
        OperandSpec::store(imag_b, out_imag_b_op);
        OperandSpec::store(imag_c, out_imag_c_op);
        OperandSpec::store(imag_d, out_imag_d_op);
        OperandSpec::store(imag_e, out_imag_e_op);
        OperandSpec::store(imag_f, out_imag_f_op);
        OperandSpec::store(imag_g, out_imag_g_op);
        OperandSpec::store(imag_h, out_imag_h_op);
        OperandSpec::store(real_a, out_real_a_op);
        OperandSpec::store(real_b, out_real_b_op);
        OperandSpec::store(real_c, out_real_c_op);
        OperandSpec::store(real_d, out_real_d_op);
        OperandSpec::store(real_e, out_real_e_op);
        OperandSpec::store(real_f, out_real_f_op);
        OperandSpec::store(real_g, out_real_g_op);
        OperandSpec::store(real_h, out_real_h_op);


        // UPDATE OFFSET
        a_offset += eight_n_samples_per_operand;
        b_offset += eight_n_samples_per_operand;
        c_offset += eight_n_samples_per_operand;
        d_offset += eight_n_samples_per_operand;
        e_offset += eight_n_samples_per_operand;
        f_offset += eight_n_samples_per_operand;
        g_offset += eight_n_samples_per_operand;
        h_offset += eight_n_samples_per_operand;

        tw_a0_offset += six_n_samples_per_operand;
        tw_a1_offset += six_n_samples_per_operand;
        tw_a2_offset += six_n_samples_per_operand;
        tw_b0_offset += six_n_samples_per_operand;
        tw_b1_offset += six_n_samples_per_operand;
        tw_b2_offset += six_n_samples_per_operand;
        tw_c0_offset += six_n_samples_per_operand;
        tw_c1_offset += six_n_samples_per_operand;
        tw_c2_offset += six_n_samples_per_operand;
        tw_d0_offset += six_n_samples_per_operand;
        tw_d1_offset += six_n_samples_per_operand;
        tw_d2_offset += six_n_samples_per_operand;
    }
}