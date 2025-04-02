template<typename Sample, typename OperandSpec, bool rotating_output, bool stage_is_core, bool second_stage_is_regular>
inline void do_radix4_ditime_init_core_stage(
    Sample* real, 
    Sample* imag, 
    const Sample* tw0_real_0, 
    const Sample* tw0_imag_0, 
    const Sample* tw1_real_0, 
    const Sample* tw1_imag_0,
    std::size_t subfft_id_start,
    std::size_t subfft_id_end
) {
    using Operand = typename OperandSpec::Value;
    constexpr std::size_t n_samples_per_operand = sizeof(Operand) / sizeof(Sample);
    const std::size_t subfft_len = 4 * subtwiddle_len;

    // DECLARE
    Operand inout_real_a_op;
    Operand inout_real_b_op;
    Operand inout_imag_a_op;
    Operand inout_imag_b_op;
    Operand inter_real_a_op;
    Operand inter_real_b_op;
    Operand inter_imag_a_op;
    Operand inter_imag_b_op;

    Operand inout_real_c_op;
    Operand inout_real_d_op;
    Operand inout_imag_c_op;
    Operand inout_imag_d_op;
    Operand inter_real_c_op;
    Operand inter_real_d_op;
    Operand inter_imag_c_op;
    Operand inter_imag_d_op;

    Operand tw0_real_op;
    Operand tw0_imag_op;
    Operand tw1_real_a_op;
    Operand tw1_imag_a_op;
    Operand tw1_real_b_op;
    Operand tw1_imag_b_op;

    OperandSpec::load(tw0_real_0, tw0_real_op);
    OperandSpec::load(tw0_imag_0, tw0_imag_op);
    OperandSpec::load(tw1_real_0, tw1_real_a_op);
    OperandSpec::load(tw1_imag_0, tw1_imag_a_op);

    if (second_stage_is_regular) {
        OperandSpec::load(tw1_real_0 + n_samples_per_operand, tw1_real_b_op);
        OperandSpec::load(tw1_imag_0 + n_samples_per_operand, tw1_imag_b_op);
    }

    std::size_t a_offset = subfft_id_start * subfft_len;
    std::size_t b_offset = a_offset + subtwiddle_len;
    std::size_t c_offset = a_offset + 2 * subtwiddle_len;
    std::size_t d_offset = a_offset + 3 * subtwiddle_len;

    for (
        std::size_t subfft_id = subfft_id_start;
        subfft_id < subfft_id_end;
        subfft_id++
    ) {
        // OFFSET
        auto real_a = real + a_offset;
        auto imag_a = imag + a_offset;
        auto real_b = real + b_offset;
        auto imag_b = imag + b_offset;
        auto real_c = real + c_offset;
        auto imag_c = imag + c_offset;
        auto real_d = real + d_offset;
        auto imag_d = imag + d_offset;
        
        //LOAD
        OperandSpec::load(real_a, inout_real_a_op);
        OperandSpec::load(imag_a, inout_imag_a_op);
        OperandSpec::load(real_b, inout_real_b_op);
        OperandSpec::load(imag_b, inout_imag_b_op);
        OperandSpec::load(real_c, inout_real_c_op);
        OperandSpec::load(imag_c, inout_imag_c_op);
        OperandSpec::load(real_d, inout_real_d_op);
        OperandSpec::load(imag_d, inout_imag_d_op);

        // COMPUTE  
        Operand::deinterleave(inout_real_a_op, inout_real_b_op);
        Operand::deinterleave(inout_imag_a_op, inout_imag_b_op);
        Operand::deinterleave(inout_real_c_op, inout_real_d_op);
        Operand::deinterleave(inout_imag_c_op, inout_imag_d_op);

        if (stage_is_core) {
            inter_real_b_op = tw0_imag_op * inout_imag_b_op;
            inter_imag_b_op = tw0_imag_op * inout_real_b_op;
            inter_real_d_op = tw0_imag_op * inout_imag_d_op;
            inter_imag_d_op = tw0_imag_op * inout_real_d_op;

            inout_real_b_op *= tw0_real_op;
            inout_imag_b_op *= tw0_real_op;
            inout_real_d_op *= tw0_real_op;
            inout_imag_d_op *= tw0_real_op;

            inout_real_b_op -= inter_real_b_op;
            inout_imag_b_op += inter_imag_b_op;
            inout_real_d_op -= inter_real_d_op;
            inout_imag_d_op += inter_imag_d_op;
        }          
        
        inter_real_a_op = inout_real_a_op + inout_real_b_op;
        inter_real_b_op = inout_real_a_op - inout_real_b_op;
        inter_imag_a_op = inout_imag_a_op + inout_imag_b_op;
        inter_imag_b_op = inout_imag_a_op - inout_imag_b_op; 

        inter_real_c_op = inout_real_c_op + inout_real_d_op;
        inter_real_d_op = inout_real_c_op - inout_real_d_op;
        inter_imag_c_op = inout_imag_c_op + inout_imag_d_op;
        inter_imag_d_op = inout_imag_c_op - inout_imag_d_op; 

        if (second_stage_is_regular) {
            inout_real_c_op = tw1_imag_a_op * inter_imag_c_op;
            inout_imag_c_op = tw1_imag_a_op * inter_real_c_op;
            inout_real_d_op = tw1_imag_b_op * inter_imag_d_op;
            inout_imag_d_op = tw1_imag_b_op * inter_real_d_op;

            inter_real_c_op *= tw1_real_a_op;
            inter_imag_c_op *= tw1_real_a_op;
            inter_real_d_op *= tw1_real_b_op;
            inter_imag_d_op *= tw1_real_b_op;

            inter_real_c_op -= inout_real_c_op;
            inter_imag_c_op += inout_imag_c_op;
            inter_real_d_op -= inout_real_d_op;
            inter_imag_d_op += inout_imag_d_op;

            inout_real_a_op = inter_real_a_op + inter_real_c_op;
            inout_real_c_op = inter_real_a_op - inter_real_c_op;
            inout_imag_a_op = inter_imag_a_op + inter_imag_c_op;
            inout_imag_c_op = inter_imag_a_op - inter_imag_c_op; 

            inout_real_b_op = inter_real_b_op + inter_real_d_op;
            inout_real_d_op = inter_real_b_op - inter_real_d_op;
            inout_imag_b_op = inter_imag_b_op + inter_imag_d_op;
            inout_imag_d_op = inter_imag_b_op - inter_imag_d_op; 
        } else {
            Operand::deinterleave(inter_real_a_op, inter_real_b_op);
            Operand::deinterleave(inter_imag_a_op, inter_imag_b_op);
            Operand::deinterleave(inter_real_c_op, inter_real_d_op);
            Operand::deinterleave(inter_imag_c_op, inter_imag_d_op);

            inout_real_b_op = tw1_imag_a_op * inter_imag_b_op;
            inout_imag_b_op = tw1_imag_a_op * inter_real_b_op;
            inout_real_d_op = tw1_imag_a_op * inter_imag_d_op;
            inout_imag_d_op = tw1_imag_a_op * inter_real_d_op;

            inter_real_b_op *= tw1_real_a_op;
            inter_imag_b_op *= tw1_real_a_op;
            inter_real_d_op *= tw1_real_a_op;
            inter_imag_d_op *= tw1_real_a_op;

            inter_real_b_op -= inout_real_b_op;
            inter_imag_b_op += inout_imag_b_op;
            inter_real_d_op -= inout_real_d_op;
            inter_imag_d_op += inout_imag_d_op;
                    
            inout_real_a_op = inter_real_a_op + inter_real_b_op;
            inout_real_b_op = inter_real_a_op - inter_real_b_op;
            inout_imag_a_op = inter_imag_a_op + inter_imag_b_op;
            inout_imag_b_op = inter_imag_a_op - inter_imag_b_op; 

            inout_real_c_op = inter_real_c_op + inter_real_d_op;
            inout_real_d_op = inter_real_c_op - inter_real_d_op;
            inout_imag_c_op = inter_imag_c_op + inter_imag_d_op;
            inout_imag_d_op = inter_imag_c_op - inter_imag_d_op; 
        }

        // STORE
        if (rotating_output) {
            OperandSpec::store(real_a, inout_imag_a_op);
            OperandSpec::store(imag_a, inout_real_a_op);
            OperandSpec::store(real_b, inout_imag_b_op);
            OperandSpec::store(imag_b, inout_real_b_op);
            OperandSpec::store(real_c, inout_imag_c_op);
            OperandSpec::store(imag_c, inout_real_c_op);
            OperandSpec::store(real_d, inout_imag_d_op);
            OperandSpec::store(imag_d, inout_real_d_op);
        } else {
            OperandSpec::store(real_a, inout_real_a_op);
            OperandSpec::store(imag_a, inout_imag_a_op);
            OperandSpec::store(real_b, inout_real_b_op);
            OperandSpec::store(imag_b, inout_imag_b_op);
            OperandSpec::store(real_c, inout_real_c_op);
            OperandSpec::store(imag_c, inout_imag_c_op);
            OperandSpec::store(real_d, inout_real_d_op);
            OperandSpec::store(imag_d, inout_imag_d_op);
        }
        
        // UPDATE OFFSET
        real_a += subfft_len;
        imag_a += subfft_len;
        real_b += subfft_len;
        imag_b += subfft_len;
        real_c += subfft_len;
        imag_c += subfft_len;
        real_d += subfft_len;
        imag_d += subfft_len;
    }
}

template<typename Sample, typename OperandSpec, bool rotating_output, bool stage_is_core, bool second_stage_is_regular>
inline void do_radix4_ditime_init_core_oop_stage(
    Sample* real, 
    Sample* imag, 
    Sample* res_real, 
    Sample* res_imag, 
    const Sample* tw0_real_0, 
    const Sample* tw0_imag_0, 
    const Sample* tw1_real_0, 
    const Sample* tw1_imag_0,
    std::size_t subfft_id_start,
    std::size_t subfft_id_end
) {
    using Operand = typename OperandSpec::Value;
    constexpr std::size_t n_samples_per_operand = sizeof(Operand) / sizeof(Sample);
    const std::size_t subfft_len = 4 * subtwiddle_len;

    // DECLARE
    Operand inout_real_a_op;
    Operand inout_real_b_op;
    Operand inout_imag_a_op;
    Operand inout_imag_b_op;
    Operand inter_real_a_op;
    Operand inter_real_b_op;
    Operand inter_imag_a_op;
    Operand inter_imag_b_op;

    Operand inout_real_c_op;
    Operand inout_real_d_op;
    Operand inout_imag_c_op;
    Operand inout_imag_d_op;
    Operand inter_real_c_op;
    Operand inter_real_d_op;
    Operand inter_imag_c_op;
    Operand inter_imag_d_op;

    Operand tw0_real_op;
    Operand tw0_imag_op;
    Operand tw1_real_a_op;
    Operand tw1_imag_a_op;
    Operand tw1_real_b_op;
    Operand tw1_imag_b_op;

    OperandSpec::load(tw0_real_0, tw0_real_op);
    OperandSpec::load(tw0_imag_0, tw0_imag_op);
    OperandSpec::load(tw1_real_0, tw1_real_a_op);
    OperandSpec::load(tw1_imag_0, tw1_imag_a_op);

    if (second_stage_is_regular) {
        OperandSpec::load(tw1_real_0 + n_samples_per_operand, tw1_real_b_op);
        OperandSpec::load(tw1_imag_0 + n_samples_per_operand, tw1_imag_b_op);
    }

    std::size_t a_offset = subfft_id_start * subfft_len;
    std::size_t b_offset = a_offset + subtwiddle_len;
    std::size_t c_offset = a_offset + 2 * subtwiddle_len;
    std::size_t d_offset = a_offset + 3 * subtwiddle_len;

    for (
        std::size_t subfft_id = subfft_id_start;
        subfft_id < subfft_id_end;
        subfft_id++
    ) {
        // OFFSET
        auto real_a = real + a_offset;
        auto imag_a = imag + a_offset;
        auto real_b = real + b_offset;
        auto imag_b = imag + b_offset;
        auto real_c = real + c_offset;
        auto imag_c = imag + c_offset;
        auto real_d = real + d_offset;
        auto imag_d = imag + d_offset;

        auto res_real_a = res_real + a_offset;
        auto res_imag_a = res_imag + a_offset;
        auto res_real_b = res_real + b_offset;
        auto res_imag_b = res_imag + b_offset;
        auto res_real_c = res_real + c_offset;
        auto res_imag_c = res_imag + c_offset;
        auto res_real_d = res_real + d_offset;
        auto res_imag_d = res_imag + d_offset;
        
        //LOAD
        OperandSpec::load(real_a, inout_real_a_op);
        OperandSpec::load(imag_a, inout_imag_a_op);
        OperandSpec::load(real_b, inout_real_b_op);
        OperandSpec::load(imag_b, inout_imag_b_op);
        OperandSpec::load(real_c, inout_real_c_op);
        OperandSpec::load(imag_c, inout_imag_c_op);
        OperandSpec::load(real_d, inout_real_d_op);
        OperandSpec::load(imag_d, inout_imag_d_op);

        // COMPUTE  
        Operand::deinterleave(inout_real_a_op, inout_real_b_op);
        Operand::deinterleave(inout_imag_a_op, inout_imag_b_op);
        Operand::deinterleave(inout_real_c_op, inout_real_d_op);
        Operand::deinterleave(inout_imag_c_op, inout_imag_d_op);

        if (stage_is_core) {
            inter_real_b_op = tw0_imag_op * inout_imag_b_op;
            inter_imag_b_op = tw0_imag_op * inout_real_b_op;
            inter_real_d_op = tw0_imag_op * inout_imag_d_op;
            inter_imag_d_op = tw0_imag_op * inout_real_d_op;

            inout_real_b_op *= tw0_real_op;
            inout_imag_b_op *= tw0_real_op;
            inout_real_d_op *= tw0_real_op;
            inout_imag_d_op *= tw0_real_op;

            inout_real_b_op -= inter_real_b_op;
            inout_imag_b_op += inter_imag_b_op;
            inout_real_d_op -= inter_real_d_op;
            inout_imag_d_op += inter_imag_d_op;
        }          
        
        inter_real_a_op = inout_real_a_op + inout_real_b_op;
        inter_real_b_op = inout_real_a_op - inout_real_b_op;
        inter_imag_a_op = inout_imag_a_op + inout_imag_b_op;
        inter_imag_b_op = inout_imag_a_op - inout_imag_b_op; 

        inter_real_c_op = inout_real_c_op + inout_real_d_op;
        inter_real_d_op = inout_real_c_op - inout_real_d_op;
        inter_imag_c_op = inout_imag_c_op + inout_imag_d_op;
        inter_imag_d_op = inout_imag_c_op - inout_imag_d_op; 

        if (second_stage_is_regular) {
            inout_real_c_op = tw1_imag_a_op * inter_imag_c_op;
            inout_imag_c_op = tw1_imag_a_op * inter_real_c_op;
            inout_real_d_op = tw1_imag_b_op * inter_imag_d_op;
            inout_imag_d_op = tw1_imag_b_op * inter_real_d_op;

            inter_real_c_op *= tw1_real_a_op;
            inter_imag_c_op *= tw1_real_a_op;
            inter_real_d_op *= tw1_real_b_op;
            inter_imag_d_op *= tw1_real_b_op;

            inter_real_c_op -= inout_real_c_op;
            inter_imag_c_op += inout_imag_c_op;
            inter_real_d_op -= inout_real_d_op;
            inter_imag_d_op += inout_imag_d_op;

            inout_real_a_op = inter_real_a_op + inter_real_c_op;
            inout_real_c_op = inter_real_a_op - inter_real_c_op;
            inout_imag_a_op = inter_imag_a_op + inter_imag_c_op;
            inout_imag_c_op = inter_imag_a_op - inter_imag_c_op; 

            inout_real_b_op = inter_real_b_op + inter_real_d_op;
            inout_real_d_op = inter_real_b_op - inter_real_d_op;
            inout_imag_b_op = inter_imag_b_op + inter_imag_d_op;
            inout_imag_d_op = inter_imag_b_op - inter_imag_d_op; 
        } else {
            Operand::deinterleave(inter_real_a_op, inter_real_b_op);
            Operand::deinterleave(inter_imag_a_op, inter_imag_b_op);
            Operand::deinterleave(inter_real_c_op, inter_real_d_op);
            Operand::deinterleave(inter_imag_c_op, inter_imag_d_op);

            inout_real_b_op = tw1_imag_a_op * inter_imag_b_op;
            inout_imag_b_op = tw1_imag_a_op * inter_real_b_op;
            inout_real_d_op = tw1_imag_a_op * inter_imag_d_op;
            inout_imag_d_op = tw1_imag_a_op * inter_real_d_op;

            inter_real_b_op *= tw1_real_a_op;
            inter_imag_b_op *= tw1_real_a_op;
            inter_real_d_op *= tw1_real_a_op;
            inter_imag_d_op *= tw1_real_a_op;

            inter_real_b_op -= inout_real_b_op;
            inter_imag_b_op += inout_imag_b_op;
            inter_real_d_op -= inout_real_d_op;
            inter_imag_d_op += inout_imag_d_op;
                    
            inout_real_a_op = inter_real_a_op + inter_real_b_op;
            inout_real_b_op = inter_real_a_op - inter_real_b_op;
            inout_imag_a_op = inter_imag_a_op + inter_imag_b_op;
            inout_imag_b_op = inter_imag_a_op - inter_imag_b_op; 

            inout_real_c_op = inter_real_c_op + inter_real_d_op;
            inout_real_d_op = inter_real_c_op - inter_real_d_op;
            inout_imag_c_op = inter_imag_c_op + inter_imag_d_op;
            inout_imag_d_op = inter_imag_c_op - inter_imag_d_op; 
        }

        // STORE
        if (rotating_output) {
            OperandSpec::store(res_real_a, inout_imag_a_op);
            OperandSpec::store(res_imag_a, inout_real_a_op);
            OperandSpec::store(res_real_b, inout_imag_b_op);
            OperandSpec::store(res_imag_b, inout_real_b_op);
            OperandSpec::store(res_real_c, inout_imag_c_op);
            OperandSpec::store(res_imag_c, inout_real_c_op);
            OperandSpec::store(res_real_d, inout_imag_d_op);
            OperandSpec::store(res_imag_d, inout_real_d_op);
        } else {
            OperandSpec::store(res_real_a, inout_real_a_op);
            OperandSpec::store(res_imag_a, inout_imag_a_op);
            OperandSpec::store(res_real_b, inout_real_b_op);
            OperandSpec::store(res_imag_b, inout_imag_b_op);
            OperandSpec::store(res_real_c, inout_real_c_op);
            OperandSpec::store(res_imag_c, inout_imag_c_op);
            OperandSpec::store(res_real_d, inout_real_d_op);
            OperandSpec::store(res_imag_d, inout_imag_d_op);
        }
        
        // UPDATE OFFSET
        real_a += subfft_len;
        imag_a += subfft_len;
        real_b += subfft_len;
        imag_b += subfft_len;
        real_c += subfft_len;
        imag_c += subfft_len;
        real_d += subfft_len;
        imag_d += subfft_len;

        res_real_a += subfft_len;
        res_imag_a += subfft_len;
        res_real_b += subfft_len;
        res_imag_b += subfft_len;
        res_real_c += subfft_len;
        res_imag_c += subfft_len;
        res_real_d += subfft_len;
        res_imag_d += subfft_len;
    }
}