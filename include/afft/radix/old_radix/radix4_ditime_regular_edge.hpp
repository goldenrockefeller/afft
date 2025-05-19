template<typename Sample, typename OperandSpec, bool rotating_output, bool rescaling_input>
inline void do_radix4_dit_regular_compound_radix_stage(
    Sample* real, 
    Sample* imag, 
    const Sample* tw_real_b_0, 
    const Sample* tw_imag_b_0, 
    const Sample* tw_real_c_0, 
    const Sample* tw_imag_c_0, 
    const Sample* tw_real_d_0, 
    const Sample* tw_imag_d_0,
    std::size_t subfft_id_start,
    std::size_t subfft_id_end,
    Sample* scaling
) {
    using Operand = typename OperandSpec::Value;
    constexpr std::size_t n_samples_per_operand = sizeof(Operand) / sizeof(Sample);
    static_assert(
        (n_samples_per_operand==1), 
        "Operand and Sample must have the same size"
    );
    constexpr std::size_t subtwiddle_len = n_samples_per_operand;
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

    Operand scaling_op(scaling);

    std::size_t a_offset = subfft_id_start * subfft_len;
    std::size_t b_offset = a_offset + subtwiddle_len;
    std::size_t c_offset = a_offset + 2 * subtwiddle_len;
    std::size_t d_offset = a_offset + 3 * subtwiddle_len;

    // OFFSET
    auto real_a = real + a_offset;
    auto imag_a = imag + a_offset;
    auto real_b = real + b_offset;
    auto imag_b = imag + b_offset;
    auto real_c = real + c_offset;
    auto imag_c = imag + c_offset;
    auto real_d = real + d_offset;
    auto imag_d = imag + d_offset;

    for (
        std::size_t subfft_id = subfft_id_start;
        subfft_id < subfft_id_end;
        subfft_id++
    ) {            
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
        if (rescaling_input) {
            inout_real_a_op *= scaling_op;
            inout_imag_a_op *= scaling_op;
            inout_real_b_op *= scaling_op;
            inout_imag_b_op *= scaling_op;
            inout_real_c_op *= scaling_op;
            inout_imag_c_op *= scaling_op;
            inout_real_d_op *= scaling_op;
            inout_imag_d_op *= scaling_op;
        }

        inter_real_a_op = inout_real_a_op + inout_real_b_op;
        inter_imag_a_op = inout_imag_a_op + inout_imag_b_op;
        inter_real_b_op = inout_real_a_op - inout_real_b_op;
        inter_imag_b_op = inout_imag_a_op - inout_imag_b_op;

        inter_real_c_op = inout_real_c_op + inout_real_d_op;
        inter_imag_c_op = inout_imag_c_op + inout_imag_d_op;
        inter_real_d_op = inout_real_c_op - inout_real_d_op;
        inter_imag_d_op = inout_imag_c_op - inout_imag_d_op;

        inout_real_a_op = inter_real_a_op + inter_real_c_op;
        inout_real_b_op = inter_real_b_op + inter_imag_d_op;
        inout_real_c_op = inter_real_a_op - inter_real_c_op;
        inout_real_d_op = inter_real_b_op - inter_imag_d_op;
        
        inout_imag_a_op = inter_imag_a_op + inter_imag_c_op;
        inout_imag_b_op = inter_imag_b_op - inter_real_d_op;
        inout_imag_c_op = inter_imag_a_op - inter_imag_c_op;
        inout_imag_d_op = inter_imag_b_op + inter_real_d_op; 

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
        } {
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