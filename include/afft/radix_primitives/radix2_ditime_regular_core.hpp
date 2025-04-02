template<typename Sample, typename OperandSpec, bool rotating_output>
inline void do_radix2_ditime_regular_core_stage(
    Sample* real, 
    Sample* imag, 
    const Sample* tw_real_b_0, 
    const Sample* tw_imag_b_0, 
    std::size_t subtwiddle_len,
    std::size_t subtwiddle_start,
    std::size_t subtwiddle_end
) {
    using Operand = typename OperandSpec::Value;
    constexpr std::size_t n_samples_per_operand = sizeof(Operand) / sizeof(Sample);

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

    std::size_t a_offset = subtwiddle_start;
    std::size_t b_offset = a_offset + subtwiddle_len;

    // OFFSET
    auto real_a = real + a_offset;
    auto imag_a = imag + a_offset;
    auto real_b = real + b_offset;
    auto imag_b = imag + b_offset;
    auto tw_real_b = tw_real_b_0 + subtwiddle_start;
    auto tw_imag_b = tw_imag_b_0 + subtwiddle_start;

    for (
        std::size_t i = subtwiddle_start; 
        i < subtwiddle_end;
        i += n_samples_per_operand
    ) {
        
        //LOAD
        OperandSpec::load(real_a, in_real_a_op);
        OperandSpec::load(imag_a, in_imag_a_op);
        OperandSpec::load(real_b, in_real_b_op);
        OperandSpec::load(imag_b, in_imag_b_op);

        OperandSpec::load(tw_real_b, tw_real_b_op);
        OperandSpec::load(tw_imag_b, tw_imag_b_op);

        // COMPUTE            
        out_real_b_op = tw_imag_b_op * in_imag_b_op;
        out_imag_b_op = tw_imag_b_op * in_real_b_op;

        in_real_b_op *= tw_real_b_op;
        in_imag_b_op *= tw_real_b_op;

        in_real_b_op -= out_real_b_op;
        in_imag_b_op += out_imag_b_op;
        
        out_real_a_op = in_real_a_op + in_real_b_op;
        out_real_b_op = in_real_a_op - in_real_b_op;
        out_imag_a_op = in_imag_a_op + in_imag_b_op;
        out_imag_b_op = in_imag_a_op - in_imag_b_op;

        // STORE
        if (rotating_output){
            OperandSpec::store(imag_a, out_real_a_op);
            OperandSpec::store(real_a, out_imag_a_op);
            OperandSpec::store(imag_b, out_real_b_op);
            OperandSpec::store(real_b, out_imag_b_op);
        } else {
            OperandSpec::store(real_a, out_real_a_op);
            OperandSpec::store(imag_a, out_imag_a_op);
            OperandSpec::store(real_b, out_real_b_op);
            OperandSpec::store(imag_b, out_imag_b_op);
        }

        // UPDATE OFFSET
        real_a += n_samples_per_operand;
        imag_a += n_samples_per_operand;
        real_b += n_samples_per_operand;
        imag_b += n_samples_per_operand;
        tw_real_b += n_samples_per_operand;
        tw_imag_b += n_samples_per_operand;
    }
}