template<typename Sample, typename OperandSpec, bool rotating_output>
inline void do_radix2_ditime_init_core_stage(
    Sample* real, 
    Sample* imag, 
    const Sample* tw_real_b_0, 
    const Sample* tw_imag_b_0
) {
    using Operand = typename OperandSpec::Value;
    constexpr std::size_t n_samples_per_operand = sizeof(Operand) / sizeof(Sample);
    constexpr std::size_t subtwiddle_len = n_samples_per_operand;

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

  
    //LOAD
    OperandSpec::load(real, in_real_a_op);
    OperandSpec::load(imag, in_imag_a_op);
    OperandSpec::load(real + subtwiddle_len, in_real_b_op);
    OperandSpec::load(imag + subtwiddle_len, in_imag_b_op);

    OperandSpec::load(tw_real_b_0, tw_real_b_op);
    OperandSpec::load(tw_imag_b_0, tw_imag_b_op);

    // COMPUTE            
    OperandSpec::deinterleave(out_real_a_op, out_real_b_op);
    OperandSpec::deinterleave(out_imag_a_op, out_imag_b_op);
    
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
}