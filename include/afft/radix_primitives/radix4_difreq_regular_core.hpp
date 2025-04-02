template<typename Sample, typename OperandSpec, bool rotating_output>
inline void do_radix4_difreq_regular_core_stage(
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
    std::size_t subtwiddle_len,
    std::size_t subtwiddle_start,
    std::size_t subtwiddle_end
) {
    using Operand = typename OperandSpec::Value;
    constexpr std::size_t n_samples_per_operand = sizeof(Operand) / sizeof(Sample);
    const std::size_t subfft_len = 4 * subtwiddle_len;

    // DECLARE
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
    for (
        std::size_t subfft_id = subfft_id_start;
        subfft_id < subfft_id_end;
        subfft_id++
    ) {
        std::size_t a_offset = subfft_id * subfft_len + subtwiddle_start;
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
        auto tw_real_b = tw_real_b_0 + subtwiddle_start;
        auto tw_imag_b = tw_imag_b_0 + subtwiddle_start;
        auto tw_real_c = tw_real_c_0 + subtwiddle_start;
        auto tw_imag_c = tw_imag_c_0 + subtwiddle_start;
        auto tw_real_d = tw_real_d_0 + subtwiddle_start;
        auto tw_imag_d = tw_imag_d_0 + subtwiddle_start;

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
            out_real_a_op = in_real_a_op + in_real_c_op;
            out_real_b_op = in_real_b_op + in_real_d_op;
            out_real_c_op = in_real_a_op - in_real_c_op;
            out_real_d_op = in_imag_b_op - in_imag_d_op;

            out_imag_a_op = in_imag_a_op + in_imag_c_op;
            out_imag_b_op = in_imag_b_op + in_imag_d_op;
            out_imag_c_op = in_imag_a_op - in_imag_c_op;
            out_imag_d_op = in_real_d_op - in_real_b_op;  

            in_real_a_op = out_real_a_op + out_real_b_op;
            in_real_b_op = out_real_a_op - out_real_b_op;
            in_real_c_op = out_real_c_op + out_real_d_op;
            in_real_d_op = out_real_c_op - out_real_d_op;

            in_imag_a_op = out_imag_a_op + out_imag_b_op;
            in_imag_b_op = out_imag_a_op - out_imag_b_op;
            in_imag_c_op = out_imag_c_op + out_imag_d_op;
            in_imag_d_op = out_imag_c_op - out_imag_d_op;

            out_real_a_op = in_real_a_op;
            out_real_b_op = in_real_b_op * tw_real_b_op;
            out_real_c_op = in_real_c_op * tw_real_c_op;
            out_real_d_op = in_imag_d_op * tw_real_d_op;

            out_imag_a_op = in_imag_a_op;
            out_imag_b_op = in_imag_b_op * tw_real_b_op;
            out_imag_c_op = in_imag_c_op * tw_real_c_op;
            out_imag_d_op = in_real_d_op * tw_real_d_op; 

            in_real_b_op *= tw_imag_b_op;
            in_real_c_op *= tw_imag_c_op;
            in_real_d_op *= tw_imag_d_op;
            in_imag_b_op *= tw_imag_b_op;
            in_imag_c_op *= tw_imag_c_op;
            in_imag_d_op *= tw_imag_d_op;

            out_real_b_op -= in_imag_b_op;
            out_real_c_op -= in_imag_c_op;
            out_real_d_op -= in_imag_d_op;
            out_imag_b_op += in_real_b_op;
            out_imag_c_op += in_real_c_op;
            out_imag_d_op += in_real_d_op; 

            // STORE
            if (rotating_output) {
                OperandSpec::store(real_a, out_imag_a_op);
                OperandSpec::store(imag_a, out_real_a_op);
                OperandSpec::store(real_b, out_imag_b_op);
                OperandSpec::store(imag_b, out_real_b_op);
                OperandSpec::store(real_c, out_imag_c_op);
                OperandSpec::store(imag_c, out_real_c_op);
                OperandSpec::store(real_d, out_imag_d_op);
                OperandSpec::store(imag_d, out_real_d_op);
            } else {
                OperandSpec::store(real_a, out_real_a_op);
                OperandSpec::store(imag_a, out_imag_a_op);
                OperandSpec::store(real_b, out_real_b_op);
                OperandSpec::store(imag_b, out_imag_b_op);
                OperandSpec::store(real_c, out_real_c_op);
                OperandSpec::store(imag_c, out_imag_c_op);
                OperandSpec::store(real_d, out_real_d_op);
                OperandSpec::store(imag_d, out_imag_d_op);
            }
            
            // UPDATE OFFSET
            real_a += n_samples_per_operand;
            imag_a += n_samples_per_operand;
            real_b += n_samples_per_operand;
            imag_b += n_samples_per_operand;
            real_c += n_samples_per_operand;
            imag_c += n_samples_per_operand;
            real_d += n_samples_per_operand;
            imag_d += n_samples_per_operand;
            tw_real_b += n_samples_per_operand;
            tw_imag_b += n_samples_per_operand;
            tw_real_c += n_samples_per_operand;
            tw_imag_c += n_samples_per_operand;
            tw_real_d += n_samples_per_operand;
            tw_imag_d += n_samples_per_operand;
        }
    }
}