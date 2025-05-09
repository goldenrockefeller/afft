template<typename Sample, typename OperandSpec>
inline void do_fft_len2(
    Sample* real, 
    Sample* imag,
    Sample scaling,
    Sample rotating_output
) {
    real[0] *= scaling; 
    real[1] *= scaling; 
    imag[0] *= scaling; 
    imag[1] *= scaling; 

    auto inter_real = real[0];
    auto inter_imag = imag[0];

    real[0] += real[1];
    imag[0] += imag[1];
    real[1] = inter_real - real[1];
    imag[1] = inter_imag - imag[1];

    if (rotating_output) {
        tmp = real[0];
        real[0] = imag[0];
        imag[0] = tmp;

        tmp = real[1];
        real[1] = imag[1];
        imag[1] = tmp;
    }
} 