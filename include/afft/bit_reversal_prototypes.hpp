//Analyze Clang vs GCC vs MSVC output
// Tree-like bitreversal to replace recursive one
void bitreversal64_unrolled(double* out_real, double* out_imag, double* in_real, double* in_imag, std::size_t len){
    if (len != 64) {
        return;
    }
    out_real[ 63 ] = in_real[ 63 ];
    out_real[ 62 ] = in_real[ 31 ];
    out_real[ 61 ] = in_real[ 47 ];
    out_real[ 60 ] = in_real[ 15 ];
    out_real[ 59 ] = in_real[ 55 ];
    out_real[ 58 ] = in_real[ 23 ];
    out_real[ 57 ] = in_real[ 39 ];
    out_real[ 56 ] = in_real[ 7 ];
    out_real[ 55 ] = in_real[ 59 ];
    out_real[ 54 ] = in_real[ 27 ];
    out_real[ 53 ] = in_real[ 43 ];
    out_real[ 52 ] = in_real[ 11 ];
    out_real[ 51 ] = in_real[ 51 ];
    out_real[ 50 ] = in_real[ 19 ];
    out_real[ 49 ] = in_real[ 35 ];
    out_real[ 48 ] = in_real[ 3 ];
    out_real[ 47 ] = in_real[ 61 ];
    out_real[ 46 ] = in_real[ 29 ];
    out_real[ 45 ] = in_real[ 45 ];
    out_real[ 44 ] = in_real[ 13 ];
    out_real[ 43 ] = in_real[ 53 ];
    out_real[ 42 ] = in_real[ 21 ];
    out_real[ 41 ] = in_real[ 37 ];
    out_real[ 40 ] = in_real[ 5 ];
    out_real[ 39 ] = in_real[ 57 ];
    out_real[ 38 ] = in_real[ 25 ];
    out_real[ 37 ] = in_real[ 41 ];
    out_real[ 36 ] = in_real[ 9 ];
    out_real[ 35 ] = in_real[ 49 ];
    out_real[ 34 ] = in_real[ 17 ];
    out_real[ 33 ] = in_real[ 33 ];
    out_real[ 32 ] = in_real[ 1 ];
    out_real[ 31 ] = in_real[ 62 ];
    out_real[ 30 ] = in_real[ 30 ];
    out_real[ 29 ] = in_real[ 46 ];
    out_real[ 28 ] = in_real[ 14 ];
    out_real[ 27 ] = in_real[ 54 ];
    out_real[ 26 ] = in_real[ 22 ];
    out_real[ 25 ] = in_real[ 38 ];
    out_real[ 24 ] = in_real[ 6 ];
    out_real[ 23 ] = in_real[ 58 ];
    out_real[ 22 ] = in_real[ 26 ];
    out_real[ 21 ] = in_real[ 42 ];
    out_real[ 20 ] = in_real[ 10 ];
    out_real[ 19 ] = in_real[ 50 ];
    out_real[ 18 ] = in_real[ 18 ];
    out_real[ 17 ] = in_real[ 34 ];
    out_real[ 16 ] = in_real[ 2 ];
    out_real[ 15 ] = in_real[ 60 ];
    out_real[ 14 ] = in_real[ 28 ];
    out_real[ 13 ] = in_real[ 44 ];
    out_real[ 12 ] = in_real[ 12 ];
    out_real[ 11 ] = in_real[ 52 ];
    out_real[ 10 ] = in_real[ 20 ];
    out_real[ 9 ] = in_real[ 36 ];
    out_real[ 8 ] = in_real[ 4 ];
    out_real[ 7 ] = in_real[ 56 ];
    out_real[ 6 ] = in_real[ 24 ];
    out_real[ 5 ] = in_real[ 40 ];
    out_real[ 4 ] = in_real[ 8 ];
    out_real[ 3 ] = in_real[ 48 ];
    out_real[ 2 ] = in_real[ 16 ];
    out_real[ 1 ] = in_real[ 32 ];
    out_real[ 0 ] = in_real[ 0 ];

    out_imag[ 63 ] = in_imag[ 63 ];
    out_imag[ 62 ] = in_imag[ 31 ];
    out_imag[ 61 ] = in_imag[ 47 ];
    out_imag[ 60 ] = in_imag[ 15 ];
    out_imag[ 59 ] = in_imag[ 55 ];
    out_imag[ 58 ] = in_imag[ 23 ];
    out_imag[ 57 ] = in_imag[ 39 ];
    out_imag[ 56 ] = in_imag[ 7 ];
    out_imag[ 55 ] = in_imag[ 59 ];
    out_imag[ 54 ] = in_imag[ 27 ];
    out_imag[ 53 ] = in_imag[ 43 ];
    out_imag[ 52 ] = in_imag[ 11 ];
    out_imag[ 51 ] = in_imag[ 51 ];
    out_imag[ 50 ] = in_imag[ 19 ];
    out_imag[ 49 ] = in_imag[ 35 ];
    out_imag[ 48 ] = in_imag[ 3 ];
    out_imag[ 47 ] = in_imag[ 61 ];
    out_imag[ 46 ] = in_imag[ 29 ];
    out_imag[ 45 ] = in_imag[ 45 ];
    out_imag[ 44 ] = in_imag[ 13 ];
    out_imag[ 43 ] = in_imag[ 53 ];
    out_imag[ 42 ] = in_imag[ 21 ];
    out_imag[ 41 ] = in_imag[ 37 ];
    out_imag[ 40 ] = in_imag[ 5 ];
    out_imag[ 39 ] = in_imag[ 57 ];
    out_imag[ 38 ] = in_imag[ 25 ];
    out_imag[ 37 ] = in_imag[ 41 ];
    out_imag[ 36 ] = in_imag[ 9 ];
    out_imag[ 35 ] = in_imag[ 49 ];
    out_imag[ 34 ] = in_imag[ 17 ];
    out_imag[ 33 ] = in_imag[ 33 ];
    out_imag[ 32 ] = in_imag[ 1 ];
    out_imag[ 31 ] = in_imag[ 62 ];
    out_imag[ 30 ] = in_imag[ 30 ];
    out_imag[ 29 ] = in_imag[ 46 ];
    out_imag[ 28 ] = in_imag[ 14 ];
    out_imag[ 27 ] = in_imag[ 54 ];
    out_imag[ 26 ] = in_imag[ 22 ];
    out_imag[ 25 ] = in_imag[ 38 ];
    out_imag[ 24 ] = in_imag[ 6 ];
    out_imag[ 23 ] = in_imag[ 58 ];
    out_imag[ 22 ] = in_imag[ 26 ];
    out_imag[ 21 ] = in_imag[ 42 ];
    out_imag[ 20 ] = in_imag[ 10 ];
    out_imag[ 19 ] = in_imag[ 50 ];
    out_imag[ 18 ] = in_imag[ 18 ];
    out_imag[ 17 ] = in_imag[ 34 ];
    out_imag[ 16 ] = in_imag[ 2 ];
    out_imag[ 15 ] = in_imag[ 60 ];
    out_imag[ 14 ] = in_imag[ 28 ];
    out_imag[ 13 ] = in_imag[ 44 ];
    out_imag[ 12 ] = in_imag[ 12 ];
    out_imag[ 11 ] = in_imag[ 52 ];
    out_imag[ 10 ] = in_imag[ 20 ];
    out_imag[ 9 ] = in_imag[ 36 ];
    out_imag[ 8 ] = in_imag[ 4 ];
    out_imag[ 7 ] = in_imag[ 56 ];
    out_imag[ 6 ] = in_imag[ 24 ];
    out_imag[ 5 ] = in_imag[ 40 ];
    out_imag[ 4 ] = in_imag[ 8 ];
    out_imag[ 3 ] = in_imag[ 48 ];
    out_imag[ 2 ] = in_imag[ 16 ];
    out_imag[ 1 ] = in_imag[ 32 ];
    out_imag[ 0 ] = in_imag[ 0 ];
}


std::size_t lzcnt64(std::size_t v) {
    double d = double(v & ~(v >> 1));
    return (1086 - ((*(std::size_t*)&d >> 52))) - (v == 0) * 1022;
}

void autobit_reverse(double* out_real, double* out_imag, double* in_real, double* in_imag, std::size_t len){
    auto active_bits = std::size_t(len - 1);
    std::size_t positive_mask = ~(std::size_t(1) << 63);
    auto incrementer = active_bits ^ (len >> 1);

    auto all_64_bits = ~(std::size_t(0));

    auto bitrev = active_bits;
    auto half_len = len >> 1;

    for (std::size_t i = len - 1; i > 1; i-=2) {
        // Note, requires __builtin_clzll, and march=native to be competitive. Can be faster if "unrolled" to so that 1 interation of bit reversal tree traversal will move (x4) multiple values.
        out_real[i] = in_real[bitrev];
        out_imag[i] = in_imag[bitrev];


        bitrev = bitrev & incrementer;

        out_real[i - 1] = in_real[bitrev];
        out_imag[i - 1] = in_imag[bitrev];

        auto lzcnt = __builtin_clzll (bitrev);

        // clear / traverse up tree
        bitrev = (bitrev << lzcnt) & positive_mask;  
        bitrev = bitrev >> lzcnt;

        // traverse down tree
        auto traverse_down = all_64_bits << (64 - lzcnt);
        traverse_down = traverse_down & active_bits;
        bitrev = bitrev | traverse_down;

    }

    out_real[1] = in_real[bitrev];
    out_imag[1] = in_imag[bitrev];
    out_real[0] = in_real[0];
    out_imag[0] = in_imag[0];
}

void recursive_bitreversal(double* out_real, double* out_imag, double* in_real, double* in_imag, std::size_t rlen, std::size_t stride) {
    if (rlen == 2) {
        out_real[0] = in_real[0];
        out_real[1] = in_real[stride];
        out_imag[0] = in_imag[0];
        out_imag[1] = in_imag[stride];
        return;
    }
    auto half_rlen = rlen >> 1;
    auto two_stride = stride * 2;
        
    recursive_bitreversal(
        out_real,
        out_imag,
        in_real,
        in_imag,
        half_rlen,
        two_stride
    );
    recursive_bitreversal(
        out_real + half_rlen,
        out_imag + half_rlen,
        in_real + stride,
        in_imag + stride,
        half_rlen,
        two_stride
    );
}


void standard_bitreversal(double* out_real, double* out_imag, double* in_real, double* in_imag, std::size_t len, std::size_t* bit_reversed_indexes) {
    for (
        std::size_t new_index = 0;
        new_index < len; 
        new_index++
    ) {
        auto old_index_0 = bit_reversed_indexes[new_index];

        out_real[new_index] = in_real[old_index_0];
        out_imag[new_index] = in_imag[old_index_0];
    }
}

void r_standard_bitreversal(double*__restrict__ out_real, double*__restrict__ out_imag, double*__restrict__ in_real, double*__restrict__ in_imag, std::size_t len, std::size_t*__restrict__ bit_reversed_indexes) {
    for (
        std::size_t new_index = 0;
        new_index < len; 
        new_index++
    ) {
        auto old_index_0 = bit_reversed_indexes[new_index];

        out_real[new_index] = in_real[old_index_0];
        out_imag[new_index] = in_imag[old_index_0];
    }
}

void interleave_bitreversal_unrolled64(double* out_real, double* out_imag, double* in_real, double* in_imag, double* work_real, double* work_imag, std::size_t len, std::size_t* bit_reversed_indexes) {
    if (len != 64) {
        return;
    }

    using Operand = xsimd::batch<double, xsimd::avx>;

    auto quarter_len = len >> 2;
    auto half_len = len >> 1;
    auto three_quarter_len = quarter_len + half_len;

    // AAA
    auto iq0_real = in_real;
    auto iq1_real = in_real + quarter_len;
    auto iq2_real = in_real + half_len;
    auto iq3_real = in_real + three_quarter_len;

    auto iq0_imag = in_imag;
    auto iq1_imag = in_imag + quarter_len;
    auto iq2_imag = in_imag + half_len;
    auto iq3_imag = in_imag + three_quarter_len;

    auto oq0_real = work_real;
    auto oq1_real = work_real + 4;
    auto oq2_real = work_real + half_len;
    auto oq3_real = work_real + half_len + 4;

    auto oq0_imag = work_imag;
    auto oq1_imag = work_imag + 4;
    auto oq2_imag = work_imag + half_len;
    auto oq3_imag = work_imag + half_len + 4;
    
    #pragma GCC unroll 4
    for (
        std::size_t new_index = 0;
        new_index < 16; 
        new_index+=4
    ) {
        Operand q0_real = Operand::load_unaligned(iq0_real);
        Operand q1_real = Operand::load_unaligned(iq1_real);
        Operand q2_real = Operand::load_unaligned(iq2_real);
        Operand q3_real = Operand::load_unaligned(iq3_real);

        Operand q0_imag = Operand::load_unaligned(iq0_imag);
        Operand q1_imag = Operand::load_unaligned(iq1_imag);
        Operand q2_imag = Operand::load_unaligned(iq2_imag);
        Operand q3_imag = Operand::load_unaligned(iq3_imag);

        Operand p0_real = xsimd::zip_lo(q0_real, q1_real);
        Operand p1_real = xsimd::zip_hi(q0_real, q1_real);
        Operand p2_real = xsimd::zip_lo(q2_real, q3_real);
        Operand p3_real = xsimd::zip_hi(q2_real, q3_real);

        Operand p0_imag = xsimd::zip_lo(q0_imag, q1_imag);
        Operand p1_imag = xsimd::zip_hi(q0_imag, q1_imag);
        Operand p2_imag = xsimd::zip_lo(q2_imag, q3_imag);
        Operand p3_imag = xsimd::zip_hi(q2_imag, q3_imag);

        p0_real.store_unaligned(oq0_real);
        p1_real.store_unaligned(oq1_real);
        p2_real.store_unaligned(oq2_real);
        p3_real.store_unaligned(oq3_real);

        p0_imag.store_unaligned(oq0_imag);
        p1_imag.store_unaligned(oq1_imag);
        p2_imag.store_unaligned(oq2_imag);
        p3_imag.store_unaligned(oq3_real);

        // This could be optimized too!!!!
        iq0_real+=4;
        iq1_real+=4;
        iq2_real+=4;
        iq3_real+=4;

        iq0_imag += 4;
        iq1_imag += 4;
        iq2_imag += 4;
        iq3_imag += 4;

        oq0_real += 8;
        oq1_real += 8;
        oq2_real += 8;
        oq3_real += 8;

        oq0_imag += 8;
        oq1_imag += 8;
        oq2_imag += 8;
        oq3_imag += 8;
    }

    iq0_real = work_real;
    iq1_real = work_real + half_len;

    iq0_imag = work_imag;
    iq1_imag = work_imag + half_len;


    Operand q0_real;
    Operand q1_real;

    Operand q0_imag;
    Operand q1_imag;

    Operand p0_real;
    Operand p1_real;

    Operand p0_imag;
    Operand p1_imag;
    
    /////////////////////////////// n = 0 //////////////////////////////////////

        q0_real = Operand::load_unaligned(iq0_real);
        q1_real = Operand::load_unaligned(iq1_real);

        q0_imag = Operand::load_unaligned(iq0_imag);
        q1_imag = Operand::load_unaligned(iq1_imag);

        p0_real = xsimd::zip_lo(q0_real, q1_real);
        p1_real = xsimd::zip_hi(q0_real, q1_real);

        p0_imag = xsimd::zip_lo(q0_imag, q1_imag);
        p1_imag = xsimd::zip_hi(q0_imag, q1_imag);

        //intereave and drop

        p0_real.store_unaligned(out_real);
        p1_real.store_unaligned(out_real + 32);

        p0_imag.store_unaligned(out_imag);
        p1_imag.store_unaligned(out_imag + 32);

    /////////////////////////////// n = 8 //////////////////////////////////////

        q0_real = Operand::load_unaligned(iq0_real + 4);
        q1_real = Operand::load_unaligned(iq1_real + 4);

        q0_imag = Operand::load_unaligned(iq0_imag + 4);
        q1_imag = Operand::load_unaligned(iq1_imag + 4);

        p0_real = xsimd::zip_lo(q0_real, q1_real);
        p1_real = xsimd::zip_hi(q0_real, q1_real);

        p0_imag = xsimd::zip_lo(q0_imag, q1_imag);
        p1_imag = xsimd::zip_hi(q0_imag, q1_imag);

        //intereave and drop

        p0_real.store_unaligned(out_real + 16);
        p1_real.store_unaligned(out_real + 48);

        p0_imag.store_unaligned(out_imag + 16);
        p1_imag.store_unaligned(out_imag + 48);


        /////////////////////////////// n = 16 //////////////////////////////////////

        q0_real = Operand::load_unaligned(iq0_real + 8);
        q1_real = Operand::load_unaligned(iq1_real + 8);

        q0_imag = Operand::load_unaligned(iq0_imag + 8);
        q1_imag = Operand::load_unaligned(iq1_imag + 8);

        p0_real = xsimd::zip_lo(q0_real, q1_real);
        p1_real = xsimd::zip_hi(q0_real, q1_real);

        p0_imag = xsimd::zip_lo(q0_imag, q1_imag);
        p1_imag = xsimd::zip_hi(q0_imag, q1_imag);

        //intereave and drop

        p0_real.store_unaligned(out_real + 8);
        p1_real.store_unaligned(out_real + 40);

        p0_imag.store_unaligned(out_imag + 8);
        p1_imag.store_unaligned(out_imag + 40);

        /////////////////////////////// n = 24 //////////////////////////////////////


        q0_real = Operand::load_unaligned(iq0_real + 12);
        q1_real = Operand::load_unaligned(iq1_real + 12);

        q0_imag = Operand::load_unaligned(iq0_imag + 12);
        q1_imag = Operand::load_unaligned(iq1_imag + 12);

        p0_real = xsimd::zip_lo(q0_real, q1_real);
        p1_real = xsimd::zip_hi(q0_real, q1_real);

        p0_imag = xsimd::zip_lo(q0_imag, q1_imag);
        p1_imag = xsimd::zip_hi(q0_imag, q1_imag);

        //intereave and drop

        p0_real.store_unaligned(out_real + 24);
        p1_real.store_unaligned(out_real + 56);

        p0_imag.store_unaligned(out_imag + 24);
        p1_imag.store_unaligned(out_imag + 56);

        /////////////////////////////// n = 32 //////////////////////////////////////

        q0_real = Operand::load_unaligned(iq0_real + 16);
        q1_real = Operand::load_unaligned(iq1_real + 16);

        q0_imag = Operand::load_unaligned(iq0_imag + 16);
        q1_imag = Operand::load_unaligned(iq1_imag + 16);

        p0_real = xsimd::zip_lo(q0_real, q1_real);
        p1_real = xsimd::zip_hi(q0_real, q1_real);

        p0_imag = xsimd::zip_lo(q0_imag, q1_imag);
        p1_imag = xsimd::zip_hi(q0_imag, q1_imag);

        //intereave and drop

        p0_real.store_unaligned(out_real + 4);
        p1_real.store_unaligned(out_real + 36);

        p0_imag.store_unaligned(out_imag + 4);
        p1_imag.store_unaligned(out_imag + 36);


        /////////////////////////////// n = 40 //////////////////////////////////////

        q0_real = Operand::load_unaligned(iq0_real + 20);
        q1_real = Operand::load_unaligned(iq1_real + 20);

        q0_imag = Operand::load_unaligned(iq0_imag + 20);
        q1_imag = Operand::load_unaligned(iq1_imag + 20);

        p0_real = xsimd::zip_lo(q0_real, q1_real);
        p1_real = xsimd::zip_hi(q0_real, q1_real);

        p0_imag = xsimd::zip_lo(q0_imag, q1_imag);
        p1_imag = xsimd::zip_hi(q0_imag, q1_imag);

        //intereave and drop

        p0_real.store_unaligned(out_real + 20);
        p1_real.store_unaligned(out_real + 52);

        p0_imag.store_unaligned(out_imag + 20);
        p1_imag.store_unaligned(out_imag + 52);

        /////////////////////////////// n = 48 //////////////////////////////////////
        q0_real = Operand::load_unaligned(iq0_real + 24);
        q1_real = Operand::load_unaligned(iq1_real + 24);

        q0_imag = Operand::load_unaligned(iq0_imag + 24);
        q1_imag = Operand::load_unaligned(iq1_imag + 24);

        p0_real = xsimd::zip_lo(q0_real, q1_real);
        p1_real = xsimd::zip_hi(q0_real, q1_real);

        p0_imag = xsimd::zip_lo(q0_imag, q1_imag);
        p1_imag = xsimd::zip_hi(q0_imag, q1_imag);

        //intereave and drop

        p0_real.store_unaligned(out_real + 12);
        p1_real.store_unaligned(out_real + 44);

        p0_imag.store_unaligned(out_imag + 12);
        p1_imag.store_unaligned(out_imag + 44);

        /////////////////////////////// n = 56 //////////////////////////////////////

        q0_real = Operand::load_unaligned(iq0_real + 28);
        q1_real = Operand::load_unaligned(iq1_real + 28);

        q0_imag = Operand::load_unaligned(iq0_imag + 28);
        q1_imag = Operand::load_unaligned(iq1_imag + 28);

        p0_real = xsimd::zip_lo(q0_real, q1_real);
        p1_real = xsimd::zip_hi(q0_real, q1_real);

        p0_imag = xsimd::zip_lo(q0_imag, q1_imag);
        p1_imag = xsimd::zip_hi(q0_imag, q1_imag);

        //intereave and drop

        p0_real.store_unaligned(out_real + 28);
        p1_real.store_unaligned(out_real + 60);

        p0_imag.store_unaligned(out_imag + 28);
        p1_imag.store_unaligned(out_imag + 60);
    
}

void interleave_bitreversal(double* out_real, double* out_imag, double* in_real, double* in_imag, double* work_real, double* work_imag, std::size_t len, std::size_t* bit_reversed_indexes) {
    // On len 64, "unrolling" instead of using lookup table gives 15% performance boost, (save 1 to 2 ns)
    
    using Operand = xsimd::batch<double, xsimd::avx>;

        //         x = xsimd::batch<double, xsimd::avx>::load_unaligned(ptr);
        // }

        // static inline void store(double* ptr, const Value& x) {
        //     x.store_unaligned(ptr);
    auto quarter_len = len >> 2;
    auto half_len = len >> 1;
    auto three_quarter_len = quarter_len + half_len;

    // AAA
    auto iq0_real = in_real;
    auto iq1_real = in_real + quarter_len;
    auto iq2_real = in_real + half_len;
    auto iq3_real = in_real + three_quarter_len;

    auto iq0_imag = in_imag;
    auto iq1_imag = in_imag + quarter_len;
    auto iq2_imag = in_imag + half_len;
    auto iq3_imag = in_imag + three_quarter_len;

    auto oq0_real = work_real;
    auto oq1_real = work_real + 4;
    auto oq2_real = work_real + half_len;
    auto oq3_real = work_real + half_len + 4;

    auto oq0_imag = work_imag;
    auto oq1_imag = work_imag + 4;
    auto oq2_imag = work_imag + half_len;
    auto oq3_imag = work_imag + half_len + 4;
    
    for (
        std::size_t new_index = 0;
        new_index < quarter_len; 
        new_index+=4
    ) {
        Operand q0_real = Operand::load_unaligned(iq0_real);
        Operand q1_real = Operand::load_unaligned(iq1_real);
        Operand q2_real = Operand::load_unaligned(iq2_real);
        Operand q3_real = Operand::load_unaligned(iq3_real);

        Operand q0_imag = Operand::load_unaligned(iq0_imag);
        Operand q1_imag = Operand::load_unaligned(iq1_imag);
        Operand q2_imag = Operand::load_unaligned(iq2_imag);
        Operand q3_imag = Operand::load_unaligned(iq3_imag);

        Operand p0_real = xsimd::zip_lo(q0_real, q1_real);
        Operand p1_real = xsimd::zip_hi(q0_real, q1_real);
        Operand p2_real = xsimd::zip_lo(q2_real, q3_real);
        Operand p3_real = xsimd::zip_hi(q2_real, q3_real);

        Operand p0_imag = xsimd::zip_lo(q0_imag, q1_imag);
        Operand p1_imag = xsimd::zip_hi(q0_imag, q1_imag);
        Operand p2_imag = xsimd::zip_lo(q2_imag, q3_imag);
        Operand p3_imag = xsimd::zip_hi(q2_imag, q3_imag);

        p0_real.store_unaligned(oq0_real);
        p1_real.store_unaligned(oq1_real);
        p2_real.store_unaligned(oq2_real);
        p3_real.store_unaligned(oq3_real);

        p0_imag.store_unaligned(oq0_imag);
        p1_imag.store_unaligned(oq1_imag);
        p2_imag.store_unaligned(oq2_imag);
        p3_imag.store_unaligned(oq3_real);

        // This could be optimized too!!!!
        iq0_real+=4;
        iq1_real+=4;
        iq2_real+=4;
        iq3_real+=4;

        iq0_imag += 4;
        iq1_imag += 4;
        iq2_imag += 4;
        iq3_imag += 4;

        oq0_real += 8;
        oq1_real += 8;
        oq2_real += 8;
        oq3_real += 8;

        oq0_imag += 8;
        oq1_imag += 8;
        oq2_imag += 8;
        oq3_imag += 8;
    }

    iq0_real = work_real;
    iq1_real = work_real + half_len;

    iq0_imag = work_imag;
    iq1_imag = work_imag + half_len;
    
    double* o0_real;
    double* o1_real;

    double* o0_imag;
    double* o1_imag;

    for (
        std::size_t new_index = 0;
        new_index < quarter_len; 
        new_index+=2
    ) {
        auto old_index_0 = bit_reversed_indexes[new_index];
        auto old_index_1 = bit_reversed_indexes[new_index + 1];

        o0_real  = out_real + old_index_0;
        o1_real  = out_real + old_index_1;

        o0_imag  = out_imag + old_index_0;
        o1_imag  = out_imag + old_index_1;

        Operand q0_real = Operand::load_unaligned(iq0_real);
        Operand q1_real = Operand::load_unaligned(iq1_real);

        Operand q0_imag = Operand::load_unaligned(iq0_imag);
        Operand q1_imag = Operand::load_unaligned(iq1_imag);

        Operand p0_real = xsimd::zip_lo(q0_real, q1_real);
        Operand p1_real = xsimd::zip_hi(q0_real, q1_real);

        Operand p0_imag = xsimd::zip_lo(q0_imag, q1_imag);
        Operand p1_imag = xsimd::zip_hi(q0_imag, q1_imag);

        //intereave and drop

        p0_real.store_unaligned(o0_real);
        p1_real.store_unaligned(o1_real);

        p0_imag.store_unaligned(o0_imag);
        p1_imag.store_unaligned(o1_imag);


        iq0_real+=4;
        iq1_real+=4;

        iq0_imag += 4;
        iq1_imag += 4;
    }
}


void interleave_bitreversal_single_pass(double* out_real, double* out_imag, double* in_real, double* in_imag, std::size_t len, std::size_t* bit_reversed_indexes) {
    // On len 64, "unrolling" instead of using lookup table gives 15% performance boost, (save 1 to 2 ns)
    
    using Operand = xsimd::batch<double, xsimd::avx>;

        //         x = xsimd::batch<double, xsimd::avx>::load_unaligned(ptr);
        // }

        // static inline void store(double* ptr, const Value& x) {
        //     x.store_unaligned(ptr);
    auto quarter_len = len >> 2;
    auto half_len = len >> 1;
    auto three_quarter_len = quarter_len + half_len;

    // AAA
    auto iq0_real = in_real;
    auto iq1_real = in_real + quarter_len;
    auto iq2_real = in_real + half_len;
    auto iq3_real = in_real + three_quarter_len;

    auto iq0_imag = in_imag;
    auto iq1_imag = in_imag + quarter_len;
    auto iq2_imag = in_imag + half_len;
    auto iq3_imag = in_imag + three_quarter_len;

    for (
        std::size_t new_index = 0;
        new_index < quarter_len; 
        new_index+=4
    ) {
        auto old_index_0 = bit_reversed_indexes[new_index];
        auto old_index_1 = bit_reversed_indexes[new_index + 1];
        auto old_index_2 = bit_reversed_indexes[new_index + 2];
        auto old_index_3 = bit_reversed_indexes[new_index + 3];

        auto o0_real  = out_real + old_index_0;
        auto o1_real  = out_real + old_index_1;
        auto o2_real  = out_real + old_index_2;
        auto o3_real  = out_real + old_index_3;

        auto o0_imag  = out_imag + old_index_0;
        auto o1_imag  = out_imag + old_index_1;
        auto o2_imag  = out_imag + old_index_2;
        auto o3_imag  = out_imag + old_index_3;

        Operand q0_real = Operand::load_unaligned(iq0_real);
        Operand q1_real = Operand::load_unaligned(iq1_real);
        Operand q2_real = Operand::load_unaligned(iq2_real);
        Operand q3_real = Operand::load_unaligned(iq3_real);

        Operand q0_imag = Operand::load_unaligned(iq0_imag);
        Operand q1_imag = Operand::load_unaligned(iq1_imag);
        Operand q2_imag = Operand::load_unaligned(iq2_imag);
        Operand q3_imag = Operand::load_unaligned(iq3_imag);

        Operand p0_real = xsimd::zip_lo(q0_real, q1_real);
        Operand p1_real = xsimd::zip_hi(q0_real, q1_real);
        Operand p2_real = xsimd::zip_lo(q2_real, q3_real);
        Operand p3_real = xsimd::zip_hi(q2_real, q3_real);

        Operand p0_imag = xsimd::zip_lo(q0_imag, q1_imag);
        Operand p1_imag = xsimd::zip_hi(q0_imag, q1_imag);
        Operand p2_imag = xsimd::zip_lo(q2_imag, q3_imag);
        Operand p3_imag = xsimd::zip_hi(q2_imag, q3_imag);
    
        Operand r0_real = xsimd::zip_lo(p0_real, p2_real);
        Operand r1_real = xsimd::zip_hi(p0_real, p2_real);
        Operand r2_real = xsimd::zip_lo(p1_real, p3_real);
        Operand r3_real = xsimd::zip_hi(p1_real, p3_real);

        Operand r0_imag = xsimd::zip_lo(p0_imag, p2_imag);
        Operand r1_imag = xsimd::zip_hi(p0_imag, p2_imag);
        Operand r2_imag = xsimd::zip_lo(p1_imag, p3_imag);
        Operand r3_imag = xsimd::zip_hi(p1_imag, p3_imag);

        r0_real.store_unaligned(o0_real);
        r1_real.store_unaligned(o1_real);
        r2_real.store_unaligned(o2_real);
        r3_real.store_unaligned(o3_real);

        r0_imag.store_unaligned(o0_imag);
        r1_imag.store_unaligned(o1_imag);
        r2_imag.store_unaligned(o2_imag);
        r3_imag.store_unaligned(o3_imag);

        iq0_real+=4;
        iq1_real+=4;
        iq2_real+=4;
        iq3_real+=4;

        iq0_imag += 4;
        iq1_imag += 4;
        iq2_imag += 4;
        iq3_imag += 4;
    }
}


void interleave_bitreversal_single_pass_unrolled64(double* out_real, double* out_imag, double* in_real, double* in_imag) {
    using Operand = xsimd::batch<double, xsimd::avx>;
    Operand q0_real;
    Operand q1_real;
    Operand q2_real;
    Operand q3_real;

    Operand q0_imag;
    Operand q1_imag;
    Operand q2_imag;
    Operand q3_imag;

    Operand p0_real;
    Operand p1_real;
    Operand p2_real;
    Operand p3_real;

    Operand p0_imag;
    Operand p1_imag;
    Operand p2_imag;
    Operand p3_imag;

    Operand r0_real;
    Operand r1_real;
    Operand r2_real;
    Operand r3_real;

    Operand r0_imag;
    Operand r1_imag;
    Operand r2_imag;
    Operand r3_imag;

    q0_real = Operand::load_unaligned(in_real + 0);
q1_real = Operand::load_unaligned(in_real + 16);
q2_real = Operand::load_unaligned(in_real + 32);
q3_real = Operand::load_unaligned(in_real + 18);
q0_imag = Operand::load_unaligned(in_imag + 0);
q1_imag = Operand::load_unaligned(in_imag + 16);
q2_imag = Operand::load_unaligned(in_imag + 32);
q3_imag = Operand::load_unaligned(in_imag + 18);
p0_real = xsimd::zip_lo(q0_real, q1_real);
p1_real = xsimd::zip_hi(q0_real, q1_real);
p2_real = xsimd::zip_lo(q2_real, q3_real);
p3_real = xsimd::zip_hi(q2_real, q3_real);
p0_imag = xsimd::zip_lo(q0_imag, q1_imag);
p1_imag = xsimd::zip_hi(q0_imag, q1_imag);
p2_imag = xsimd::zip_lo(q2_imag, q3_imag);
p3_imag = xsimd::zip_hi(q2_imag, q3_imag);
r0_real = xsimd::zip_lo(p0_real, p2_real);
r1_real = xsimd::zip_hi(p0_real, p2_real);
r2_real = xsimd::zip_lo(p1_real, p3_real);
r3_real = xsimd::zip_hi(p1_real, p3_real);
r0_imag = xsimd::zip_lo(p0_imag, p2_imag);
r1_imag = xsimd::zip_hi(p0_imag, p2_imag);
r2_imag = xsimd::zip_lo(p1_imag, p3_imag);
r3_imag = xsimd::zip_hi(p1_imag, p3_imag);
r0_real.store_unaligned(out_real + 0);
r1_real.store_unaligned(out_real + 32);
r2_real.store_unaligned(out_real + 16);
r3_real.store_unaligned(out_real + 48);
r0_imag.store_unaligned(out_imag + 0);
r1_imag.store_unaligned(out_imag + 32);
r2_imag.store_unaligned(out_imag + 16);
r3_imag.store_unaligned(out_imag + 48);
q0_real = Operand::load_unaligned(in_real + 4);
q1_real = Operand::load_unaligned(in_real + 20);
q2_real = Operand::load_unaligned(in_real + 36);
q3_real = Operand::load_unaligned(in_real + 22);
q0_imag = Operand::load_unaligned(in_imag + 4);
q1_imag = Operand::load_unaligned(in_imag + 20);
q2_imag = Operand::load_unaligned(in_imag + 36);
q3_imag = Operand::load_unaligned(in_imag + 22);
p0_real = xsimd::zip_lo(q0_real, q1_real);
p1_real = xsimd::zip_hi(q0_real, q1_real);
p2_real = xsimd::zip_lo(q2_real, q3_real);
p3_real = xsimd::zip_hi(q2_real, q3_real);
p0_imag = xsimd::zip_lo(q0_imag, q1_imag);
p1_imag = xsimd::zip_hi(q0_imag, q1_imag);
p2_imag = xsimd::zip_lo(q2_imag, q3_imag);
p3_imag = xsimd::zip_hi(q2_imag, q3_imag);
r0_real = xsimd::zip_lo(p0_real, p2_real);
r1_real = xsimd::zip_hi(p0_real, p2_real);
r2_real = xsimd::zip_lo(p1_real, p3_real);
r3_real = xsimd::zip_hi(p1_real, p3_real);
r0_imag = xsimd::zip_lo(p0_imag, p2_imag);
r1_imag = xsimd::zip_hi(p0_imag, p2_imag);
r2_imag = xsimd::zip_lo(p1_imag, p3_imag);
r3_imag = xsimd::zip_hi(p1_imag, p3_imag);
r0_real.store_unaligned(out_real + 8);
r1_real.store_unaligned(out_real + 40);
r2_real.store_unaligned(out_real + 24);
r3_real.store_unaligned(out_real + 56);
r0_imag.store_unaligned(out_imag + 8);
r1_imag.store_unaligned(out_imag + 40);
r2_imag.store_unaligned(out_imag + 24);
r3_imag.store_unaligned(out_imag + 56);
q0_real = Operand::load_unaligned(in_real + 8);
q1_real = Operand::load_unaligned(in_real + 24);
q2_real = Operand::load_unaligned(in_real + 40);
q3_real = Operand::load_unaligned(in_real + 26);
q0_imag = Operand::load_unaligned(in_imag + 8);
q1_imag = Operand::load_unaligned(in_imag + 24);
q2_imag = Operand::load_unaligned(in_imag + 40);
q3_imag = Operand::load_unaligned(in_imag + 26);
p0_real = xsimd::zip_lo(q0_real, q1_real);
p1_real = xsimd::zip_hi(q0_real, q1_real);
p2_real = xsimd::zip_lo(q2_real, q3_real);
p3_real = xsimd::zip_hi(q2_real, q3_real);
p0_imag = xsimd::zip_lo(q0_imag, q1_imag);
p1_imag = xsimd::zip_hi(q0_imag, q1_imag);
p2_imag = xsimd::zip_lo(q2_imag, q3_imag);
p3_imag = xsimd::zip_hi(q2_imag, q3_imag);
r0_real = xsimd::zip_lo(p0_real, p2_real);
r1_real = xsimd::zip_hi(p0_real, p2_real);
r2_real = xsimd::zip_lo(p1_real, p3_real);
r3_real = xsimd::zip_hi(p1_real, p3_real);
r0_imag = xsimd::zip_lo(p0_imag, p2_imag);
r1_imag = xsimd::zip_hi(p0_imag, p2_imag);
r2_imag = xsimd::zip_lo(p1_imag, p3_imag);
r3_imag = xsimd::zip_hi(p1_imag, p3_imag);
r0_real.store_unaligned(out_real + 4);
r1_real.store_unaligned(out_real + 36);
r2_real.store_unaligned(out_real + 20);
r3_real.store_unaligned(out_real + 52);
r0_imag.store_unaligned(out_imag + 4);
r1_imag.store_unaligned(out_imag + 36);
r2_imag.store_unaligned(out_imag + 20);
r3_imag.store_unaligned(out_imag + 52);
q0_real = Operand::load_unaligned(in_real + 12);
q1_real = Operand::load_unaligned(in_real + 28);
q2_real = Operand::load_unaligned(in_real + 44);
q3_real = Operand::load_unaligned(in_real + 30);
q0_imag = Operand::load_unaligned(in_imag + 12);
q1_imag = Operand::load_unaligned(in_imag + 28);
q2_imag = Operand::load_unaligned(in_imag + 44);
q3_imag = Operand::load_unaligned(in_imag + 30);
p0_real = xsimd::zip_lo(q0_real, q1_real);
p1_real = xsimd::zip_hi(q0_real, q1_real);
p2_real = xsimd::zip_lo(q2_real, q3_real);
p3_real = xsimd::zip_hi(q2_real, q3_real);
p0_imag = xsimd::zip_lo(q0_imag, q1_imag);
p1_imag = xsimd::zip_hi(q0_imag, q1_imag);
p2_imag = xsimd::zip_lo(q2_imag, q3_imag);
p3_imag = xsimd::zip_hi(q2_imag, q3_imag);
r0_real = xsimd::zip_lo(p0_real, p2_real);
r1_real = xsimd::zip_hi(p0_real, p2_real);
r2_real = xsimd::zip_lo(p1_real, p3_real);
r3_real = xsimd::zip_hi(p1_real, p3_real);
r0_imag = xsimd::zip_lo(p0_imag, p2_imag);
r1_imag = xsimd::zip_hi(p0_imag, p2_imag);
r2_imag = xsimd::zip_lo(p1_imag, p3_imag);
r3_imag = xsimd::zip_hi(p1_imag, p3_imag);
r0_real.store_unaligned(out_real + 12);
r1_real.store_unaligned(out_real + 44);
r2_real.store_unaligned(out_real + 28);
r3_real.store_unaligned(out_real + 60);
r0_imag.store_unaligned(out_imag + 12);
r1_imag.store_unaligned(out_imag + 44);
r2_imag.store_unaligned(out_imag + 28);
r3_imag.store_unaligned(out_imag + 60);
}