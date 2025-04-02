//Analyze Clang vs GCC vs MSVC output




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

void inline interleave_bitreversal_single_pass(double* out_real, double* out_imag, double* in_real, double* in_imag, std::size_t len, const std::size_t* bit_reversed_indexes) {
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

void interleave_bitreversal_single_pass_by_16(double* out_real, double* out_imag, double* in_real, double* in_imag, std::size_t len, const std::size_t* bit_reversed_indexes_16) {
    for (std::size_t i = 0; i < len; i+=16) {
        interleave_bitreversal_single_pass(out_real + i, out_imag + i, in_real + i, in_imag + i, 16, bit_reversed_indexes_16);
    }
}

// TODO : Math common functions file
inline std::size_t int_log_2b(std::size_t n) {
    if (n == 0) {
        return 0;
    }
    std::size_t res = 0;
    while (n > 0) {
        res ++;
        n = n >> 1;
    }
    return res - 1;
}

template<std::size_t arr_size>
inline void copy_size_t_array(std::size_t* dest_arr, const std::size_t* src_arr) {
    dest_arr[arr_size - 1] = src_arr[arr_size - 1];
    copy_size_t_array<arr_size - 1>(dest_arr, src_arr); 
}

template<>
inline void copy_size_t_array<1>(std::size_t* dest_arr, const std::size_t* src_arr) {
    dest_arr[0] = src_arr[0];
}

template<typename Sample, typename OperandSpec, std::size_t arr_size>
struct LoadOperandsWithOffset{
    using Operand = typename OperandSpec::Value;
    static inline void call(Operand* dest_arr, const Sample* src_arr,  const std::size_t* offsets) {
        OperandSpec::load(src_arr + offsets[arr_size - 1], dest_arr[arr_size - 1]);
        LoadOperandsWithOffset<Sample, OperandSpec, arr_size - 1>::call(dest_arr, src_arr, offsets); 
    }
};

template<typename Sample, typename OperandSpec>
struct LoadOperandsWithOffset<Sample, OperandSpec, 1>{
    using Operand = typename OperandSpec::Value;
    static inline void call (Operand* dest_arr, const Sample* src_arr,  const std::size_t* offsets) {
        OperandSpec::load(src_arr + offsets[0], dest_arr[0]);
    }
};

template<typename Sample, typename OperandSpec, std::size_t arr_size>
struct StoreOperandsWithOffset{
    using Operand = typename OperandSpec::Value;
    static inline void call (Sample* dest_arr, const Operand* src_arr,  const std::size_t* offsets) {
        OperandSpec::store(dest_arr + offsets[arr_size - 1], src_arr[arr_size - 1]);
        StoreOperandsWithOffset<Sample, OperandSpec, arr_size - 1>::call(dest_arr, src_arr, offsets); 
    }
};

template<typename Sample, typename OperandSpec>
struct StoreOperandsWithOffset<Sample, OperandSpec, 1>{
    using Operand = typename OperandSpec::Value;
    static inline void call (Sample* dest_arr, const Operand* src_arr,  const std::size_t* offsets) {
        OperandSpec::store(dest_arr + offsets[0], src_arr[0]);
    }
};



template<typename Sample, typename OperandSpec, std::size_t arr_size>
struct Interleave{
    using Operand = typename OperandSpec::Value;
    static inline void call (Operand* arr_a, Operand* arr_b) {
        constexpr std::size_t half_n_samples_per_operand = sizeof(Operand) / sizeof(Sample) / 2;
        constexpr std::size_t half_arr_size = arr_size / 2;
        OperandSpec::interleave(arr_a[arr_size-2], arr_a[arr_size-1], arr_b[half_arr_size - 1], arr_b[half_arr_size - 1 + half_n_samples_per_operand]);
        Interleave<Sample, OperandSpec, arr_size - 2>::call(arr_a, arr_b);
    }
};

template<typename Sample, typename OperandSpec>
struct Interleave<Sample, OperandSpec, 2>{
    using Operand = typename OperandSpec::Value;
    static inline void call (Operand* arr_a, const Operand* arr_b) {
        constexpr std::size_t half_n_samples_per_operand = sizeof(Operand) / sizeof(Sample) / 2;
        OperandSpec::interleave(arr_a[0], arr_a[1], arr_b[0], arr_b[half_n_samples_per_operand]);
    }
};

template<typename Sample, typename OperandSpec>
struct Interleave<Sample, OperandSpec, 1>{
    using Operand = typename OperandSpec::Value;
    static inline void call(Operand* arr_a, const Operand* arr_b) {
        // Do Nothing
    }
};

template<typename Sample, typename OperandSpec, std::size_t arr_size, std::size_t interleave_factor>
struct ComputeTranspose{
    using Operand = typename OperandSpec::Value;
    static inline void call (Operand* arr_a, Operand* arr_b) {
        Interleave<Sample, OperandSpec, arr_size>::call(arr_a, arr_b);
        ComputeTranspose<Sample, OperandSpec, arr_size, interleave_factor / 2>::call(arr_b, arr_a);
    }
};

template<typename Sample, typename OperandSpec, std::size_t arr_size>
struct ComputeTranspose<Sample, OperandSpec, arr_size, 1>{
    using Operand = typename OperandSpec::Value;
    static inline void call (Operand* arr_a, Operand* arr_b) {
        Interleave<Sample, OperandSpec, arr_size>::call(arr_a, arr_b);
    }
};

template<typename OperandSpec, std::size_t alternate_factor>
struct AlternateOut{
    using Operand = typename OperandSpec::Value;
    static inline const Operand* call (const Operand* arr_a, const Operand* arr_b) {
        return AlternateOut<OperandSpec, alternate_factor / 2>::call(arr_b, arr_a);
    }
};

template<typename OperandSpec>
struct AlternateOut<OperandSpec, 1>{
    using Operand = typename OperandSpec::Value;
    static inline const Operand* call (const Operand* arr_a, const Operand* arr_b) {
        return arr_a;
    }
};

template<typename Sample, typename OperandSpec>
static inline void transpose_diagonal(double* real, double* imag, const std::size_t* indexes) {
    using Operand = typename OperandSpec::Value;
    constexpr std::size_t n_samples_per_operand = sizeof(Operand) / sizeof(Sample);

    Operand row_x[n_samples_per_operand];
    Operand row_y[n_samples_per_operand];
    Operand row_u[n_samples_per_operand];
    Operand row_v[n_samples_per_operand];

    std::size_t ind[n_samples_per_operand];
    copy_size_t_array<n_samples_per_operand>(ind, indexes);

    LoadOperandsWithOffset<Sample, OperandSpec, n_samples_per_operand>::call(row_x, real, ind);
    LoadOperandsWithOffset<Sample, OperandSpec, n_samples_per_operand>::call(row_u, imag, ind);
    ComputeTranspose<Sample, OperandSpec, n_samples_per_operand, n_samples_per_operand>::call(row_y, row_x);
    ComputeTranspose<Sample, OperandSpec, n_samples_per_operand, n_samples_per_operand>::call(row_v, row_u);
    StoreOperandsWithOffset<Sample, OperandSpec, n_samples_per_operand>::call(real, AlternateOut<OperandSpec, n_samples_per_operand>::call(row_x, row_y), ind);
    StoreOperandsWithOffset<Sample, OperandSpec, n_samples_per_operand>::call(imag, AlternateOut<OperandSpec, n_samples_per_operand>::call(row_u, row_v), ind);
}

template<typename Sample, typename OperandSpec>
static inline void transpose_off_diagonal(Sample* real, Sample* imag, const std::size_t* indexes) {
    using Operand = typename OperandSpec::Value;
    constexpr std::size_t n_samples_per_operand = sizeof(Operand) / sizeof(Sample);

    Operand row_x[n_samples_per_operand];
    Operand row_u[n_samples_per_operand];
    Operand row_y[n_samples_per_operand];
    Operand row_v[n_samples_per_operand];

    std::size_t ind_xy[n_samples_per_operand];
    std::size_t ind_uv[n_samples_per_operand];

    copy_size_t_array<n_samples_per_operand>(ind_xy, indexes);
    copy_size_t_array<n_samples_per_operand>(ind_uv, indexes + n_samples_per_operand);

    LoadOperandsWithOffset<Sample, OperandSpec, n_samples_per_operand>::call(row_x, real, ind_xy);
    LoadOperandsWithOffset<Sample, OperandSpec, n_samples_per_operand>::call(row_u, real, ind_uv);
    ComputeTranspose<Sample, OperandSpec, n_samples_per_operand, n_samples_per_operand>::call(row_y, row_x);
    ComputeTranspose<Sample, OperandSpec, n_samples_per_operand, n_samples_per_operand>::call(row_v, row_u);
    StoreOperandsWithOffset<Sample, OperandSpec, n_samples_per_operand>::call(real, AlternateOut<OperandSpec, n_samples_per_operand>::call(row_x, row_y), ind_uv);
    StoreOperandsWithOffset<Sample, OperandSpec, n_samples_per_operand>::call(real, AlternateOut<OperandSpec, n_samples_per_operand>::call(row_u, row_v), ind_xy);

    LoadOperandsWithOffset<Sample, OperandSpec, n_samples_per_operand>::call(row_x, imag, ind_xy);
    LoadOperandsWithOffset<Sample, OperandSpec, n_samples_per_operand>::call(row_u, imag, ind_uv);
    ComputeTranspose<Sample, OperandSpec, n_samples_per_operand, n_samples_per_operand>::call(row_y, row_x);
    ComputeTranspose<Sample, OperandSpec, n_samples_per_operand, n_samples_per_operand>::call(row_v, row_u);
    StoreOperandsWithOffset<Sample, OperandSpec, n_samples_per_operand>::call(imag, AlternateOut<OperandSpec, n_samples_per_operand>::call(row_x, row_y), ind_uv);
    StoreOperandsWithOffset<Sample, OperandSpec, n_samples_per_operand>::call(imag, AlternateOut<OperandSpec, n_samples_per_operand>::call(row_u, row_v), ind_xy);
}

std::vector<std::vector<std::vector<std::size_t>>> get_indexes_as_mats(std::size_t n_indexes) {
    std::size_t n_bits = int_log_2b(n_indexes);
    std::vector<std::vector<std::size_t>> mat;

    if (n_bits % 2 == 0) {
        // This is a square matrix
        std::size_t n_rows = 1 << (n_bits / 2);
        for (std::size_t row_id = 0; row_id < n_rows; ++row_id) {
            std::vector<std::size_t> row(n_rows, 0);
            for (std::size_t col_id = 0; col_id < n_rows; ++col_id) {
                row[col_id] = row_id * n_rows + col_id;
            }
            mat.push_back(row);
        }
        return {mat};  // Return a 2D vector in a vector (equivalent to Python list of lists)
    } else {
        // This is a rectangular matrix
        std::size_t n_rows = 1 << ((n_bits - 1) / 2);
        std::vector<std::vector<std::size_t>> mat_a, mat_b;
        for (std::size_t row_id = 0; row_id < n_rows; ++row_id) {
            std::vector<std::size_t> row_a(n_rows, 0), row_b(n_rows, 0);
            for (std::size_t col_id = 0; col_id < n_rows; ++col_id) {
                row_a[col_id] = row_id * 2 * n_rows + col_id;
                row_b[col_id] = row_id * 2 * n_rows + col_id + n_rows;
            }
            mat_a.push_back(row_a);
            mat_b.push_back(row_b);
        }
        return {mat_a, mat_b};  // Return two matrices in a vector
    }
}

std::vector<std::size_t> bit_reversed_indexesb(std::size_t n_indexes) {
    std::vector<std::size_t> bit_reversed_indexes_(n_indexes);
    std::size_t n_bits = int_log_2b(n_indexes);

    for (
        std::size_t id = 0;
        id < n_indexes;
        id++
    ) {
        auto work = id;
        std::size_t bit_reversed_id = 0;
        for (
            std::size_t bit_id = 0;
            bit_id < n_bits;
            bit_id++
        ) {
            bit_reversed_id = bit_reversed_id << 1;
            bit_reversed_id += work & 1;
            work = work >> 1;
        }
        bit_reversed_indexes_[id] = bit_reversed_id;
    }
    return bit_reversed_indexes_;
}


std::vector<std::vector<std::size_t>> bit_rev_permute_rows(const std::vector<std::vector<std::size_t>>& mat) {
    std::size_t n_rows = mat.size();
    std::vector<std::vector<std::size_t>> new_mat;

    std::vector<std::size_t> scrambled_rows = bit_reversed_indexesb(n_rows);

    for (std::size_t row_id = 0; row_id < n_rows; ++row_id) {
        new_mat.push_back(mat[scrambled_rows[row_id]]);
    }

    return new_mat;
}

std::vector<std::vector<std::size_t>> get_corner_mat(
    const std::vector<std::vector<std::size_t>>& mat, 
    bool bottom_row, 
    bool right_col
) {
    std::size_t new_n_rows = mat.size() / 2;

    std::vector<std::vector<std::size_t>> new_mat;

    for (std::size_t row_id = 0; row_id < new_n_rows; ++row_id) {
        // Determine the starting row based on bottom_row
        std::size_t old_row_index = new_n_rows * std::size_t(bottom_row) + row_id;
        const std::vector<std::size_t>& old_row = mat[old_row_index];
        
        std::vector<std::size_t> new_row;
        
        for (std::size_t col_id = 0; col_id < new_n_rows; ++col_id) {
            // Determine the starting column based on right_col
            new_row.push_back(old_row[new_n_rows * std::size_t(right_col) + col_id]);
        }
        
        new_mat.push_back(new_row);
    }

    return new_mat;
}

std::vector<std::vector<std::vector<std::size_t>>> transpose_off_diagonal_indexes(
    const std::vector<std::vector<std::size_t>>& mat_a, 
    const std::vector<std::vector<std::size_t>>& mat_b, 
    std::size_t base_size) 
{
    if (mat_a.size() == base_size) {
        std::vector<std::size_t> destinations_a, destinations_b;

        for (std::size_t row_id = 0; row_id < base_size; ++row_id) {
            destinations_a.push_back(mat_a[row_id][0]);
            destinations_b.push_back(mat_b[row_id][0]);
        }

        return {{destinations_a, destinations_b}};
    }

    std::vector<std::vector<std::vector<std::size_t>>> destinations;

    // Call transpose_off_diagonal_indexes recursively on different corners
    auto corner_a_00 = get_corner_mat(mat_a, 0, 0);
    auto corner_b_00 = get_corner_mat(mat_b, 0, 0);
    auto corner_a_01 = get_corner_mat(mat_a, 0, 1);
    auto corner_b_10 = get_corner_mat(mat_b, 1, 0);
    auto corner_a_10 = get_corner_mat(mat_a, 1, 0);
    auto corner_b_01 = get_corner_mat(mat_b, 0, 1);
    auto corner_a_11 = get_corner_mat(mat_a, 1, 1);
    auto corner_b_11 = get_corner_mat(mat_b, 1, 1);

    // Recurse on different corner combinations
    std::vector<std::vector<std::vector<std::size_t>>> part_1 = transpose_off_diagonal_indexes(corner_a_00, corner_b_00, base_size);
    std::vector<std::vector<std::vector<std::size_t>>> part_2 = transpose_off_diagonal_indexes(corner_a_01, corner_b_10, base_size);
    std::vector<std::vector<std::vector<std::size_t>>> part_3 = transpose_off_diagonal_indexes(corner_a_10, corner_b_01, base_size);
    std::vector<std::vector<std::vector<std::size_t>>> part_4 = transpose_off_diagonal_indexes(corner_a_11, corner_b_11, base_size);

    destinations.insert(destinations.end(), part_1.begin(), part_1.end());
    destinations.insert(destinations.end(), part_2.begin(), part_2.end());
    destinations.insert(destinations.end(), part_3.begin(), part_3.end());
    destinations.insert(destinations.end(), part_4.begin(), part_4.end());

    return destinations;
}

std::vector<std::vector<std::vector<std::size_t>>> transpose_diagonal_indexes(
    const std::vector<std::vector<std::size_t>>& mat, 
    std::size_t base_size) 
{
    if (mat.size() == base_size) {
        std::vector<std::size_t> destinations;

        for (std::size_t row_id = 0; row_id < base_size; ++row_id) {
            destinations.push_back(mat[row_id][0]);
        }

        return {{destinations, destinations}};
    }

    std::vector<std::vector<std::vector<std::size_t>>> destinations;

    // Recurse on the corners
    auto corner_00 = get_corner_mat(mat, 0, 0);
    auto corner_11 = get_corner_mat(mat, 1, 1);    
    auto corner_01 = get_corner_mat(mat, 0, 1);
    auto corner_10 = get_corner_mat(mat, 1, 0);


    std::vector<std::vector<std::vector<std::size_t>>> part_1 = transpose_diagonal_indexes(corner_00, base_size);
    std::vector<std::vector<std::vector<std::size_t>>> part_2 = transpose_off_diagonal_indexes(corner_01, corner_10, base_size);
    std::vector<std::vector<std::vector<std::size_t>>> part_3 = transpose_diagonal_indexes(corner_11, base_size);

    destinations.insert(destinations.end(), part_1.begin(), part_1.end());
    destinations.insert(destinations.end(), part_2.begin(), part_2.end());
    destinations.insert(destinations.end(), part_3.begin(), part_3.end());

    return destinations;
}

std::vector<std::size_t> get_off_diagonal_streak_lens(const std::vector<std::vector<std::vector<std::size_t>>>& transpose_pairs) {
    std::vector<std::size_t> streak_lens;
    std::size_t counter = 0;

    for (const auto& pair : transpose_pairs) {
        if (pair[0][0] == pair[1][0]) {
            if (counter != 0) {
                if (counter != 1) {
                    streak_lens.push_back(counter);
                }
                counter = 0;
            }
        } else {
            counter += 1;
        }
    }

    return streak_lens;
}

enum class BitRevPermPlanType {
    N_INDEXES_BASE_SIZE_SQR,
    N_INDEXES_2_BASE_SIZE_SQR,
    N_INDEXES_4_BASE_SIZE_SQR,
    N_INDEXES_8_BASE_SIZE_SQR,
    MAT_IS_SQUARE,
    MAT_IS_NONSQUARE
};

class BitRevPermPlan {
public:
    BitRevPermPlanType type;
    std::vector<std::size_t> plan_indexes;
    std::vector<std::size_t> off_diagonal_streak_lens;

    BitRevPermPlan() : type(BitRevPermPlanType::MAT_IS_SQUARE) {}
};

BitRevPermPlan get_bit_rev_perm_plan(std::size_t n_indexes, std::size_t base_size) {
    std::size_t n_bits = int_log_2b(n_indexes);
    bool mat_is_square = n_bits % 2 == 0;
    BitRevPermPlan plan;

    if (n_indexes == base_size * base_size) {
        plan.type = BitRevPermPlanType::N_INDEXES_BASE_SIZE_SQR;
    } else if (n_indexes == 2 * base_size * base_size) {
        plan.type = BitRevPermPlanType::N_INDEXES_2_BASE_SIZE_SQR;
    } else if (n_indexes == 4 * base_size * base_size) {
        plan.type = BitRevPermPlanType::N_INDEXES_4_BASE_SIZE_SQR;
    } else if (n_indexes == 8 * base_size * base_size) {
        plan.type = BitRevPermPlanType::N_INDEXES_8_BASE_SIZE_SQR;
    } else if (mat_is_square) {
        plan.type = BitRevPermPlanType::MAT_IS_SQUARE;
    } else {
        plan.type = BitRevPermPlanType::MAT_IS_NONSQUARE;
    }

    // Get matrices as an example, the actual implementation may vary
    std::vector<std::vector<std::vector<std::size_t>>> mats = get_indexes_as_mats(n_indexes);

    std::vector<std::vector<std::vector<std::vector<std::size_t>>>> pre_plans_indexes;
    for (auto& mat : mats) {
        mat = bit_rev_permute_rows(mat);
        pre_plans_indexes.push_back(transpose_diagonal_indexes(mat, base_size));
    }

    std::vector<std::size_t> off_diagonal_streak_lens = get_off_diagonal_streak_lens(pre_plans_indexes[0]);

    std::vector<std::size_t> plan_indexes;
    std::size_t transpose_pair_id = 0;

    // Compress diagonal
    for (auto& pre_plan : pre_plans_indexes) {
        for (std::size_t j = 0; j < base_size; ++j) {
            plan_indexes.push_back(pre_plan[transpose_pair_id][0][j]);
        }
    }
    
    ++transpose_pair_id;

    if (n_indexes == base_size * base_size || n_indexes == 2 * base_size * base_size) {
        plan.plan_indexes = plan_indexes;
        plan.off_diagonal_streak_lens = {};
        return plan;
    }

    for (auto& pre_plan : pre_plans_indexes) {
        for (std::size_t j = 0; j < base_size; ++j) {
            plan_indexes.push_back(pre_plan[transpose_pair_id][0][j]);
        }
        for (std::size_t j = 0; j < base_size; ++j) {
            plan_indexes.push_back(pre_plan[transpose_pair_id][1][j]);
        }
    }
    
    ++transpose_pair_id;

    for (auto& pre_plan : pre_plans_indexes) {
        for (std::size_t j = 0; j < base_size; ++j) {
            plan_indexes.push_back(pre_plan[transpose_pair_id][0][j]);
        }
    }

    ++transpose_pair_id;

    // Process off-diagonal streaks
    for (auto streak_len : off_diagonal_streak_lens) {
        for (std::size_t off_diagonal_id = 0; off_diagonal_id < streak_len; ++off_diagonal_id) {
            for (auto& pre_plan : pre_plans_indexes) {
                for (std::size_t j = 0; j < base_size; ++j) {
                    plan_indexes.push_back(pre_plan[transpose_pair_id][0][j]);
                }
                for (std::size_t j = 0; j < base_size; ++j) {
                    plan_indexes.push_back(pre_plan[transpose_pair_id][1][j]);
                }
            }
            
            ++transpose_pair_id;
        }

        

        for (auto& pre_plan : pre_plans_indexes) {
            for (std::size_t j = 0; j < base_size; ++j) {
                plan_indexes.push_back(pre_plan[transpose_pair_id][0][j]);
            }
        }
        ++transpose_pair_id;

        for (auto& pre_plan : pre_plans_indexes) {
            for (std::size_t j = 0; j < base_size; ++j) {
                plan_indexes.push_back(pre_plan[transpose_pair_id][0][j]);
            }
            for (std::size_t j = 0; j < base_size; ++j) {
                plan_indexes.push_back(pre_plan[transpose_pair_id][1][j]);
            }
        }
        ++transpose_pair_id;

        for (auto& pre_plan : pre_plans_indexes) {
            for (std::size_t j = 0; j < base_size; ++j) {
                plan_indexes.push_back(pre_plan[transpose_pair_id][0][j]);
            }
        }
        ++transpose_pair_id;
    }

    plan.plan_indexes = plan_indexes;
    plan.off_diagonal_streak_lens = off_diagonal_streak_lens;

    return plan;
}


void data_transpose_diagonal(std::size_t* data, std::size_t* indexes, std::size_t base_size) {
    for (std::size_t i = 0; i < base_size; ++i) {
        for (std::size_t j = i + 1; j < base_size; ++j) {
            // Swapping data at positions (indexes[i] + j) and (indexes[j] + i)
            std::size_t temp = data[indexes[i] + j];
            data[indexes[i] + j] = data[indexes[j] + i];
            data[indexes[j] + i] = temp;
        }
    }
}

void data_transpose_off_diagonal(std::size_t* data, std::size_t* indexes0, std::size_t* indexes1, std::size_t base_size) {
    for (std::size_t i = 0; i < base_size; ++i) {
        for (std::size_t j = 0; j < base_size; ++j) {
            // Swapping data at positions (indexes0[i] + j) and (indexes1[j] + i)
            std::size_t temp = data[indexes0[i] + j];
            data[indexes0[i] + j] = data[indexes1[j] + i];
            data[indexes1[j] + i] = temp;
        }
    }
}

std::vector<std::size_t> transpose_scrambled_indexes(std::size_t n_indexes, std::size_t base_size) {
    std::vector<std::size_t> indexes_vec;

    for (std::size_t i = 0; i < n_indexes; i++) {
        indexes_vec.push_back(i);
    }

    std::size_t* indexes = indexes_vec.data();

    BitRevPermPlan plan = get_bit_rev_perm_plan(n_indexes, base_size);
    BitRevPermPlanType plan_type = plan.type;
    std::vector<std::size_t>& plan_indexes = plan.plan_indexes;
    std::vector<std::size_t>& off_diagonal_streak_lens = plan.off_diagonal_streak_lens;

    std::size_t transpose_id = 0;

    // Handling different BitRevPermPlanTypes
    switch (plan_type){
    case BitRevPermPlanType::N_INDEXES_BASE_SIZE_SQR:
        data_transpose_diagonal(indexes, plan_indexes.data(), base_size);
        return indexes_vec;

    case BitRevPermPlanType::N_INDEXES_2_BASE_SIZE_SQR:
        data_transpose_diagonal(indexes, plan_indexes.data(), base_size);
        data_transpose_diagonal(indexes, plan_indexes.data() + base_size, base_size);
        return indexes_vec;

    case BitRevPermPlanType::N_INDEXES_4_BASE_SIZE_SQR:
        data_transpose_diagonal(indexes, plan_indexes.data(), base_size);
        data_transpose_off_diagonal(indexes, plan_indexes.data() + base_size, plan_indexes.data() + 2 * base_size, base_size);
        data_transpose_diagonal(indexes, plan_indexes.data() + 3 * base_size, base_size);
        return indexes_vec;

    case BitRevPermPlanType::N_INDEXES_8_BASE_SIZE_SQR:
        data_transpose_diagonal(indexes, plan_indexes.data(), base_size);
        data_transpose_diagonal(indexes, plan_indexes.data() + base_size, base_size);
        data_transpose_off_diagonal(indexes, plan_indexes.data() + 2 * base_size, plan_indexes.data() + 3 * base_size, base_size);
        data_transpose_off_diagonal(indexes, plan_indexes.data() + 4 * base_size, plan_indexes.data() + 5 * base_size, base_size);
        data_transpose_diagonal(indexes, plan_indexes.data() + 6 * base_size, base_size);
        data_transpose_diagonal(indexes, plan_indexes.data() + 7 * base_size, base_size);
        return indexes_vec;

    case BitRevPermPlanType::MAT_IS_SQUARE:
        data_transpose_diagonal(indexes, plan_indexes.data(), base_size);
        data_transpose_off_diagonal(indexes, plan_indexes.data() + base_size, plan_indexes.data() + 2 * base_size, base_size);
        data_transpose_diagonal(indexes, plan_indexes.data() + 3 * base_size, base_size);
        transpose_id = 4;

        for (std::size_t streak_len : off_diagonal_streak_lens) {
            for (std::size_t off_diagonal_id = 0; off_diagonal_id < streak_len; ++off_diagonal_id) {
                data_transpose_off_diagonal(indexes, plan_indexes.data() + transpose_id * base_size, plan_indexes.data() + (transpose_id + 1) * base_size, base_size);
                transpose_id += 2;
            }

            data_transpose_diagonal(indexes, plan_indexes.data() + transpose_id * base_size, base_size);
            data_transpose_off_diagonal(indexes, plan_indexes.data() + (transpose_id + 1) * base_size, plan_indexes.data() + (transpose_id + 2) * base_size, base_size);
            data_transpose_diagonal(indexes, plan_indexes.data() + (transpose_id + 3) * base_size, base_size);

            transpose_id += 4;
        }
        return indexes_vec;

    default:
        data_transpose_diagonal(indexes, plan_indexes.data(), base_size);
        data_transpose_diagonal(indexes, plan_indexes.data() + base_size, base_size);
        data_transpose_off_diagonal(indexes, plan_indexes.data() + 2 * base_size, plan_indexes.data() + 3 * base_size, base_size);
        data_transpose_off_diagonal(indexes, plan_indexes.data() + 4 * base_size, plan_indexes.data() + 5 * base_size, base_size);
        data_transpose_diagonal(indexes, plan_indexes.data() + 6 * base_size, base_size);
        data_transpose_diagonal(indexes, plan_indexes.data() + 7 * base_size, base_size);

        transpose_id = 8;

        for (std::size_t streak_len : off_diagonal_streak_lens) {
            for (std::size_t off_diagonal_id = 0; off_diagonal_id < streak_len; ++off_diagonal_id) {
                data_transpose_off_diagonal(indexes, plan_indexes.data() + transpose_id * base_size, plan_indexes.data() + (transpose_id + 1) * base_size, base_size);
                data_transpose_off_diagonal(indexes, plan_indexes.data() + (transpose_id + 2) * base_size, plan_indexes.data() + (transpose_id + 3) * base_size, base_size);
                transpose_id += 4;
            }

            data_transpose_diagonal(indexes, plan_indexes.data() + transpose_id * base_size, base_size);
            data_transpose_diagonal(indexes, plan_indexes.data() + (transpose_id + 1) * base_size, base_size);
            data_transpose_off_diagonal(indexes, plan_indexes.data() + (transpose_id + 2) * base_size, plan_indexes.data() + (transpose_id + 3) * base_size, base_size);
            data_transpose_off_diagonal(indexes, plan_indexes.data() + (transpose_id + 4) * base_size, plan_indexes.data() + (transpose_id + 5) * base_size, base_size);
            data_transpose_diagonal(indexes, plan_indexes.data() + (transpose_id + 6) * base_size, base_size);
            data_transpose_diagonal(indexes, plan_indexes.data() + (transpose_id + 7) * base_size, base_size);

            transpose_id += 8;
        }

        return indexes_vec;
    }
}

template<typename Sample, typename OperandSpec>
void cache_oblivious_bit_reversal_permutation(
    Sample* real, 
    Sample* imag,
    const BitRevPermPlan& plan
) {
    using Operand = typename OperandSpec::Value;
    constexpr size_t base_size = sizeof(Operand) / sizeof(Sample);
    BitRevPermPlanType plan_type = plan.type;
    const std::vector<std::size_t>& plan_indexes = plan.plan_indexes;
    const std::vector<std::size_t>& off_diagonal_streak_lens = plan.off_diagonal_streak_lens;
    auto plan_indexes_data = plan_indexes.data();
    
    std::size_t transpose_id = 0;

    switch (plan_type){
    case BitRevPermPlanType::N_INDEXES_BASE_SIZE_SQR:
        transpose_diagonal<Sample, OperandSpec>(real, imag, plan_indexes_data);
        return;

    case BitRevPermPlanType::N_INDEXES_2_BASE_SIZE_SQR:
        transpose_diagonal<Sample, OperandSpec>(real, imag, plan_indexes_data);
        transpose_diagonal<Sample, OperandSpec>(real, imag, plan_indexes_data + base_size);
        return;

    case BitRevPermPlanType::N_INDEXES_4_BASE_SIZE_SQR:
        transpose_diagonal<Sample, OperandSpec>(real, imag, plan_indexes_data);
        transpose_off_diagonal<Sample, OperandSpec>(real, imag, plan_indexes_data + base_size);
        transpose_diagonal<Sample, OperandSpec>(real, imag, plan_indexes_data + 3 * base_size);
        return;

    case BitRevPermPlanType::N_INDEXES_8_BASE_SIZE_SQR:
        transpose_diagonal<Sample, OperandSpec>(real, imag, plan_indexes_data);
        transpose_diagonal<Sample, OperandSpec>(real, imag, plan_indexes_data + base_size);
        transpose_off_diagonal<Sample, OperandSpec>(real, imag, plan_indexes_data + 2 * base_size);
        transpose_off_diagonal<Sample, OperandSpec>(real, imag, plan_indexes_data + 4 * base_size);
        transpose_diagonal<Sample, OperandSpec>(real, imag, plan_indexes_data + 6 * base_size);
        transpose_diagonal<Sample, OperandSpec>(real, imag, plan_indexes_data + 7 * base_size);
        return;

    case BitRevPermPlanType::MAT_IS_SQUARE:
        transpose_diagonal<Sample, OperandSpec>(real, imag, plan_indexes_data);
        transpose_off_diagonal<Sample, OperandSpec>(real, imag, plan_indexes_data + base_size);
        transpose_diagonal<Sample, OperandSpec>(real, imag, plan_indexes_data + 3 * base_size);
        transpose_id = 4;

        for (std::size_t streak_len : off_diagonal_streak_lens) {
            for (std::size_t off_diagonal_id = 0; off_diagonal_id < streak_len; ++off_diagonal_id) {
                transpose_off_diagonal<Sample, OperandSpec>(real, imag, plan_indexes_data + transpose_id * base_size);
                transpose_id += 2;
            }

            transpose_diagonal<Sample, OperandSpec>(real, imag, plan_indexes_data + transpose_id * base_size);
            transpose_off_diagonal<Sample, OperandSpec>(real, imag, plan_indexes_data + (transpose_id + 1) * base_size);
            transpose_diagonal<Sample, OperandSpec>(real, imag, plan_indexes_data + (transpose_id + 3) * base_size);

            transpose_id += 4;
        }
        return;

    default:
        transpose_diagonal<Sample, OperandSpec>(real, imag, plan_indexes_data);
        transpose_diagonal<Sample, OperandSpec>(real, imag, plan_indexes_data + base_size);
        transpose_off_diagonal<Sample, OperandSpec>(real, imag, plan_indexes_data + 2 * base_size);
        transpose_off_diagonal<Sample, OperandSpec>(real, imag, plan_indexes_data + 4 * base_size);
        transpose_diagonal<Sample, OperandSpec>(real, imag, plan_indexes_data + 6 * base_size);
        transpose_diagonal<Sample, OperandSpec>(real, imag, plan_indexes_data + 7 * base_size);

        transpose_id = 8;

        for (std::size_t streak_len : off_diagonal_streak_lens) {
            for (std::size_t off_diagonal_id = 0; off_diagonal_id < streak_len; ++off_diagonal_id) {
                transpose_off_diagonal<Sample, OperandSpec>(real, imag, plan_indexes_data + transpose_id * base_size);
                transpose_off_diagonal<Sample, OperandSpec>(real, imag, plan_indexes_data + (transpose_id + 2) * base_size);
                transpose_id += 4;
            }

            transpose_diagonal<Sample, OperandSpec>(real, imag, plan_indexes_data + transpose_id * base_size);
            transpose_diagonal<Sample, OperandSpec>(real, imag, plan_indexes_data + (transpose_id + 1) * base_size);
            transpose_off_diagonal<Sample, OperandSpec>(real, imag, plan_indexes_data + (transpose_id + 2) * base_size);
            transpose_off_diagonal<Sample, OperandSpec>(real, imag, plan_indexes_data + (transpose_id + 4) * base_size);
            transpose_diagonal<Sample, OperandSpec>(real, imag, plan_indexes_data + (transpose_id + 6) * base_size);
            transpose_diagonal<Sample, OperandSpec>(real, imag, plan_indexes_data + (transpose_id + 7) * base_size);

            transpose_id += 8;
        }
    }
}


// Function to convert BitRevPermPlanType to string for human-readable printing
std::string plan_type_to_string(BitRevPermPlanType type) {
    switch (type) {
        case BitRevPermPlanType::N_INDEXES_BASE_SIZE_SQR: return "N_INDEXES_BASE_SIZE_SQR";
        case BitRevPermPlanType::N_INDEXES_2_BASE_SIZE_SQR: return "N_INDEXES_2_BASE_SIZE_SQR";
        case BitRevPermPlanType::N_INDEXES_4_BASE_SIZE_SQR: return "N_INDEXES_4_BASE_SIZE_SQR";
        case BitRevPermPlanType::N_INDEXES_8_BASE_SIZE_SQR: return "N_INDEXES_8_BASE_SIZE_SQR";
        case BitRevPermPlanType::MAT_IS_SQUARE: return "MAT_IS_SQUARE";
        case BitRevPermPlanType::MAT_IS_NONSQUARE: return "MAT_IS_NONSQUARE";
        default: return "Unknown";
    }
}

// Function to print the contents of BitRevPermPlan in a human-readable format
void print_bit_rev_perm_plan(const BitRevPermPlan& plan) {
    // Print the plan type
    std::cout << "Plan Type: " << plan_type_to_string(plan.type) << std::endl;

    // Print plan_indexes
    std::cout << "Plan Indexes: ";
    for (const auto& idx : plan.plan_indexes) {
        std::cout << idx << " ";
    }
    std::cout << std::endl;

    // Print off_diagonal_streaks
    std::cout << "Off-Diagonal Streaks: ";
    for (const auto& streak : plan.off_diagonal_streak_lens) {
        std::cout << streak << " ";
    }
    std::cout << std::endl;


    std::cout << "---------------" << std::endl;
}

// Use to determine COBRA buffer size. 2^pgfft_brc_thresh should fit in 
// cache.
static inline std::size_t pgfft_brc_qb() {
    return 7;
}


template<typename Sample, typename OperandSpec>
void cobra(Sample* signal_real, Sample* signal_imag, Sample* transform_real, Sample* transform_imag, Sample* work_real, Sample* work_imag,
        const std::vector<std::size_t>& bit_reversed_indexes_, const std::vector<std::size_t>& bit_reversed_indexes_2_, size_t log_reversal_len_
        ) {
    using Operand = typename OperandSpec::Value;
    constexpr std::size_t k_N_SAMPLES_PER_OPERAND = sizeof(Operand) / sizeof(Sample);
    const std::size_t* rev_flex = bit_reversed_indexes_.data();
    const std::size_t* rev_fixed = bit_reversed_indexes_2_.data();

    const auto A_real = signal_real;
    const auto A_imag = signal_imag;
    auto B_real = transform_real;
    auto B_imag = transform_imag;

    std::size_t q = pgfft_brc_qb();
    
    for (
        std::size_t b = 0; 
        b < bit_reversed_indexes_.size(); 
        b++
    ) {
        std::size_t b1 = rev_flex[b]; 
        for (
            std::size_t a = 0; 
            a < bit_reversed_indexes_2_.size(); 
            a++
        ) {
            std::size_t a1 = rev_fixed[a]; 

            Sample* T_p_real = work_real + (a1 << q);
            Sample* T_p_imag = work_imag + (a1 << q);

            const Sample* A_p_real 
                = A_real 
                + (a << (log_reversal_len_ +q)) 
                + (b << q);
            
            const Sample* A_p_imag 
                = A_imag 
                + (a << (log_reversal_len_ +q)) 
                + (b << q);
                
            for (
                long c = 0; 
                c < bit_reversed_indexes_2_.size(); 
                c+=k_N_SAMPLES_PER_OPERAND
            ) {
                Operand store_real;
                Operand store_imag;
                OperandSpec::load(A_p_real + c, store_real);
                OperandSpec::load(A_p_imag + c, store_imag);
                OperandSpec::store(T_p_real + c, store_real);
                OperandSpec::store(T_p_imag + c, store_imag);
            }
        }

        for (
            long c = 0; 
            c < bit_reversed_indexes_2_.size(); 
            c++
        ) {
            long c1 = rev_fixed[c];

            Sample* B_p_real 
                = B_real 
                + (c1 << (log_reversal_len_ +q)) 
                + (b1 << q);
            
            Sample* B_p_imag 
                = B_imag 
                + (c1 << (log_reversal_len_ +q)) 
                + (b1 << q);

            Sample* T_p_real = work_real + c;
            Sample* T_p_imag = work_imag + c;
            
            for (
                long a1 = 0; 
                a1 < bit_reversed_indexes_2_.size(); 
                a1+=1
            ) { 
                B_p_real[a1] = T_p_real[a1 << q];
                B_p_imag[a1] = T_p_imag[a1 << q];
            }
        }
    }
}

void print_mats(std::vector<std::vector<std::vector<std::size_t>>> result) {
    for (const auto& mat : result) {
        for (const auto& row : mat) {
            for (size_t value : row) {
                std::cout << value << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "-------" << std::endl; 
        std::cout << std::endl;
    }
}

void print_mat(std::vector<std::vector<std::size_t>> mat) {
    for (const auto& row : mat) {
        for (int value : row) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}