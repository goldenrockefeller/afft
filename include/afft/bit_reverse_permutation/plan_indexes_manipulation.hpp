#ifndef AFFT_PLAN_INDEXES_MANIPULATION_HPP
#define AFFT_PLAN_INDEXES_MANIPULATION_HPP

#include <cstddef>
#include <vector>
#include "afft/common_math.hpp"

namespace afft
{
    namespace plan_indexes_manipulation
    {
        using common_math::int_log_2;

        std::vector<std::vector<std::vector<std::size_t>>> indexes_as_mats(std::size_t n_indexes)
        {
            std::size_t n_bits = int_log_2(n_indexes);
            std::vector<std::vector<std::size_t>> mat;

            if (n_bits % 2 == 0)
            {
                // This is a square matrix
                std::size_t n_rows = 1 << (n_bits / 2);
                for (std::size_t row_id = 0; row_id < n_rows; ++row_id)
                {
                    std::vector<std::size_t> row(n_rows, 0);
                    for (std::size_t col_id = 0; col_id < n_rows; ++col_id)
                    {
                        row[col_id] = row_id * n_rows + col_id;
                    }
                    mat.push_back(row);
                }
                return {mat}; // Return a 2D vector in a vector (equivalent to Python list of lists)
            }
            else
            {
                // This is a rectangular matrix
                std::size_t n_rows = 1 << ((n_bits - 1) / 2);
                std::vector<std::vector<std::size_t>> mat_a, mat_b;
                for (std::size_t row_id = 0; row_id < n_rows; ++row_id)
                {
                    std::vector<std::size_t> row_a(n_rows, 0), row_b(n_rows, 0);
                    for (std::size_t col_id = 0; col_id < n_rows; ++col_id)
                    {
                        row_a[col_id] = row_id * 2 * n_rows + col_id;
                        row_b[col_id] = row_id * 2 * n_rows + col_id + n_rows;
                    }
                    mat_a.push_back(row_a);
                    mat_b.push_back(row_b);
                }
                return {mat_a, mat_b}; // Return two matrices in a vector
            }
        }

        std::vector<std::size_t> bit_reversed_indexes(std::size_t n_indexes)
        {
            std::vector<std::size_t> bit_reversed_indexes_(n_indexes);
            std::size_t n_bits = int_log_2(n_indexes);

            for (
                std::size_t id = 0;
                id < n_indexes;
                id++)
            {
                auto work = id;
                std::size_t bit_reversed_id = 0;
                for (
                    std::size_t bit_id = 0;
                    bit_id < n_bits;
                    bit_id++)
                {
                    bit_reversed_id = bit_reversed_id << 1;
                    bit_reversed_id += work & 1;
                    work = work >> 1;
                }
                bit_reversed_indexes_[id] = bit_reversed_id;
            }
            return bit_reversed_indexes_;
        }

        std::vector<std::vector<std::size_t>> bit_rev_permute_rows(const std::vector<std::vector<std::size_t>> &mat)
        {
            std::size_t n_rows = mat.size();
            std::vector<std::vector<std::size_t>> new_mat;

            std::vector<std::size_t> scrambled_rows = bit_reversed_indexes(n_rows);

            for (std::size_t row_id = 0; row_id < n_rows; ++row_id)
            {
                new_mat.push_back(mat[scrambled_rows[row_id]]);
            }

            return new_mat;
        }

        std::vector<std::vector<std::size_t>> get_corner_mat(
            const std::vector<std::vector<std::size_t>> &mat,
            bool bottom_row,
            bool right_col)
        {
            std::size_t new_n_rows = mat.size() / 2;

            std::vector<std::vector<std::size_t>> new_mat;

            for (std::size_t row_id = 0; row_id < new_n_rows; ++row_id)
            {
                // Determine the starting row based on bottom_row
                std::size_t old_row_index = new_n_rows * std::size_t(bottom_row) + row_id;
                const std::vector<std::size_t> &old_row = mat[old_row_index];

                std::vector<std::size_t> new_row;

                for (std::size_t col_id = 0; col_id < new_n_rows; ++col_id)
                {
                    // Determine the starting column based on right_col
                    new_row.push_back(old_row[new_n_rows * std::size_t(right_col) + col_id]);
                }

                new_mat.push_back(new_row);
            }

            return new_mat;
        }

        std::vector<std::vector<std::vector<std::size_t>>> plan_transpose_off_diagonal_indexes(
            const std::vector<std::vector<std::size_t>> &mat_a,
            const std::vector<std::vector<std::size_t>> &mat_b,
            std::size_t base_size)
        {
            if (mat_a.size() == base_size)
            {
                std::vector<std::size_t> destinations_a, destinations_b;

                for (std::size_t row_id = 0; row_id < base_size; ++row_id)
                {
                    destinations_a.push_back(mat_a[row_id][0]);
                    destinations_b.push_back(mat_b[row_id][0]);
                }

                return {{destinations_a, destinations_b}};
            }

            std::vector<std::vector<std::vector<std::size_t>>> destinations;

            // Call plan_transpose_off_diagonal_indexes recursively on different corners
            auto corner_a_00 = get_corner_mat(mat_a, 0, 0);
            auto corner_b_00 = get_corner_mat(mat_b, 0, 0);
            auto corner_a_01 = get_corner_mat(mat_a, 0, 1);
            auto corner_b_10 = get_corner_mat(mat_b, 1, 0);
            auto corner_a_10 = get_corner_mat(mat_a, 1, 0);
            auto corner_b_01 = get_corner_mat(mat_b, 0, 1);
            auto corner_a_11 = get_corner_mat(mat_a, 1, 1);
            auto corner_b_11 = get_corner_mat(mat_b, 1, 1);

            // Recurse on different corner combinations
            std::vector<std::vector<std::vector<std::size_t>>> part_1 = plan_transpose_off_diagonal_indexes(corner_a_00, corner_b_00, base_size);
            std::vector<std::vector<std::vector<std::size_t>>> part_2 = plan_transpose_off_diagonal_indexes(corner_a_01, corner_b_10, base_size);
            std::vector<std::vector<std::vector<std::size_t>>> part_3 = plan_transpose_off_diagonal_indexes(corner_a_10, corner_b_01, base_size);
            std::vector<std::vector<std::vector<std::size_t>>> part_4 = plan_transpose_off_diagonal_indexes(corner_a_11, corner_b_11, base_size);

            destinations.insert(destinations.end(), part_1.begin(), part_1.end());
            destinations.insert(destinations.end(), part_2.begin(), part_2.end());
            destinations.insert(destinations.end(), part_3.begin(), part_3.end());
            destinations.insert(destinations.end(), part_4.begin(), part_4.end());

            return destinations;
        }

        std::vector<std::vector<std::vector<std::size_t>>> plan_transpose_diagonal_indexes(
            const std::vector<std::vector<std::size_t>> &mat,
            std::size_t base_size)
        {
            if (mat.size() == base_size)
            {
                std::vector<std::size_t> destinations;

                for (std::size_t row_id = 0; row_id < base_size; ++row_id)
                {
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

            std::vector<std::vector<std::vector<std::size_t>>> part_1 = plan_transpose_diagonal_indexes(corner_00, base_size);
            std::vector<std::vector<std::vector<std::size_t>>> part_2 = plan_transpose_off_diagonal_indexes(corner_01, corner_10, base_size);
            std::vector<std::vector<std::vector<std::size_t>>> part_3 = plan_transpose_diagonal_indexes(corner_11, base_size);

            destinations.insert(destinations.end(), part_1.begin(), part_1.end());
            destinations.insert(destinations.end(), part_2.begin(), part_2.end());
            destinations.insert(destinations.end(), part_3.begin(), part_3.end());

            return destinations;
        }

        std::pair<std::vector<std::size_t>, std::vector<std::size_t>> ordered_bit_rev_indexes(std::size_t n_indexes){
            auto mats = indexes_as_mats(n_indexes);

            std::vector<std::size_t> in_indexes;
            std::vector<std::size_t> out_indexes;

            for (auto mat : mats){
                mat = bit_rev_permute_rows(mat);
                auto indexes = plan_transpose_diagonal_indexes(mat, 1);
                for (auto pair_indexes : indexes) {
                    if (pair_indexes[0][0] == pair_indexes[1][0]) {
                        in_indexes.push_back(pair_indexes[0][0]);
                        out_indexes.push_back(pair_indexes[0][0]);
                    }
                    else
                    {
                        in_indexes.push_back(pair_indexes[1][0]);
                        out_indexes.push_back(pair_indexes[0][0]);
                        in_indexes.push_back(pair_indexes[0][0]);
                        out_indexes.push_back(pair_indexes[1][0]);
                    }
                }
            }
            return {in_indexes, out_indexes};
        }
    }
}

#endif