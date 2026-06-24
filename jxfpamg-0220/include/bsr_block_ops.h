//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2024        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  bsr_block_ops.h -- Header file for BSR block-level small matrix operations
 *  Date: 2026/04/29
 *  Created for extracting block operations from jx_BSRMatrix
 */

#ifndef BSR_BLOCK_OPS_HEADER
#define BSR_BLOCK_OPS_HEADER

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef JX_REAL_MIN
#define JX_REAL_MIN 1.0e-300
#endif

#ifndef JXF_Int
#define JXF_Int int
#endif

#ifndef JXF_Real
#define JXF_Real double
#endif

#ifndef JX_Complex
#define JX_Complex double
#endif

/*----------------------------------------------------------------*
 *             BSR Block-Level Small Matrix Operations              *
 *----------------------------------------------------------------*/

/*!
 * \brief Small matrix multiplication: C = A * B (block_size x block_size)
 * \param A Input matrix A (row-major, block_size x block_size)
 * \param B Input matrix B (row-major, block_size x block_size)
 * \param C Output matrix C (row-major, block_size x block_size)
 * \param block_size Size of the small square matrix
 */
void jxf_bsr_block_matmul(const JXF_Real* A, const JXF_Real* B, JXF_Real* C, JXF_Int block_size);

/*!
 * \brief Small matrix multiplication with accumulate: C = A * B + beta * C
 * \param A Input matrix A (row-major, block_size x block_size)
 * \param B Input matrix B (row-major, block_size x block_size)
 * \param beta Scalar multiplier for C
 * \param C Output matrix C (row-major, block_size x block_size)
 * \param block_size Size of the small square matrix
 */
void jxf_bsr_block_matmul_add(const JXF_Real* A, const JXF_Real* B, JXF_Real beta, JXF_Real* C, JXF_Int block_size);

/*!
 * \brief Inverse of 2x2 matrix (in-place)
 * \param a Pointer to 2x2 matrix in row-major order (4 elements)
 */
void jxf_bsr_block_inv_2x2(JXF_Real* a);

/*!
 * \brief Inverse of 3x3 matrix (in-place)
 * \param a Pointer to 3x3 matrix in row-major order (9 elements)
 */
void jxf_bsr_block_inv_3x3(JXF_Real* a);

/*!
 * \brief Inverse of 4x4 matrix (in-place)
 * \param a Pointer to 4x4 matrix in row-major order (16 elements)
 */
void jxf_bsr_block_inv_4x4(JXF_Real* a);

/*!
 * \brief Inverse of 5x5 matrix (in-place)
 * \param a Pointer to 5x5 matrix in row-major order (25 elements)
 */
void jxf_bsr_block_inv_5x5(JXF_Real* a);

/*!
 * \brief Inverse of n x n matrix using Gauss elimination (no pivoting, in-place)
 * \param a Pointer to n x n matrix in row-major order (n*n elements)
 * \param n Dimension of the matrix
 */
void jxf_bsr_block_inv_n(JXF_Real* a, JXF_Int n);

/*!
 * \brief Inverse of n x n matrix using Gauss elimination with pivoting (in-place)
 * \param a Pointer to n x n matrix in row-major order (n*n elements)
 * \param n Dimension of the matrix
 * \return 1 on success, 0 on failure
 */
JXF_Int jxf_bsr_block_inv_pivot(JXF_Real* a, JXF_Int n);

/*!
 * \brief General small matrix inverse (dispatches to specialized or generic)
 * \param a Pointer to n x n matrix in row-major order
 * \param n Dimension of the matrix
 * \return 1 on success, 0 on failure
 */
JXF_Int jxf_bsr_block_inv(JXF_Real* a, JXF_Int n);

/*!
 * \brief Copy block data: dst = alpha * src
 * \param src Source block data
 * \param dst Destination block data
 * \param alpha Scaling factor
 * \param block_size Size of the block (block_size x block_size)
 */
void jxf_bsr_block_copy(const JXF_Real* src, JXF_Real* dst, JXF_Real alpha, JXF_Int block_size);

/*!
 * \brief Compute determinant of a small matrix
 * \param a Pointer to n x n matrix in row-major order
 * \param n Dimension of the matrix
 * \return Determinant value
 */
JXF_Real jxf_bsr_block_det(const JXF_Real* a, JXF_Int n);

#endif /* BSR_BLOCK_OPS_HEADER */
