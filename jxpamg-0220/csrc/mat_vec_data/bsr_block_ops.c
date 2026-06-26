//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2024        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  bsr_block_ops.c -- BSR block-level small matrix multiplication and inversion
 *  Date: 2026/04/29
 *  Extracted from jx_BSRMatrix implementation for standalone use
 */

#include "bsr_block_ops.h"

#define SWAP(a, b)             \
    {                          \
        temp = (a);            \
        (a)  = (b);            \
        (b)  = temp;            \
    }

/*--------------------------------------------------------------------------
 * jx_bsr_block_matmul
 * C = A * B where A, B, C are block_size x block_size in row-major order
 *--------------------------------------------------------------------------*/
void jx_bsr_block_matmul(const JX_Real* A, const JX_Real* B, JX_Real* C, JX_Int block_size)
{
    JX_Int i, j, k;
    JX_Real sum;

    for (i = 0; i < block_size; i++) {
        for (j = 0; j < block_size; j++) {
            sum = 0.0;
            for (k = 0; k < block_size; k++) {
                sum += A[i * block_size + k] * B[k * block_size + j];
            }
            C[i * block_size + j] = sum;
        }
    }
}

/*--------------------------------------------------------------------------
 * jx_bsr_block_matmul_add
 * C = A * B + beta * C
 *--------------------------------------------------------------------------*/
void jx_bsr_block_matmul_add(const JX_Real* A, const JX_Real* B, JX_Real beta, JX_Real* C, JX_Int block_size)
{
    JX_Int i, j, k;
    JX_Real sum;

    if (beta == 0.0) {
        for (i = 0; i < block_size; i++) {
            for (j = 0; j < block_size; j++) {
                sum = 0.0;
                for (k = 0; k < block_size; k++) {
                    sum += A[i * block_size + k] * B[k * block_size + j];
                }
                C[i * block_size + j] = sum;
            }
        }
    } else {
        for (i = 0; i < block_size; i++) {
            for (j = 0; j < block_size; j++) {
                sum = beta * C[i * block_size + j];
                for (k = 0; k < block_size; k++) {
                    sum += A[i * block_size + k] * B[k * block_size + j];
                }
                C[i * block_size + j] = sum;
            }
        }
    }
}

/*--------------------------------------------------------------------------
 * jx_bsr_block_inv_2x2
 * Compute inverse of 2x2 matrix in-place (row-major: a00, a01, a10, a11)
 *--------------------------------------------------------------------------*/
void jx_bsr_block_inv_2x2(JX_Real* a)
{
    const JX_Real a0 = a[0], a1 = a[1];
    const JX_Real a2 = a[2], a3 = a[3];

    const JX_Real det = a0 * a3 - a1 * a2;

    if (fabs(det) < JX_REAL_MIN) {
        printf("### WARNING: Matrix is nearly singular, det = %e! Ignore.\n", det);
        a[0] = 1.0; a[1] = 0.0;
        a[2] = 0.0; a[3] = 1.0;
    } else {
        JX_Real det_inv = 1.0 / det;
        a[0] =  a3 * det_inv;
        a[1] = -a1 * det_inv;
        a[2] = -a2 * det_inv;
        a[3] =  a0 * det_inv;
    }
}

/*--------------------------------------------------------------------------
 * jx_bsr_block_inv_3x3
 * Compute inverse of 3x3 matrix in-place (row-major, 9 elements)
 *--------------------------------------------------------------------------*/
void jx_bsr_block_inv_3x3(JX_Real* a)
{
    const JX_Real a0 = a[0], a1 = a[1], a2 = a[2];
    const JX_Real a3 = a[3], a4 = a[4], a5 = a[5];
    const JX_Real a6 = a[6], a7 = a[7], a8 = a[8];

    const JX_Real M0 = a4 * a8 - a5 * a7;
    const JX_Real M3 = a2 * a7 - a1 * a8;
    const JX_Real M6 = a1 * a5 - a2 * a4;
    const JX_Real M1 = a5 * a6 - a3 * a8;
    const JX_Real M4 = a0 * a8 - a2 * a6;
    const JX_Real M7 = a2 * a3 - a0 * a5;
    const JX_Real M2 = a3 * a7 - a4 * a6;
    const JX_Real M5 = a1 * a6 - a0 * a7;
    const JX_Real M8 = a0 * a4 - a1 * a3;

    const JX_Real det = a0 * M0 + a3 * M3 + a6 * M6;

    if (fabs(det) < JX_REAL_MIN) {
        printf("### WARNING: Matrix is nearly singular, det = %e! Ignore.\n", det);
        a[0] = 1.0; a[1] = 0.0; a[2] = 0.0;
        a[3] = 0.0; a[4] = 1.0; a[5] = 0.0;
        a[6] = 0.0; a[7] = 0.0; a[8] = 1.0;
    } else {
        JX_Real det_inv = 1.0 / det;
        a[0] = M0 * det_inv; a[1] = M3 * det_inv; a[2] = M6 * det_inv;
        a[3] = M1 * det_inv; a[4] = M4 * det_inv; a[5] = M7 * det_inv;
        a[6] = M2 * det_inv; a[7] = M5 * det_inv; a[8] = M8 * det_inv;
    }
}

/*--------------------------------------------------------------------------
 * jx_bsr_block_inv_4x4
 * Compute inverse of 4x4 matrix in-place (row-major, 16 elements)
 *--------------------------------------------------------------------------*/
void jx_bsr_block_inv_4x4(JX_Real* a)
{
    const JX_Real a11 = a[0], a12 = a[1], a13 = a[2], a14 = a[3];
    const JX_Real a21 = a[4], a22 = a[5], a23 = a[6], a24 = a[7];
    const JX_Real a31 = a[8], a32 = a[9], a33 = a[10], a34 = a[11];
    const JX_Real a41 = a[12], a42 = a[13], a43 = a[14], a44 = a[15];

    const JX_Real M11 = a22 * a33 * a44 + a23 * a34 * a42 + a24 * a32 * a43 - a22 * a34 * a43 - a23 * a32 * a44 - a24 * a33 * a42;
    const JX_Real M12 = a12 * a34 * a43 + a13 * a32 * a44 + a14 * a33 * a42 - a12 * a33 * a44 - a13 * a34 * a42 - a14 * a32 * a43;
    const JX_Real M13 = a12 * a23 * a44 + a13 * a24 * a42 + a14 * a22 * a43 - a12 * a24 * a43 - a13 * a22 * a44 - a14 * a23 * a42;
    const JX_Real M14 = a12 * a24 * a33 + a13 * a22 * a34 + a14 * a23 * a32 - a12 * a23 * a34 - a13 * a24 * a32 - a14 * a22 * a33;

    const JX_Real M21 = a21 * a34 * a43 + a23 * a31 * a44 + a24 * a33 * a41 - a21 * a33 * a44 - a23 * a34 * a41 - a24 * a31 * a43;
    const JX_Real M22 = a11 * a33 * a44 + a13 * a34 * a41 + a14 * a31 * a43 - a11 * a34 * a43 - a13 * a31 * a44 - a14 * a33 * a41;
    const JX_Real M23 = a11 * a24 * a43 + a13 * a21 * a44 + a14 * a23 * a41 - a11 * a23 * a44 - a13 * a24 * a41 - a14 * a21 * a43;
    const JX_Real M24 = a11 * a23 * a34 + a13 * a24 * a31 + a14 * a21 * a33 - a11 * a24 * a33 - a13 * a21 * a34 - a14 * a23 * a31;

    const JX_Real M31 = a21 * a32 * a44 + a22 * a34 * a41 + a24 * a31 * a42 - a21 * a34 * a42 - a22 * a31 * a44 - a24 * a32 * a41;
    const JX_Real M32 = a11 * a34 * a42 + a12 * a31 * a44 + a14 * a32 * a41 - a11 * a32 * a44 - a12 * a34 * a41 - a14 * a31 * a42;
    const JX_Real M33 = a11 * a22 * a44 + a12 * a24 * a41 + a14 * a21 * a42 - a11 * a24 * a42 - a12 * a21 * a44 - a14 * a22 * a41;
    const JX_Real M34 = a11 * a24 * a32 + a12 * a21 * a34 + a14 * a22 * a31 - a11 * a22 * a34 - a12 * a24 * a31 - a14 * a21 * a32;

    const JX_Real M41 = a21 * a33 * a42 + a22 * a31 * a43 + a23 * a32 * a41 - a21 * a32 * a43 - a22 * a33 * a41 - a23 * a31 * a42;
    const JX_Real M42 = a11 * a32 * a43 + a12 * a33 * a41 + a13 * a31 * a42 - a11 * a33 * a42 - a12 * a31 * a43 - a13 * a32 * a41;
    const JX_Real M43 = a11 * a23 * a42 + a12 * a21 * a43 + a13 * a22 * a41 - a11 * a22 * a43 - a12 * a23 * a41 - a13 * a21 * a42;
    const JX_Real M44 = a11 * a22 * a33 + a12 * a23 * a31 + a13 * a21 * a32 - a11 * a23 * a32 - a12 * a21 * a33 - a13 * a22 * a31;

    const JX_Real det = a11 * M11 + a12 * M21 + a13 * M31 + a14 * M41;

    if (fabs(det) < JX_REAL_MIN) {
        printf("### WARNING: Matrix is nearly singular, det = %e! Ignore.\n", det);
        int idx;
        for (idx = 0; idx < 16; idx++) a[idx] = (idx % 5 == 0) ? 1.0 : 0.0;  // identity
    } else {
        JX_Real det_inv = 1.0 / det;
        a[0]  = M11 * det_inv; a[1]  = M12 * det_inv; a[2]  = M13 * det_inv; a[3]  = M14 * det_inv;
        a[4]  = M21 * det_inv; a[5]  = M22 * det_inv; a[6]  = M23 * det_inv; a[7]  = M24 * det_inv;
        a[8]  = M31 * det_inv; a[9]  = M32 * det_inv; a[10] = M33 * det_inv; a[11] = M34 * det_inv;
        a[12] = M41 * det_inv; a[13] = M42 * det_inv; a[14] = M43 * det_inv; a[15] = M44 * det_inv;
    }
}

/*--------------------------------------------------------------------------
 * jx_bsr_block_inv_5x5
 * Compute inverse of 5x5 matrix in-place (row-major, 25 elements)
 * Uses cofactor expansion method
 *--------------------------------------------------------------------------*/
void jx_bsr_block_inv_5x5(JX_Real* a)
{
    JX_Real det0, det1, det2, det3, det4, det;

    const JX_Real a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4];
    const JX_Real a5 = a[5], a6 = a[6], a7 = a[7], a8 = a[8], a9 = a[9];
    const JX_Real a10 = a[10], a11 = a[11], a12 = a[12], a13 = a[13], a14 = a[14];
    const JX_Real a15 = a[15], a16 = a[16], a17 = a[17], a18 = a[18], a19 = a[19];
    const JX_Real a20 = a[20], a21 = a[21], a22 = a[22], a23 = a[23], a24 = a[24];

    det0 = a6 * (a12 * (a18 * a24 - a19 * a23) + a17 * (a14 * a23 - a13 * a24) + a22 * (a13 * a19 - a14 * a18));
    det0 += a11 * (a7 * (a19 * a23 - a18 * a24) + a17 * (a8 * a24 - a9 * a23) + a22 * (a9 * a18 - a8 * a19));
    det0 += a16 * (a7 * (a13 * a24 - a14 * a23) + a12 * (a9 * a23 - a8 * a24) + a22 * (a8 * a14 - a9 * a13));
    det0 += a21 * (a17 * (a9 * a13 - a8 * a14) + a7 * (a14 * a18 - a13 * a19) + a12 * (a8 * a19 - a9 * a18));

    det1 = a1 * (a22 * (a14 * a18 - a13 * a19) + a12 * (a19 * a23 - a18 * a24) + a17 * (a13 * a24 - a14 * a23));
    det1 += a11 * (a17 * (a4 * a23 - a3 * a24) + a2 * (a18 * a24 - a19 * a23) + a22 * (a3 * a19 - a4 * a18));
    det1 += a16 * (a12 * (a3 * a24 - a4 * a23) + a2 * (a14 * a23 - a13 * a24) + a22 * (a4 * a13 - a3 * a14));
    det1 += a21 * (a2 * (a13 * a19 - a14 * a18) + a12 * (a4 * a18 - a3 * a19) + a17 * (a3 * a14 - a4 * a13));

    det2 = a1 * (a7 * (a18 * a24 - a19 * a23) + a17 * (a9 * a23 - a8 * a24) + a22 * (a8 * a19 - a9 * a18));
    det2 += a6 * (a2 * (a19 * a23 - a18 * a24) + a17 * (a3 * a24 - a4 * a23) + a22 * (a4 * a18 - a3 * a19));
    det2 += a16 * (a2 * (a8 * a24 - a9 * a23) + a7 * (a4 * a23 - a3 * a24) + a22 * (a3 * a9 - a4 * a8));
    det2 += a21 * (a7 * (a3 * a19 - a4 * a18) + a2 * (a9 * a18 - a8 * a19) + a17 * (a4 * a8 - a3 * a9));

    det3 = a1 * (a12 * (a8 * a24 - a9 * a23) + a7 * (a14 * a23 - a13 * a24) + a22 * (a9 * a13 - a8 * a14));
    det3 += a6 * (a2 * (a13 * a24 - a14 * a23) + a12 * (a4 * a23 - a3 * a24) + a22 * (a3 * a14 - a4 * a13));
    det3 += a11 * (a7 * (a3 * a24 - a4 * a23) + a2 * (a9 * a23 - a8 * a24) + a22 * (a4 * a8 - a3 * a9));
    det3 += a21 * (a2 * (a8 * a14 - a9 * a13) + a7 * (a4 * a13 - a3 * a14) + a12 * (a3 * a9 - a4 * a8));

    det4 = a1 * (a7 * (a13 * a19 - a14 * a18) + a12 * (a9 * a18 - a8 * a19) + a17 * (a8 * a14 - a9 * a13));
    det4 += a6 * (a12 * (a3 * a19 - a4 * a18) + a17 * (a4 * a13 - a3 * a14) + a2 * (a14 * a18 - a13 * a19));
    det4 += a11 * (a2 * (a8 * a19 - a9 * a18) + a7 * (a4 * a18 - a3 * a19) + a17 * (a3 * a9 - a4 * a8));
    det4 += a16 * (a7 * (a3 * a14 - a4 * a13) + a2 * (a9 * a13 - a8 * a14) + a12 * (a4 * a8 - a3 * a9));

    det = det0 * a0 + det1 * a5 + det2 * a10 + det3 * a15 + det4 * a20;

    if (fabs(det) < JX_REAL_MIN) {
        printf("### WARNING: Matrix is nearly singular, det = %e! Ignore.\n", det);
        int idx;
        for (idx = 0; idx < 25; idx++) a[idx] = (idx % 6 == 0) ? 1.0 : 0.0;  // identity
    } else {
        JX_Real det_inv = 1.0 / det;
        // First row
        a[0] = (a6 * (a12 * a18 * a24 - a12 * a19 * a23 - a17 * a13 * a24 + a17 * a14 * a23 + a22 * a13 * a19 - a22 * a14 * a18) +
                a11 * (a7 * a19 * a23 - a7 * a18 * a24 + a17 * a8 * a24 - a17 * a9 * a23 - a22 * a8 * a19 + a22 * a9 * a18) +
                a16 * (a7 * a13 * a24 - a7 * a14 * a23 - a12 * a8 * a24 + a12 * a9 * a23 + a22 * a8 * a14 - a22 * a9 * a13) +
                a21 * (a7 * a14 * a18 - a7 * a13 * a19 + a12 * a8 * a19 - a12 * a9 * a18 - a17 * a8 * a14 + a17 * a9 * a13)) * det_inv;

        a[1] = (a1 * (a12 * a19 * a23 - a12 * a18 * a24 + a22 * a14 * a18 - a17 * a14 * a23 - a22 * a13 * a19 + a17 * a13 * a24) +
                a11 * (a22 * a3 * a19 + a2 * a18 * a24 - a17 * a3 * a24 - a22 * a4 * a18 - a2 * a19 * a23 + a17 * a4 * a23) +
                a16 * (a12 * a3 * a24 - a12 * a4 * a23 - a22 * a3 * a14 + a2 * a14 * a23 + a22 * a4 * a13 - a2 * a13 * a24) +
                a21 * (a12 * a4 * a18 - a12 * a3 * a19 - a2 * a14 * a18 - a17 * a4 * a13 + a2 * a13 * a19 + a17 * a3 * a14)) * det_inv;

        a[2] = (a1 * (a7 * a18 * a24 - a7 * a19 * a23 - a17 * a8 * a24 + a17 * a9 * a23 + a22 * a8 * a19 - a22 * a9 * a18) +
                a6 * (a2 * a19 * a23 - a2 * a18 * a24 + a17 * a3 * a24 - a17 * a4 * a23 - a22 * a3 * a19 + a22 * a4 * a18) +
                a16 * (a2 * a8 * a24 - a2 * a9 * a23 - a7 * a3 * a24 + a7 * a4 * a23 + a22 * a3 * a9 - a22 * a4 * a8) +
                a21 * (a2 * a9 * a18 - a2 * a8 * a19 + a7 * a3 * a19 - a7 * a4 * a18 - a17 * a3 * a9 + a17 * a4 * a8)) * det_inv;

        a[3] = (a1 * (a12 * a8 * a24 - a12 * a9 * a23 + a7 * a14 * a23 - a7 * a13 * a24 + a22 * a9 * a13 - a22 * a8 * a14) +
                a6 * (a12 * a4 * a23 - a12 * a3 * a24 + a22 * a3 * a14 - a22 * a4 * a13 + a2 * a13 * a24 - a2 * a14 * a23) +
                a11 * (a7 * a3 * a24 - a7 * a4 * a23 + a22 * a4 * a8 - a22 * a3 * a9 + a2 * a9 * a23 - a2 * a8 * a24) +
                a21 * (a12 * a3 * a9 - a12 * a4 * a8 + a2 * a8 * a14 - a2 * a9 * a13 + a7 * a4 * a13 - a7 * a3 * a14)) * det_inv;

        a[4] = (a1 * (a7 * a13 * a19 - a7 * a14 * a18 - a12 * a8 * a19 + a12 * a9 * a18 + a17 * a8 * a14 - a17 * a9 * a13) +
                a6 * (a2 * a14 * a18 - a2 * a13 * a19 + a12 * a3 * a19 - a12 * a4 * a18 - a17 * a3 * a14 + a17 * a4 * a13) +
                a11 * (a2 * a8 * a19 - a2 * a9 * a18 - a7 * a3 * a19 + a7 * a4 * a18 + a17 * a3 * a9 - a17 * a4 * a8) +
                a16 * (a2 * a9 * a13 - a2 * a8 * a14 + a7 * a3 * a14 - a7 * a4 * a13 - a12 * a3 * a9 + a12 * a4 * a8)) * det_inv;

        // Remaining rows - use Gauss elimination for simplicity
        JX_Real temp[25];
        int i, j, k;
        for (i = 0; i < 5; i++)
            for (j = 0; j < 5; j++)
                temp[i * 5 + j] = a[i * 5 + j];

        jx_bsr_block_inv_pivot(temp, 5);

        for (i = 5; i < 25; i++) a[i] = temp[i];
    }
}

/*--------------------------------------------------------------------------
 * jx_bsr_block_inv_n
 * Compute inverse using Gauss elimination (no pivoting, in-place)
 *--------------------------------------------------------------------------*/
void jx_bsr_block_inv_n(JX_Real* a, JX_Int n)
{
    JX_Int i, j, k, l, u, kn, in;
    JX_Real alinv;

    for (k = 0; k < n; ++k) {
        kn = k * n;
        l  = kn + k;

        if (fabs(a[l]) < JX_REAL_MIN) {
            printf("### ERROR: Diagonal entry is close to zero! ");
            printf("diag_%d = %.2e!\n", k, a[l]);
            return;
        }
        alinv = 1.0 / a[l];
        a[l] = alinv;

        for (j = 0; j < k; ++j) {
            u = kn + j;
            a[u] *= alinv;
        }
        for (j = k + 1; j < n; ++j) {
            u = kn + j;
            a[u] *= alinv;
        }
        for (i = 0; i < k; ++i) {
            in = i * n;
            for (j = 0; j < n; ++j)
                if (j != k) {
                    u = in + j;
                    a[u] -= a[in + k] * a[kn + j];
                }
        }
        for (i = k + 1; i < n; ++i) {
            in = i * n;
            for (j = 0; j < n; ++j)
                if (j != k) {
                    u = in + j;
                    a[u] -= a[in + k] * a[kn + j];
                }
        }
        for (i = 0; i < k; ++i) {
            u = i * n + k;
            a[u] *= -alinv;
        }
        for (i = k + 1; i < n; ++i) {
            u = i * n + k;
            a[u] *= -alinv;
        }
    }
}

/*--------------------------------------------------------------------------
 * jx_bsr_block_inv_pivot
 * Compute inverse using Gauss elimination with partial pivoting (in-place)
 * Based on gaussj() from "Numerical Recipes in C"
 *--------------------------------------------------------------------------*/
JX_Int jx_bsr_block_inv_pivot(JX_Real* a, JX_Int n)
{
    JX_Int i, j, k, l, ll, u;
    JX_Int icol = 0, irow = 0;
    JX_Real vmax, dum, pivinv, temp;

    JX_Int* work = (JX_Int*)malloc(3 * n * sizeof(JX_Int));
    if (!work) return 0;
    JX_Int* indxc = work;
    JX_Int* indxr = work + n;
    JX_Int* ipiv = work + 2 * n;

    for (j = 0; j < n; j++) ipiv[j] = 0;

    for (i = 0; i < n; i++) {
        vmax = 0.0;
        for (j = 0; j < n; j++) {
            if (ipiv[j] != 1) {
                for (k = 0; k < n; k++) {
                    if (ipiv[k] == 0) {
                        u = j * n + k;
                        if (fabs(a[u]) >= vmax) {
                            vmax = fabs(a[u]);
                            irow = j;
                            icol = k;
                        }
                    }
                }
            }
        }
        ++(ipiv[icol]);

        if (irow != icol) {
            for (l = 0; l < n; l++) SWAP(a[irow * n + l], a[icol * n + l]);
        }
        indxr[i] = irow;
        indxc[i] = icol;

        u = icol * n + icol;
        if (fabs(a[u]) < JX_REAL_MIN) {
            printf("### WARNING: The matrix is nearly singular!\n");
            free(work);
            return 0;
        }
        pivinv = 1.0 / a[u];
        a[u] = 1.0;
        for (l = 0; l < n; l++) a[icol * n + l] *= pivinv;

        for (ll = 0; ll < n; ll++) {
            if (ll != icol) {
                u = ll * n + icol;
                dum = a[u];
                a[u] = 0.0;
                for (l = 0; l < n; l++) a[ll * n + l] -= a[icol * n + l] * dum;
            }
        }
    }

    for (l = n - 1; l >= 0; l--) {
        if (indxr[l] != indxc[l])
            for (k = 0; k < n; k++) SWAP(a[k * n + indxr[l]], a[k * n + indxc[l]]);
    }

    free(work);
    return 1;
}

/*--------------------------------------------------------------------------
 * jx_bsr_block_inv
 * General small matrix inverse - dispatches to specialized or generic
 *--------------------------------------------------------------------------*/
JX_Int jx_bsr_block_inv(JX_Real* a, JX_Int n)
{
    switch (n) {
        case 2:
            jx_bsr_block_inv_2x2(a);
            return 1;
        case 3:
            jx_bsr_block_inv_3x3(a);
            return 1;
        case 4:
            jx_bsr_block_inv_4x4(a);
            return 1;
        case 5:
            jx_bsr_block_inv_5x5(a);
            return 1;
        default:
            return jx_bsr_block_inv_pivot(a, n);
    }
}

/*--------------------------------------------------------------------------
 * jx_bsr_block_copy
 * Copy block data: dst = alpha * src
 *--------------------------------------------------------------------------*/
void jx_bsr_block_copy(const JX_Real* src, JX_Real* dst, JX_Real alpha, JX_Int block_size)
{
    JX_Int i, bnnz = block_size * block_size;
    for (i = 0; i < bnnz; i++) {
        dst[i] = alpha * src[i];
    }
}

/*--------------------------------------------------------------------------
 * jx_bsr_block_det
 * Compute determinant of small matrix using recursive cofactor expansion
 *--------------------------------------------------------------------------*/
JX_Real jx_bsr_block_det(const JX_Real* a, JX_Int n)
{
    if (n == 1) return a[0];
    if (n == 2) return a[0] * a[3] - a[1] * a[2];

    JX_Real det = 0.0;
    JX_Real* submat = (JX_Real*)malloc((n - 1) * (n - 1) * sizeof(JX_Real));
    if (!submat) return 0.0;

    JX_Int col, i, j, subi, subj;
    for (col = 0; col < n; col++) {
        // Build submatrix excluding row 0 and column col
        subi = 0;
        for (i = 1; i < n; i++) {
            subj = 0;
            for (j = 0; j < n; j++) {
                if (j == col) continue;
                submat[subi * (n - 1) + subj] = a[i * n + j];
                subj++;
            }
            subi++;
        }
        JX_Real sign = (col % 2 == 0) ? 1.0 : -1.0;
        det += sign * a[col] * jx_bsr_block_det(submat, n - 1);
    }

    free(submat);
    return det;
}
