//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2024        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  seq_bsr_matop.c --  Matrix operation functions for jx_BSRMatrix.
 *  Date: 2025/10/08
 */ 


#include "jx_bsr_mv.h"

/*-----------------------------------------------*/
/* compare on w, move v and blk_array */
/*-----------------------------------------------*/

void jx_block_qsort(JX_Int* v, JX_Complex* w, JX_Complex* blk_array, JX_Int block_size, JX_Int left, JX_Int right)
{
    JX_Int i, last;

    if (left >= right) {
        return;
    }

    jx_swap2(v, w, left, (left + right) / 2);
    jx_swap_blk(blk_array, block_size, left, (left + right) / 2);
    last = left;
    for (i = left + 1; i <= right; i++)
        if (jx_cabs(w[i]) > jx_cabs(w[left])) {
            jx_swap2(v, w, ++last, i);
            jx_swap_blk(blk_array, block_size, last, i);
        }
    jx_swap2(v, w, left, last);
    jx_swap_blk(blk_array, block_size, left, last);
    jx_block_qsort(v, w, blk_array, block_size, left, last - 1);
    jx_block_qsort(v, w, blk_array, block_size, last + 1, right);
}

void jx_swap_blk(JX_Complex* v, JX_Int block_size, JX_Int i, JX_Int j)
{
    JX_Int   bnnz = block_size * block_size;
    JX_Real* temp;

    temp = jx_CTAlloc(JX_Real, bnnz);

    /*temp = v[i];*/
    jx_BSRMatrixBlockCopyData(&v[i * bnnz], temp, 1.0, block_size);
    /*v[i] = v[j];*/
    jx_BSRMatrixBlockCopyData(&v[j * bnnz], &v[i * bnnz], 1.0, block_size);
    /* v[j] = temp; */
    jx_BSRMatrixBlockCopyData(temp, &v[j * bnnz], 1.0, block_size);

    jx_TFree(temp);
}

/*--------------------------------------------------------------------------
 * jx_BSRMatrixAdd:
 * adds two BSR Matrices A and B and returns a BSR Matrix C;
 * Note: The routine does not check for 0-elements which might be generated
 *       through cancellation of elements in A and B or already contained
         in A and B. To remove those, use jx_BSRMatrixDeleteZeros
 *--------------------------------------------------------------------------*/

jx_BSRMatrix* jx_BSRMatrixAdd(jx_BSRMatrix* A, jx_BSRMatrix* B)
{
    JX_Complex*   A_data  = jx_BSRMatrixData(A);
    JX_Int*       A_i     = jx_BSRMatrixI(A);
    JX_Int*       A_j     = jx_BSRMatrixJ(A);
    JX_Int        nrows_A = jx_BSRMatrixNumRows(A);
    JX_Int        ncols_A = jx_BSRMatrixNumCols(A);
    JX_Complex*   B_data  = jx_BSRMatrixData(B);
    JX_Int*       B_i     = jx_BSRMatrixI(B);
    JX_Int*       B_j     = jx_BSRMatrixJ(B);
    JX_Int        nrows_B = jx_BSRMatrixNumRows(B);
    JX_Int        ncols_B = jx_BSRMatrixNumCols(B);
    jx_BSRMatrix* C;
    JX_Complex*   C_data;
    JX_Int*       C_i;
    JX_Int*       C_j;

    JX_Int  block_size  = jx_BSRMatrixBlockSize(A);
    JX_Int  block_sizeB = jx_BSRMatrixBlockSize(B);
    JX_Int  ia, ib, ic, ii, jcol, num_nonzeros, bnnz;
    JX_Int  pos;
    JX_Int* marker;

    if (nrows_A != nrows_B || ncols_A != ncols_B) {
        jx_printf("Warning! incompatible matrix dimensions!\n");
        return NULL;
    }
    if (block_size != block_sizeB) {
        jx_printf("Warning! incompatible matrix block size!\n");
        return NULL;
    }

    bnnz   = block_size * block_size;
    marker = jx_CTAlloc(JX_Int, ncols_A);
    C_i    = jx_CTAlloc(JX_Int, nrows_A + 1);

    for (ia = 0; ia < ncols_A; ia++) {
        marker[ia] = -1;
    }

    num_nonzeros = 0;
    C_i[0]       = 0;
    for (ic = 0; ic < nrows_A; ic++) {
        for (ia = A_i[ic]; ia < A_i[ic + 1]; ia++) {
            jcol         = A_j[ia];
            marker[jcol] = ic;
            num_nonzeros++;
        }
        for (ib = B_i[ic]; ib < B_i[ic + 1]; ib++) {
            jcol = B_j[ib];
            if (marker[jcol] != ic) {
                marker[jcol] = ic;
                num_nonzeros++;
            }
        }
        C_i[ic + 1] = num_nonzeros;
    }

    C                    = jx_BSRMatrixCreate(block_size, nrows_A, ncols_A, num_nonzeros);
    jx_BSRMatrixI(C) = C_i;
    /* Manual allocation: Initialize frees and zeroes C_i, so skip it */
    C_j    = jx_CTAlloc(JX_Int, num_nonzeros);
    C_data = jx_CTAlloc(JX_Complex, num_nonzeros * bnnz);
    jx_BSRMatrixJ(C)    = C_j;
    jx_BSRMatrixData(C) = C_data;

    for (ia = 0; ia < ncols_A; ia++) {
        marker[ia] = -1;
    }

    pos = 0;
    for (ic = 0; ic < nrows_A; ic++) {
        for (ia = A_i[ic]; ia < A_i[ic + 1]; ia++) {
            jcol     = A_j[ia];
            C_j[pos] = jcol;
            for (ii = 0; ii < bnnz; ii++) {
                C_data[pos * bnnz + ii] = A_data[ia * bnnz + ii];
            }
            marker[jcol] = pos;
            pos++;
        }
        for (ib = B_i[ic]; ib < B_i[ic + 1]; ib++) {
            jcol = B_j[ib];
            if (marker[jcol] < C_i[ic]) {
                C_j[pos] = jcol;
                for (ii = 0; ii < bnnz; ii++) {
                    C_data[pos * bnnz + ii] = B_data[ib * bnnz + ii];
                }
                marker[jcol] = pos;
                pos++;
            } else {
                for (ii = 0; ii < bnnz; ii++) {
                    C_data[marker[jcol] * bnnz + ii] = B_data[ib * bnnz + ii];
                }
            }
        }
    }
    jx_TFree(marker);
    return C;
}

/*--------------------------------------------------------------------------
 * jx_BSRMatrixMultiply
 * multiplies two BSR Matrices A and B and returns a BSR Matrix C;
 * Note: The routine does not check for 0-elements which might be generated
 *       through cancellation of elements in A and B or already contained
         in A and B. To remove those, use jx_BSRMatrixDeleteZeros
 *--------------------------------------------------------------------------*/

jx_BSRMatrix* jx_BSRMatrixMultiply(jx_BSRMatrix* A, jx_BSRMatrix* B)
{
    JX_Complex*   A_data      = jx_BSRMatrixData(A);
    JX_Int*       A_i         = jx_BSRMatrixI(A);
    JX_Int*       A_j         = jx_BSRMatrixJ(A);
    JX_Int        nrows_A     = jx_BSRMatrixNumRows(A);
    JX_Int        ncols_A     = jx_BSRMatrixNumCols(A);
    JX_Int        block_size  = jx_BSRMatrixBlockSize(A);
    JX_Complex*   B_data      = jx_BSRMatrixData(B);
    JX_Int*       B_i         = jx_BSRMatrixI(B);
    JX_Int*       B_j         = jx_BSRMatrixJ(B);
    JX_Int        nrows_B     = jx_BSRMatrixNumRows(B);
    JX_Int        ncols_B     = jx_BSRMatrixNumCols(B);
    JX_Int        block_sizeB = jx_BSRMatrixBlockSize(B);
    jx_BSRMatrix* C;
    JX_Complex*   C_data;
    JX_Int*       C_i;
    JX_Int*       C_j;

    JX_Int      ia, ib, ic, ja, jb, num_nonzeros = 0, bnnz;
    JX_Int      row_start, counter;
    JX_Complex *a_entries, *b_entries, *c_entries, dzero = 0.0, done = 1.0;
    JX_Int*     B_marker;

    if (ncols_A != nrows_B) {
        jx_printf("Warning! incompatible matrix dimensions!\n");
        return NULL;
    }
    if (block_size != block_sizeB) {
        jx_printf("Warning! incompatible matrix block size!\n");
        return NULL;
    }

    bnnz     = block_size * block_size;
    B_marker = jx_CTAlloc(JX_Int, ncols_B);
    C_i      = jx_CTAlloc(JX_Int, nrows_A + 1);

    for (ib = 0; ib < ncols_B; ib++) {
        B_marker[ib] = -1;
    }

    for (ic = 0; ic < nrows_A; ic++) {
        for (ia = A_i[ic]; ia < A_i[ic + 1]; ia++) {
            ja = A_j[ia];
            for (ib = B_i[ja]; ib < B_i[ja + 1]; ib++) {
                jb = B_j[ib];
                if (B_marker[jb] != ic) {
                    B_marker[jb] = ic;
                    num_nonzeros++;
                }
            }
        }
        C_i[ic + 1] = num_nonzeros;
    }

    C                    = jx_BSRMatrixCreate(block_size, nrows_A, ncols_B, num_nonzeros);
    jx_BSRMatrixI(C) = C_i;
    /* Manual allocation: Initialize frees and zeroes C_i, so skip it */
    C_j    = jx_CTAlloc(JX_Int, num_nonzeros);
    C_data = jx_CTAlloc(JX_Complex, num_nonzeros * bnnz);
    jx_BSRMatrixJ(C)    = C_j;
    jx_BSRMatrixData(C) = C_data;

    for (ib = 0; ib < ncols_B; ib++) {
        B_marker[ib] = -1;
    }

    counter = 0;
    for (ic = 0; ic < nrows_A; ic++) {
        row_start = C_i[ic];
        for (ia = A_i[ic]; ia < A_i[ic + 1]; ia++) {
            ja        = A_j[ia];
            a_entries = &(A_data[ia * bnnz]);
            for (ib = B_i[ja]; ib < B_i[ja + 1]; ib++) {
                jb        = B_j[ib];
                b_entries = &(B_data[ib * bnnz]);
                if (B_marker[jb] < row_start) {
                    B_marker[jb]      = counter;
                    C_j[B_marker[jb]] = jb;
                    c_entries         = &(C_data[B_marker[jb] * bnnz]);
                    jx_BSRMatrixBlockMultAdd(a_entries, b_entries, dzero, c_entries, block_size);
                    counter++;
                } else {
                    c_entries = &(C_data[B_marker[jb] * bnnz]);
                    jx_BSRMatrixBlockMultAdd(a_entries, b_entries, done, c_entries, block_size);
                }
            }
        }
    }
    jx_TFree(B_marker);
    return C;
}

/*--------------------------------------------------------------------------
 * jx_BSRMatrixReorder:
 *
 * Reorders the column and data arrays of a square BSR matrix, such that the
 * first entry in each row is the diagonal one.
 *--------------------------------------------------------------------------*/

JX_Int jx_BSRMatrixReorder(jx_BSRMatrix* A)
{
    JX_Complex* A_data = jx_BSRMatrixData(A);
    JX_Int*     A_i    = jx_BSRMatrixI(A);
    JX_Int*     A_j    = jx_BSRMatrixJ(A);
    // JX_Int*     rownnz_A   = jx_CSRMatrixRownnz(A);
    // JX_Int      nnzrows_A  = jx_CSRMatrixNumRownnz(A);
    JX_Int num_rows_A = jx_BSRMatrixNumRows(A);
    JX_Int num_cols_A = jx_BSRMatrixNumCols(A);
    JX_Int block_size = jx_BSRMatrixBlockSize(A);

    JX_Int i, j;

    /* the matrix should be square */
    if (num_rows_A != num_cols_A) {
        return -1;
    }

#ifdef JX_USING_OPENMP
#pragma omp parallel for private(i, j)
#endif
    for (i = 0; i < num_rows_A; i++) {
        for (j = A_i[i]; j < A_i[i + 1]; j++) {
            if (A_j[j] == i) {
                if (j != A_i[i]) {
                    jx_swap(A_j, A_i[i], j);
                    jx_swap_blk(A_data, block_size, A_i[i], j);
                }
                break;
            }
        }
    }

    return jx_error_flag;
}
