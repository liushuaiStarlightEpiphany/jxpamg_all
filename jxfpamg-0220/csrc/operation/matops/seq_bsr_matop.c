//========================================================================//
//  JXFPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2024        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  seq_bsr_matop.c --  Matrix operation functions for jxf_BSRMatrix.
 *  Date: 2025/10/08
 */ 


#include "jxf_bsr_mv.h"

/*-----------------------------------------------*/
/* compare on w, move v and blk_array */
/*-----------------------------------------------*/

void jxf_block_qsort(JXF_Int* v, JXF_Complex* w, JXF_Complex* blk_array, JXF_Int block_size, JXF_Int left, JXF_Int right)
{
    JXF_Int i, last;

    if (left >= right) {
        return;
    }

    jxf_swap2(v, w, left, (left + right) / 2);
    jxf_swap_blk(blk_array, block_size, left, (left + right) / 2);
    last = left;
    for (i = left + 1; i <= right; i++)
        if (jxf_cabs(w[i]) > jxf_cabs(w[left])) {
            jxf_swap2(v, w, ++last, i);
            jxf_swap_blk(blk_array, block_size, last, i);
        }
    jxf_swap2(v, w, left, last);
    jxf_swap_blk(blk_array, block_size, left, last);
    jxf_block_qsort(v, w, blk_array, block_size, left, last - 1);
    jxf_block_qsort(v, w, blk_array, block_size, last + 1, right);
}

void jxf_swap_blk(JXF_Complex* v, JXF_Int block_size, JXF_Int i, JXF_Int j)
{
    JXF_Int   bnnz = block_size * block_size;
    JXF_Real* temp;

    temp = jxf_CTAlloc(JXF_Real, bnnz);

    /*temp = v[i];*/
    jxf_BSRMatrixBlockCopyData(&v[i * bnnz], temp, 1.0, block_size);
    /*v[i] = v[j];*/
    jxf_BSRMatrixBlockCopyData(&v[j * bnnz], &v[i * bnnz], 1.0, block_size);
    /* v[j] = temp; */
    jxf_BSRMatrixBlockCopyData(temp, &v[j * bnnz], 1.0, block_size);

    jxf_TFree(temp);
}

/*--------------------------------------------------------------------------
 * jxf_BSRMatrixAdd:
 * adds two BSR Matrices A and B and returns a BSR Matrix C;
 * Note: The routine does not check for 0-elements which might be generated
 *       through cancellation of elements in A and B or already contained
         in A and B. To remove those, use jxf_BSRMatrixDeleteZeros
 *--------------------------------------------------------------------------*/

jxf_BSRMatrix* jxf_BSRMatrixAdd(jxf_BSRMatrix* A, jxf_BSRMatrix* B)
{
    JXF_Complex*   A_data  = jxf_BSRMatrixData(A);
    JXF_Int*       A_i     = jxf_BSRMatrixI(A);
    JXF_Int*       A_j     = jxf_BSRMatrixJ(A);
    JXF_Int        nrows_A = jxf_BSRMatrixNumRows(A);
    JXF_Int        ncols_A = jxf_BSRMatrixNumCols(A);
    JXF_Complex*   B_data  = jxf_BSRMatrixData(B);
    JXF_Int*       B_i     = jxf_BSRMatrixI(B);
    JXF_Int*       B_j     = jxf_BSRMatrixJ(B);
    JXF_Int        nrows_B = jxf_BSRMatrixNumRows(B);
    JXF_Int        ncols_B = jxf_BSRMatrixNumCols(B);
    jxf_BSRMatrix* C;
    JXF_Complex*   C_data;
    JXF_Int*       C_i;
    JXF_Int*       C_j;

    JXF_Int  block_size  = jxf_BSRMatrixBlockSize(A);
    JXF_Int  block_sizeB = jxf_BSRMatrixBlockSize(B);
    JXF_Int  ia, ib, ic, ii, jcol, num_nonzeros, bnnz;
    JXF_Int  pos;
    JXF_Int* marker;

    if (nrows_A != nrows_B || ncols_A != ncols_B) {
        jxf_printf("Warning! incompatible matrix dimensions!\n");
        return NULL;
    }
    if (block_size != block_sizeB) {
        jxf_printf("Warning! incompatible matrix block size!\n");
        return NULL;
    }

    bnnz   = block_size * block_size;
    marker = jxf_CTAlloc(JXF_Int, ncols_A);
    C_i    = jxf_CTAlloc(JXF_Int, nrows_A + 1);

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

    C                    = jxf_BSRMatrixCreate(block_size, nrows_A, ncols_A, num_nonzeros);
    jxf_BSRMatrixI(C) = C_i;
    jxf_BSRMatrixInitialize(C);
    C_j    = jxf_BSRMatrixJ(C);
    C_data = jxf_BSRMatrixData(C);

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
    jxf_TFree(marker);
    return C;
}

/*--------------------------------------------------------------------------
 * jxf_BSRMatrixMultiply
 * multiplies two BSR Matrices A and B and returns a BSR Matrix C;
 * Note: The routine does not check for 0-elements which might be generated
 *       through cancellation of elements in A and B or already contained
         in A and B. To remove those, use jxf_BSRMatrixDeleteZeros
 *--------------------------------------------------------------------------*/

jxf_BSRMatrix* jxf_BSRMatrixMultiply(jxf_BSRMatrix* A, jxf_BSRMatrix* B)
{
    JXF_Complex*   A_data      = jxf_BSRMatrixData(A);
    JXF_Int*       A_i         = jxf_BSRMatrixI(A);
    JXF_Int*       A_j         = jxf_BSRMatrixJ(A);
    JXF_Int        nrows_A     = jxf_BSRMatrixNumRows(A);
    JXF_Int        ncols_A     = jxf_BSRMatrixNumCols(A);
    JXF_Int        block_size  = jxf_BSRMatrixBlockSize(A);
    JXF_Complex*   B_data      = jxf_BSRMatrixData(B);
    JXF_Int*       B_i         = jxf_BSRMatrixI(B);
    JXF_Int*       B_j         = jxf_BSRMatrixJ(B);
    JXF_Int        nrows_B     = jxf_BSRMatrixNumRows(B);
    JXF_Int        ncols_B     = jxf_BSRMatrixNumCols(B);
    JXF_Int        block_sizeB = jxf_BSRMatrixBlockSize(B);
    jxf_BSRMatrix* C;
    JXF_Complex*   C_data;
    JXF_Int*       C_i;
    JXF_Int*       C_j;

    JXF_Int      ia, ib, ic, ja, jb, num_nonzeros = 0, bnnz;
    JXF_Int      row_start, counter;
    JXF_Complex *a_entries, *b_entries, *c_entries, dzero = 0.0, done = 1.0;
    JXF_Int*     B_marker;

    if (ncols_A != nrows_B) {
        jxf_printf("Warning! incompatible matrix dimensions!\n");
        return NULL;
    }
    if (block_size != block_sizeB) {
        jxf_printf("Warning! incompatible matrix block size!\n");
        return NULL;
    }

    bnnz     = block_size * block_size;
    B_marker = jxf_CTAlloc(JXF_Int, ncols_B);
    C_i      = jxf_CTAlloc(JXF_Int, nrows_A + 1);

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

    C                    = jxf_BSRMatrixCreate(block_size, nrows_A, ncols_B, num_nonzeros);
    jxf_BSRMatrixI(C) = C_i;
    jxf_BSRMatrixInitialize(C);
    C_j    = jxf_BSRMatrixJ(C);
    C_data = jxf_BSRMatrixData(C);

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
                    jxf_BSRMatrixBlockMultAdd(a_entries, b_entries, dzero, c_entries, block_size);
                    counter++;
                } else {
                    c_entries = &(C_data[B_marker[jb] * bnnz]);
                    jxf_BSRMatrixBlockMultAdd(a_entries, b_entries, done, c_entries, block_size);
                }
            }
        }
    }
    jxf_TFree(B_marker);
    return C;
}

/*--------------------------------------------------------------------------
 * jxf_BSRMatrixReorder:
 *
 * Reorders the column and data arrays of a square BSR matrix, such that the
 * first entry in each row is the diagonal one.
 *--------------------------------------------------------------------------*/

JXF_Int jxf_BSRMatrixReorder(jxf_BSRMatrix* A)
{
    JXF_Complex* A_data = jxf_BSRMatrixData(A);
    JXF_Int*     A_i    = jxf_BSRMatrixI(A);
    JXF_Int*     A_j    = jxf_BSRMatrixJ(A);
    // JXF_Int*     rownnz_A   = jxf_CSRMatrixRownnz(A);
    // JXF_Int      nnzrows_A  = jxf_CSRMatrixNumRownnz(A);
    JXF_Int num_rows_A = jxf_BSRMatrixNumRows(A);
    JXF_Int num_cols_A = jxf_BSRMatrixNumCols(A);
    JXF_Int block_size = jxf_BSRMatrixBlockSize(A);

    JXF_Int i, j;

    /* the matrix should be square */
    if (num_rows_A != num_cols_A) {
        return -1;
    }

#ifdef JXF_USING_OPENMP
#pragma omp parallel for private(i, j)
#endif
    for (i = 0; i < num_rows_A; i++) {
        for (j = A_i[i]; j < A_i[i + 1]; j++) {
            if (A_j[j] == i) {
                if (j != A_i[i]) {
                    jxf_swap(A_j, A_i[i], j);
                    jxf_swap_blk(A_data, block_size, A_i[i], j);
                }
                break;
            }
        }
    }

    return jxf_error_flag;
}
