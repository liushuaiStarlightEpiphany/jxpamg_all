//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//


/*!
 *  seq_bsr_matrix.c -- basic operations for BSR matrices.
*  Date: 2025/10/08
*  Created by zlj
*/

#include "jx_mv.h"
#include "jx_bsr_mv.h"

/******************************************************************************
 *
 * Member functions for jx_BSRMatrix class.
 *
 *****************************************************************************/

// #include "_jx_parcsr_block_mv.h"

#define LB_VERSION 0 // Using the LAPACK/Blas library for inverse calculation

/*---------------------------------*/
/*--  Declare Private Functions  --*/
/*---------------------------------*/

#include "bsr_matrix_inv.inl"

/*--------------------------------------------------------------------------
 * jx_BSRMatrixCreate
 *--------------------------------------------------------------------------*/

jx_BSRMatrix* jx_BSRMatrixCreate(JX_Int block_size, JX_Int num_rows, JX_Int num_cols, JX_Int num_nonzeros)
{
    jx_BSRMatrix* matrix;

    matrix = jx_CTAlloc(jx_BSRMatrix, 1);

    jx_BSRMatrixData(matrix)        = NULL;
    jx_BSRMatrixI(matrix)           = NULL;
    jx_BSRMatrixJ(matrix)           = NULL;
    jx_BSRMatrixBigJ(matrix)        = NULL;
    jx_BSRMatrixBlockSize(matrix)   = block_size;
    jx_BSRMatrixNumRows(matrix)     = num_rows;
    jx_BSRMatrixNumCols(matrix)     = num_cols;
    jx_BSRMatrixNumNonzeros(matrix) = num_nonzeros;

    /* set defaults */
    jx_BSRMatrixOwnsData(matrix) = 1;

    return matrix;
}

/*--------------------------------------------------------------------------
 * jx_BSRMatrixDestroy
 *--------------------------------------------------------------------------*/

JX_Int jx_BSRMatrixDestroy(jx_BSRMatrix* matrix)
{
    JX_Int ierr = 0;

    if (matrix) {
        jx_TFree(jx_BSRMatrixI(matrix));
        if (jx_BSRMatrixOwnsData(matrix)) {
            jx_TFree(jx_BSRMatrixData(matrix));
            jx_TFree(jx_BSRMatrixJ(matrix));
            jx_TFree(jx_BSRMatrixBigJ(matrix));
        }
        jx_TFree(matrix);
    }

    return ierr;
}

/*--------------------------------------------------------------------------
 * jx_BSRMatrixInitialize
 *--------------------------------------------------------------------------*/

JX_Int jx_BSRMatrixInitialize(jx_BSRMatrix* matrix)
{
    JX_Int block_size   = jx_BSRMatrixBlockSize(matrix);
    JX_Int num_rows     = jx_BSRMatrixNumRows(matrix);
    JX_Int num_nonzeros = jx_BSRMatrixNumNonzeros(matrix);
    JX_Int ierr         = 0, nnz;

    if (jx_BSRMatrixI(matrix)) {
        jx_TFree(jx_BSRMatrixI(matrix));
    }
    if (jx_BSRMatrixJ(matrix)) {
        jx_TFree(jx_BSRMatrixJ(matrix));
    }
    if (jx_BSRMatrixBigJ(matrix)) {
        jx_TFree(jx_BSRMatrixBigJ(matrix));
    }
    if (jx_BSRMatrixData(matrix)) {
        jx_TFree(jx_BSRMatrixData(matrix));
    }

    nnz                       = num_nonzeros * block_size * block_size;
    jx_BSRMatrixI(matrix) = jx_CTAlloc(JX_Int, num_rows + 1);
    if (nnz) {
        jx_BSRMatrixData(matrix) = jx_CTAlloc(JX_Complex, nnz);
    } else {
        jx_BSRMatrixData(matrix) = NULL;
    }
    if (nnz) {
        jx_BSRMatrixJ(matrix) = jx_CTAlloc(JX_Int, num_nonzeros);
    } else {
        jx_BSRMatrixJ(matrix) = NULL;
    }

    return ierr;
}

/*--------------------------------------------------------------------------
 * jx_BSRMatrixBigInitialize
 *--------------------------------------------------------------------------*/

JX_Int jx_BSRMatrixBigInitialize(jx_BSRMatrix* matrix)
{
    JX_Int block_size   = jx_BSRMatrixBlockSize(matrix);
    JX_Int num_rows     = jx_BSRMatrixNumRows(matrix);
    JX_Int num_nonzeros = jx_BSRMatrixNumNonzeros(matrix);
    JX_Int ierr         = 0, nnz;

    if (jx_BSRMatrixI(matrix)) {
        jx_TFree(jx_BSRMatrixI(matrix));
    }
    if (jx_BSRMatrixJ(matrix)) {
        jx_TFree(jx_BSRMatrixJ(matrix));
    }
    if (jx_BSRMatrixBigJ(matrix)) {
        jx_TFree(jx_BSRMatrixBigJ(matrix));
    }
    if (jx_BSRMatrixData(matrix)) {
        jx_TFree(jx_BSRMatrixData(matrix));
    }

    nnz                       = num_nonzeros * block_size * block_size;
    jx_BSRMatrixI(matrix) = jx_CTAlloc(JX_Int, num_rows + 1);
    if (nnz) {
        jx_BSRMatrixData(matrix) = jx_CTAlloc(JX_Complex, nnz);
    } else {
        jx_BSRMatrixData(matrix) = NULL;
    }
    if (nnz) {
        jx_BSRMatrixBigJ(matrix) = jx_CTAlloc(JX_Int, num_nonzeros);
    } else {
        jx_BSRMatrixJ(matrix) = NULL;
    }

    return ierr;
}

/*--------------------------------------------------------------------------
 * jx_BSRMatrixSetDataOwner
 *--------------------------------------------------------------------------*/

JX_Int jx_BSRMatrixSetDataOwner(jx_BSRMatrix* matrix, JX_Int owns_data)
{
    JX_Int ierr = 0;

    jx_BSRMatrixOwnsData(matrix) = owns_data;

    return ierr;
}

/*--------------------------------------------------------------------------
 * jx_BSRMatrixCompress, take the F-norm for each sub block
 *--------------------------------------------------------------------------*/

jx_CSRMatrix* jx_BSRMatrixCompress(jx_BSRMatrix* matrix)
{
    JX_Int        block_size   = jx_BSRMatrixBlockSize(matrix);
    JX_Int        num_rows     = jx_BSRMatrixNumRows(matrix);
    JX_Int        num_cols     = jx_BSRMatrixNumCols(matrix);
    JX_Int        num_nonzeros = jx_BSRMatrixNumNonzeros(matrix);
    JX_Int*       matrix_i     = jx_BSRMatrixI(matrix);
    JX_Int*       matrix_j     = jx_BSRMatrixJ(matrix);
    JX_Complex*   matrix_data  = jx_BSRMatrixData(matrix);
    jx_CSRMatrix* matrix_C;
    JX_Int *      matrix_C_i, *matrix_C_j, i, j, bnnz;
    JX_Complex *  matrix_C_data, ddata;

    matrix_C = jx_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
    jx_CSRMatrixInitialize(matrix_C);
    matrix_C_i    = jx_CSRMatrixI(matrix_C);
    matrix_C_j    = jx_CSRMatrixJ(matrix_C);
    matrix_C_data = jx_CSRMatrixData(matrix_C);

    bnnz = block_size * block_size;
    for (i = 0; i < num_rows + 1; i++) {
        matrix_C_i[i] = matrix_i[i];
    }
    for (i = 0; i < num_nonzeros; i++) {
        matrix_C_j[i] = matrix_j[i];
        ddata         = 0.0;
        for (j = 0; j < bnnz; j++) {
            ddata += matrix_data[i * bnnz + j] * matrix_data[i * bnnz + j];
        }
        matrix_C_data[i] = jx_sqrt(ddata);
    }
    return matrix_C;
}

/*--------------------------------------------------------------------------
 * jx_BSRMatrixGetSubmatrix, Aij = Aij_[subposition][subposition]
 *--------------------------------------------------------------------------*/

jx_CSRMatrix* jx_BSRMatrixGetSubmatrix(jx_BSRMatrix* matrix, JX_Int subposition)
{
    JX_Int        block_size   = jx_BSRMatrixBlockSize(matrix);
    JX_Int        num_rows     = jx_BSRMatrixNumRows(matrix);
    JX_Int        num_cols     = jx_BSRMatrixNumCols(matrix);
    JX_Int        num_nonzeros = jx_BSRMatrixNumNonzeros(matrix);
    JX_Int*       matrix_i     = jx_BSRMatrixI(matrix);
    JX_Int*       matrix_j     = jx_BSRMatrixJ(matrix);
    JX_Complex*   matrix_data  = jx_BSRMatrixData(matrix);
    jx_CSRMatrix* matrix_C;
    JX_Int *      matrix_C_i, *matrix_C_j, i, j, bnnz;
    JX_Complex*   matrix_C_data;

    matrix_C = jx_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
    jx_CSRMatrixInitialize(matrix_C);
    matrix_C_i    = jx_CSRMatrixI(matrix_C);
    matrix_C_j    = jx_CSRMatrixJ(matrix_C);
    matrix_C_data = jx_CSRMatrixData(matrix_C);

    bnnz = block_size * block_size;
    for (i = 0; i < num_rows + 1; i++) {
        matrix_C_i[i] = matrix_i[i];
    }

    JX_Int pos = subposition * block_size + subposition; // Aij_[subposition][subposition]
    for (i = 0; i < num_nonzeros; i++) {
        matrix_C_j[i]    = matrix_j[i];
        matrix_C_data[i] = matrix_data[i * bnnz + pos]; // Aij_[subposition][subposition]
    }

    // printf(" [%s:%d] CSR = %p \n",__FUNCTION__, __LINE__, (void*)matrix_C);
    // fflush(stdout);  
    // printf(" [%s:%d] jx_CSRMatrixI(CSR) = %p \n",__FUNCTION__, __LINE__, (void*)jx_CSRMatrixI(matrix_C));
    // fflush(stdout);  


    return matrix_C;
}

/*--------------------------------------------------------------------------
 * jx_BSRMatrixConvertToCSRMatrix
 *--------------------------------------------------------------------------*/

jx_CSRMatrix* jx_BSRMatrixConvertToCSRMatrix(jx_BSRMatrix* matrix)
{
    JX_Int      block_size   = jx_BSRMatrixBlockSize(matrix);
    JX_Int      num_rows     = jx_BSRMatrixNumRows(matrix);
    JX_Int      num_cols     = jx_BSRMatrixNumCols(matrix);
    JX_Int      num_nonzeros = jx_BSRMatrixNumNonzeros(matrix);
    JX_Int*     matrix_i     = jx_BSRMatrixI(matrix);
    JX_Int*     matrix_j     = jx_BSRMatrixJ(matrix);
    JX_Complex* matrix_data  = jx_BSRMatrixData(matrix);

    jx_CSRMatrix* matrix_C;
    JX_Int        i, j, k, ii, C_ii, bnnz, new_nrows, new_ncols, new_num_nonzeros;
    JX_Int *      matrix_C_i, *matrix_C_j;
    JX_Complex*   matrix_C_data;

    bnnz             = block_size * block_size;
    new_nrows        = num_rows * block_size;
    new_ncols        = num_cols * block_size;
    new_num_nonzeros = block_size * block_size * num_nonzeros;
    matrix_C         = jx_CSRMatrixCreate(new_nrows, new_ncols, new_num_nonzeros);
    jx_CSRMatrixInitialize(matrix_C);
    matrix_C_i    = jx_CSRMatrixI(matrix_C);
    matrix_C_j    = jx_CSRMatrixJ(matrix_C);
    matrix_C_data = jx_CSRMatrixData(matrix_C);
    for (i = 0; i < num_rows; i++) {
        for (j = 0; j < block_size; j++) matrix_C_i[i * block_size + j] = matrix_i[i] * bnnz + j * (matrix_i[i + 1] - matrix_i[i]) * block_size;
    }
    matrix_C_i[new_nrows] = matrix_i[num_rows] * bnnz;

    C_ii = 0;
    for (i = 0; i < num_rows; i++) {
        for (j = 0; j < block_size; j++) {
            for (ii = matrix_i[i]; ii < matrix_i[i + 1]; ii++) {
                k                   = j;
                matrix_C_j[C_ii]    = matrix_j[ii] * block_size + k;
                matrix_C_data[C_ii] = matrix_data[ii * bnnz + j * block_size + k];
                C_ii++;
                for (k = 0; k < block_size; k++) {
                    if (j != k) {
                        matrix_C_j[C_ii]    = matrix_j[ii] * block_size + k;
                        matrix_C_data[C_ii] = matrix_data[ii * bnnz + j * block_size + k];
                        C_ii++;
                    }
                }
            }
        }
    }
    return matrix_C;
}

/*--------------------------------------------------------------------------
 * jx_BSRMatrixConvertFromCSRMatrix

 * this doesn't properly convert the parcsr off_diag matrices
   (because here we assume the matrix is square - we don't check what the
    number of columns should be ) - it can only be used for the diag part
 *--------------------------------------------------------------------------*/

jx_BSRMatrix* jx_BSRMatrixConvertFromCSRMatrix(jx_CSRMatrix* matrix, JX_Int matrix_C_block_size)
{
    JX_Int      num_rows    = jx_CSRMatrixNumRows(matrix);
    JX_Int      num_cols    = jx_CSRMatrixNumCols(matrix);
    JX_Int*     matrix_i    = jx_CSRMatrixI(matrix);
    JX_Int*     matrix_j    = jx_CSRMatrixJ(matrix);
    JX_Complex* matrix_data = jx_CSRMatrixData(matrix);

    jx_BSRMatrix* matrix_C;
    JX_Int *      matrix_C_i, *matrix_C_j;
    JX_Complex*   matrix_C_data;
    JX_Int        matrix_C_num_rows, matrix_C_num_cols, matrix_C_num_nonzeros;
    JX_Int        i, j, ii, jj, s_jj, index, *counter;

    matrix_C_num_rows = num_rows / matrix_C_block_size;
    matrix_C_num_cols = num_cols / matrix_C_block_size;

    counter = jx_CTAlloc(JX_Int, matrix_C_num_cols);
    for (i = 0; i < matrix_C_num_cols; i++) {
        counter[i] = -1;
    }
    matrix_C_num_nonzeros = 0;
    for (i = 0; i < matrix_C_num_rows; i++) {
        for (j = 0; j < matrix_C_block_size; j++) {
            for (ii = matrix_i[i * matrix_C_block_size + j]; ii < matrix_i[i * matrix_C_block_size + j + 1]; ii++) {
                if (counter[matrix_j[ii] / matrix_C_block_size] < i) {
                    counter[matrix_j[ii] / matrix_C_block_size] = i;
                    matrix_C_num_nonzeros++;
                }
            }
        }
    }
    matrix_C = jx_BSRMatrixCreate(matrix_C_block_size, matrix_C_num_rows, matrix_C_num_cols, matrix_C_num_nonzeros);
    jx_BSRMatrixInitialize(matrix_C);
    matrix_C_i    = jx_BSRMatrixI(matrix_C);
    matrix_C_j    = jx_BSRMatrixJ(matrix_C);
    matrix_C_data = jx_BSRMatrixData(matrix_C);

    for (i = 0; i < matrix_C_num_cols; i++) {
        counter[i] = -1;
    }
    jj = s_jj = 0;
    for (i = 0; i < matrix_C_num_rows; i++) {
        matrix_C_i[i] = jj;
        for (j = 0; j < matrix_C_block_size; j++) {
            for (ii = matrix_i[i * matrix_C_block_size + j]; ii < matrix_i[i * matrix_C_block_size + j + 1]; ii++) {
                if (counter[matrix_j[ii] / matrix_C_block_size] < s_jj) {
                    counter[matrix_j[ii] / matrix_C_block_size] = jj;
                    matrix_C_j[jj]                              = matrix_j[ii] / matrix_C_block_size;
                    jj++;
                }
                index = counter[matrix_j[ii] / matrix_C_block_size] * matrix_C_block_size * matrix_C_block_size + j * matrix_C_block_size +
                        matrix_j[ii] % matrix_C_block_size;
                matrix_C_data[index] = matrix_data[ii];
            }
        }
        s_jj = jj;
    }
    matrix_C_i[matrix_C_num_rows] = matrix_C_num_nonzeros;

    jx_TFree(counter);

    return matrix_C;
}

/*--------------------------------------------------------------------------
 * jx_BSRMatrixBlockAdd
 * (o = i1 + i2)
 *--------------------------------------------------------------------------*/
JX_Int jx_BSRMatrixBlockAdd(JX_Complex* i1, JX_Complex* i2, JX_Complex* o, JX_Int block_size)
{
    JX_Int i;
    JX_Int sz = block_size * block_size;

    for (i = 0; i < sz; i++) {
        o[i] = i1[i] + i2[i];
    }

    return 0;
}

/*--------------------------------------------------------------------------
 * jx_BSRMatrixBlockAddAccumulate
 * (o = i1 + o)
 *--------------------------------------------------------------------------*/
JX_Int jx_BSRMatrixBlockAddAccumulate(JX_Complex* i1, JX_Complex* o, JX_Int block_size)
{
    JX_Int i;
    JX_Int sz = block_size * block_size;

    for (i = 0; i < sz; i++) {
        o[i] += i1[i];
    }

    return 0;
}

/*--------------------------------------------------------------------------
 * jx_BSRMatrixBlockAddAccumulateDiag
 * (diag(o) = diag(i1) + diag(o))
 *--------------------------------------------------------------------------*/
JX_Int jx_BSRMatrixBlockAddAccumulateDiag(JX_Complex* i1, JX_Complex* o, JX_Int block_size)
{
    JX_Int i;

    for (i = 0; i < block_size; i++) {
        o[i * block_size + i] += i1[i * block_size + i];
    }
    return 0;
}

/*--------------------------------------------------------------------------
 * jx_BSRMatrixBlockAddAccumulateDiagCheckSign
 * only add elements of sign*i1 that are negative (sign is size block_size)
 * (diag(o) = diag(i1) + diag(o))
 *--------------------------------------------------------------------------*/
JX_Int jx_BSRMatrixBlockAddAccumulateDiagCheckSign(JX_Complex* i1, JX_Complex* o, JX_Int block_size, JX_Real* sign)
{
    JX_Int  i;
    JX_Real tmp;

    for (i = 0; i < block_size; i++) {
        tmp = (JX_Real)i1[i * block_size + i] * sign[i];
        if (tmp < 0) {
            o[i * block_size + i] += i1[i * block_size + i];
        }
    }

    return 0;
}

/*--------------------------------------------------------------------------
 *  jx_BSRMatrixComputeSign

 * o = sign(diag(i1))
 *--------------------------------------------------------------------------*/

JX_Int jx_BSRMatrixComputeSign(JX_Complex* i1, JX_Complex* o, JX_Int block_size)
{
    JX_Int i;

    for (i = 0; i < block_size; i++) {
        if ((JX_Real)i1[i * block_size + i] < 0) {
            o[i] = -1;
        } else {
            o[i] = 1;
        }
    }

    return 0;
}

/*--------------------------------------------------------------------------
 * jx_BSRMatrixBlockSetScalar
 * (each entry in block o is set to beta )
 *--------------------------------------------------------------------------*/
JX_Int jx_BSRMatrixBlockSetScalar(JX_Complex* o, JX_Complex beta, JX_Int block_size)
{
    JX_Int i;
    JX_Int sz = block_size * block_size;

    for (i = 0; i < sz; i++) {
        o[i] = beta;
    }

    return 0;
}

/*--------------------------------------------------------------------------
 * jx_BSRMatrixBlockCopyData
 * (o = beta*i1 )
 *--------------------------------------------------------------------------*/
JX_Int jx_BSRMatrixBlockCopyData(JX_Complex* i1, JX_Complex* o, JX_Complex beta, JX_Int block_size)
{
    JX_Int i;
    JX_Int sz = block_size * block_size;

    for (i = 0; i < sz; i++) {
        o[i] = beta * i1[i];
    }

    return 0;
}

/*--------------------------------------------------------------------------
 * jx_BSRMatrixBlockCopyDataDiag - zeros off-diag entries
 * (o = beta*diag(i1))
 *--------------------------------------------------------------------------*/
JX_Int jx_BSRMatrixBlockCopyDataDiag(JX_Complex* i1, JX_Complex* o, JX_Complex beta, JX_Int block_size)
{
    JX_Int i;

    JX_Int sz = block_size * block_size;

    for (i = 0; i < sz; i++) {
        o[i] = 0.0;
    }

    for (i = 0; i < block_size; i++) {
        o[i * block_size + i] = beta * i1[i * block_size + i];
    }

    return 0;
}

/*--------------------------------------------------------------------------
 * jx_BSRMatrixBlockTranspose
 * (o = i1' )
 *--------------------------------------------------------------------------*/
JX_Int jx_BSRMatrixBlockTranspose(JX_Complex* i1, JX_Complex* o, JX_Int block_size)
{
    JX_Int i, j;

    for (i = 0; i < block_size; i++)
        for (j = 0; j < block_size; j++) {
            o[i * block_size + j] = i1[j * block_size + i];
        }
    return 0;
}

/*--------------------------------------------------------------------------
 * jx_BSRMatrixBlockNorm
 * (out = norm(data) )
 *
 *  (note: these are not all actually "norms")
 *
 *--------------------------------------------------------------------------*/
JX_Int jx_BSRMatrixBlockNorm(JX_Int norm_type, JX_Complex* data, JX_Real* out, JX_Int block_size)
{
    JX_Int   ierr = 0;
    JX_Int   i, j;
    JX_Real  sum = 0.0;
    JX_Real* totals;
    JX_Int   sz = block_size * block_size;

    switch (norm_type) {
        case 8: /* pp: pressure entry in each block for reservior  */
            {
                sum = data[0];
                break;
            }
        case 7: /* |pp|: abs of pressure entry in each block for reservior  */
            {
                sum = jx_cabs(data[0]);
                break;
            }
        case 6: /* sum of all elements in the block  */
            {
                for (i = 0; i < sz; i++) {
                    sum += (JX_Real)(data[i]);
                }
                break;
            }
        case 5: /* one norm  - max col sum*/
            {

                totals = jx_CTAlloc(JX_Real, block_size);
                for (i = 0; i < block_size; i++) /* row */
                {
                    for (j = 0; j < block_size; j++) /* col */
                    {
                        totals[j] += jx_cabs(data[i * block_size + j]);
                    }
                }

                sum = totals[0];
                for (j = 1; j < block_size; j++) /* col */
                {
                    if (totals[j] > sum) {
                        sum = totals[j];
                    }
                }
                jx_TFree(totals);

                break;
            }
        case 4: /* inf norm - max row sum */
            {

                totals = jx_CTAlloc(JX_Real, block_size);
                for (i = 0; i < block_size; i++) /* row */
                {
                    for (j = 0; j < block_size; j++) /* col */
                    {
                        totals[i] += jx_cabs(data[i * block_size + j]);
                    }
                }

                sum = totals[0];
                for (i = 1; i < block_size; i++) /* row */
                {
                    if (totals[i] > sum) {
                        sum = totals[i];
                    }
                }
                jx_TFree(totals);

                break;
            }

        case 3: /* largest element of block (return value includes sign) */
            {

                sum = (JX_Real)data[0];

                for (i = 0; i < sz; i++) {
                    if (jx_cabs(data[i]) > jx_cabs(sum)) {
                        sum = (JX_Real)data[i];
                    }
                }

                break;
            }
        case 2: /* sum of abs values of all elements in the block  */
            {
                for (i = 0; i < sz; i++) {
                    sum += jx_cabs(data[i]);
                }
                break;
            }

        default: /* 1 = frobenius*/
            {
                for (i = 0; i < sz; i++) {
                    sum += ((JX_Real)data[i]) * ((JX_Real)data[i]);
                }
                sum = jx_sqrt(sum);
            }
    }

    *out = sum;

    return ierr;
}

/*--------------------------------------------------------------------------
 * jx_BSRMatrixBlockMultAdd
 * (o = i1 * i2 + beta * o)
 *--------------------------------------------------------------------------*/
JX_Int jx_BSRMatrixBlockMultAdd(JX_Complex* i1, JX_Complex* i2, JX_Complex beta, JX_Complex* o, JX_Int block_size)
{

#if LB_VERSION
    {
        JX_Complex alp = 1.0;
        dgemm_("N", "N", &block_size, &block_size, &block_size, &alp, i2, &block_size, i1, &block_size, &beta, o, &block_size);
    }
#else
    {
        JX_Int     i, j, k;
        JX_Complex ddata;

        if (beta == 0.0) {
            for (i = 0; i < block_size; i++) {
                for (j = 0; j < block_size; j++) {
                    ddata = 0.0;
                    for (k = 0; k < block_size; k++) {
                        ddata += i1[i * block_size + k] * i2[k * block_size + j];
                    }
                    o[i * block_size + j] = ddata;
                }
            }
        } else if (beta == 1.0) {
            for (i = 0; i < block_size; i++) {
                for (j = 0; j < block_size; j++) {
                    ddata = o[i * block_size + j];
                    for (k = 0; k < block_size; k++) {
                        ddata += i1[i * block_size + k] * i2[k * block_size + j];
                    }
                    o[i * block_size + j] = ddata;
                }
            }
        } else {
            for (i = 0; i < block_size; i++) {
                for (j = 0; j < block_size; j++) {
                    ddata = beta * o[i * block_size + j];
                    for (k = 0; k < block_size; k++) {
                        ddata += i1[i * block_size + k] * i2[k * block_size + j];
                    }
                    o[i * block_size + j] = ddata;
                }
            }
        }
    }

#endif

    return 0;
}

/*--------------------------------------------------------------------------
 * jx_BSRMatrixBlockMultAddDiag
 * (diag(o) = diag(i1) * diag(i2) + beta * diag(o))
 *--------------------------------------------------------------------------*/
JX_Int jx_BSRMatrixBlockMultAddDiag(JX_Complex* i1, JX_Complex* i2, JX_Complex beta, JX_Complex* o, JX_Int block_size)
{
    JX_Int i;

    if (beta == 0.0) {
        for (i = 0; i < block_size; i++) {
            o[i * block_size + i] = i1[i * block_size + i] * i2[i * block_size + i];
        }
    } else if (beta == 1.0) {
        for (i = 0; i < block_size; i++) {
            o[i * block_size + i] = o[i * block_size + i] + i1[i * block_size + i] * i2[i * block_size + i];
        }
    } else {
        for (i = 0; i < block_size; i++) {
            o[i * block_size + i] = beta * o[i * block_size + i] + i1[i * block_size + i] * i2[i * block_size + i];
        }
    }
    return 0;
}

/*--------------------------------------------------------------------------
 * jx_BSRMatrixBlockMultAddDiagCheckSign
 *
 *  only mult elements if sign*diag(i2) is negative
 *(diag(o) = diag(i1) * diag(i2) + beta * diag(o))
 *--------------------------------------------------------------------------*/
JX_Int jx_BSRMatrixBlockMultAddDiagCheckSign(JX_Complex* i1, JX_Complex* i2, JX_Complex beta, JX_Complex* o, JX_Int block_size,
                                                     JX_Real* sign)
{
    JX_Int  i;
    JX_Real tmp;

    if (beta == 0.0) {
        for (i = 0; i < block_size; i++) {
            tmp = (JX_Real)i2[i * block_size + i] * sign[i];
            if (tmp < 0) {
                o[i * block_size + i] = i1[i * block_size + i] * i2[i * block_size + i];
            }
        }
    } else if (beta == 1.0) {
        for (i = 0; i < block_size; i++) {
            tmp = (JX_Real)i2[i * block_size + i] * sign[i];
            if (tmp < 0) {
                o[i * block_size + i] = o[i * block_size + i] + i1[i * block_size + i] * i2[i * block_size + i];
            }
        }
    } else {
        for (i = 0; i < block_size; i++) {
            tmp = i2[i * block_size + i] * sign[i];
            if (tmp < 0) {
                o[i * block_size + i] = beta * o[i * block_size + i] + i1[i * block_size + i] * i2[i * block_size + i];
            }
        }
    }
    return 0;
}

/*--------------------------------------------------------------------------
 * jx_BSRMatrixBlockMultAddDiag2 (scales cols of il by diag of i2)
 * ((o) = (i1) * diag(i2) + beta * (o))
 *--------------------------------------------------------------------------*/
JX_Int jx_BSRMatrixBlockMultAddDiag2(JX_Complex* i1, JX_Complex* i2, JX_Complex beta, JX_Complex* o, JX_Int block_size)
{
    JX_Int i, j;

    if (beta == 0.0) {
        for (i = 0; i < block_size; i++) {
            for (j = 0; j < block_size; j++) {
                o[i * block_size + j] = i1[i * block_size + j] * i2[j * block_size + j];
            }
        }
    } else if (beta == 1.0) {
        for (i = 0; i < block_size; i++) {
            for (j = 0; j < block_size; j++) {
                o[i * block_size + j] = o[i * block_size + j] + i1[i * block_size + j] * i2[j * block_size + j];
            }
        }

    } else {
        for (i = 0; i < block_size; i++) {
            for (j = 0; j < block_size; j++) {
                o[i * block_size + j] = beta * o[i * block_size + j] + i1[i * block_size + j] * i2[j * block_size + j];
            }
        }
    }
    return 0;
}

/*--------------------------------------------------------------------------
 * jx_BSRMatrixBlockMultAddDiag3 (scales cols of il by i2 -
                                          whose diag elements are row sums)
 * ((o) = (i1) * diag(i2) + beta * (o))
 *--------------------------------------------------------------------------*/
JX_Int jx_BSRMatrixBlockMultAddDiag3(JX_Complex* i1, JX_Complex* i2, JX_Complex beta, JX_Complex* o, JX_Int block_size)
{
    JX_Int i, j;

    JX_Complex* row_sum;

    row_sum = jx_CTAlloc(JX_Complex, block_size);
    for (i = 0; i < block_size; i++) {
        for (j = 0; j < block_size; j++) {
            row_sum[i] += i2[i * block_size + j];
        }
    }

    if (beta == 0.0) {
        for (i = 0; i < block_size; i++) {
            for (j = 0; j < block_size; j++) {
                o[i * block_size + j] = i1[i * block_size + j] * row_sum[j];
            }
        }
    } else if (beta == 1.0) {
        for (i = 0; i < block_size; i++) {
            for (j = 0; j < block_size; j++) {
                o[i * block_size + j] = o[i * block_size + j] + i1[i * block_size + j] * row_sum[j];
            }
        }
    } else {
        for (i = 0; i < block_size; i++) {
            for (j = 0; j < block_size; j++) {
                o[i * block_size + j] = beta * o[i * block_size + j] + i1[i * block_size + j] * row_sum[j];
            }
        }
    }

    jx_TFree(row_sum);

    return 0;
}
/*--------------------------------------------------------------------------
 * jx_BSRMatrixBlockMatvec
 * (ov = alpha* mat * v + beta * ov)
 * mat is the matrix - size is block_size^2
 * alpha and beta are scalars
 *--------------------------------------------------------------------------*/

JX_Int jx_BSRMatrixBlockMatvec(JX_Complex alpha, JX_Complex* mat, JX_Complex* v, JX_Complex beta, JX_Complex* ov,
                                       JX_Int block_size)
{
    JX_Int ierr = 0;

#if LB_VERSION
    {
        JX_Int one = 1;

        dgemv_("T", &block_size, &block_size, &alpha, mat, &block_size, v, &one, &beta, ov, &one);
    }

#else
    {
        JX_Int     i, j;
        JX_Complex ddata;

        /* if alpha = 0, then no matvec */
        if (alpha == 0.0) {
            for (j = 0; j < block_size; j++) {
                ov[j] *= beta;
            }
            return ierr;
        }

        /* ov = (beta/alpha) * ov; */
        ddata = beta / alpha;
        if (ddata != 1.0) {
            if (ddata == 0.0) {
                for (j = 0; j < block_size; j++) {
                    ov[j] = 0.0;
                }
            } else {
                for (j = 0; j < block_size; j++) {
                    ov[j] *= ddata;
                }
            }
        }

        /* ov = ov + mat*v */
        for (i = 0; i < block_size; i++) {
            ddata = ov[i];
            for (j = 0; j < block_size; j++) {
                ddata += mat[i * block_size + j] * v[j];
            }
            ov[i] = ddata;
        }

        /* ov = alpha*ov */
        if (alpha != 1.0) {
            for (j = 0; j < block_size; j++) {
                ov[j] *= alpha;
            }
        }
    }

#endif

    return ierr;
}

/*--------------------------------------------------------------------------
 * jx_BSRMatrixBlockInvMatvec
 * (ov = mat^{-1} * v)
 * o and v are vectors
 * mat is the matrix - size is block_size^2
 *--------------------------------------------------------------------------*/
JX_Int jx_BSRMatrixBlockInvMatvec(JX_Complex* mat, JX_Complex* v, JX_Complex* ov, JX_Int block_size)
{
    JX_Int      ierr = 0;
    JX_Complex* mat_i;

    mat_i = jx_CTAlloc(JX_Complex, block_size * block_size);

#if LB_VERSION
    {

        JX_Int  one, info;
        JX_Int* piv;
        JX_Int  sz;

        one = 1;
        piv = jx_CTAlloc(JX_Int, block_size);
        sz  = block_size * block_size;

        /* copy v to ov and  mat to mat_i*/

        dcopy_(&sz, mat, &one, mat_i, &one);
        dcopy_(&block_size, v, &one, ov, &one);

        /* writes over mat_i with LU */
        dgetrf_(&block_size, &block_size, mat_i, &block_size, piv, &info);
        if (info) {
            jx_TFree(mat_i);
            jx_TFree(piv);
            return (-1);
        }

        /* writes over ov */
        dgetrs_("T", &block_size, &one, mat_i, &block_size, piv, ov, &block_size, &info);
        if (info) {
            jx_TFree(mat_i);
            jx_TFree(piv);
            return (-1);
        }

        jx_TFree(piv);
    }

#else
    {
        JX_Int     m, j, k;
        JX_Int     piv_row;
        JX_Real    eps;
        JX_Complex factor;
        JX_Complex piv, tmp;
        eps = 1.0e-6;

        if (block_size == 1) {
            if (jx_cabs(mat[0]) > 1e-10) {
                ov[0] = v[0] / mat[0];
                jx_TFree(mat_i);
                return (ierr);
            } else {
                /* jx_printf("GE zero pivot error\n"); */
                jx_TFree(mat_i);
                return (-1);
            }
        }
#if 0 //! Zhao Li
        else if (block_size > 1) {
            /* copy v to ov and mat to mat_i*/
            for (k = 0; k < block_size; k++) {
                // ov[k] = v[k];
                for (j = 0; j < block_size; j++) {
                    mat_i[k * block_size + j] = mat[k * block_size + j];
                }
            }

            // compute mat_i = mat^{-1}
            jx_smat_inv(mat_i, block_size);

            // ov = mat^{-1} * v
            for (k = 0; k < block_size; k++) {
                ov[k] = 0.0;
                for (j = 0; j < block_size; j++) {
                    ov[k] += mat_i[k * block_size + j] * v[j];
                }
            }
        }
#endif
        else {
            /* copy v to ov and mat to mat_i*/
            for (k = 0; k < block_size; k++) {
                ov[k] = v[k];
                for (j = 0; j < block_size; j++) {
                    mat_i[k * block_size + j] = mat[k * block_size + j];
                }
            }
            /* start ge  - turning m_i into U factor (don't save L - just apply to
               rhs - which is ov)*/
            /* we do partial pivoting for size */

            /* loop through the rows (row k) */
            for (k = 0; k < block_size - 1; k++) {
                piv     = mat_i[k * block_size + k];
                piv_row = k;

                /* find the largest pivot in position k*/
                for (j = k + 1; j < block_size; j++) {
                    if (jx_cabs(mat_i[j * block_size + k]) > jx_cabs(piv)) {
                        piv     = mat_i[j * block_size + k];
                        piv_row = j;
                    }
                }
                if (piv_row != k) /* do a row exchange  - rows k and piv_row*/
                {
                    for (j = 0; j < block_size; j++) {
                        tmp                             = mat_i[k * block_size + j];
                        mat_i[k * block_size + j]       = mat_i[piv_row * block_size + j];
                        mat_i[piv_row * block_size + j] = tmp;
                    }
                    tmp         = ov[k];
                    ov[k]       = ov[piv_row];
                    ov[piv_row] = tmp;
                }
                /* end of pivoting */

                if (jx_cabs(piv) > eps) {
                    /* now we can factor into U */
                    for (j = k + 1; j < block_size; j++) {
                        factor = mat_i[j * block_size + k] / piv;
                        for (m = k + 1; m < block_size; m++) {
                            mat_i[j * block_size + m] -= factor * mat_i[k * block_size + m];
                        }
                        /* Elimination step for rhs */
                        ov[j] -= factor * ov[k];
                    }
                } else {
                    /* jx_printf("Block of matrix is nearly singular: zero pivot error\n");  */
                    jx_TFree(mat_i);
                    return (-1);
                }
            }

            /* we also need to check the pivot in the last row to see if it is zero */
            k = block_size - 1; /* last row */
            if (jx_cabs(mat_i[k * block_size + k]) < eps) {
                /* jx_printf("Block of matrix is nearly singular: zero pivot error\n");  */
                jx_TFree(mat_i);
                return (-1);
            }

            /* Back Substitution  - do rhs (U is now in m_i1)*/
            for (k = block_size - 1; k > 0; --k) {
                ov[k] /= mat_i[k * block_size + k];
                for (j = 0; j < k; j++) {
                    if (mat_i[j * block_size + k] != 0.0) {
                        ov[j] -= ov[k] * mat_i[j * block_size + k];
                    }
                }
            }
            ov[0] /= mat_i[0];
        }
    }
#endif

    jx_TFree(mat_i);

    return (ierr);
}

/*--------------------------------------------------------------------------
 * jx_BSRMatrixBlockInvMult
 * (o = i1^{-1} * i2)
 *--------------------------------------------------------------------------*/
JX_Int jx_BSRMatrixBlockInvMult(JX_Complex* i1, JX_Complex* i2, JX_Complex* o, JX_Int block_size)
{

    JX_Int      ierr = 0;
    JX_Int      i, j;
    JX_Complex* m_i1;

    m_i1 = jx_CTAlloc(JX_Complex, block_size * block_size);

#if LB_VERSION
    {

        JX_Int  one, info;
        JX_Int* piv;
        JX_Int  sz;

        JX_Complex* i2_t;

        one  = 1;
        i2_t = jx_CTAlloc(JX_Complex, block_size * block_size);
        piv  = jx_CTAlloc(JX_Int, block_size);

        /* copy i1 to m_i1*/
        sz = block_size * block_size;
        dcopy_(&sz, i1, &one, m_i1, &one);

        /* writes over m_i1 with LU */
        dgetrf_(&block_size, &block_size, m_i1, &block_size, piv, &info);
        if (info) {
            jx_TFree(m_i1);
            jx_TFree(i2_t);
            jx_TFree(piv);
            return (-1);
        }

        /* need the transpose of i_2*/
        for (i = 0; i < block_size; i++) {
            for (j = 0; j < block_size; j++) {
                i2_t[i * block_size + j] = i2[j * block_size + i];
            }
        }

        /* writes over i2_t */
        dgetrs_("T", &block_size, &block_size, m_i1, &block_size, piv, i2_t, &block_size, &info);
        if (info) {
            jx_TFree(m_i1);
            jx_TFree(i2_t);
            jx_TFree(piv);
            return (-1);
        }

        /* ans. is the transpose of i2_t*/
        for (i = 0; i < block_size; i++) {
            for (j = 0; j < block_size; j++) {
                o[i * block_size + j] = i2_t[j * block_size + i];
            }
        }

        jx_TFree(i2_t);
        jx_TFree(piv);
    }

#else
    {
        JX_Int     m, k;
        JX_Int     piv_row;
        JX_Real    eps;
        JX_Complex factor;
        JX_Complex piv, tmp;

        eps = 1.0e-6;

        if (block_size == 1) {
            if (jx_cabs(m_i1[0]) > 1e-10) {
                o[0] = i2[0] / i1[0];
                jx_TFree(m_i1);
                return (ierr);
            } else {
                /* jx_printf("GE zero pivot error\n"); */
                jx_TFree(m_i1);
                return (-1);
            }
        } else {
            /* copy i2 to o and i1 to m_i1*/
            for (k = 0; k < block_size * block_size; k++) {
                o[k]    = i2[k];
                m_i1[k] = i1[k];
            }

            /* start ge  - turning m_i1 into U factor (don't save L - just apply to
               rhs - which is o)*/
            /* we do partial pivoting for size */

            /* loop through the rows (row k) */
            for (k = 0; k < block_size - 1; k++) {
                piv     = m_i1[k * block_size + k];
                piv_row = k;

                /* find the largest pivot in position k*/
                for (j = k + 1; j < block_size; j++) {
                    if (jx_cabs(m_i1[j * block_size + k]) > jx_cabs(piv)) {
                        piv     = m_i1[j * block_size + k];
                        piv_row = j;
                    }
                }
                if (piv_row != k) /* do a row exchange  - rows k and piv_row*/
                {
                    for (j = 0; j < block_size; j++) {
                        tmp                            = m_i1[k * block_size + j];
                        m_i1[k * block_size + j]       = m_i1[piv_row * block_size + j];
                        m_i1[piv_row * block_size + j] = tmp;

                        tmp                         = o[k * block_size + j];
                        o[k * block_size + j]       = o[piv_row * block_size + j];
                        o[piv_row * block_size + j] = tmp;
                    }
                }
                /* end of pivoting */

                if (jx_cabs(piv) > eps) {
                    /* now we can factor into U */
                    for (j = k + 1; j < block_size; j++) {
                        factor = m_i1[j * block_size + k] / piv;
                        for (m = k + 1; m < block_size; m++) {
                            m_i1[j * block_size + m] -= factor * m_i1[k * block_size + m];
                        }
                        /* Elimination step for rhs */
                        /* do for each of the "rhs" */
                        for (i = 0; i < block_size; i++) {
                            /* o(row, col) = o(row*block_size + col) */
                            o[j * block_size + i] -= factor * o[k * block_size + i];
                        }
                    }
                } else {
                    /* jx_printf("Block of matrix is nearly singular: zero pivot error\n"); */
                    jx_TFree(m_i1);
                    return (-1);
                }
            }

            /* we also need to check the pivot in the last row to see if it is zero */
            k = block_size - 1; /* last row */
            if (jx_cabs(m_i1[k * block_size + k]) < eps) {
                /* jx_printf("Block of matrix is nearly singular: zero pivot error\n"); */
                jx_TFree(m_i1);
                return (-1);
            }

            /* Back Substitution  - do for each "rhs" (U is now in m_i1)*/
            for (i = 0; i < block_size; i++) {
                for (k = block_size - 1; k > 0; --k) {
                    o[k * block_size + i] /= m_i1[k * block_size + k];
                    for (j = 0; j < k; j++) {
                        if (m_i1[j * block_size + k] != 0.0) {
                            o[j * block_size + i] -= o[k * block_size + i] * m_i1[j * block_size + k];
                        }
                    }
                }
                o[0 * block_size + i] /= m_i1[0];
            }
        }
    }

#endif
    jx_TFree(m_i1);

    return ierr;
}

/*--------------------------------------------------------------------------
 * jx_BSRMatrixBlockMultInv
 * (o = i2*il^(-1))
 *--------------------------------------------------------------------------*/
JX_Int jx_BSRMatrixBlockMultInv(JX_Complex* i1, JX_Complex* i2, JX_Complex* o, JX_Int block_size)
{

    JX_Int ierr = 0;

#if LB_VERSION

    {
        /* same as solving A^T C^T = B^T */
        JX_Complex* m_i1;
        JX_Int      info;
        JX_Int*     piv;
        JX_Int      sz, one;

        piv  = jx_CTAlloc(JX_Int, block_size);
        m_i1 = jx_CTAlloc(JX_Complex, block_size * block_size);
        one  = 1;
        sz   = block_size * block_size;

        /* copy i1 to m_i1 and i2 to o*/

        dcopy_(&sz, i1, &one, m_i1, &one);
        dcopy_(&sz, i2, &one, o, &one);

        /* writes over m_i1 with LU */
        dgetrf_(&block_size, &block_size, m_i1, &block_size, piv, &info);
        if (info) {
            jx_TFree(m_i1);
            jx_TFree(piv);
            return (-1);
        }
        /* writes over B */
        dgetrs_("N", &block_size, &block_size, m_i1, &block_size, piv, o, &block_size, &info);
        if (info) {
            jx_TFree(m_i1);
            jx_TFree(piv);
            return (-1);
        }

        jx_TFree(m_i1);
        jx_TFree(piv);
    }

#else
    {
        JX_Real     eps;
        JX_Complex *i1_t, *i2_t, *o_t;

        eps = 1.0e-12;

        if (block_size == 1) {
            if (jx_cabs(i1[0]) > eps) {
                o[0] = i2[0] / i1[0];
                return (ierr);
            } else {
                /* jx_printf("GE zero pivot error\n"); */
                return (-1);
            }
        } else {

            i1_t = jx_CTAlloc(JX_Complex, block_size * block_size);
            i2_t = jx_CTAlloc(JX_Complex, block_size * block_size);
            o_t  = jx_CTAlloc(JX_Complex, block_size * block_size);

            /* TO DO:: this could be done more efficiently! */
            jx_BSRMatrixBlockTranspose(i1, i1_t, block_size);
            jx_BSRMatrixBlockTranspose(i2, i2_t, block_size);
            ierr = jx_BSRMatrixBlockInvMult(i1_t, i2_t, o_t, block_size);

            if (!ierr) {
                jx_BSRMatrixBlockTranspose(o_t, o, block_size);
            }

            jx_TFree(i1_t);
            jx_TFree(i2_t);
            jx_TFree(o_t);
        }
    }

#endif
    return (ierr);
}

/*--------------------------------------------------------------------------
 * jx_BSRMatrixBlockInvMultDiag - zeros off-d entires
 * (o = diag(i1)^{-1} * diag(i2))
 *--------------------------------------------------------------------------*/
JX_Int jx_BSRMatrixBlockInvMultDiag(JX_Complex* i1, JX_Complex* i2, JX_Complex* o, JX_Int block_size)
{

    JX_Int  ierr = 0;
    JX_Int  i;
    JX_Int  sz  = block_size * block_size;
    JX_Real eps = 1.0e-8;

    for (i = 0; i < sz; i++) {
        o[i] = 0.0;
    }

    for (i = 0; i < block_size; i++) {
        if (jx_cabs(i1[i * block_size + i]) > eps) {
            o[i * block_size + i] = i2[i * block_size + i] / i1[i * block_size + i];
        } else {
            /* jx_printf("GE zero pivot error\n"); */
            return (-1);
        }
    }

    return (ierr);
}

/*--------------------------------------------------------------------------
 * jx_BSRMatrixBlockInvMultDiag2
 * (o = (i1)* diag(i2)^-1) - so this scales the cols of il by
                             the diag entries in i2
 *--------------------------------------------------------------------------*/
JX_Int jx_BSRMatrixBlockInvMultDiag2(JX_Complex* i1, JX_Complex* i2, JX_Complex* o, JX_Int block_size)
{

    JX_Int ierr = 0;
    JX_Int i, j;

    JX_Real    eps = 1.0e-8;
    JX_Complex tmp;

    for (i = 0; i < block_size; i++) {
        if (jx_cabs(i2[i * block_size + i]) > eps) {
            tmp = 1 / i2[i * block_size + i];
        } else {
            tmp = 1.0;
        }
        for (j = 0; j < block_size; j++) /* this should be re-written to access by row (not col)! */
        {
            o[j * block_size + i] = i1[j * block_size + i] * tmp;
        }
    }

    return (ierr);
}

/*--------------------------------------------------------------------------
 * jx_BSRMatrixBlockInvMultDiag3
 * (o = (i1)* diag(i2)^-1) - so this scales the cols of il by
                             the i2 whose diags are row sums
 *--------------------------------------------------------------------------*/
JX_Int jx_BSRMatrixBlockInvMultDiag3(JX_Complex* i1, JX_Complex* i2, JX_Complex* o, JX_Int block_size)
{

    JX_Int     ierr = 0;
    JX_Int     i, j;
    JX_Real    eps = 1.0e-8;
    JX_Complex tmp, row_sum;

    for (i = 0; i < block_size; i++) {
        /* get row sum of i2, row i */
        row_sum = 0.0;
        for (j = 0; j < block_size; j++) {
            row_sum += i2[i * block_size + j];
        }

        /* invert */
        if (jx_cabs(row_sum) > eps) {
            tmp = 1 / row_sum;
        } else {
            tmp = 1.0;
        }
        /* scale col of i1 */
        for (j = 0; j < block_size; j++) /* this should be re-written to access by row (not col)! */
        {
            o[j * block_size + i] = i1[j * block_size + i] * tmp;
        }
    }

    return (ierr);
}

/*--------------------------------------------------------------------------
 * jx_BSRMatrixTranspose
 *--------------------------------------------------------------------------*/

JX_Int jx_BSRMatrixTranspose(jx_BSRMatrix* A, jx_BSRMatrix** AT, JX_Int data)

{
    JX_Complex* A_data        = jx_BSRMatrixData(A);
    JX_Int*     A_i           = jx_BSRMatrixI(A);
    JX_Int*     A_j           = jx_BSRMatrixJ(A);
    JX_Int      num_rowsA     = jx_BSRMatrixNumRows(A);
    JX_Int      num_colsA     = jx_BSRMatrixNumCols(A);
    JX_Int      num_nonzerosA = jx_BSRMatrixNumNonzeros(A);
    JX_Int      block_size    = jx_BSRMatrixBlockSize(A);

    JX_Complex* AT_data;
    JX_Int*     AT_i;
    JX_Int*     AT_j;
    JX_Int      num_rowsAT;
    JX_Int      num_colsAT;
    JX_Int      num_nonzerosAT;

    JX_Int max_col;
    JX_Int i, j, k, m, offset, bnnz;

    /*--------------------------------------------------------------
     * First, ascertain that num_cols and num_nonzeros has been set.
     * If not, set them.
     *--------------------------------------------------------------*/

    if (!num_nonzerosA) {
        num_nonzerosA = A_i[num_rowsA];
    }
    if (num_rowsA && !num_colsA) {
        max_col = -1;
        for (i = 0; i < num_rowsA; ++i)
            for (j = A_i[i]; j < A_i[i + 1]; j++)
                if (A_j[j] > max_col) {
                    max_col = A_j[j];
                }
        num_colsA = max_col + 1;
    }
    num_rowsAT     = num_colsA;
    num_colsAT     = num_rowsA;
    num_nonzerosAT = num_nonzerosA;
    bnnz           = block_size * block_size;

    *AT = jx_BSRMatrixCreate(block_size, num_rowsAT, num_colsAT, num_nonzerosAT);

    AT_i                   = jx_CTAlloc(JX_Int, num_rowsAT + 1);
    AT_j                   = jx_CTAlloc(JX_Int, num_nonzerosAT);
    jx_BSRMatrixI(*AT) = AT_i;
    jx_BSRMatrixJ(*AT) = AT_j;
    if (data) {
        AT_data                   = jx_CTAlloc(JX_Complex, num_nonzerosAT * bnnz);
        jx_BSRMatrixData(*AT) = AT_data;
    }

    /*-----------------------------------------------------------------
     * Count the number of entries in each column of A (row of AT)
     * and fill the AT_i array.
     *-----------------------------------------------------------------*/

    for (i = 0; i < num_nonzerosA; i++) {
        ++AT_i[A_j[i] + 1];
    }
    for (i = 2; i <= num_rowsAT; i++) {
        AT_i[i] += AT_i[i - 1];
    }

    /*----------------------------------------------------------------
     * Load the data and column numbers of AT
     *----------------------------------------------------------------*/

    for (i = 0; i < num_rowsA; i++) {
        for (j = A_i[i]; j < A_i[i + 1]; j++) {
            AT_j[AT_i[A_j[j]]] = i;
            if (data) {
                offset = AT_i[A_j[j]] * bnnz;
                for (k = 0; k < block_size; k++)
                    for (m = 0; m < block_size; m++) AT_data[offset + k * block_size + m] = A_data[j * bnnz + m * block_size + k];
            }
            AT_i[A_j[j]]++;
        }
    }

    /*------------------------------------------------------------
     * AT_i[j] now points to the *end* of the jth row of entries
     * instead of the beginning.  Restore AT_i to front of row.
     *------------------------------------------------------------*/

    for (i = num_rowsAT; i > 0; i--) {
        AT_i[i] = AT_i[i - 1];
    }
    AT_i[0] = 0;

    return (0);
}

/*--------------------------------------------------------------------------
 * jx_BSRMatrixRead: Read a BSR matrix from a file.
 * This file format is compatible with faspsolver
 * File format:
 *   - ROW, COL, NNZ
 *   - block_size: size of each block
 *   - storage_manner: storage manner of each block
 *   - ROW+1: length of IA
 *   - IA(i), i=0:ROW
 *   - NNZ: length of JA
 *   - JA(i), i=0:NNZ-1
 *   - NNZ*nb*nb: length of val
 *   - val(i), i=0:NNZ*nb*nb-1
 *
 * \author zlj
 * \date   2025/10/08
 *--------------------------------------------------------------------------*/

jx_BSRMatrix* jx_BSRMatrixRead(const char* file_name)
{
    jx_BSRMatrix* matrix;

    FILE* fp = NULL;

    JX_Complex* matrix_data;
    JX_Int*     matrix_i;
    JX_Int*     matrix_j;
    // JX_Int*     matrix_bigj;
    JX_Int block_size;
    JX_Int num_rows;
    JX_Int num_cols;
    JX_Int num_nonzeros;
    JX_Int max_col        = 0;
    JX_Int storage_manner = 0; //! storage manner for each sub-block, 0: row-major order, 1: column-major order
    JX_Int file_base      = 0; // default is 0-based index, it automatically judged based on the file_name
    JX_Int j, n;

    /*----------------------------------------------------------
     * Read in the data
     *----------------------------------------------------------*/
    fp = fopen(file_name, "r");

    jx_fscanf(fp, "%d %d %d", &num_rows, &num_cols, &num_nonzeros); // dimensions of the problem
    jx_fscanf(fp, "%d", &block_size);                               // read the size of each block
    jx_fscanf(fp, "%d", &storage_manner);                           // read the storage_manner

    // allocate memory for IA, JA, and AA
    matrix_i    = jx_CTAlloc(JX_Int, num_rows + 1);
    matrix_j    = jx_CTAlloc(JX_Int, num_nonzeros);
    matrix_data = jx_CTAlloc(JX_Complex, num_nonzeros * block_size * block_size);

    // read IA
    jx_fscanf(fp, "%d", &n); // n = rows + 1
    jx_fscanf(fp, "%d", &matrix_i[0]);
    if (matrix_i[0] == 1) {
        file_base = 1;
        matrix_i[0] -= file_base;
    }
    for (j = 1; j < n; j++) {
        jx_fscanf(fp, "%d", &matrix_i[j]);
        matrix_i[j] -= file_base;
    }

    // read JA
    jx_fscanf(fp, "%d", &n); // n = nnz
    for (j = 0; j < n; j++) {
        jx_fscanf(fp, "%d", &matrix_j[j]);
        matrix_j[j] -= file_base;

        if (matrix_j[j] > max_col) {
            max_col = matrix_j[j];
        }
    }
    if (max_col + 1 != num_cols) {
        ERROR_PRINTF("%s %d, max_col: %d != num_cols: %d\n", __FUNCTION__, __LINE__, max_col + 1, num_cols);
        exit(jx_error_flag);
    }

    // read val
    jx_fscanf(fp, "%d", &n); // n = nnz * block_size * block_size
    if (storage_manner == 0) {   // row-major order
        for (j = 0; j < n; j++) {
            jx_fscanf(fp, "%le", &matrix_data[j]);
        }
    } else { // column-major order
        //! Note that storage manner is row-major order for each sub-block in jx_BSRMatrix
        JX_Int ind, loc_ind, bi, bj, bnnz = block_size * block_size;
        for (j = 0; j < n; j++) {
            loc_ind = j % bnnz;
            bi      = loc_ind / block_size; // local row index for each sub-block
            bj      = loc_ind % block_size; // local col index for each sub-block
            // (bi, bj) = bi * block_size + bj ==> (bj, bi) =  bj * block_size + bi
            ind = (j / bnnz) * bnnz + bj * block_size + bi;
            jx_fscanf(fp, "%le", &matrix_data[ind]);
        }
    }
    fclose(fp);

    // Create BSR matrix
    matrix                       = jx_BSRMatrixCreate(block_size, num_rows, num_cols, num_nonzeros);
    jx_BSRMatrixI(matrix)    = matrix_i;
    jx_BSRMatrixJ(matrix)    = matrix_j;
    jx_BSRMatrixData(matrix) = matrix_data;

    return matrix;
}

/*--------------------------------------------------------------------------
 * jx_BSRMatrixPrint: Write a file (file_base = 1) for BSR matrix.
 *
 * File format:
 *   - ROW, COL, NNZ
 *   - block_size: size of each block
 *   - storage_manner: storage manner of each block
 *   - ROW+1: length of IA
 *   - IA(i), i=0:ROW
 *   - NNZ: length of JA
 *   - JA(i), i=0:NNZ-1
 *   - NNZ*nb*nb: length of val
 *   - val(i), i=0:NNZ*nb*nb-1
 *
 * \author zlj
 * \date   2025/10/08
 *--------------------------------------------------------------------------*/
JX_Int jx_BSRMatrixPrint(jx_BSRMatrix* matrix, const char* file_name)
{
    FILE* fp = NULL;

    JX_Complex* matrix_data;
    JX_Int*     matrix_i;
    JX_Int*     matrix_j;
    JX_Int*  matrix_bigj;
    JX_Int      num_rows;
    JX_Int      num_cols;
    JX_Int      num_nonzeros;
    JX_Int      block_size;

    JX_Int storage_manner = 0; // storage manner for each sub-block, 0: row-major order, 1: column-major order
    JX_Int file_base      = 1; // default is 1-based index

    JX_Int j;

    JX_Int ierr = 0;

    /*----------------------------------------------------------
     * Print the matrix data
     *----------------------------------------------------------*/

    matrix_data  = jx_BSRMatrixData(matrix);
    matrix_i     = jx_BSRMatrixI(matrix);
    matrix_j     = jx_BSRMatrixJ(matrix);
    matrix_bigj  = jx_BSRMatrixBigJ(matrix);
    num_rows     = jx_BSRMatrixNumRows(matrix);
    num_cols     = jx_BSRMatrixNumCols(matrix);
    num_nonzeros = jx_BSRMatrixNumNonzeros(matrix);
    block_size   = jx_BSRMatrixBlockSize(matrix);

    fp = fopen(file_name, "w");

    jx_fprintf(fp, "%d %d %d\n", num_rows, num_cols, num_nonzeros);
    jx_fprintf(fp, "%d\n", block_size);
    jx_fprintf(fp, "%d\n", storage_manner);

    jx_fprintf(fp, "%d\n", num_rows + 1);
    for (j = 0; j <= num_rows; j++) {
        jx_fprintf(fp, "%d\n", matrix_i[j] + file_base);
    }

    jx_fprintf(fp, "%d\n", num_nonzeros);
    if (matrix_j) {
        for (j = 0; j < num_nonzeros; j++) {
            jx_fprintf(fp, "%d\n", matrix_j[j] + file_base);
        }
    }

    if (matrix_bigj) {
        for (j = 0; j < num_nonzeros; j++) {
            jx_fprintf(fp, "%d\n", matrix_bigj[j] + file_base);
        }
    }

    JX_Int num = num_nonzeros * block_size * block_size;
    jx_fprintf(fp, "%d\n", num);
    if (matrix_data) {
        for (j = 0; j < num; j++) {
#ifdef JX_COMPLEX
            jx_fprintf(fp, "%.14e , %.14e\n", jx_creal(matrix_data[j]), jx_cimag(matrix_data[j]));
#else
            jx_fprintf(fp, "%.14e\n", matrix_data[j]);
#endif
        }
    } else {
        jx_fprintf(fp, "Warning: No matrix data!\n");
    }

    fclose(fp);

    return ierr;
}

/*--------------------------------------------------------------------------
 * jx_BSRMatrixRead_Binary: Read a BSR matrix from a binary file.
 * This file format is compatible with faspsolver
 * File format:
 *   - ROW, COL, NNZ
 *   - block_size: size of each block
 *   - storage_manner: storage manner of each block
 *   - ROW+1: length of IA
 *   - IA(i), i=0:ROW
 *   - NNZ: length of JA
 *   - JA(i), i=0:NNZ-1
 *   - NNZ*nb*nb: length of val
 *   - val(i), i=0:NNZ*nb*nb-1
 *
 * \author zlj
 * \date   2025/10/08
 *--------------------------------------------------------------------------*/

jx_BSRMatrix* jx_BSRMatrixRead_Binary(const char* file_name)
{
    jx_BSRMatrix* matrix;

    FILE* fp = NULL;

    JX_Complex* matrix_data;
    JX_Int*     matrix_i;
    JX_Int*     matrix_j;
    // JX_Int*     matrix_bigj;
    JX_Int block_size;
    JX_Int num_rows;
    JX_Int num_cols;
    JX_Int num_nonzeros;
    JX_Int storage_manner = 0; //! storage manner for each sub-block, 0: row-major order, 1: column-major order
    JX_Int file_base      = 0; // default is 0-based index, it automatically judged based on the file_name
    JX_Int j, n;

    /*----------------------------------------------------------
     * Read in the data
     *----------------------------------------------------------*/
    fp = fopen(file_name, "rb");

    // dimensions of the problem
    jx_fread(&num_rows, sizeof(JX_Int), 1, fp);
    jx_fread(&num_cols, sizeof(JX_Int), 1, fp);
    jx_fread(&num_nonzeros, sizeof(JX_Int), 1, fp);

    jx_fread(&block_size, sizeof(JX_Int), 1, fp);     // read the size of each block
    jx_fread(&storage_manner, sizeof(JX_Int), 1, fp); // read the storage_manner

    // allocate memory for IA, JA, and AA
    matrix_i    = jx_CTAlloc(JX_Int, num_rows + 1);
    matrix_j    = jx_CTAlloc(JX_Int, num_nonzeros);
    matrix_data = jx_CTAlloc(JX_Complex, num_nonzeros * block_size * block_size);

    // read IA
    jx_fread(&n, sizeof(JX_Int), 1, fp); // n = rows + 1
    jx_fread(matrix_i, sizeof(JX_Int), n, fp);
    if (matrix_i[0] == 1) {
        file_base = 1;
        for (j = 0; j < n; j++) {
            matrix_i[j] -= file_base;
        }
    }

    // read JA
    jx_fread(&n, sizeof(JX_Int), 1, fp); // n = nnz
    jx_fread(matrix_j, sizeof(JX_Int), n, fp);
    if (file_base == 1) {
        for (j = 0; j < n; j++) {
            matrix_j[j] -= file_base;
        }
    }

    // read val
    jx_fread(&n, sizeof(JX_Int), 1, fp); // n = nnz * block_size * block_size

    if (storage_manner == 0) { // row-major order
        jx_fread(matrix_data, sizeof(JX_Complex), n, fp);
    } else { // column-major order
        //! Note that storage manner is row-major order for each sub-block in jx_BSRMatrix
        JX_Int      ind, loc_ind, bi, bj, bnnz = block_size * block_size;
        JX_Complex* matrix_data_tmp = jx_CTAlloc(JX_Complex, n);
        for (j = 0; j < n; j++) {
            loc_ind = j % bnnz;
            bi      = loc_ind / block_size; // local row index for each sub-block
            bj      = loc_ind % block_size; // local col index for each sub-block
            // (bi, bj) = bi * block_size + bj ==> (bj, bi) =  bj * block_size + bi
            ind              = (j / bnnz) * bnnz + bj * block_size + bi;
            matrix_data[ind] = matrix_data_tmp[j];
        }
        jx_TFree(matrix_data_tmp);
    }

    fclose(fp);

    // Create BSR matrix
    matrix                       = jx_BSRMatrixCreate(block_size, num_rows, num_cols, num_nonzeros);
    jx_BSRMatrixI(matrix)    = matrix_i;
    jx_BSRMatrixJ(matrix)    = matrix_j;
    jx_BSRMatrixData(matrix) = matrix_data;

    return matrix;
}

/*--------------------------------------------------------------------------
 * jx_BSRMatrixPrint_Binary: Write a binary file (file_base = 1) for BSR matrix.
 *
 * File format:
 *   - ROW, COL, NNZ
 *   - block_size: size of each block
 *   - storage_manner: storage manner of each block
 *   - ROW+1: length of IA
 *   - IA(i), i=0:ROW
 *   - NNZ: length of JA
 *   - JA(i), i=0:NNZ-1
 *   - NNZ*nb*nb: length of val
 *   - val(i), i=0:NNZ*nb*nb-1
 *
 * \author zlj
 * \date   08/25/2024
 *--------------------------------------------------------------------------*/
JX_Int jx_BSRMatrixPrint_Binary(jx_BSRMatrix* matrix, const char* file_name)
{
    FILE* fp = NULL;

    JX_Complex* matrix_data;
    JX_Int*     matrix_i;
    JX_Int*     matrix_j;
    JX_Int*  matrix_bigj;
    JX_Int      num_rows;
    JX_Int      num_cols;
    JX_Int      num_nonzeros;
    JX_Int      block_size;

    JX_Int storage_manner = 0; // storage manner for each sub-block, 0: row-major order, 1: column-major order
    JX_Int file_base      = 0; // default is 1-based index

    JX_Int j, temp = 0;

    JX_Int ierr = 0;

    /*----------------------------------------------------------
     * Print the matrix data
     *----------------------------------------------------------*/

    matrix_data  = jx_BSRMatrixData(matrix);
    matrix_i     = jx_BSRMatrixI(matrix);
    matrix_j     = jx_BSRMatrixJ(matrix);
    matrix_bigj  = jx_BSRMatrixBigJ(matrix);
    num_rows     = jx_BSRMatrixNumRows(matrix);
    num_cols     = jx_BSRMatrixNumCols(matrix);
    num_nonzeros = jx_BSRMatrixNumNonzeros(matrix);
    block_size   = jx_BSRMatrixBlockSize(matrix);

    fp = fopen(file_name, "wb");

    // write dimension of the block matrix
    jx_fwrite(&num_rows, sizeof(JX_Int), 1, fp);
    jx_fwrite(&num_cols, sizeof(JX_Int), 1, fp);
    jx_fwrite(&num_nonzeros, sizeof(JX_Int), 1, fp);
    jx_fwrite(&block_size, sizeof(JX_Int), 1, fp); // write block_size
    jx_fwrite(&storage_manner, sizeof(JX_Int), 1, fp);

    temp = num_rows + 1;
    jx_fwrite(&temp, sizeof(JX_Int), 1, fp);
    for (j = 0; j <= num_rows; j++) {
        temp = matrix_i[j] + file_base;
        jx_fwrite(&temp, sizeof(JX_Int), 1, fp);
    }

    jx_fwrite(&num_nonzeros, sizeof(JX_Int), 1, fp);
    if (matrix_j) {
        for (j = 0; j < num_nonzeros; j++) {
            temp = matrix_j[j] + file_base;
            jx_fwrite(&temp, sizeof(JX_Int), 1, fp);
        }
    }

    if (matrix_bigj) {
        JX_Int Big_temp = 0;
        for (j = 0; j < num_nonzeros; j++) {
            Big_temp = matrix_bigj[j] + file_base;
            jx_fwrite(&Big_temp, sizeof(JX_Int), 1, fp);
        }
    }

    JX_Int num = num_nonzeros * block_size * block_size;
    jx_fwrite(&num, sizeof(JX_Int), 1, fp);
    if (matrix_data) {
        for (j = 0; j < num; j++) {
#ifdef JX_COMPLEX
            // jx_fwrite(&jx_creal(matrix_data[j]), sizeof(JX_Real), 1, fp);
            // jx_fwrite(&jx_cimag(matrix_data[j]), sizeof(JX_Real), 1, fp);
#else
            jx_fwrite(&matrix_data[j], sizeof(JX_Complex), 1, fp);
#endif
        }
    } else {
        jx_fprintf(fp, "Warning: No matrix data!\n");
    }

    fclose(fp);

    return ierr;
}