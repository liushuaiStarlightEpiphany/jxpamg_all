//========================================================================//
//  JXFPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//


/*!
 *  seq_bsr_matrix.c -- basic operations for BSR matrices.
*  Date: 2025/10/08
*  Created by zlj
*/

#include "jxf_mv.h"
#include "jxf_bsr_mv.h"

/******************************************************************************
 *
 * Member functions for jxf_BSRMatrix class.
 *
 *****************************************************************************/

// #include "_jxf_parcsr_block_mv.h"

#define LB_VERSION 0 // Using the LAPACK/Blas library for inverse calculation

/*---------------------------------*/
/*--  Declare Private Functions  --*/
/*---------------------------------*/

#include "bsr_matrix_inv.inl"

/*--------------------------------------------------------------------------
 * jxf_BSRMatrixCreate
 *--------------------------------------------------------------------------*/

jxf_BSRMatrix* jxf_BSRMatrixCreate(JXF_Int block_size, JXF_Int num_rows, JXF_Int num_cols, JXF_Int num_nonzeros)
{
    jxf_BSRMatrix* matrix;

    matrix = jxf_CTAlloc(jxf_BSRMatrix, 1);

    jxf_BSRMatrixData(matrix)        = NULL;
    jxf_BSRMatrixI(matrix)           = NULL;
    jxf_BSRMatrixJ(matrix)           = NULL;
    jxf_BSRMatrixBigJ(matrix)        = NULL;
    jxf_BSRMatrixBlockSize(matrix)   = block_size;
    jxf_BSRMatrixNumRows(matrix)     = num_rows;
    jxf_BSRMatrixNumCols(matrix)     = num_cols;
    jxf_BSRMatrixNumNonzeros(matrix) = num_nonzeros;

    /* set defaults */
    jxf_BSRMatrixOwnsData(matrix) = 1;

    return matrix;
}

/*--------------------------------------------------------------------------
 * jxf_BSRMatrixDestroy
 *--------------------------------------------------------------------------*/

JXF_Int jxf_BSRMatrixDestroy(jxf_BSRMatrix* matrix)
{
    JXF_Int ierr = 0;

    if (matrix) {
        jxf_TFree(jxf_BSRMatrixI(matrix));
        if (jxf_BSRMatrixOwnsData(matrix)) {
            jxf_TFree(jxf_BSRMatrixData(matrix));
            jxf_TFree(jxf_BSRMatrixJ(matrix));
            jxf_TFree(jxf_BSRMatrixBigJ(matrix));
        }
        jxf_TFree(matrix);
    }

    return ierr;
}

/*--------------------------------------------------------------------------
 * jxf_BSRMatrixInitialize
 *--------------------------------------------------------------------------*/

JXF_Int jxf_BSRMatrixInitialize(jxf_BSRMatrix* matrix)
{
    JXF_Int block_size   = jxf_BSRMatrixBlockSize(matrix);
    JXF_Int num_rows     = jxf_BSRMatrixNumRows(matrix);
    JXF_Int num_nonzeros = jxf_BSRMatrixNumNonzeros(matrix);
    JXF_Int ierr         = 0, nnz;

    if (jxf_BSRMatrixI(matrix)) {
        jxf_TFree(jxf_BSRMatrixI(matrix));
    }
    if (jxf_BSRMatrixJ(matrix)) {
        jxf_TFree(jxf_BSRMatrixJ(matrix));
    }
    if (jxf_BSRMatrixBigJ(matrix)) {
        jxf_TFree(jxf_BSRMatrixBigJ(matrix));
    }
    if (jxf_BSRMatrixData(matrix)) {
        jxf_TFree(jxf_BSRMatrixData(matrix));
    }

    nnz                       = num_nonzeros * block_size * block_size;
    jxf_BSRMatrixI(matrix) = jxf_CTAlloc(JXF_Int, num_rows + 1);
    if (nnz) {
        jxf_BSRMatrixData(matrix) = jxf_CTAlloc(JXF_Complex, nnz);
    } else {
        jxf_BSRMatrixData(matrix) = NULL;
    }
    if (nnz) {
        jxf_BSRMatrixJ(matrix) = jxf_CTAlloc(JXF_Int, num_nonzeros);
    } else {
        jxf_BSRMatrixJ(matrix) = NULL;
    }

    return ierr;
}

/*--------------------------------------------------------------------------
 * jxf_BSRMatrixBigInitialize
 *--------------------------------------------------------------------------*/

JXF_Int jxf_BSRMatrixBigInitialize(jxf_BSRMatrix* matrix)
{
    JXF_Int block_size   = jxf_BSRMatrixBlockSize(matrix);
    JXF_Int num_rows     = jxf_BSRMatrixNumRows(matrix);
    JXF_Int num_nonzeros = jxf_BSRMatrixNumNonzeros(matrix);
    JXF_Int ierr         = 0, nnz;

    if (jxf_BSRMatrixI(matrix)) {
        jxf_TFree(jxf_BSRMatrixI(matrix));
    }
    if (jxf_BSRMatrixJ(matrix)) {
        jxf_TFree(jxf_BSRMatrixJ(matrix));
    }
    if (jxf_BSRMatrixBigJ(matrix)) {
        jxf_TFree(jxf_BSRMatrixBigJ(matrix));
    }
    if (jxf_BSRMatrixData(matrix)) {
        jxf_TFree(jxf_BSRMatrixData(matrix));
    }

    nnz                       = num_nonzeros * block_size * block_size;
    jxf_BSRMatrixI(matrix) = jxf_CTAlloc(JXF_Int, num_rows + 1);
    if (nnz) {
        jxf_BSRMatrixData(matrix) = jxf_CTAlloc(JXF_Complex, nnz);
    } else {
        jxf_BSRMatrixData(matrix) = NULL;
    }
    if (nnz) {
        jxf_BSRMatrixBigJ(matrix) = jxf_CTAlloc(JXF_Int, num_nonzeros);
    } else {
        jxf_BSRMatrixJ(matrix) = NULL;
    }

    return ierr;
}

/*--------------------------------------------------------------------------
 * jxf_BSRMatrixSetDataOwner
 *--------------------------------------------------------------------------*/

JXF_Int jxf_BSRMatrixSetDataOwner(jxf_BSRMatrix* matrix, JXF_Int owns_data)
{
    JXF_Int ierr = 0;

    jxf_BSRMatrixOwnsData(matrix) = owns_data;

    return ierr;
}

/*--------------------------------------------------------------------------
 * jxf_BSRMatrixCompress, take the F-norm for each sub block
 *--------------------------------------------------------------------------*/

jxf_CSRMatrix* jxf_BSRMatrixCompress(jxf_BSRMatrix* matrix)
{
    JXF_Int        block_size   = jxf_BSRMatrixBlockSize(matrix);
    JXF_Int        num_rows     = jxf_BSRMatrixNumRows(matrix);
    JXF_Int        num_cols     = jxf_BSRMatrixNumCols(matrix);
    JXF_Int        num_nonzeros = jxf_BSRMatrixNumNonzeros(matrix);
    JXF_Int*       matrix_i     = jxf_BSRMatrixI(matrix);
    JXF_Int*       matrix_j     = jxf_BSRMatrixJ(matrix);
    JXF_Complex*   matrix_data  = jxf_BSRMatrixData(matrix);
    jxf_CSRMatrix* matrix_C;
    JXF_Int *      matrix_C_i, *matrix_C_j, i, j, bnnz;
    JXF_Complex *  matrix_C_data, ddata;

    matrix_C = jxf_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
    jxf_CSRMatrixInitialize(matrix_C);
    matrix_C_i    = jxf_CSRMatrixI(matrix_C);
    matrix_C_j    = jxf_CSRMatrixJ(matrix_C);
    matrix_C_data = jxf_CSRMatrixData(matrix_C);

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
        matrix_C_data[i] = jxf_sqrt(ddata);
    }
    return matrix_C;
}

/*--------------------------------------------------------------------------
 * jxf_BSRMatrixGetSubmatrix, Aij = Aij_[subposition][subposition]
 *--------------------------------------------------------------------------*/

jxf_CSRMatrix* jxf_BSRMatrixGetSubmatrix(jxf_BSRMatrix* matrix, JXF_Int subposition)
{
    JXF_Int        block_size   = jxf_BSRMatrixBlockSize(matrix);
    JXF_Int        num_rows     = jxf_BSRMatrixNumRows(matrix);
    JXF_Int        num_cols     = jxf_BSRMatrixNumCols(matrix);
    JXF_Int        num_nonzeros = jxf_BSRMatrixNumNonzeros(matrix);
    JXF_Int*       matrix_i     = jxf_BSRMatrixI(matrix);
    JXF_Int*       matrix_j     = jxf_BSRMatrixJ(matrix);
    JXF_Complex*   matrix_data  = jxf_BSRMatrixData(matrix);
    jxf_CSRMatrix* matrix_C;
    JXF_Int *      matrix_C_i, *matrix_C_j, i, j, bnnz;
    JXF_Complex*   matrix_C_data;

    matrix_C = jxf_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
    jxf_CSRMatrixInitialize(matrix_C);
    matrix_C_i    = jxf_CSRMatrixI(matrix_C);
    matrix_C_j    = jxf_CSRMatrixJ(matrix_C);
    matrix_C_data = jxf_CSRMatrixData(matrix_C);

    bnnz = block_size * block_size;
    for (i = 0; i < num_rows + 1; i++) {
        matrix_C_i[i] = matrix_i[i];
    }

    JXF_Int pos = subposition * block_size + subposition; // Aij_[subposition][subposition]
    for (i = 0; i < num_nonzeros; i++) {
        matrix_C_j[i]    = matrix_j[i];
        matrix_C_data[i] = matrix_data[i * bnnz + pos]; // Aij_[subposition][subposition]
    }

    // printf(" [%s:%d] CSR = %p \n",__FUNCTION__, __LINE__, (void*)matrix_C);
    // fflush(stdout);  
    // printf(" [%s:%d] jxf_CSRMatrixI(CSR) = %p \n",__FUNCTION__, __LINE__, (void*)jxf_CSRMatrixI(matrix_C));
    // fflush(stdout);  


    return matrix_C;
}

/*--------------------------------------------------------------------------
 * jxf_BSRMatrixConvertToCSRMatrix
 *--------------------------------------------------------------------------*/

jxf_CSRMatrix* jxf_BSRMatrixConvertToCSRMatrix(jxf_BSRMatrix* matrix)
{
    JXF_Int      block_size   = jxf_BSRMatrixBlockSize(matrix);
    JXF_Int      num_rows     = jxf_BSRMatrixNumRows(matrix);
    JXF_Int      num_cols     = jxf_BSRMatrixNumCols(matrix);
    JXF_Int      num_nonzeros = jxf_BSRMatrixNumNonzeros(matrix);
    JXF_Int*     matrix_i     = jxf_BSRMatrixI(matrix);
    JXF_Int*     matrix_j     = jxf_BSRMatrixJ(matrix);
    JXF_Complex* matrix_data  = jxf_BSRMatrixData(matrix);

    jxf_CSRMatrix* matrix_C;
    JXF_Int        i, j, k, ii, C_ii, bnnz, new_nrows, new_ncols, new_num_nonzeros;
    JXF_Int *      matrix_C_i, *matrix_C_j;
    JXF_Complex*   matrix_C_data;

    bnnz             = block_size * block_size;
    new_nrows        = num_rows * block_size;
    new_ncols        = num_cols * block_size;
    new_num_nonzeros = block_size * block_size * num_nonzeros;
    matrix_C         = jxf_CSRMatrixCreate(new_nrows, new_ncols, new_num_nonzeros);
    jxf_CSRMatrixInitialize(matrix_C);
    matrix_C_i    = jxf_CSRMatrixI(matrix_C);
    matrix_C_j    = jxf_CSRMatrixJ(matrix_C);
    matrix_C_data = jxf_CSRMatrixData(matrix_C);
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
 * jxf_BSRMatrixConvertFromCSRMatrix

 * this doesn't properly convert the parcsr off_diag matrices
   (because here we assume the matrix is square - we don't check what the
    number of columns should be ) - it can only be used for the diag part
 *--------------------------------------------------------------------------*/

jxf_BSRMatrix* jxf_BSRMatrixConvertFromCSRMatrix(jxf_CSRMatrix* matrix, JXF_Int matrix_C_block_size)
{
    JXF_Int      num_rows    = jxf_CSRMatrixNumRows(matrix);
    JXF_Int      num_cols    = jxf_CSRMatrixNumCols(matrix);
    JXF_Int*     matrix_i    = jxf_CSRMatrixI(matrix);
    JXF_Int*     matrix_j    = jxf_CSRMatrixJ(matrix);
    JXF_Complex* matrix_data = jxf_CSRMatrixData(matrix);

    jxf_BSRMatrix* matrix_C;
    JXF_Int *      matrix_C_i, *matrix_C_j;
    JXF_Complex*   matrix_C_data;
    JXF_Int        matrix_C_num_rows, matrix_C_num_cols, matrix_C_num_nonzeros;
    JXF_Int        i, j, ii, jj, s_jj, index, *counter;

    matrix_C_num_rows = num_rows / matrix_C_block_size;
    matrix_C_num_cols = num_cols / matrix_C_block_size;

    counter = jxf_CTAlloc(JXF_Int, matrix_C_num_cols);
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
    matrix_C = jxf_BSRMatrixCreate(matrix_C_block_size, matrix_C_num_rows, matrix_C_num_cols, matrix_C_num_nonzeros);
    jxf_BSRMatrixInitialize(matrix_C);
    matrix_C_i    = jxf_BSRMatrixI(matrix_C);
    matrix_C_j    = jxf_BSRMatrixJ(matrix_C);
    matrix_C_data = jxf_BSRMatrixData(matrix_C);

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

    jxf_TFree(counter);

    return matrix_C;
}

/*--------------------------------------------------------------------------
 * jxf_BSRMatrixBlockAdd
 * (o = i1 + i2)
 *--------------------------------------------------------------------------*/
JXF_Int jxf_BSRMatrixBlockAdd(JXF_Complex* i1, JXF_Complex* i2, JXF_Complex* o, JXF_Int block_size)
{
    JXF_Int i;
    JXF_Int sz = block_size * block_size;

    for (i = 0; i < sz; i++) {
        o[i] = i1[i] + i2[i];
    }

    return 0;
}

/*--------------------------------------------------------------------------
 * jxf_BSRMatrixBlockAddAccumulate
 * (o = i1 + o)
 *--------------------------------------------------------------------------*/
JXF_Int jxf_BSRMatrixBlockAddAccumulate(JXF_Complex* i1, JXF_Complex* o, JXF_Int block_size)
{
    JXF_Int i;
    JXF_Int sz = block_size * block_size;

    for (i = 0; i < sz; i++) {
        o[i] += i1[i];
    }

    return 0;
}

/*--------------------------------------------------------------------------
 * jxf_BSRMatrixBlockAddAccumulateDiag
 * (diag(o) = diag(i1) + diag(o))
 *--------------------------------------------------------------------------*/
JXF_Int jxf_BSRMatrixBlockAddAccumulateDiag(JXF_Complex* i1, JXF_Complex* o, JXF_Int block_size)
{
    JXF_Int i;

    for (i = 0; i < block_size; i++) {
        o[i * block_size + i] += i1[i * block_size + i];
    }
    return 0;
}

/*--------------------------------------------------------------------------
 * jxf_BSRMatrixBlockAddAccumulateDiagCheckSign
 * only add elements of sign*i1 that are negative (sign is size block_size)
 * (diag(o) = diag(i1) + diag(o))
 *--------------------------------------------------------------------------*/
JXF_Int jxf_BSRMatrixBlockAddAccumulateDiagCheckSign(JXF_Complex* i1, JXF_Complex* o, JXF_Int block_size, JXF_Real* sign)
{
    JXF_Int  i;
    JXF_Real tmp;

    for (i = 0; i < block_size; i++) {
        tmp = (JXF_Real)i1[i * block_size + i] * sign[i];
        if (tmp < 0) {
            o[i * block_size + i] += i1[i * block_size + i];
        }
    }

    return 0;
}

/*--------------------------------------------------------------------------
 *  jxf_BSRMatrixComputeSign

 * o = sign(diag(i1))
 *--------------------------------------------------------------------------*/

JXF_Int jxf_BSRMatrixComputeSign(JXF_Complex* i1, JXF_Complex* o, JXF_Int block_size)
{
    JXF_Int i;

    for (i = 0; i < block_size; i++) {
        if ((JXF_Real)i1[i * block_size + i] < 0) {
            o[i] = -1;
        } else {
            o[i] = 1;
        }
    }

    return 0;
}

/*--------------------------------------------------------------------------
 * jxf_BSRMatrixBlockSetScalar
 * (each entry in block o is set to beta )
 *--------------------------------------------------------------------------*/
JXF_Int jxf_BSRMatrixBlockSetScalar(JXF_Complex* o, JXF_Complex beta, JXF_Int block_size)
{
    JXF_Int i;
    JXF_Int sz = block_size * block_size;

    for (i = 0; i < sz; i++) {
        o[i] = beta;
    }

    return 0;
}

/*--------------------------------------------------------------------------
 * jxf_BSRMatrixBlockCopyData
 * (o = beta*i1 )
 *--------------------------------------------------------------------------*/
JXF_Int jxf_BSRMatrixBlockCopyData(JXF_Complex* i1, JXF_Complex* o, JXF_Complex beta, JXF_Int block_size)
{
    JXF_Int i;
    JXF_Int sz = block_size * block_size;

    for (i = 0; i < sz; i++) {
        o[i] = beta * i1[i];
    }

    return 0;
}

/*--------------------------------------------------------------------------
 * jxf_BSRMatrixBlockCopyDataDiag - zeros off-diag entries
 * (o = beta*diag(i1))
 *--------------------------------------------------------------------------*/
JXF_Int jxf_BSRMatrixBlockCopyDataDiag(JXF_Complex* i1, JXF_Complex* o, JXF_Complex beta, JXF_Int block_size)
{
    JXF_Int i;

    JXF_Int sz = block_size * block_size;

    for (i = 0; i < sz; i++) {
        o[i] = 0.0;
    }

    for (i = 0; i < block_size; i++) {
        o[i * block_size + i] = beta * i1[i * block_size + i];
    }

    return 0;
}

/*--------------------------------------------------------------------------
 * jxf_BSRMatrixBlockTranspose
 * (o = i1' )
 *--------------------------------------------------------------------------*/
JXF_Int jxf_BSRMatrixBlockTranspose(JXF_Complex* i1, JXF_Complex* o, JXF_Int block_size)
{
    JXF_Int i, j;

    for (i = 0; i < block_size; i++)
        for (j = 0; j < block_size; j++) {
            o[i * block_size + j] = i1[j * block_size + i];
        }
    return 0;
}

/*--------------------------------------------------------------------------
 * jxf_BSRMatrixBlockNorm
 * (out = norm(data) )
 *
 *  (note: these are not all actually "norms")
 *
 *--------------------------------------------------------------------------*/
JXF_Int jxf_BSRMatrixBlockNorm(JXF_Int norm_type, JXF_Complex* data, JXF_Real* out, JXF_Int block_size)
{
    JXF_Int   ierr = 0;
    JXF_Int   i, j;
    JXF_Real  sum = 0.0;
    JXF_Real* totals;
    JXF_Int   sz = block_size * block_size;

    switch (norm_type) {
        case 8: /* pp: pressure entry in each block for reservior  */
            {
                sum = data[0];
                break;
            }
        case 7: /* |pp|: abs of pressure entry in each block for reservior  */
            {
                sum = jxf_cabs(data[0]);
                break;
            }
        case 6: /* sum of all elements in the block  */
            {
                for (i = 0; i < sz; i++) {
                    sum += (JXF_Real)(data[i]);
                }
                break;
            }
        case 5: /* one norm  - max col sum*/
            {

                totals = jxf_CTAlloc(JXF_Real, block_size);
                for (i = 0; i < block_size; i++) /* row */
                {
                    for (j = 0; j < block_size; j++) /* col */
                    {
                        totals[j] += jxf_cabs(data[i * block_size + j]);
                    }
                }

                sum = totals[0];
                for (j = 1; j < block_size; j++) /* col */
                {
                    if (totals[j] > sum) {
                        sum = totals[j];
                    }
                }
                jxf_TFree(totals);

                break;
            }
        case 4: /* inf norm - max row sum */
            {

                totals = jxf_CTAlloc(JXF_Real, block_size);
                for (i = 0; i < block_size; i++) /* row */
                {
                    for (j = 0; j < block_size; j++) /* col */
                    {
                        totals[i] += jxf_cabs(data[i * block_size + j]);
                    }
                }

                sum = totals[0];
                for (i = 1; i < block_size; i++) /* row */
                {
                    if (totals[i] > sum) {
                        sum = totals[i];
                    }
                }
                jxf_TFree(totals);

                break;
            }

        case 3: /* largest element of block (return value includes sign) */
            {

                sum = (JXF_Real)data[0];

                for (i = 0; i < sz; i++) {
                    if (jxf_cabs(data[i]) > jxf_cabs(sum)) {
                        sum = (JXF_Real)data[i];
                    }
                }

                break;
            }
        case 2: /* sum of abs values of all elements in the block  */
            {
                for (i = 0; i < sz; i++) {
                    sum += jxf_cabs(data[i]);
                }
                break;
            }

        default: /* 1 = frobenius*/
            {
                for (i = 0; i < sz; i++) {
                    sum += ((JXF_Real)data[i]) * ((JXF_Real)data[i]);
                }
                sum = jxf_sqrt(sum);
            }
    }

    *out = sum;

    return ierr;
}

/*--------------------------------------------------------------------------
 * jxf_BSRMatrixBlockMultAdd
 * (o = i1 * i2 + beta * o)
 *--------------------------------------------------------------------------*/
JXF_Int jxf_BSRMatrixBlockMultAdd(JXF_Complex* i1, JXF_Complex* i2, JXF_Complex beta, JXF_Complex* o, JXF_Int block_size)
{

#if LB_VERSION
    {
        JXF_Complex alp = 1.0;
        dgemm_("N", "N", &block_size, &block_size, &block_size, &alp, i2, &block_size, i1, &block_size, &beta, o, &block_size);
    }
#else
    {
        JXF_Int     i, j, k;
        JXF_Complex ddata;

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
 * jxf_BSRMatrixBlockMultAddDiag
 * (diag(o) = diag(i1) * diag(i2) + beta * diag(o))
 *--------------------------------------------------------------------------*/
JXF_Int jxf_BSRMatrixBlockMultAddDiag(JXF_Complex* i1, JXF_Complex* i2, JXF_Complex beta, JXF_Complex* o, JXF_Int block_size)
{
    JXF_Int i;

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
 * jxf_BSRMatrixBlockMultAddDiagCheckSign
 *
 *  only mult elements if sign*diag(i2) is negative
 *(diag(o) = diag(i1) * diag(i2) + beta * diag(o))
 *--------------------------------------------------------------------------*/
JXF_Int jxf_BSRMatrixBlockMultAddDiagCheckSign(JXF_Complex* i1, JXF_Complex* i2, JXF_Complex beta, JXF_Complex* o, JXF_Int block_size,
                                                     JXF_Real* sign)
{
    JXF_Int  i;
    JXF_Real tmp;

    if (beta == 0.0) {
        for (i = 0; i < block_size; i++) {
            tmp = (JXF_Real)i2[i * block_size + i] * sign[i];
            if (tmp < 0) {
                o[i * block_size + i] = i1[i * block_size + i] * i2[i * block_size + i];
            }
        }
    } else if (beta == 1.0) {
        for (i = 0; i < block_size; i++) {
            tmp = (JXF_Real)i2[i * block_size + i] * sign[i];
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
 * jxf_BSRMatrixBlockMultAddDiag2 (scales cols of il by diag of i2)
 * ((o) = (i1) * diag(i2) + beta * (o))
 *--------------------------------------------------------------------------*/
JXF_Int jxf_BSRMatrixBlockMultAddDiag2(JXF_Complex* i1, JXF_Complex* i2, JXF_Complex beta, JXF_Complex* o, JXF_Int block_size)
{
    JXF_Int i, j;

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
 * jxf_BSRMatrixBlockMultAddDiag3 (scales cols of il by i2 -
                                          whose diag elements are row sums)
 * ((o) = (i1) * diag(i2) + beta * (o))
 *--------------------------------------------------------------------------*/
JXF_Int jxf_BSRMatrixBlockMultAddDiag3(JXF_Complex* i1, JXF_Complex* i2, JXF_Complex beta, JXF_Complex* o, JXF_Int block_size)
{
    JXF_Int i, j;

    JXF_Complex* row_sum;

    row_sum = jxf_CTAlloc(JXF_Complex, block_size);
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

    jxf_TFree(row_sum);

    return 0;
}

// 改进的块矩阵-向量乘积
JXF_Int jxf_BSRMatrixBlockMatvec_Stable(JXF_Real alpha, JXF_Complex* mat, 
                                       JXF_Complex* v, JXF_Complex beta, 
                                       JXF_Complex* ov, JXF_Int block_size)
{
    JXF_Int i, j;
    JXF_Real comp;  // 补偿项
    
    if (alpha == 0.0) {
        for (j = 0; j < block_size; j++) {
            ov[j] *= beta;
        }
        return 0;
    }
    
    // 临时使用双精度计算
    double* tmp_ov = (double*)malloc(block_size * sizeof(double));
    for (j = 0; j < block_size; j++) {
        tmp_ov[j] = (double)(ov[j] * beta);
    }
    
    for (i = 0; i < block_size; i++) {
        double sum = 0.0;
        double c = 0.0;  // Kahan补偿
        
        for (j = 0; j < block_size; j++) {
            double y = (double)mat[i*block_size + j] * (double)v[j] - c;
            double t = sum + y;
            c = (t - sum) - y;  // 补偿下一项
            sum = t;
        }
        
        tmp_ov[i] += (double)alpha * sum;
        ov[i] = (JXF_Complex)tmp_ov[i];
    }
    
    free(tmp_ov);
    return 0;
}
/*--------------------------------------------------------------------------
 * jxf_BSRMatrixBlockMatvec
 * (ov = alpha* mat * v + beta * ov)
 * mat is the matrix - size is block_size^2
 * alpha and beta are scalars
 *--------------------------------------------------------------------------*/

JXF_Int jxf_BSRMatrixBlockMatvec(JXF_Complex alpha, JXF_Complex* mat, JXF_Complex* v, JXF_Complex beta, JXF_Complex* ov,
                                       JXF_Int block_size)
{
    JXF_Int ierr = 0;

#if LB_VERSION
    {
        JXF_Int one = 1;

        dgemv_("T", &block_size, &block_size, &alpha, mat, &block_size, v, &one, &beta, ov, &one);
    }

#else
    {
        JXF_Int     i, j;
        JXF_Complex ddata;

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

// 混合精度的块LU分解求解
JXF_Int jxf_BSRMatrixBlockInvMatvec_MixedPrec(JXF_Complex* mat_f, 
                                             JXF_Complex* v_f, 
                                             JXF_Complex* ov_f, 
                                             JXF_Int block_size)
{
    // 转换为双精度
    double* mat_d = (double*)malloc(block_size * block_size * sizeof(double));
    double* v_d = (double*)malloc(block_size * sizeof(double));
    double* ov_d = (double*)malloc(block_size * sizeof(double));
    
    for (int i = 0; i < block_size * block_size; i++) {
        mat_d[i] = (double)mat_f[i];
    }
    for (int i = 0; i < block_size; i++) {
        v_d[i] = (double)v_f[i];
        ov_d[i] = 0.0;
    }
    
    // 使用双精度LU分解
    int* ipiv = (int*)malloc(block_size * sizeof(int));
    int info = 0;
    
    // 使用LAPACK的dgetrf/dgetrs（如果有）
    // 或者自己实现稳定的LU分解
    
    // 简化的稳定LU分解
    for (int i = 0; i < block_size; i++) {
        // 选主元
        int max_row = i;
        double max_val = fabs(mat_d[i*block_size + i]);
        for (int k = i+1; k < block_size; k++) {
            if (fabs(mat_d[k*block_size + i]) > max_val) {
                max_val = fabs(mat_d[k*block_size + i]);
                max_row = k;
            }
        }
        
        if (max_val < 1e-15) {
            // 奇异矩阵，使用对角近似
            for (int j = 0; j < block_size; j++) {
                double diag = mat_d[j*block_size + j];
                ov_d[j] = (fabs(diag) > 1e-12) ? v_d[j] / diag : 0.0;
            }
            break;
        }
        
        // 行交换
        if (max_row != i) {
            for (int j = 0; j < block_size; j++) {
                double temp = mat_d[i*block_size + j];
                mat_d[i*block_size + j] = mat_d[max_row*block_size + j];
                mat_d[max_row*block_size + j] = temp;
            }
            double temp_v = v_d[i];
            v_d[i] = v_d[max_row];
            v_d[max_row] = temp_v;
        }
        
        ipiv[i] = max_row;
        
        // 消元
        double pivot = mat_d[i*block_size + i];
        for (int k = i+1; k < block_size; k++) {
            double factor = mat_d[k*block_size + i] / pivot;
            mat_d[k*block_size + i] = factor;
            for (int j = i+1; j < block_size; j++) {
                mat_d[k*block_size + j] -= factor * mat_d[i*block_size + j];
            }
        }
    }
    
    // 前向替换（L*y = Pb）
    for (int i = 0; i < block_size; i++) {
        ov_d[i] = v_d[i];
        for (int j = 0; j < i; j++) {
            ov_d[i] -= mat_d[i*block_size + j] * ov_d[j];
        }
    }
    
    // 后向替换（U*x = y）
    for (int i = block_size-1; i >= 0; i--) {
        for (int j = i+1; j < block_size; j++) {
            ov_d[i] -= mat_d[i*block_size + j] * ov_d[j];
        }
        double diag = mat_d[i*block_size + i];
        if (fabs(diag) > 1e-15) {
            ov_d[i] /= diag;
        } else {
            ov_d[i] = 0.0;
        }
    }
    
    // 转换回单精度
    for (int i = 0; i < block_size; i++) {
        ov_f[i] = (JXF_Complex)ov_d[i];
    }
    
    free(mat_d);
    free(v_d);
    free(ov_d);
    free(ipiv);
    
    return 0;
}

// 使用双精度计算的块求逆
JXF_Int jxf_BSRMatrixBlockInvMatvec_Stable(JXF_Complex* mat, 
                                          JXF_Complex* v, 
                                          JXF_Complex* ov, 
                                          JXF_Int block_size)
{
    if (block_size == 2) {
        // 对于2×2块，使用双精度直接计算
        double a11 = (double)mat[0];
        double a12 = (double)mat[1];
        double a21 = (double)mat[2];
        double a22 = (double)mat[3];
        
        double det = a11 * a22 - a12 * a21;
        
        // 检查条件数
        if (fabs(det) < 1e-12 * (fabs(a11*a22) + fabs(a12*a21))) {
            // 病态矩阵，使用对角近似
            double b1 = (double)v[0];
            double b2 = (double)v[1];
            
            ov[0] = (JXF_Complex)(b1 / (a11 + 1e-12));
            ov[1] = (JXF_Complex)(b2 / (a22 + 1e-12));
            return 0;
        }
        
        double inv_det = 1.0 / det;
        double b1 = (double)v[0];
        double b2 = (double)v[1];
        
        ov[0] = (JXF_Complex)((a22 * b1 - a12 * b2) * inv_det);
        ov[1] = (JXF_Complex)((-a21 * b1 + a11 * b2) * inv_det);
        return 0;
    }
    
    // 对于更大的块，使用混合精度LU分解
    return jxf_BSRMatrixBlockInvMatvec_MixedPrec(mat, v, ov, block_size);
}



/*--------------------------------------------------------------------------
 * jxf_BSRMatrixBlockInvMatvec
 * (ov = mat^{-1} * v)
 * o and v are vectors
 * mat is the matrix - size is block_size^2
 *--------------------------------------------------------------------------*/
JXF_Int jxf_BSRMatrixBlockInvMatvec(JXF_Complex* mat, JXF_Complex* v, JXF_Complex* ov, JXF_Int block_size)
{
    JXF_Int      ierr = 0;
    JXF_Complex* mat_i;

    mat_i = jxf_CTAlloc(JXF_Complex, block_size * block_size);

#if LB_VERSION
    {

        JXF_Int  one, info;
        JXF_Int* piv;
        JXF_Int  sz;

        one = 1;
        piv = jxf_CTAlloc(JXF_Int, block_size);
        sz  = block_size * block_size;

        /* copy v to ov and  mat to mat_i*/

        dcopy_(&sz, mat, &one, mat_i, &one);
        dcopy_(&block_size, v, &one, ov, &one);

        /* writes over mat_i with LU */
        dgetrf_(&block_size, &block_size, mat_i, &block_size, piv, &info);
        if (info) {
            jxf_TFree(mat_i);
            jxf_TFree(piv);
            return (-1);
        }

        /* writes over ov */
        dgetrs_("T", &block_size, &one, mat_i, &block_size, piv, ov, &block_size, &info);
        if (info) {
            jxf_TFree(mat_i);
            jxf_TFree(piv);
            return (-1);
        }

        jxf_TFree(piv);
    }

#else
    {
        JXF_Int     m, j, k;
        JXF_Int     piv_row;
        JXF_Real    eps;
        JXF_Complex factor;
        JXF_Complex piv, tmp;
        eps = 1.0e-6;

        if (block_size == 1) {
            if (jxf_cabs(mat[0]) > 1e-10) {
                ov[0] = v[0] / mat[0];
                jxf_TFree(mat_i);
                return (ierr);
            } else {
                /* jxf_printf("GE zero pivot error\n"); */
                jxf_TFree(mat_i);
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
            jxf_smat_inv(mat_i, block_size);

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
                    if (jxf_cabs(mat_i[j * block_size + k]) > jxf_cabs(piv)) {
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

                if (jxf_cabs(piv) > eps) {
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
                    /* jxf_printf("Block of matrix is nearly singular: zero pivot error\n");  */
                    jxf_TFree(mat_i);
                    return (-1);
                }
            }

            /* we also need to check the pivot in the last row to see if it is zero */
            k = block_size - 1; /* last row */
            if (jxf_cabs(mat_i[k * block_size + k]) < eps) {
                /* jxf_printf("Block of matrix is nearly singular: zero pivot error\n");  */
                jxf_TFree(mat_i);
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

    jxf_TFree(mat_i);

    return (ierr);
}

/*--------------------------------------------------------------------------
 * jxf_BSRMatrixBlockInvMult
 * (o = i1^{-1} * i2)
 *--------------------------------------------------------------------------*/
JXF_Int jxf_BSRMatrixBlockInvMult(JXF_Complex* i1, JXF_Complex* i2, JXF_Complex* o, JXF_Int block_size)
{

    JXF_Int      ierr = 0;
    JXF_Int      i, j;
    JXF_Complex* m_i1;

    m_i1 = jxf_CTAlloc(JXF_Complex, block_size * block_size);

#if LB_VERSION
    {

        JXF_Int  one, info;
        JXF_Int* piv;
        JXF_Int  sz;

        JXF_Complex* i2_t;

        one  = 1;
        i2_t = jxf_CTAlloc(JXF_Complex, block_size * block_size);
        piv  = jxf_CTAlloc(JXF_Int, block_size);

        /* copy i1 to m_i1*/
        sz = block_size * block_size;
        dcopy_(&sz, i1, &one, m_i1, &one);

        /* writes over m_i1 with LU */
        dgetrf_(&block_size, &block_size, m_i1, &block_size, piv, &info);
        if (info) {
            jxf_TFree(m_i1);
            jxf_TFree(i2_t);
            jxf_TFree(piv);
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
            jxf_TFree(m_i1);
            jxf_TFree(i2_t);
            jxf_TFree(piv);
            return (-1);
        }

        /* ans. is the transpose of i2_t*/
        for (i = 0; i < block_size; i++) {
            for (j = 0; j < block_size; j++) {
                o[i * block_size + j] = i2_t[j * block_size + i];
            }
        }

        jxf_TFree(i2_t);
        jxf_TFree(piv);
    }

#else
    {
        JXF_Int     m, k;
        JXF_Int     piv_row;
        JXF_Real    eps;
        JXF_Complex factor;
        JXF_Complex piv, tmp;

        eps = 1.0e-6;

        if (block_size == 1) {
            if (jxf_cabs(m_i1[0]) > 1e-10) {
                o[0] = i2[0] / i1[0];
                jxf_TFree(m_i1);
                return (ierr);
            } else {
                /* jxf_printf("GE zero pivot error\n"); */
                jxf_TFree(m_i1);
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
                    if (jxf_cabs(m_i1[j * block_size + k]) > jxf_cabs(piv)) {
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

                if (jxf_cabs(piv) > eps) {
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
                    /* jxf_printf("Block of matrix is nearly singular: zero pivot error\n"); */
                    jxf_TFree(m_i1);
                    return (-1);
                }
            }

            /* we also need to check the pivot in the last row to see if it is zero */
            k = block_size - 1; /* last row */
            if (jxf_cabs(m_i1[k * block_size + k]) < eps) {
                /* jxf_printf("Block of matrix is nearly singular: zero pivot error\n"); */
                jxf_TFree(m_i1);
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
    jxf_TFree(m_i1);

    return ierr;
}

/*--------------------------------------------------------------------------
 * jxf_BSRMatrixBlockMultInv
 * (o = i2*il^(-1))
 *--------------------------------------------------------------------------*/
JXF_Int jxf_BSRMatrixBlockMultInv(JXF_Complex* i1, JXF_Complex* i2, JXF_Complex* o, JXF_Int block_size)
{

    JXF_Int ierr = 0;

#if LB_VERSION

    {
        /* same as solving A^T C^T = B^T */
        JXF_Complex* m_i1;
        JXF_Int      info;
        JXF_Int*     piv;
        JXF_Int      sz, one;

        piv  = jxf_CTAlloc(JXF_Int, block_size);
        m_i1 = jxf_CTAlloc(JXF_Complex, block_size * block_size);
        one  = 1;
        sz   = block_size * block_size;

        /* copy i1 to m_i1 and i2 to o*/

        dcopy_(&sz, i1, &one, m_i1, &one);
        dcopy_(&sz, i2, &one, o, &one);

        /* writes over m_i1 with LU */
        dgetrf_(&block_size, &block_size, m_i1, &block_size, piv, &info);
        if (info) {
            jxf_TFree(m_i1);
            jxf_TFree(piv);
            return (-1);
        }
        /* writes over B */
        dgetrs_("N", &block_size, &block_size, m_i1, &block_size, piv, o, &block_size, &info);
        if (info) {
            jxf_TFree(m_i1);
            jxf_TFree(piv);
            return (-1);
        }

        jxf_TFree(m_i1);
        jxf_TFree(piv);
    }

#else
    {
        JXF_Real     eps;
        JXF_Complex *i1_t, *i2_t, *o_t;

        eps = 1.0e-12;

        if (block_size == 1) {
            if (jxf_cabs(i1[0]) > eps) {
                o[0] = i2[0] / i1[0];
                return (ierr);
            } else {
                /* jxf_printf("GE zero pivot error\n"); */
                return (-1);
            }
        } else {

            i1_t = jxf_CTAlloc(JXF_Complex, block_size * block_size);
            i2_t = jxf_CTAlloc(JXF_Complex, block_size * block_size);
            o_t  = jxf_CTAlloc(JXF_Complex, block_size * block_size);

            /* TO DO:: this could be done more efficiently! */
            jxf_BSRMatrixBlockTranspose(i1, i1_t, block_size);
            jxf_BSRMatrixBlockTranspose(i2, i2_t, block_size);
            ierr = jxf_BSRMatrixBlockInvMult(i1_t, i2_t, o_t, block_size);

            if (!ierr) {
                jxf_BSRMatrixBlockTranspose(o_t, o, block_size);
            }

            jxf_TFree(i1_t);
            jxf_TFree(i2_t);
            jxf_TFree(o_t);
        }
    }

#endif
    return (ierr);
}

/*--------------------------------------------------------------------------
 * jxf_BSRMatrixBlockInvMultDiag - zeros off-d entires
 * (o = diag(i1)^{-1} * diag(i2))
 *--------------------------------------------------------------------------*/
JXF_Int jxf_BSRMatrixBlockInvMultDiag(JXF_Complex* i1, JXF_Complex* i2, JXF_Complex* o, JXF_Int block_size)
{

    JXF_Int  ierr = 0;
    JXF_Int  i;
    JXF_Int  sz  = block_size * block_size;
    JXF_Real eps = 1.0e-8;

    for (i = 0; i < sz; i++) {
        o[i] = 0.0;
    }

    for (i = 0; i < block_size; i++) {
        if (jxf_cabs(i1[i * block_size + i]) > eps) {
            o[i * block_size + i] = i2[i * block_size + i] / i1[i * block_size + i];
        } else {
            /* jxf_printf("GE zero pivot error\n"); */
            return (-1);
        }
    }

    return (ierr);
}

/*--------------------------------------------------------------------------
 * jxf_BSRMatrixBlockInvMultDiag2
 * (o = (i1)* diag(i2)^-1) - so this scales the cols of il by
                             the diag entries in i2
 *--------------------------------------------------------------------------*/
JXF_Int jxf_BSRMatrixBlockInvMultDiag2(JXF_Complex* i1, JXF_Complex* i2, JXF_Complex* o, JXF_Int block_size)
{

    JXF_Int ierr = 0;
    JXF_Int i, j;

    JXF_Real    eps = 1.0e-8;
    JXF_Complex tmp;

    for (i = 0; i < block_size; i++) {
        if (jxf_cabs(i2[i * block_size + i]) > eps) {
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
 * jxf_BSRMatrixBlockInvMultDiag3
 * (o = (i1)* diag(i2)^-1) - so this scales the cols of il by
                             the i2 whose diags are row sums
 *--------------------------------------------------------------------------*/
JXF_Int jxf_BSRMatrixBlockInvMultDiag3(JXF_Complex* i1, JXF_Complex* i2, JXF_Complex* o, JXF_Int block_size)
{

    JXF_Int     ierr = 0;
    JXF_Int     i, j;
    JXF_Real    eps = 1.0e-8;
    JXF_Complex tmp, row_sum;

    for (i = 0; i < block_size; i++) {
        /* get row sum of i2, row i */
        row_sum = 0.0;
        for (j = 0; j < block_size; j++) {
            row_sum += i2[i * block_size + j];
        }

        /* invert */
        if (jxf_cabs(row_sum) > eps) {
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
 * jxf_BSRMatrixTranspose
 *--------------------------------------------------------------------------*/

JXF_Int jxf_BSRMatrixTranspose(jxf_BSRMatrix* A, jxf_BSRMatrix** AT, JXF_Int data)

{
    JXF_Complex* A_data        = jxf_BSRMatrixData(A);
    JXF_Int*     A_i           = jxf_BSRMatrixI(A);
    JXF_Int*     A_j           = jxf_BSRMatrixJ(A);
    JXF_Int      num_rowsA     = jxf_BSRMatrixNumRows(A);
    JXF_Int      num_colsA     = jxf_BSRMatrixNumCols(A);
    JXF_Int      num_nonzerosA = jxf_BSRMatrixNumNonzeros(A);
    JXF_Int      block_size    = jxf_BSRMatrixBlockSize(A);

    JXF_Complex* AT_data;
    JXF_Int*     AT_i;
    JXF_Int*     AT_j;
    JXF_Int      num_rowsAT;
    JXF_Int      num_colsAT;
    JXF_Int      num_nonzerosAT;

    JXF_Int max_col;
    JXF_Int i, j, k, m, offset, bnnz;

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

    *AT = jxf_BSRMatrixCreate(block_size, num_rowsAT, num_colsAT, num_nonzerosAT);

    AT_i                   = jxf_CTAlloc(JXF_Int, num_rowsAT + 1);
    AT_j                   = jxf_CTAlloc(JXF_Int, num_nonzerosAT);
    jxf_BSRMatrixI(*AT) = AT_i;
    jxf_BSRMatrixJ(*AT) = AT_j;
    if (data) {
        AT_data                   = jxf_CTAlloc(JXF_Complex, num_nonzerosAT * bnnz);
        jxf_BSRMatrixData(*AT) = AT_data;
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
 * jxf_BSRMatrixRead: Read a BSR matrix from a file.
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

jxf_BSRMatrix* jxf_BSRMatrixRead(const char* file_name)
{
    jxf_BSRMatrix* matrix;

    FILE* fp = NULL;

    JXF_Complex* matrix_data;
    JXF_Int*     matrix_i;
    JXF_Int*     matrix_j;
    // JXF_Int*     matrix_bigj;
    JXF_Int block_size;
    JXF_Int num_rows;
    JXF_Int num_cols;
    JXF_Int num_nonzeros;
    JXF_Int max_col        = 0;
    JXF_Int storage_manner = 0; //! storage manner for each sub-block, 0: row-major order, 1: column-major order
    JXF_Int file_base      = 0; // default is 0-based index, it automatically judged based on the file_name
    JXF_Int j, n;

    /*----------------------------------------------------------
     * Read in the data
     *----------------------------------------------------------*/
    fp = fopen(file_name, "r");

    jxf_fscanf(fp, "%d %d %d", &num_rows, &num_cols, &num_nonzeros); // dimensions of the problem
    jxf_fscanf(fp, "%d", &block_size);                               // read the size of each block
    jxf_fscanf(fp, "%d", &storage_manner);                           // read the storage_manner

    // allocate memory for IA, JA, and AA
    matrix_i    = jxf_CTAlloc(JXF_Int, num_rows + 1);
    matrix_j    = jxf_CTAlloc(JXF_Int, num_nonzeros);
    matrix_data = jxf_CTAlloc(JXF_Complex, num_nonzeros * block_size * block_size);

    // read IA
    jxf_fscanf(fp, "%d", &n); // n = rows + 1
    jxf_fscanf(fp, "%d", &matrix_i[0]);
    if (matrix_i[0] == 1) {
        file_base = 1;
        matrix_i[0] -= file_base;
    }
    for (j = 1; j < n; j++) {
        jxf_fscanf(fp, "%d", &matrix_i[j]);
        matrix_i[j] -= file_base;
    }

    // read JA
    jxf_fscanf(fp, "%d", &n); // n = nnz
    for (j = 0; j < n; j++) {
        jxf_fscanf(fp, "%d", &matrix_j[j]);
        matrix_j[j] -= file_base;

        if (matrix_j[j] > max_col) {
            max_col = matrix_j[j];
        }
    }
    if (max_col + 1 != num_cols) {
        ERROR_PRINTF("%s %d, max_col: %d != num_cols: %d\n", __FUNCTION__, __LINE__, max_col + 1, num_cols);
        exit(jxf_error_flag);
    }

    // read val
    jxf_fscanf(fp, "%d", &n); // n = nnz * block_size * block_size
    if (storage_manner == 0) {   // row-major order
        for (j = 0; j < n; j++) {
            jxf_fscanf(fp, "%le", &matrix_data[j]);
        }
    } else { // column-major order
        //! Note that storage manner is row-major order for each sub-block in jxf_BSRMatrix
        JXF_Int ind, loc_ind, bi, bj, bnnz = block_size * block_size;
        for (j = 0; j < n; j++) {
            loc_ind = j % bnnz;
            bi      = loc_ind / block_size; // local row index for each sub-block
            bj      = loc_ind % block_size; // local col index for each sub-block
            // (bi, bj) = bi * block_size + bj ==> (bj, bi) =  bj * block_size + bi
            ind = (j / bnnz) * bnnz + bj * block_size + bi;
            jxf_fscanf(fp, "%le", &matrix_data[ind]);
        }
    }
    fclose(fp);

    // Create BSR matrix
    matrix                       = jxf_BSRMatrixCreate(block_size, num_rows, num_cols, num_nonzeros);
    jxf_BSRMatrixI(matrix)    = matrix_i;
    jxf_BSRMatrixJ(matrix)    = matrix_j;
    jxf_BSRMatrixData(matrix) = matrix_data;

    return matrix;
}

/*--------------------------------------------------------------------------
 * jxf_BSRMatrixPrint: Write a file (file_base = 1) for BSR matrix.
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
JXF_Int jxf_BSRMatrixPrint(jxf_BSRMatrix* matrix, const char* file_name)
{
    FILE* fp = NULL;

    JXF_Complex* matrix_data;
    JXF_Int*     matrix_i;
    JXF_Int*     matrix_j;
    JXF_Int*  matrix_bigj;
    JXF_Int      num_rows;
    JXF_Int      num_cols;
    JXF_Int      num_nonzeros;
    JXF_Int      block_size;

    JXF_Int storage_manner = 0; // storage manner for each sub-block, 0: row-major order, 1: column-major order
    JXF_Int file_base      = 1; // default is 1-based index

    JXF_Int j;

    JXF_Int ierr = 0;

    /*----------------------------------------------------------
     * Print the matrix data
     *----------------------------------------------------------*/

    matrix_data  = jxf_BSRMatrixData(matrix);
    matrix_i     = jxf_BSRMatrixI(matrix);
    matrix_j     = jxf_BSRMatrixJ(matrix);
    matrix_bigj  = jxf_BSRMatrixBigJ(matrix);
    num_rows     = jxf_BSRMatrixNumRows(matrix);
    num_cols     = jxf_BSRMatrixNumCols(matrix);
    num_nonzeros = jxf_BSRMatrixNumNonzeros(matrix);
    block_size   = jxf_BSRMatrixBlockSize(matrix);

    fp = fopen(file_name, "w");

    jxf_fprintf(fp, "%d %d %d\n", num_rows, num_cols, num_nonzeros);
    jxf_fprintf(fp, "%d\n", block_size);
    jxf_fprintf(fp, "%d\n", storage_manner);

    jxf_fprintf(fp, "%d\n", num_rows + 1);
    for (j = 0; j <= num_rows; j++) {
        jxf_fprintf(fp, "%d\n", matrix_i[j] + file_base);
    }

    jxf_fprintf(fp, "%d\n", num_nonzeros);
    if (matrix_j) {
        for (j = 0; j < num_nonzeros; j++) {
            jxf_fprintf(fp, "%d\n", matrix_j[j] + file_base);
        }
    }

    if (matrix_bigj) {
        for (j = 0; j < num_nonzeros; j++) {
            jxf_fprintf(fp, "%d\n", matrix_bigj[j] + file_base);
        }
    }

    JXF_Int num = num_nonzeros * block_size * block_size;
    jxf_fprintf(fp, "%d\n", num);
    if (matrix_data) {
        for (j = 0; j < num; j++) {
#ifdef JXF_COMPLEX
            jxf_fprintf(fp, "%.14e , %.14e\n", jxf_creal(matrix_data[j]), jxf_cimag(matrix_data[j]));
#else
            jxf_fprintf(fp, "%.14e\n", matrix_data[j]);
#endif
        }
    } else {
        jxf_fprintf(fp, "Warning: No matrix data!\n");
    }

    fclose(fp);

    return ierr;
}

/*--------------------------------------------------------------------------
 * jxf_BSRMatrixRead_Binary: Read a BSR matrix from a binary file.
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

jxf_BSRMatrix* jxf_BSRMatrixRead_Binary(const char* file_name)
{
    jxf_BSRMatrix* matrix;

    FILE* fp = NULL;

    JXF_Complex* matrix_data;
    JXF_Int*     matrix_i;
    JXF_Int*     matrix_j;
    // JXF_Int*     matrix_bigj;
    JXF_Int block_size;
    JXF_Int num_rows;
    JXF_Int num_cols;
    JXF_Int num_nonzeros;
    JXF_Int storage_manner = 0; //! storage manner for each sub-block, 0: row-major order, 1: column-major order
    JXF_Int file_base      = 0; // default is 0-based index, it automatically judged based on the file_name
    JXF_Int j, n;

    /*----------------------------------------------------------
     * Read in the data
     *----------------------------------------------------------*/
    fp = fopen(file_name, "rb");

    // dimensions of the problem
    jxf_fread(&num_rows, sizeof(JXF_Int), 1, fp);
    jxf_fread(&num_cols, sizeof(JXF_Int), 1, fp);
    jxf_fread(&num_nonzeros, sizeof(JXF_Int), 1, fp);

    jxf_fread(&block_size, sizeof(JXF_Int), 1, fp);     // read the size of each block
    jxf_fread(&storage_manner, sizeof(JXF_Int), 1, fp); // read the storage_manner

    // allocate memory for IA, JA, and AA
    matrix_i    = jxf_CTAlloc(JXF_Int, num_rows + 1);
    matrix_j    = jxf_CTAlloc(JXF_Int, num_nonzeros);
    matrix_data = jxf_CTAlloc(JXF_Complex, num_nonzeros * block_size * block_size);

    // read IA
    jxf_fread(&n, sizeof(JXF_Int), 1, fp); // n = rows + 1
    jxf_fread(matrix_i, sizeof(JXF_Int), n, fp);
    if (matrix_i[0] == 1) {
        file_base = 1;
        for (j = 0; j < n; j++) {
            matrix_i[j] -= file_base;
        }
    }

    // read JA
    jxf_fread(&n, sizeof(JXF_Int), 1, fp); // n = nnz
    jxf_fread(matrix_j, sizeof(JXF_Int), n, fp);
    if (file_base == 1) {
        for (j = 0; j < n; j++) {
            matrix_j[j] -= file_base;
        }
    }

    // read val
    jxf_fread(&n, sizeof(JXF_Int), 1, fp); // n = nnz * block_size * block_size

    if (storage_manner == 0) { // row-major order
        jxf_fread(matrix_data, sizeof(JXF_Complex), n, fp);
    } else { // column-major order
        //! Note that storage manner is row-major order for each sub-block in jxf_BSRMatrix
        JXF_Int      ind, loc_ind, bi, bj, bnnz = block_size * block_size;
        JXF_Complex* matrix_data_tmp = jxf_CTAlloc(JXF_Complex, n);
        for (j = 0; j < n; j++) {
            loc_ind = j % bnnz;
            bi      = loc_ind / block_size; // local row index for each sub-block
            bj      = loc_ind % block_size; // local col index for each sub-block
            // (bi, bj) = bi * block_size + bj ==> (bj, bi) =  bj * block_size + bi
            ind              = (j / bnnz) * bnnz + bj * block_size + bi;
            matrix_data[ind] = matrix_data_tmp[j];
        }
        jxf_TFree(matrix_data_tmp);
    }

    fclose(fp);

    // Create BSR matrix
    matrix                       = jxf_BSRMatrixCreate(block_size, num_rows, num_cols, num_nonzeros);
    jxf_BSRMatrixI(matrix)    = matrix_i;
    jxf_BSRMatrixJ(matrix)    = matrix_j;
    jxf_BSRMatrixData(matrix) = matrix_data;

    return matrix;
}

/*--------------------------------------------------------------------------
 * jxf_BSRMatrixPrint_Binary: Write a binary file (file_base = 1) for BSR matrix.
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
JXF_Int jxf_BSRMatrixPrint_Binary(jxf_BSRMatrix* matrix, const char* file_name)
{
    FILE* fp = NULL;

    JXF_Complex* matrix_data;
    JXF_Int*     matrix_i;
    JXF_Int*     matrix_j;
    JXF_Int*  matrix_bigj;
    JXF_Int      num_rows;
    JXF_Int      num_cols;
    JXF_Int      num_nonzeros;
    JXF_Int      block_size;

    JXF_Int storage_manner = 0; // storage manner for each sub-block, 0: row-major order, 1: column-major order
    JXF_Int file_base      = 0; // default is 1-based index

    JXF_Int j, temp = 0;

    JXF_Int ierr = 0;

    /*----------------------------------------------------------
     * Print the matrix data
     *----------------------------------------------------------*/

    matrix_data  = jxf_BSRMatrixData(matrix);
    matrix_i     = jxf_BSRMatrixI(matrix);
    matrix_j     = jxf_BSRMatrixJ(matrix);
    matrix_bigj  = jxf_BSRMatrixBigJ(matrix);
    num_rows     = jxf_BSRMatrixNumRows(matrix);
    num_cols     = jxf_BSRMatrixNumCols(matrix);
    num_nonzeros = jxf_BSRMatrixNumNonzeros(matrix);
    block_size   = jxf_BSRMatrixBlockSize(matrix);

    fp = fopen(file_name, "wb");

    // write dimension of the block matrix
    jxf_fwrite(&num_rows, sizeof(JXF_Int), 1, fp);
    jxf_fwrite(&num_cols, sizeof(JXF_Int), 1, fp);
    jxf_fwrite(&num_nonzeros, sizeof(JXF_Int), 1, fp);
    jxf_fwrite(&block_size, sizeof(JXF_Int), 1, fp); // write block_size
    jxf_fwrite(&storage_manner, sizeof(JXF_Int), 1, fp);

    temp = num_rows + 1;
    jxf_fwrite(&temp, sizeof(JXF_Int), 1, fp);
    for (j = 0; j <= num_rows; j++) {
        temp = matrix_i[j] + file_base;
        jxf_fwrite(&temp, sizeof(JXF_Int), 1, fp);
    }

    jxf_fwrite(&num_nonzeros, sizeof(JXF_Int), 1, fp);
    if (matrix_j) {
        for (j = 0; j < num_nonzeros; j++) {
            temp = matrix_j[j] + file_base;
            jxf_fwrite(&temp, sizeof(JXF_Int), 1, fp);
        }
    }

    if (matrix_bigj) {
        JXF_Int Big_temp = 0;
        for (j = 0; j < num_nonzeros; j++) {
            Big_temp = matrix_bigj[j] + file_base;
            jxf_fwrite(&Big_temp, sizeof(JXF_Int), 1, fp);
        }
    }

    JXF_Int num = num_nonzeros * block_size * block_size;
    jxf_fwrite(&num, sizeof(JXF_Int), 1, fp);
    if (matrix_data) {
        for (j = 0; j < num; j++) {
#ifdef JXF_COMPLEX
            // jxf_fwrite(&jxf_creal(matrix_data[j]), sizeof(JXF_Real), 1, fp);
            // jxf_fwrite(&jxf_cimag(matrix_data[j]), sizeof(JXF_Real), 1, fp);
#else
            jxf_fwrite(&matrix_data[j], sizeof(JXF_Complex), 1, fp);
#endif
        }
    } else {
        jxf_fprintf(fp, "Warning: No matrix data!\n");
    }

    fclose(fp);

    return ierr;
}