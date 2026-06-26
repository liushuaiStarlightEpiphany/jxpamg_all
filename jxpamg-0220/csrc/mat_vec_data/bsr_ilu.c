//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2024        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  bsr_ilu.c -- Symbolic factorization, numerical factoration, and forward/backward sweeps for BSR matrix
 *  Date: 2025/10/08
 */ 


#include "jx_bsr_mv.h"

#include "bsr_matrix_inv.inl"

// Symbolic factorization of a BSR matrix A
JX_Int jx_BSRMatrixSymbFactor(jx_BSRMatrix* A, JX_Int levfill, JX_Int nzmax, JX_Int* nzlu, JX_Int* ijlu, JX_Int* uptr)
{
    JX_Int  row = jx_BSRMatrixNumRows(A);
    JX_Int* A_i = jx_BSRMatrixI(A);
    JX_Int* A_j = jx_BSRMatrixJ(A);

    return jx_SymbFactor(row, A_i, A_j, levfill, nzmax, nzlu, ijlu, uptr);
}

/**
 * \fn JX_Int jx_BSRMatrixNumFactor(jx_BSRMatrix* A, JX_Real* luval, JX_Int* jlu, JX_Int* uptr)
 *
 * \brief Get numerical ILU decoposition of a BSR matrix A
 *
 * \param A        Pointer to dBSRmat matrix
 * \param luval    Pointer to numerical value of ILU
 * \param jlu      Pointer to the nonzero pattern of ILU
 * \param uptr     Pointer to the diagnal position of ILU
 *
 * \author Li Zhao
 * \date   11/03/2024
 *
 * \note Works for general nb
 */
JX_Int jx_BSRMatrixNumFactor(jx_BSRMatrix* A, JX_Real* luval, JX_Int* jlu, JX_Int* uptr)
{

    JX_Int      n      = jx_BSRMatrixNumRows(A);
    JX_Int      nb     = jx_BSRMatrixBlockSize(A);
    JX_Int*     A_i    = jx_BSRMatrixI(A);
    JX_Int*     A_j    = jx_BSRMatrixJ(A);
    JX_Complex* A_data = jx_BSRMatrixData(A);

    JX_Int   nb2 = nb * nb, ib, ibstart, ibstart1;
    JX_Int   k, indj, inds, indja, jluj, jlus, ijaj;
    JX_Real *mult, *mult1;
    JX_Int*  colptrs;
    JX_Int   status = JX_SUCCESS;

    colptrs = (JX_Int*)calloc(n, sizeof(JX_Int));
    mult    = (JX_Real*)calloc(nb2, sizeof(JX_Real));
    mult1   = (JX_Real*)calloc(nb2, sizeof(JX_Real));

    /**
     *     colptrs is used to hold the indices of entries in LU of row k.
     *     It is initialized to zero here, and then reset after each row's
     *     work. The first segment of the loop on indj effectively solves
     *     the transposed upper triangular system
     *            U(1:k-1, 1:k-1)'L(k,1:k-1)' = A(k,1:k-1)'
     *     via sparse saxpy operations, throwing away disallowed fill.
     *     When the loop index indj reaches the k-th column (i.e., the diag
     *     entry), then the innermost sparse saxpy operation effectively is
     *     applying the previous updates to the corresponding part of U via
     *     sparse vec*mat, discarding disallowed fill-in entries, i.e.
     *            U(k,k:n) = A(k,k:n) - U(1:k-1,k:n)*L(k,1:k-1)
     */

    // for (k=0;k<n;k++) colptrs[k]=0;
    // memset(colptrs, 0, sizeof(JX_Int) * n);

    switch (nb) {

        case 1:

            for (k = 0; k < n; ++k) {

                for (indj = jlu[k]; indj < jlu[k + 1]; ++indj) {
                    colptrs[jlu[indj]] = indj;
                    ibstart            = indj * nb2;
                    for (ib = 0; ib < nb2; ++ib) luval[ibstart + ib] = 0;
                }

                colptrs[k] = k;

                for (indja = A_i[k]; indja < A_i[k + 1]; ++indja) {
                    ijaj     = A_j[indja];
                    ibstart  = colptrs[ijaj] * nb2;
                    ibstart1 = indja * nb2;
                    for (ib = 0; ib < nb2; ++ib) luval[ibstart + ib] = A_data[ibstart1 + ib];
                }

                for (indj = jlu[k]; indj < uptr[k]; ++indj) {

                    jluj = jlu[indj];

                    luval[indj] = luval[indj] * luval[jluj];
                    mult[0]     = luval[indj];

                    for (inds = uptr[jluj]; inds < jlu[jluj + 1]; ++inds) {
                        jlus = jlu[inds];
                        if (colptrs[jlus] != 0) luval[colptrs[jlus]] = luval[colptrs[jlus]] - mult[0] * luval[inds];
                    }
                }

                for (indj = jlu[k]; indj < jlu[k + 1]; ++indj) colptrs[jlu[indj]] = 0;

                colptrs[k] = 0;
                luval[k]   = 1.0 / luval[k];
            }

            break;

        case 3:

            for (k = 0; k < n; ++k) {

                for (indj = jlu[k]; indj < jlu[k + 1]; ++indj) {
                    colptrs[jlu[indj]] = indj;
                    ibstart            = indj * nb2;
                    for (ib = 0; ib < nb2; ++ib) luval[ibstart + ib] = 0;
                }

                colptrs[k] = k;

                for (indja = A_i[k]; indja < A_i[k + 1]; ++indja) {
                    ijaj     = A_j[indja];
                    ibstart  = colptrs[ijaj] * nb2;
                    ibstart1 = indja * nb2;
                    for (ib = 0; ib < nb2; ++ib) luval[ibstart + ib] = A_data[ibstart1 + ib];
                }

                for (indj = jlu[k]; indj < uptr[k]; ++indj) {
                    jluj = jlu[indj];

                    ibstart = indj * nb2;
                    // jx_blas_smat_mul_nc3(&(luval[ibstart]), &(luval[jluj * nb2]), mult);
                    jx_BSRMatrixBlockMultAdd(&(luval[ibstart]), &(luval[jluj * nb2]), 0, mult, nb);
                    for (ib = 0; ib < nb2; ++ib) luval[ibstart + ib] = mult[ib];

                    for (inds = uptr[jluj]; inds < jlu[jluj + 1]; ++inds) {
                        jlus = jlu[inds];
                        if (colptrs[jlus] != 0) {
                            // jx_blas_smat_mul_nc3(mult, &(luval[inds * nb2]), mult1);
                            jx_BSRMatrixBlockMultAdd(mult, &(luval[inds * nb2]), 0, mult1, nb);
                            ibstart = colptrs[jlus] * nb2;
                            for (ib = 0; ib < nb2; ++ib) luval[ibstart + ib] -= mult1[ib];
                        }
                    }
                }

                for (indj = jlu[k]; indj < jlu[k + 1]; ++indj) colptrs[jlu[indj]] = 0;

                colptrs[k] = 0;

                jx_smat_inv_nc3(&(luval[k * nb2]));
            }

            break;

        case -5:

            for (k = 0; k < n; ++k) {

                for (indj = jlu[k]; indj < jlu[k + 1]; ++indj) {
                    colptrs[jlu[indj]] = indj;
                    ibstart            = indj * nb2;
                    for (ib = 0; ib < nb2; ++ib) luval[ibstart + ib] = 0;
                }

                colptrs[k] = k;

                for (indja = A_i[k]; indja < A_i[k + 1]; ++indja) {
                    ijaj     = A_j[indja];
                    ibstart  = colptrs[ijaj] * nb2;
                    ibstart1 = indja * nb2;
                    for (ib = 0; ib < nb2; ++ib) luval[ibstart + ib] = A_data[ibstart1 + ib];
                }

                for (indj = jlu[k]; indj < uptr[k]; ++indj) {
                    jluj = jlu[indj];

                    ibstart = indj * nb2;
                    // jx_blas_smat_mul_nc5(&(luval[ibstart]), &(luval[jluj * nb2]), mult);
                    jx_BSRMatrixBlockMultAdd(&(luval[ibstart]), &(luval[jluj * nb2]), 0, mult, nb);
                    for (ib = 0; ib < nb2; ++ib) luval[ibstart + ib] = mult[ib];

                    for (inds = uptr[jluj]; inds < jlu[jluj + 1]; ++inds) {
                        jlus = jlu[inds];
                        if (colptrs[jlus] != 0) {
                            // jx_blas_smat_mul_nc5(mult, &(luval[inds * nb2]), mult1);
                            jx_BSRMatrixBlockMultAdd(mult, &(luval[inds * nb2]), 0, mult1, nb);
                            ibstart = colptrs[jlus] * nb2;
                            for (ib = 0; ib < nb2; ++ib) luval[ibstart + ib] -= mult1[ib];
                        }
                    }
                }

                for (indj = jlu[k]; indj < jlu[k + 1]; ++indj) colptrs[jlu[indj]] = 0;

                colptrs[k] = 0;

                // jx_smat_inv_nc5(&(luval[k*nb2])); // not numerically stable --zcs
                // 04/26/2021
                status = jx_smat_invp_nc(&(luval[k * nb2]), 5);
            }

            break;

        case -7:

            for (k = 0; k < n; ++k) {

                for (indj = jlu[k]; indj < jlu[k + 1]; ++indj) {
                    colptrs[jlu[indj]] = indj;
                    ibstart            = indj * nb2;
                    for (ib = 0; ib < nb2; ++ib) luval[ibstart + ib] = 0;
                }

                colptrs[k] = k;

                for (indja = A_i[k]; indja < A_i[k + 1]; ++indja) {
                    ijaj     = A_j[indja];
                    ibstart  = colptrs[ijaj] * nb2;
                    ibstart1 = indja * nb2;
                    for (ib = 0; ib < nb2; ++ib) luval[ibstart + ib] = A_data[ibstart1 + ib];
                }

                for (indj = jlu[k]; indj < uptr[k]; ++indj) {
                    jluj = jlu[indj];

                    ibstart = indj * nb2;
                    // jx_blas_smat_mul_nc7(&(luval[ibstart]), &(luval[jluj * nb2]), mult);
                    jx_BSRMatrixBlockMultAdd(&(luval[ibstart]), &(luval[jluj * nb2]), 0, mult, nb);
                    for (ib = 0; ib < nb2; ++ib) luval[ibstart + ib] = mult[ib];

                    for (inds = uptr[jluj]; inds < jlu[jluj + 1]; ++inds) {
                        jlus = jlu[inds];
                        if (colptrs[jlus] != 0) {
                            // jx_blas_smat_mul_nc7(mult, &(luval[inds * nb2]), mult1);
                            jx_BSRMatrixBlockMultAdd(mult, &(luval[inds * nb2]), 0, mult1, nb);
                            ibstart = colptrs[jlus] * nb2;
                            for (ib = 0; ib < nb2; ++ib) luval[ibstart + ib] -= mult1[ib];
                        }
                    }
                }

                for (indj = jlu[k]; indj < jlu[k + 1]; ++indj) colptrs[jlu[indj]] = 0;

                colptrs[k] = 0;

                // jx_smat_inv(&(luval[k*nb2]),nb); // not numerically stable --zcs
                // 04/26/2021
                status = jx_smat_invp_nc(&(luval[k * nb2]), nb);
            }

            break;

        default:

            for (k = 0; k < n; k++) {

                for (indj = jlu[k]; indj < jlu[k + 1]; ++indj) {
                    colptrs[jlu[indj]] = indj;
                    ibstart            = indj * nb2;
                    for (ib = 0; ib < nb2; ++ib) luval[ibstart + ib] = 0;
                }

                colptrs[k] = k;

                for (indja = A_i[k]; indja < A_i[k + 1]; indja++) {
                    ijaj     = A_j[indja];
                    ibstart  = colptrs[ijaj] * nb2;
                    ibstart1 = indja * nb2;
                    for (ib = 0; ib < nb2; ++ib) luval[ibstart + ib] = A_data[ibstart1 + ib];
                }

                for (indj = jlu[k]; indj < uptr[k]; ++indj) {
                    jluj = jlu[indj];

                    ibstart = indj * nb2;
                    // jx_blas_smat_mul(&(luval[ibstart]), &(luval[jluj * nb2]), mult, nb);            //!
                    jx_BSRMatrixBlockMultAdd(&(luval[ibstart]), &(luval[jluj * nb2]), 0, mult, nb); // mult = luval * luval

                    for (ib = 0; ib < nb2; ++ib) luval[ibstart + ib] = mult[ib];

                    for (inds = uptr[jluj]; inds < jlu[jluj + 1]; inds++) {
                        jlus = jlu[inds];
                        if (colptrs[jlus] != 0) {
                            // jx_blas_smat_mul(mult, &(luval[inds * nb2]), mult1, nb);
                            jx_BSRMatrixBlockMultAdd(mult, &(luval[inds * nb2]), 0, mult1, nb);
                            ibstart = colptrs[jlus] * nb2;
                            for (ib = 0; ib < nb2; ++ib) luval[ibstart + ib] -= mult1[ib];
                        }
                    }
                }

                for (indj = jlu[k]; indj < jlu[k + 1]; ++indj) colptrs[jlu[indj]] = 0;

                colptrs[k] = 0;

                status = jx_smat_invp_nc(&(luval[k * nb2]), nb);
            }
    }

    free(colptrs);
    colptrs = NULL;
    free(mult);
    mult = NULL;
    free(mult1);
    mult1 = NULL;

    return status;
}
