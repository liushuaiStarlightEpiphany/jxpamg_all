//========================================================================//
//  JXFPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2024        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  bsr_ilu.c -- Symbolic factorization, numerical factoration, and forward/backward sweeps for BSR matrix
 *  Date: 2025/10/08
 */ 


#include "jxf_bsr_mv.h"

#include "bsr_matrix_inv.inl"

// Symbolic factorization of a BSR matrix A
JXF_Int jxf_BSRMatrixSymbFactor(jxf_BSRMatrix* A, JXF_Int levfill, JXF_Int nzmax, JXF_Int* nzlu, JXF_Int* ijlu, JXF_Int* uptr)
{
    JXF_Int  row = jxf_BSRMatrixNumRows(A);
    JXF_Int* A_i = jxf_BSRMatrixI(A);
    JXF_Int* A_j = jxf_BSRMatrixJ(A);

    return jxf_SymbFactor(row, A_i, A_j, levfill, nzmax, nzlu, ijlu, uptr);
}

/**
 * \fn JXF_Int jxf_BSRMatrixNumFactor(jxf_BSRMatrix* A, JXF_Real* luval, JXF_Int* jlu, JXF_Int* uptr)
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
JXF_Int jxf_BSRMatrixNumFactor(jxf_BSRMatrix* A, JXF_Real* luval, JXF_Int* jlu, JXF_Int* uptr)
{

    JXF_Int      n      = jxf_BSRMatrixNumRows(A);
    JXF_Int      nb     = jxf_BSRMatrixBlockSize(A);
    JXF_Int*     A_i    = jxf_BSRMatrixI(A);
    JXF_Int*     A_j    = jxf_BSRMatrixJ(A);
    JXF_Complex* A_data = jxf_BSRMatrixData(A);

    JXF_Int   nb2 = nb * nb, ib, ibstart, ibstart1;
    JXF_Int   k, indj, inds, indja, jluj, jlus, ijaj;
    JXF_Real *mult, *mult1;
    JXF_Int*  colptrs;
    JXF_Int   status = JXF_SUCCESS;

    colptrs = (JXF_Int*)calloc(n, sizeof(JXF_Int));
    mult    = (JXF_Real*)calloc(nb2, sizeof(JXF_Real));
    mult1   = (JXF_Real*)calloc(nb2, sizeof(JXF_Real));

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
    // memset(colptrs, 0, sizeof(JXF_Int) * n);

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
                    // jxf_blas_smat_mul_nc3(&(luval[ibstart]), &(luval[jluj * nb2]), mult);
                    jxf_BSRMatrixBlockMultAdd(&(luval[ibstart]), &(luval[jluj * nb2]), 0, mult, nb);
                    for (ib = 0; ib < nb2; ++ib) luval[ibstart + ib] = mult[ib];

                    for (inds = uptr[jluj]; inds < jlu[jluj + 1]; ++inds) {
                        jlus = jlu[inds];
                        if (colptrs[jlus] != 0) {
                            // jxf_blas_smat_mul_nc3(mult, &(luval[inds * nb2]), mult1);
                            jxf_BSRMatrixBlockMultAdd(mult, &(luval[inds * nb2]), 0, mult1, nb);
                            ibstart = colptrs[jlus] * nb2;
                            for (ib = 0; ib < nb2; ++ib) luval[ibstart + ib] -= mult1[ib];
                        }
                    }
                }

                for (indj = jlu[k]; indj < jlu[k + 1]; ++indj) colptrs[jlu[indj]] = 0;

                colptrs[k] = 0;

                jxf_smat_inv_nc3(&(luval[k * nb2]));
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
                    // jxf_blas_smat_mul_nc5(&(luval[ibstart]), &(luval[jluj * nb2]), mult);
                    jxf_BSRMatrixBlockMultAdd(&(luval[ibstart]), &(luval[jluj * nb2]), 0, mult, nb);
                    for (ib = 0; ib < nb2; ++ib) luval[ibstart + ib] = mult[ib];

                    for (inds = uptr[jluj]; inds < jlu[jluj + 1]; ++inds) {
                        jlus = jlu[inds];
                        if (colptrs[jlus] != 0) {
                            // jxf_blas_smat_mul_nc5(mult, &(luval[inds * nb2]), mult1);
                            jxf_BSRMatrixBlockMultAdd(mult, &(luval[inds * nb2]), 0, mult1, nb);
                            ibstart = colptrs[jlus] * nb2;
                            for (ib = 0; ib < nb2; ++ib) luval[ibstart + ib] -= mult1[ib];
                        }
                    }
                }

                for (indj = jlu[k]; indj < jlu[k + 1]; ++indj) colptrs[jlu[indj]] = 0;

                colptrs[k] = 0;

                // jxf_smat_inv_nc5(&(luval[k*nb2])); // not numerically stable --zcs
                // 04/26/2021
                status = jxf_smat_invp_nc(&(luval[k * nb2]), 5);
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
                    // jxf_blas_smat_mul_nc7(&(luval[ibstart]), &(luval[jluj * nb2]), mult);
                    jxf_BSRMatrixBlockMultAdd(&(luval[ibstart]), &(luval[jluj * nb2]), 0, mult, nb);
                    for (ib = 0; ib < nb2; ++ib) luval[ibstart + ib] = mult[ib];

                    for (inds = uptr[jluj]; inds < jlu[jluj + 1]; ++inds) {
                        jlus = jlu[inds];
                        if (colptrs[jlus] != 0) {
                            // jxf_blas_smat_mul_nc7(mult, &(luval[inds * nb2]), mult1);
                            jxf_BSRMatrixBlockMultAdd(mult, &(luval[inds * nb2]), 0, mult1, nb);
                            ibstart = colptrs[jlus] * nb2;
                            for (ib = 0; ib < nb2; ++ib) luval[ibstart + ib] -= mult1[ib];
                        }
                    }
                }

                for (indj = jlu[k]; indj < jlu[k + 1]; ++indj) colptrs[jlu[indj]] = 0;

                colptrs[k] = 0;

                // jxf_smat_inv(&(luval[k*nb2]),nb); // not numerically stable --zcs
                // 04/26/2021
                status = jxf_smat_invp_nc(&(luval[k * nb2]), nb);
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
                    // jxf_blas_smat_mul(&(luval[ibstart]), &(luval[jluj * nb2]), mult, nb);            //!
                    jxf_BSRMatrixBlockMultAdd(&(luval[ibstart]), &(luval[jluj * nb2]), 0, mult, nb); // mult = luval * luval

                    for (ib = 0; ib < nb2; ++ib) luval[ibstart + ib] = mult[ib];

                    for (inds = uptr[jluj]; inds < jlu[jluj + 1]; inds++) {
                        jlus = jlu[inds];
                        if (colptrs[jlus] != 0) {
                            // jxf_blas_smat_mul(mult, &(luval[inds * nb2]), mult1, nb);
                            jxf_BSRMatrixBlockMultAdd(mult, &(luval[inds * nb2]), 0, mult1, nb);
                            ibstart = colptrs[jlus] * nb2;
                            for (ib = 0; ib < nb2; ++ib) luval[ibstart + ib] -= mult1[ib];
                        }
                    }
                }

                for (indj = jlu[k]; indj < jlu[k + 1]; ++indj) colptrs[jlu[indj]] = 0;

                colptrs[k] = 0;

                status = jxf_smat_invp_nc(&(luval[k * nb2]), nb);
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
