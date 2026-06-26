//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//


/*! \file  bsr_matrix_inv.inl
*
*  \brief Find inversion of small dense matrices in row-major format
*
*  Date: 2025/10/08
*  Created by zlj
*/

/*---------------------------------*/
/*--      Private Functions       --*/
/*---------------------------------*/

#define SWAP(a, b)                                                                                                                                             \
    {                                                                                                                                                          \
        temp = (a);                                                                                                                                            \
        (a)  = (b);                                                                                                                                            \
        (b)  = temp;                                                                                                                                           \
    } /**< swap two numbers */

/**
 * \fn static void jx_smat_inv_nc2 (JX_Real *a)
 *
 * \brief Compute the inverse matrix of a 2*2 full matrix A (in place)
 *
 * \param a Pointer to the JX_Real array which stands a 2*2 matrix
 *
 * \author zlj
 * \date   2025/10/08
 */
static void jx_smat_inv_nc2(JX_Real* a)
{
    const JX_Real a0 = a[0], a1 = a[1];
    const JX_Real a2 = a[2], a3 = a[3];

    const JX_Real det = a0 * a3 - a1 * a2;

    if (jx_cabs(det) < JX_REAL_MIN) {
        printf("### WARNING: Matrix is nearly singular, det = %e! Ignore.\n", det);
        printf("##----------------------------------------------\n");
        printf("## %12.5e %12.5e \n", a0, a1);
        printf("## %12.5e %12.5e \n", a2, a3);
        printf("##----------------------------------------------\n");

        a[0] = 1.0;
        a[1] = 0.0;
        a[2] = 0.0;
        a[3] = 1.0;
    } else {
        JX_Real det_inv = 1.0 / det;
        a[0]                = a3 * det_inv;
        a[1]                = -a1 * det_inv;
        a[2]                = -a2 * det_inv;
        a[3]                = a0 * det_inv;
    }
}

/**
 * \fn static void jx_smat_inv_nc3 (JX_Real *a)
 *
 * \brief Compute the inverse matrix of a 3*3 full matrix A (in place)
 *
 * \param a  Pointer to the JX_Real array which stands a 3*3 matrix
 *
 * \author zlj
 * \date   2025/10/08
 */
static void jx_smat_inv_nc3(JX_Real* a)
{
    const JX_Real a0 = a[0], a1 = a[1], a2 = a[2];
    const JX_Real a3 = a[3], a4 = a[4], a5 = a[5];
    const JX_Real a6 = a[6], a7 = a[7], a8 = a[8];

    const JX_Real M0 = a4 * a8 - a5 * a7, M3 = a2 * a7 - a1 * a8, M6 = a1 * a5 - a2 * a4;
    const JX_Real M1 = a5 * a6 - a3 * a8, M4 = a0 * a8 - a2 * a6, M7 = a2 * a3 - a0 * a5;
    const JX_Real M2 = a3 * a7 - a4 * a6, M5 = a1 * a6 - a0 * a7, M8 = a0 * a4 - a1 * a3;

    const JX_Real det = a0 * M0 + a3 * M3 + a6 * M6;

    if (jx_cabs(det) < JX_REAL_MIN) {
        printf("### WARNING: Matrix is nearly singular, det = %e! Ignore.\n", det);
        printf("##----------------------------------------------\n");
        printf("## %12.5e %12.5e %12.5e \n", a0, a1, a2);
        printf("## %12.5e %12.5e %12.5e \n", a3, a4, a5);
        printf("## %12.5e %12.5e %12.5e \n", a6, a7, a8);
        printf("##----------------------------------------------\n");

        a[0] = 1.0;
        a[1] = 0.0;
        a[2] = 0.0;
        a[3] = 0.0;
        a[4] = 1.0;
        a[5] = 0.0;
        a[6] = 0.0;
        a[7] = 0.0;
        a[8] = 1.0;
    } else {
        JX_Real det_inv = 1.0 / det;
        a[0]                = M0 * det_inv;
        a[1]                = M3 * det_inv;
        a[2]                = M6 * det_inv;
        a[3]                = M1 * det_inv;
        a[4]                = M4 * det_inv;
        a[5]                = M7 * det_inv;
        a[6]                = M2 * det_inv;
        a[7]                = M5 * det_inv;
        a[8]                = M8 * det_inv;
    }
}

/**
 * \fn static void jx_smat_inv_nc4 (JX_Real *a)
 *
 * \brief Compute the inverse matrix of a 4*4 full matrix A (in place)
 *
 * \param a  Pointer to the JX_Real array which stands a 4*4 matrix
 *
 * \author zlj
 * \date   2025/10/08
 *
 */
static void jx_smat_inv_nc4(JX_Real* a)
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

    if (jx_cabs(det) < JX_REAL_MIN) {
        printf("### WARNING: Matrix is nearly singular, det = %e! Ignore.\n", det);
        printf("##----------------------------------------------\n");
        printf("## %12.5e %12.5e %12.5e %12.5e\n", a11, a12, a13, a14);
        printf("## %12.5e %12.5e %12.5e %12.5e\n", a21, a22, a23, a24);
        printf("## %12.5e %12.5e %12.5e %12.5e\n", a31, a32, a33, a34);
        printf("## %12.5e %12.5e %12.5e %12.5e\n", a41, a42, a43, a44);
        printf("##----------------------------------------------\n");

        a[0]  = 1.0;
        a[1]  = 0.0;
        a[2]  = 0.0;
        a[3]  = 0.0;
        a[4]  = 0.0;
        a[5]  = 1.0;
        a[6]  = 0.0;
        a[7]  = 0.0;
        a[8]  = 0.0;
        a[9]  = 0.0;
        a[10] = 1.0;
        a[11] = 0.0;
        a[12] = 0.0;
        a[13] = 0.0;
        a[14] = 0.0;
        a[15] = 1.0;
    } else {
        JX_Real det_inv = 1.0 / det;
        a[0]                = M11 * det_inv;
        a[1]                = M12 * det_inv;
        a[2]                = M13 * det_inv;
        a[3]                = M14 * det_inv;
        a[4]                = M21 * det_inv;
        a[5]                = M22 * det_inv;
        a[6]                = M23 * det_inv;
        a[7]                = M24 * det_inv;
        a[8]                = M31 * det_inv;
        a[9]                = M32 * det_inv;
        a[10]               = M33 * det_inv;
        a[11]               = M34 * det_inv;
        a[12]               = M41 * det_inv;
        a[13]               = M42 * det_inv;
        a[14]               = M43 * det_inv;
        a[15]               = M44 * det_inv;
    }
}

/**
 * \fn static void jx_smat_inv_nc5 (JX_Real *a)
 *
 * \brief Compute the inverse matrix of a 5*5 full matrix A (in place)
 *
 * \param a  Pointer to the JX_Real array which stands a 5*5 matrix
 *
 * \author zlj
 * \date   2025/10/08
 */
static void jx_smat_inv_nc5(JX_Real* a)
{
    const JX_Real a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4];
    const JX_Real a5 = a[5], a6 = a[6], a7 = a[7], a8 = a[8], a9 = a[9];
    const JX_Real a10 = a[10], a11 = a[11], a12 = a[12], a13 = a[13], a14 = a[14];
    const JX_Real a15 = a[15], a16 = a[16], a17 = a[17], a18 = a[18], a19 = a[19];
    const JX_Real a20 = a[20], a21 = a[21], a22 = a[22], a23 = a[23], a24 = a[24];

    JX_Real det0, det1, det2, det3, det4, det;

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

    if (jx_cabs(det) < JX_REAL_MIN) {
        printf("### WARNING: Matrix is nearly singular, det = %e! Ignore.\n", det);
        printf("##----------------------------------------------\n");
        printf("## %12.5e %12.5e %12.5e %12.5e %12.5e\n", a0, a1, a2, a3, a4);
        printf("## %12.5e %12.5e %12.5e %12.5e %12.5e\n", a5, a6, a7, a8, a9);
        printf("## %12.5e %12.5e %12.5e %12.5e %12.5e\n", a10, a11, a12, a13, a14);
        printf("## %12.5e %12.5e %12.5e %12.5e %12.5e\n", a15, a16, a17, a18, a19);
        printf("## %12.5e %12.5e %12.5e %12.5e %12.5e\n", a20, a21, a22, a23, a24);
        printf("##----------------------------------------------\n");

        a[0]  = 1.0;
        a[1]  = 0.0;
        a[2]  = 0.0;
        a[3]  = 0.0;
        a[4]  = 0.0;
        a[5]  = 0.0;
        a[6]  = 1.0;
        a[7]  = 0.0;
        a[8]  = 0.0;
        a[9]  = 0.0;
        a[10] = 0.0;
        a[11] = 0.0;
        a[12] = 1.0;
        a[13] = 0.0;
        a[14] = 0.0;
        a[15] = 0.0;
        a[16] = 0.0;
        a[17] = 0.0;
        a[18] = 1.0;
        a[19] = 0.0;
        a[20] = 0.0;
        a[21] = 0.0;
        a[22] = 0.0;
        a[23] = 0.0;
        a[24] = 1.0;
    } else {
        JX_Real det_inv = 1 / det;

        a[0] = a6 * (a12 * a18 * a24 - a12 * a19 * a23 - a17 * a13 * a24 + a17 * a14 * a23 + a22 * a13 * a19 - a22 * a14 * a18);
        a[0] += a11 * (a7 * a19 * a23 - a7 * a18 * a24 + a17 * a8 * a24 - a17 * a9 * a23 - a22 * a8 * a19 + a22 * a9 * a18);
        a[0] += a16 * (a7 * a13 * a24 - a7 * a14 * a23 - a12 * a8 * a24 + a12 * a9 * a23 + a22 * a8 * a14 - a22 * a9 * a13);
        a[0] += a21 * (a7 * a14 * a18 - a7 * a13 * a19 + a12 * a8 * a19 - a12 * a9 * a18 - a17 * a8 * a14 + a17 * a9 * a13);
        a[0] *= det_inv;

        a[1] = a1 * (a12 * a19 * a23 - a12 * a18 * a24 + a22 * a14 * a18 - a17 * a14 * a23 - a22 * a13 * a19 + a17 * a13 * a24);
        a[1] += a11 * (a22 * a3 * a19 + a2 * a18 * a24 - a17 * a3 * a24 - a22 * a4 * a18 - a2 * a19 * a23 + a17 * a4 * a23);
        a[1] += a16 * (a12 * a3 * a24 - a12 * a4 * a23 - a22 * a3 * a14 + a2 * a14 * a23 + a22 * a4 * a13 - a2 * a13 * a24);
        a[1] += a21 * (a12 * a4 * a18 - a12 * a3 * a19 - a2 * a14 * a18 - a17 * a4 * a13 + a2 * a13 * a19 + a17 * a3 * a14);
        a[1] *= det_inv;

        a[2] = a1 * (a7 * a18 * a24 - a7 * a19 * a23 - a17 * a8 * a24 + a17 * a9 * a23 + a22 * a8 * a19 - a22 * a9 * a18);
        a[2] += a6 * (a2 * a19 * a23 - a2 * a18 * a24 + a17 * a3 * a24 - a17 * a4 * a23 - a22 * a3 * a19 + a22 * a4 * a18);
        a[2] += a16 * (a2 * a8 * a24 - a2 * a9 * a23 - a7 * a3 * a24 + a7 * a4 * a23 + a22 * a3 * a9 - a22 * a4 * a8);
        a[2] += a21 * (a2 * a9 * a18 - a2 * a8 * a19 + a7 * a3 * a19 - a7 * a4 * a18 - a17 * a3 * a9 + a17 * a4 * a8);
        a[2] *= det_inv;

        a[3] = a1 * (a12 * a8 * a24 - a12 * a9 * a23 + a7 * a14 * a23 - a7 * a13 * a24 + a22 * a9 * a13 - a22 * a8 * a14);
        a[3] += a6 * (a12 * a4 * a23 - a12 * a3 * a24 + a22 * a3 * a14 - a22 * a4 * a13 + a2 * a13 * a24 - a2 * a14 * a23);
        a[3] += a11 * (a7 * a3 * a24 - a7 * a4 * a23 + a22 * a4 * a8 - a22 * a3 * a9 + a2 * a9 * a23 - a2 * a8 * a24);
        a[3] += a21 * (a12 * a3 * a9 - a12 * a4 * a8 + a2 * a8 * a14 - a2 * a9 * a13 + a7 * a4 * a13 - a7 * a3 * a14);
        a[3] *= det_inv;

        a[4] = a1 * (a7 * a13 * a19 - a7 * a14 * a18 - a12 * a8 * a19 + a12 * a9 * a18 + a17 * a8 * a14 - a17 * a9 * a13);
        a[4] += a6 * (a2 * a14 * a18 - a2 * a13 * a19 + a12 * a3 * a19 - a12 * a4 * a18 - a17 * a3 * a14 + a17 * a4 * a13);
        a[4] += a11 * (a2 * a8 * a19 - a2 * a9 * a18 - a7 * a3 * a19 + a7 * a4 * a18 + a17 * a3 * a9 - a17 * a4 * a8);
        a[4] += a16 * (a2 * a9 * a13 - a2 * a8 * a14 + a7 * a3 * a14 - a7 * a4 * a13 - a12 * a3 * a9 + a12 * a4 * a8);
        a[4] *= det_inv;

        a[5] = a5 * (a12 * a19 * a23 - a12 * a18 * a24 + a22 * a14 * a18 - a22 * a13 * a19 + a17 * a13 * a24 - a17 * a14 * a23);
        a[5] += a20 * (a12 * a9 * a18 - a12 * a8 * a19 + a7 * a13 * a19 - a18 * a7 * a14 + a17 * a8 * a14 - a9 * a17 * a13);
        a[5] += a15 * (a22 * a9 * a13 - a12 * a9 * a23 + a12 * a24 * a8 + a7 * a14 * a23 - a24 * a7 * a13 - a22 * a14 * a8);
        a[5] += a10 * (a18 * a7 * a24 - a18 * a22 * a9 - a17 * a8 * a24 + a17 * a9 * a23 + a22 * a8 * a19 - a19 * a23 * a7);
        a[5] *= det_inv;

        a[6] = a2 * (a19 * a23 * a10 - a14 * a23 * a15 - a18 * a24 * a10 + a18 * a14 * a20 - a13 * a19 * a20 + a24 * a13 * a15);
        a[6] += a12 * (a18 * a0 * a24 - a18 * a20 * a4 + a3 * a19 * a20 - a19 * a23 * a0 + a4 * a23 * a15 - a24 * a15 * a3);
        a[6] += a17 * (a4 * a13 * a20 - a13 * a24 * a0 + a14 * a23 * a0 - a3 * a14 * a20 + a24 * a3 * a10 - a4 * a23 * a10);
        a[6] += a22 * (a14 * a15 * a3 - a18 * a14 * a0 + a18 * a4 * a10 - a4 * a13 * a15 + a13 * a19 * a0 - a3 * a19 * a10);
        a[6] *= det_inv;

        a[7] = a0 * (a18 * a9 * a22 - a18 * a24 * a7 + a19 * a23 * a7 - a9 * a23 * a17 + a24 * a8 * a17 - a8 * a19 * a22);
        a[7] += a5 * (a2 * a18 * a24 - a2 * a19 * a23 + a17 * a4 * a23 - a17 * a3 * a24 + a22 * a3 * a19 - a22 * a4 * a18);
        a[7] += a15 * (a4 * a8 * a22 - a3 * a9 * a22 - a24 * a8 * a2 + a9 * a23 * a2 - a4 * a23 * a7 + a24 * a3 * a7);
        a[7] += a20 * (a18 * a4 * a7 - a18 * a9 * a2 + a9 * a3 * a17 - a4 * a8 * a17 + a8 * a19 * a2 - a3 * a19 * a7);
        a[7] *= det_inv;

        a[8] = a0 * (a12 * a9 * a23 - a12 * a24 * a8 + a22 * a14 * a8 - a7 * a14 * a23 + a24 * a7 * a13 - a9 * a22 * a13);
        a[8] += a5 * (a12 * a3 * a24 - a12 * a4 * a23 - a22 * a3 * a14 + a2 * a14 * a23 - a2 * a13 * a24 + a22 * a4 * a13);
        a[8] += a10 * (a22 * a9 * a3 - a4 * a22 * a8 + a4 * a7 * a23 - a2 * a9 * a23 + a24 * a2 * a8 - a7 * a24 * a3);
        a[8] += a20 * (a7 * a14 * a3 - a4 * a7 * a13 + a9 * a2 * a13 + a12 * a4 * a8 - a12 * a9 * a3 - a2 * a14 * a8);
        a[8] *= det_inv;

        a[9] = a0 * (a12 * a8 * a19 - a12 * a18 * a9 + a18 * a7 * a14 - a8 * a17 * a14 + a17 * a13 * a9 - a7 * a13 * a19);
        a[9] += a5 * (a2 * a13 * a19 - a2 * a14 * a18 - a12 * a3 * a19 + a12 * a4 * a18 + a17 * a3 * a14 - a17 * a4 * a13);
        a[9] += a10 * (a18 * a2 * a9 - a18 * a7 * a4 + a3 * a7 * a19 - a2 * a8 * a19 + a17 * a8 * a4 - a3 * a17 * a9);
        a[9] += a15 * (a8 * a2 * a14 - a12 * a8 * a4 + a12 * a3 * a9 - a3 * a7 * a14 + a7 * a13 * a4 - a2 * a13 * a9);
        a[9] *= det_inv;

        a[10] = a5 * (a18 * a24 * a11 - a24 * a13 * a16 + a14 * a23 * a16 - a19 * a23 * a11 + a13 * a19 * a21 - a18 * a14 * a21);
        a[10] += a10 * (a19 * a23 * a6 - a9 * a23 * a16 + a24 * a8 * a16 - a8 * a19 * a21 + a18 * a9 * a21 - a18 * a24 * a6);
        a[10] += a15 * (a24 * a13 * a6 - a14 * a23 * a6 - a24 * a8 * a11 + a9 * a23 * a11 + a14 * a8 * a21 - a13 * a9 * a21);
        a[10] += a20 * (a18 * a14 * a6 - a18 * a9 * a11 + a8 * a19 * a11 - a13 * a19 * a6 + a9 * a13 * a16 - a14 * a8 * a16);
        a[10] *= det_inv;

        a[11] = a4 * (a21 * a13 * a15 - a11 * a23 * a15 + a16 * a23 * a10 - a13 * a16 * a20 + a18 * a11 * a20 - a18 * a21 * a10);
        a[11] += a14 * (a18 * a0 * a21 - a1 * a18 * a20 + a16 * a3 * a20 - a23 * a0 * a16 + a1 * a23 * a15 - a21 * a3 * a15);
        a[11] += a19 * (a1 * a13 * a20 - a1 * a23 * a10 + a23 * a0 * a11 + a21 * a3 * a10 - a11 * a3 * a20 - a13 * a0 * a21);
        a[11] += a24 * (a13 * a0 * a16 - a18 * a0 * a11 + a11 * a3 * a15 + a1 * a18 * a10 - a1 * a13 * a15 - a16 * a3 * a10);
        a[11] *= det_inv;

        a[12] = a4 * (a5 * a21 * a18 - a18 * a20 * a6 + a20 * a16 * a8 - a5 * a16 * a23 + a15 * a6 * a23 - a21 * a15 * a8);
        a[12] += a9 * (a1 * a20 * a18 - a1 * a15 * a23 + a0 * a16 * a23 - a18 * a0 * a21 - a20 * a16 * a3 + a15 * a21 * a3);
        a[12] += a19 * (a20 * a6 * a3 - a5 * a21 * a3 + a0 * a21 * a8 - a23 * a0 * a6 + a1 * a5 * a23 - a1 * a20 * a8);
        a[12] += a24 * (a1 * a15 * a8 - a0 * a16 * a8 + a18 * a0 * a6 - a1 * a5 * a18 + a5 * a16 * a3 - a6 * a15 * a3);
        a[12] *= det_inv;

        a[13] = a0 * (a24 * a11 * a8 - a6 * a24 * a13 + a21 * a9 * a13 - a11 * a9 * a23 + a14 * a6 * a23 - a14 * a21 * a8);
        a[13] += a1 * (a5 * a13 * a24 - a5 * a14 * a23 + a14 * a20 * a8 + a10 * a9 * a23 - a24 * a10 * a8 - a20 * a9 * a13);
        a[13] += a3 * (a6 * a10 * a24 - a10 * a9 * a21 + a5 * a14 * a21 - a5 * a24 * a11 + a20 * a9 * a11 - a14 * a6 * a20);
        a[13] += a4 * (a5 * a11 * a23 - a5 * a21 * a13 + a21 * a10 * a8 - a6 * a10 * a23 + a20 * a6 * a13 - a11 * a20 * a8);
        a[13] *= det_inv;

        a[14] = a0 * (a13 * a19 * a6 - a14 * a18 * a6 - a11 * a19 * a8 + a14 * a16 * a8 + a11 * a18 * a9 - a13 * a16 * a9);
        a[14] += a1 * (a14 * a18 * a5 - a13 * a19 * a5 + a10 * a19 * a8 - a14 * a15 * a8 - a10 * a18 * a9 + a13 * a15 * a9);
        a[14] += a3 * (a11 * a19 * a5 - a11 * a15 * a9 + a10 * a16 * a9 - a10 * a19 * a6 + a14 * a15 * a6 - a14 * a16 * a5);
        a[14] += a4 * (a11 * a15 * a8 - a11 * a18 * a5 + a13 * a16 * a5 - a13 * a15 * a6 + a10 * a18 * a6 - a10 * a16 * a8);
        a[14] *= det_inv;

        a[15] = a5 * (a19 * a22 * a11 - a24 * a17 * a11 + a12 * a24 * a16 - a22 * a14 * a16 - a12 * a19 * a21 + a17 * a14 * a21);
        a[15] += a10 * (a24 * a17 * a6 - a19 * a22 * a6 - a24 * a7 * a16 + a22 * a9 * a16 + a19 * a7 * a21 - a17 * a9 * a21);
        a[15] += a15 * (a22 * a14 * a6 - a9 * a22 * a11 + a24 * a7 * a11 - a12 * a24 * a6 - a7 * a14 * a21 + a12 * a9 * a21);
        a[15] += a20 * (a12 * a19 * a6 - a17 * a14 * a6 - a19 * a7 * a11 + a9 * a17 * a11 + a7 * a14 * a16 - a12 * a9 * a16);
        a[15] *= det_inv;

        a[16] = a0 * (a11 * a17 * a24 - a11 * a19 * a22 - a12 * a16 * a24 + a12 * a19 * a21 + a14 * a16 * a22 - a14 * a17 * a21);
        a[16] += a1 * (a10 * a19 * a22 - a10 * a17 * a24 + a12 * a15 * a24 - a12 * a19 * a20 - a14 * a15 * a22 + a14 * a17 * a20);
        a[16] += a2 * (a10 * a16 * a24 - a10 * a19 * a21 - a11 * a15 * a24 + a11 * a19 * a20 + a14 * a15 * a21 - a14 * a16 * a20);
        a[16] += a4 * (a10 * a17 * a21 * +a11 * a15 * a22 - a11 * a17 * a20 - a12 * a15 * a21 + a12 * a16 * a20 - a10 * a16 * a22);
        a[16] *= det_inv;

        a[17] = a0 * (a21 * a9 * a17 - a6 * a24 * a17 + a19 * a6 * a22 - a0 * a16 * a9 * a22 + a24 * a16 * a7 - a19 * a21 * a7);
        a[17] += a1 * (a5 * a24 * a17 - a5 * a19 * a22 + a19 * a20 * a7 - a20 * a9 * a17 + a15 * a9 * a22 - a24 * a15 * a7);
        a[17] += a2 * (a5 * a19 * a21 - a19 * a6 * a20 - a5 * a24 * a16 + a24 * a6 * a15 - a15 * a9 * a21 + a20 * a9 * a16);
        a[17] += a4 * (a16 * a5 * a22 - a6 * a15 * a22 + a20 * a6 * a17 - a5 * a21 * a17 - a6 * a15 * a22 + a21 * a15 * a7 - a16 * a20 * a7);
        a[17] *= det_inv;

        a[18] = a0 * (a12 * a24 * a6 - a14 * a22 * a6 - a11 * a24 * a7 + a14 * a21 * a7 + a11 * a22 * a9 - a12 * a21 * a9);
        a[18] += a1 * (a14 * a22 * a5 - a12 * a24 * a5 + a10 * a24 * a7 - a14 * a20 * a7 - a10 * a22 * a9 + a12 * a20 * a9);
        a[18] += a2 * (a11 * a24 * a5 - a11 * a20 * a9 + a14 * a20 * a6 - a14 * a21 * a5 + a10 * a21 * a9 - a10 * a24 * a6);
        a[18] += a4 * (a11 * a20 * a7 - a11 * a22 * a5 + a12 * a21 * a5 + a10 * a22 * a6 - a12 * a20 * a6 - a10 * a21 * a7);
        a[18] *= det_inv;

        a[19] = a0 * (a12 * a16 * a9 - a6 * a12 * a19 + a6 * a17 * a14 - a17 * a11 * a9 + a11 * a7 * a19 - a16 * a7 * a14);
        a[19] += a1 * (a5 * a12 * a19 - a5 * a17 * a14 - a12 * a15 * a9 + a17 * a10 * a9 + a15 * a7 * a14 - a10 * a7 * a19);
        a[19] += a2 * (a11 * a15 * a9 - a5 * a11 * a19 + a5 * a16 * a14 - a6 * a15 * a14 + a6 * a10 * a19 - a16 * a10 * a9);
        a[19] += a4 * (a5 * a17 * a11 - a5 * a12 * a16 + a12 * a6 * a15 + a10 * a7 * a16 - a17 * a6 * a10 - a15 * a7 * a11);
        a[19] *= det_inv;

        a[20] = a5 * (a12 * a18 * a21 - a12 * a23 * a16 + a22 * a13 * a16 - a18 * a22 * a11 + a23 * a17 * a11 - a17 * a13 * a21);
        a[20] += a15 * (a12 * a23 * a6 - a12 * a8 * a21 + a8 * a22 * a11 - a23 * a7 * a11 + a7 * a13 * a21 - a22 * a13 * a6);
        a[20] += a20 * (a12 * a8 * a16 - a12 * a18 * a6 + a18 * a7 * a11 - a8 * a17 * a11 + a17 * a13 * a6 - a7 * a13 * a16);
        a[20] += a10 * (a17 * a8 * a21 - a22 * a8 * a16 - a18 * a7 * a21 + a18 * a22 * a6 + a23 * a7 * a16 - a23 * a17 * a6);
        a[20] *= det_inv;

        a[21] = a0 * (a12 * a23 * a16 - a12 * a18 * a21 + a17 * a13 * a21 + a18 * a22 * a11 - a23 * a17 * a11 - a22 * a13 * a16);
        a[21] += a1 * (a12 * a18 * a20 - a12 * a23 * a15 + a22 * a13 * a15 + a23 * a17 * a10 - a17 * a13 * a20 - a18 * a22 * a10);
        a[21] += a2 * (a18 * a21 * a10 - a18 * a11 * a20 - a21 * a13 * a15 + a16 * a13 * a20 - a23 * a16 * a10 + a23 * a11 * a15);
        a[21] += a3 * (a17 * a11 * a20 - a12 * a16 * a20 + a12 * a21 * a15 - a21 * a17 * a10 - a22 * a11 * a15 + a16 * a22 * a10);
        a[21] *= det_inv;

        a[22] = a0 * (a18 * a21 * a7 - a18 * a6 * a22 + a23 * a6 * a17 + a16 * a8 * a22 - a21 * a8 * a17 - a23 * a16 * a7);
        a[22] += a1 * (a5 * a18 * a22 - a5 * a23 * a17 - a15 * a8 * a22 + a20 * a8 * a17 - a18 * a20 * a7 + a23 * a15 * a7);
        a[22] += a3 * (a16 * a20 * a7 + a6 * a15 * a22 - a6 * a20 * a17 - a5 * a16 * a22 + a5 * a21 * a17 - a21 * a15 * a7);
        a[22] += a2 * (a5 * a23 * a16 - a5 * a18 * a21 + a18 * a6 * a20 + a15 * a8 * a21 - a20 * a8 * a16 - a23 * a6 * a15);
        a[22] *= det_inv;

        a[23] = a0 * (a12 * a21 * a8 - a22 * a11 * a8 + a11 * a7 * a23 - a6 * a12 * a23 - a21 * a7 * a13 + a6 * a22 * a13);
        a[23] += a1 * (a5 * a12 * a23 - a5 * a22 * a13 - a10 * a7 * a23 + a20 * a7 * a13 + a22 * a10 * a8 - a12 * a20 * a8);
        a[23] += a2 * (a5 * a21 * a13 + a11 * a20 * a8 + a6 * a10 * a23 - a5 * a11 * a23 - a21 * a10 * a8 - a6 * a20 * a13);
        a[23] += a3 * (a5 * a22 * a11 - a5 * a12 * a21 + a10 * a7 * a21 - a22 * a6 * a10 - a20 * a7 * a11 + a12 * a6 * a20);
        a[23] *= det_inv;

        a[24] = a0 * (a17 * a11 * a8 - a11 * a7 * a18 + a6 * a12 * a18 - a12 * a16 * a8 + a16 * a7 * a13 - a6 * a17 * a13);
        a[24] += a1 * (a5 * a17 * a13 - a5 * a12 * a18 + a10 * a7 * a18 + a12 * a15 * a8 - a17 * a10 * a8 - a15 * a7 * a13);
        a[24] += a2 * (a5 * a11 * a18 - a5 * a16 * a13 + a16 * a10 * a8 + a6 * a15 * a13 - a11 * a15 * a8 - a6 * a10 * a18);
        a[24] += a3 * (a5 * a12 * a16 + a17 * a6 * a10 - a5 * a17 * a11 - a12 * a6 * a15 - a10 * a7 * a16 + a15 * a7 * a11);
        a[24] *= det_inv;
    }

    printf("### DEBUG: Check inverse matrix...\n");
    printf("##----------------------------------------------\n");
    printf("## %12.5e %12.5e %12.5e %12.5e %12.5e\n", a0 * a[0] + a1 * a[5] + a2 * a[10] + a3 * a[15] + a4 * a[20],
           a0 * a[1] + a1 * a[6] + a2 * a[11] + a3 * a[16] + a4 * a[21], a0 * a[2] + a1 * a[7] + a2 * a[12] + a3 * a[17] + a4 * a[22],
           a0 * a[3] + a1 * a[8] + a2 * a[13] + a3 * a[18] + a4 * a[23], a0 * a[4] + a1 * a[9] + a2 * a[14] + a3 * a[19] + a4 * a[24]);
    printf("## %12.5e %12.5e %12.5e %12.5e %12.5e\n", a5 * a[0] + a6 * a[5] + a7 * a[10] + a8 * a[15] + a9 * a[20],
           a5 * a[1] + a6 * a[6] + a7 * a[11] + a8 * a[16] + a9 * a[21], a5 * a[2] + a6 * a[7] + a7 * a[12] + a8 * a[17] + a9 * a[22],
           a5 * a[3] + a6 * a[8] + a7 * a[13] + a8 * a[18] + a9 * a[23], a5 * a[4] + a6 * a[9] + a7 * a[14] + a8 * a[19] + a9 * a[24]);
    printf("## %12.5e %12.5e %12.5e %12.5e %12.5e\n", a10 * a[0] + a11 * a[5] + a12 * a[10] + a13 * a[15] + a14 * a[20],
           a10 * a[1] + a11 * a[6] + a12 * a[11] + a13 * a[16] + a14 * a[21], a10 * a[2] + a11 * a[7] + a12 * a[12] + a13 * a[17] + a14 * a[22],
           a10 * a[3] + a11 * a[8] + a12 * a[13] + a13 * a[18] + a14 * a[23], a10 * a[4] + a11 * a[9] + a12 * a[14] + a13 * a[19] + a14 * a[24]);
    printf("## %12.5e %12.5e %12.5e %12.5e %12.5e\n", a15 * a[0] + a16 * a[5] + a17 * a[10] + a18 * a[15] + a19 * a[20],
           a15 * a[1] + a16 * a[6] + a17 * a[11] + a18 * a[16] + a19 * a[21], a15 * a[2] + a16 * a[7] + a17 * a[12] + a18 * a[17] + a19 * a[22],
           a15 * a[3] + a16 * a[8] + a17 * a[13] + a18 * a[18] + a19 * a[23], a15 * a[4] + a16 * a[9] + a17 * a[14] + a18 * a[19] + a19 * a[24]);
    printf("## %12.5e %12.5e %12.5e %12.5e %12.5e\n", a20 * a[0] + a21 * a[5] + a22 * a[10] + a23 * a[15] + a24 * a[20],
           a20 * a[1] + a21 * a[6] + a22 * a[11] + a23 * a[16] + a24 * a[21], a20 * a[2] + a21 * a[7] + a22 * a[12] + a23 * a[17] + a24 * a[22],
           a20 * a[3] + a21 * a[8] + a22 * a[13] + a23 * a[18] + a24 * a[23], a20 * a[4] + a21 * a[9] + a22 * a[14] + a23 * a[19] + a24 * a[24]);
    printf("##----------------------------------------------\n");
}

/**
 * \fn static void jx_smat_inv_nc (JX_Real *a, const JX_Int n)
 *
 * \brief Compute the inverse of a matrix using Gauss Elimination
 *
 * \param a   Pointer to the JX_Real array which stands a n*n matrix
 * \param n   Dimension of the matrix
 *
 * \author zlj
 * \date   2025/10/08
 */
static void jx_smat_inv_nc(JX_Real* a, const JX_Int n)
{
    JX_Int  i, j, k, l, u, kn, in;
    JX_Real alinv;

    for (k = 0; k < n; ++k) {

        kn = k * n;
        l  = kn + k;

        if (jx_cabs(a[l]) < JX_REAL_MIN) {
            printf("### ERROR: Diagonal entry is close to zero! ");
            printf("diag_%d = %.2e! [%s]\n", k, a[l], __FUNCTION__);
            exit(JX_ERROR_GENERIC);
        }
        alinv = 1.0 / a[l];
        a[l]  = alinv;

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
                } // end if (j!=k)
        }

        for (i = k + 1; i < n; ++i) {
            in = i * n;
            for (j = 0; j < n; ++j)
                if (j != k) {
                    u = in + j;
                    a[u] -= a[in + k] * a[kn + j];
                } // end if (j!=k)
        }

        for (i = 0; i < k; ++i) {
            u = i * n + k;
            a[u] *= -alinv;
        }

        for (i = k + 1; i < n; ++i) {
            u = i * n + k;
            a[u] *= -alinv;
        }

    } // end for (k=0; k<n; ++k)
}

/**
 * \fn static JX_Int jx_smat_invp_nc (JX_Real *a, const JX_Int n)
 *
 * \brief Compute the inverse of a matrix using Gauss Elimination with Pivoting
 *
 * \param a   Pointer to the JX_Real array which stands a n*n matrix
 * \param n   Dimension of the matrix
 *
 * \author zlj
 * \date   2025/10/08
 *
 * \note   This routine is based on gaussj() from "Numerical Recipies in C"!
 */
static JX_Int jx_smat_invp_nc(JX_Real* a, const JX_Int n)
{
    JX_Int  i, j, k, l, ll, u;
    JX_Int  icol = 0, irow = 0;
    JX_Real vmax, dum, pivinv, temp;

    JX_Int* work  = (JX_Int*)calloc(3 * n, sizeof(JX_Int));
    JX_Int *indxc = work, *indxr = work + n, *ipiv = work + 2 * n;

    // ipiv, indxr, and indxc are used for book-keeping on the pivoting.
    for (j = 0; j < n; j++) ipiv[j] = 0;

#if DEBUG_MODE > 1
    printf("### DEBUG: Matrix block\n");
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            printf(" %10.5e,", a[i * n + j]);
        }
        printf("\n");
    }
#endif

    // This is the main loop over the columns to be reduced.
    for (i = 0; i < n; i++) {

        // This is the outer loop of the search for a pivot element.
        vmax = 0.0;
        for (j = 0; j < n; j++) {
            if (ipiv[j] != 1) {
                for (k = 0; k < n; k++) {
                    if (ipiv[k] == 0) {
                        u = j * n + k;
                        if (jx_cabs(a[u]) >= vmax) {
                            vmax = jx_cabs(a[u]);
                            irow = j;
                            icol = k;
                        }
                    }
                } // end for k
            }
        } // end for j

        ++(ipiv[icol]);

        // We now have the pivot element, so we interchange rows, if needed, to put
        // the pivot element on the diagonal. The columns are not physically
        // interchanged, only relabeled: indxc[i], the column of the ith pivot
        // element, is the ith column that is reduced, while indxr[i] is the row in
        // which that pivot element was originally located. If indxr[i] != indxc[i]
        // there is an implied column interchange. With this form of bookkeeping,
        // the inverse matrix will be scrambled by columns.
        if (irow != icol) {
            for (l = 0; l < n; l++) SWAP(a[irow * n + l], a[icol * n + l]);
        }

        indxr[i] = irow;
        indxc[i] = icol;
        u        = icol * n + icol;
        if (jx_cabs(a[u]) < JX_REAL_MIN) {
            printf("### WARNING: The matrix is nearly singular!\n");
            return JX_ERROR_GENERIC;
        }
        pivinv = 1.0 / a[u];
        a[u]   = 1.0;
        for (l = 0; l < n; l++) a[icol * n + l] *= pivinv;

        for (ll = 0; ll < n; ll++) {
            if (ll != icol) {
                u    = ll * n + icol;
                dum  = a[u];
                a[u] = 0.0;
                for (l = 0; l < n; l++) a[ll * n + l] -= a[icol * n + l] * dum;
            }
        }
    }
    // This is the end of the main loop over columns of the reduction.

    // It only remains to unscramble the matrix in view of the column interchanges.
    for (l = n - 1; l >= 0; l--) {
        if (indxr[l] != indxc[l])
            for (k = 0; k < n; k++) SWAP(a[k * n + indxr[l]], a[k * n + indxc[l]]);
    } // And we are done.

    free(work);
    work = NULL;

    return 1;
}

/**
 * \fn static JX_Int jx_smat_inv (JX_Real *a, const JX_Int n)
 *
 * \brief Compute the inverse matrix of a small full matrix a
 *
 * \param a   Pointer to the JX_Real array which stands a n*n matrix
 * \param n   Dimension of the matrix
 *
 * \author zlj
 * \date   2025/10/08
 */
static JX_Int jx_smat_inv(JX_Real* a, const JX_Int n)
{
    JX_Int status = 0;

    switch (n) {

        case 2:
            jx_smat_inv_nc2(a);
            break;

        case 3:
            jx_smat_inv_nc3(a);
            break;

        case 4:
            jx_smat_inv_nc4(a);
            break;

        case -5:
            jx_smat_inv_nc5(a);
            break;

        default:
            status = jx_smat_invp_nc(a, n);
            break;
    }

    return status;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
