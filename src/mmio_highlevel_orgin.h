#ifndef _MMIO_HIGHLEVEL_
#define _MMIO_HIGHLEVEL_

#ifndef VALUE_TYPE
#define VALUE_TYPE Real
#endif

#ifndef MAT_PTR_TYPE
#define MAT_PTR_TYPE Int
#endif
#ifndef MAT_VAL_TYPE
#define MAT_VAL_TYPE VALUE_TYPE
#endif
#include "mmio.h"
//#include "common.h"

// read matrix infomation from mtx file
Int mmio_info(Int *m, Int *n, Int *nnz, Int *isSymmetric, char *filename)
{
    Int m_tmp, n_tmp, nnz_tmp;

    Int ret_code;
    MM_typecode matcode;
    FILE *f;

    Int nnz_mtx_report;
    Int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric_tmp = 0, isComplex = 0;

    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }

    if ( mm_is_pattern( matcode ) )  { isPattern = 1; /*printf("type = Pattern\n");*/ }
    if ( mm_is_real ( matcode) )     { isReal = 1; /*printf("type = real\n");*/ }
    if ( mm_is_complex( matcode ) )  { isComplex = 1; /*printf("type = real\n");*/ }
    if ( mm_is_integer ( matcode ) ) { isInteger = 1; /*printf("type = integer\n");*/ }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);
    if (ret_code != 0)
        return -4;

    if ( mm_is_symmetric( matcode ) || mm_is_hermitian( matcode ) )
    {
        isSymmetric_tmp = 1;
        //printf("input matrix is symmetric = true\n");
    }
    else
    {
        //printf("input matrix is symmetric = false\n");
    }

    Int *csrRowPtr_counter = (Int *)malloc((m_tmp+1) * sizeof(Int));
    memset(csrRowPtr_counter, 0, (m_tmp+1) * sizeof(Int));

    Int *csrRowIdx_tmp = (Int *)malloc(nnz_mtx_report * sizeof(Int));
    Int *csrColIdx_tmp = (Int *)malloc(nnz_mtx_report * sizeof(Int));
    VALUE_TYPE *csrVal_tmp    = (VALUE_TYPE *)malloc(nnz_mtx_report * sizeof(VALUE_TYPE));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%Lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
    Int i;
    for (i = 0; i < nnz_mtx_report; i++)
    {
        Int idxi, idxj;
        Real fval, fval_im;
        Int ival;
        Int returnvalue;

        if (isReal)
        {
            returnvalue = fscanf(f, "%d %d %Lg\n", &idxi, &idxj, &fval);
        }
        else if (isComplex)
        {
            returnvalue = fscanf(f, "%d %d %Lg %Lg\n", &idxi, &idxj, &fval, &fval_im);
        }
        else if (isInteger)
        {
            returnvalue = fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval = ival;
        }
        else if (isPattern)
        {
            returnvalue = fscanf(f, "%d %d\n", &idxi, &idxj);
            fval = 1.0;
        }

        // adjust from 1-based to 0-based
        idxi--;
        idxj--;

        csrRowPtr_counter[idxi]++;
        csrRowIdx_tmp[i] = idxi;
        csrColIdx_tmp[i] = idxj;
        csrVal_tmp[i] = fval;
    }

    if (f != stdin)
        fclose(f);

    if (isSymmetric_tmp)
    {
        for (i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
        }
    }

    // exclusive scan for csrRowPtr_counter
    Int old_val, new_val;

    old_val = csrRowPtr_counter[0];
    csrRowPtr_counter[0] = 0;
    for (i = 1; i <= m_tmp; i++)
    {
        new_val = csrRowPtr_counter[i];
        csrRowPtr_counter[i] = old_val + csrRowPtr_counter[i-1];
        old_val = new_val;
    }

    nnz_tmp = csrRowPtr_counter[m_tmp];

    *m = m_tmp;
    *n = n_tmp;
    *nnz = nnz_tmp;
    *isSymmetric = isSymmetric_tmp;

    // free tmp space
    free(csrColIdx_tmp);
    free(csrVal_tmp);
    free(csrRowIdx_tmp);
    free(csrRowPtr_counter);

    return 0;
}

// read matrix infomation from mtx file
Int mmio_data(Int *csrRowPtr, Int *csrColIdx, VALUE_TYPE *csrVal, char *filename)
{
    Int m_tmp, n_tmp, nnz_tmp;

    Int ret_code;
    MM_typecode matcode;
    FILE *f;

    Int nnz_mtx_report;
    Int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric_tmp = 0, isComplex = 0;
    Int i;
    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }

    if ( mm_is_pattern( matcode ) )  { isPattern = 1; /*printf("type = Pattern\n");*/ }
    if ( mm_is_real ( matcode) )     { isReal = 1; /*printf("type = real\n");*/ }
    if ( mm_is_complex( matcode ) )  { isComplex = 1; /*printf("type = real\n");*/ }
    if ( mm_is_integer ( matcode ) ) { isInteger = 1; /*printf("type = integer\n");*/ }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);
    if (ret_code != 0)
        return -4;

    if ( mm_is_symmetric( matcode ) || mm_is_hermitian( matcode ) )
    {
        isSymmetric_tmp = 1;
        //printf("input matrix is symmetric = true\n");
    }
    else
    {
        //printf("input matrix is symmetric = false\n");
    }

    Int *csrRowPtr_counter = (Int *)malloc((m_tmp+1) * sizeof(Int));
    memset(csrRowPtr_counter, 0, (m_tmp+1) * sizeof(Int));

    Int *csrRowIdx_tmp = (Int *)malloc(nnz_mtx_report * sizeof(Int));
    Int *csrColIdx_tmp = (Int *)malloc(nnz_mtx_report * sizeof(Int));
    VALUE_TYPE *csrVal_tmp    = (VALUE_TYPE *)malloc(nnz_mtx_report * sizeof(VALUE_TYPE));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%Lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (i = 0; i < nnz_mtx_report; i++)
    {
        Int idxi, idxj;
        Real fval, fval_im;
        Int ival;
        Int returnvalue;

        if (isReal)
        {
            returnvalue = fscanf(f, "%d %d %Lg\n", &idxi, &idxj, &fval);
        }
        else if (isComplex)
        {
            returnvalue = fscanf(f, "%d %d %Lg %Lg\n", &idxi, &idxj, &fval, &fval_im);
        }
        else if (isInteger)
        {
            returnvalue = fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval = ival;
        }
        else if (isPattern)
        {
            returnvalue = fscanf(f, "%d %d\n", &idxi, &idxj);
            fval = 1.0;
        }

        // adjust from 1-based to 0-based
        idxi--;
        idxj--;

        csrRowPtr_counter[idxi]++;
        csrRowIdx_tmp[i] = idxi;
        csrColIdx_tmp[i] = idxj;
        csrVal_tmp[i] = fval;
    }

    if (f != stdin)
        fclose(f);

    if (isSymmetric_tmp)
    {
        for (i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
        }
    }

    // exclusive scan for csrRowPtr_counter
    Int old_val, new_val;

    old_val = csrRowPtr_counter[0];
    csrRowPtr_counter[0] = 0;
    for (i = 1; i <= m_tmp; i++)
    {
        new_val = csrRowPtr_counter[i];
        csrRowPtr_counter[i] = old_val + csrRowPtr_counter[i-1];
        old_val = new_val;
    }

    nnz_tmp = csrRowPtr_counter[m_tmp];
    memcpy(csrRowPtr, csrRowPtr_counter, (m_tmp+1) * sizeof(Int));
    memset(csrRowPtr_counter, 0, (m_tmp+1) * sizeof(Int));

    if (isSymmetric_tmp)
    {
        for (i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
            {
                Int offset = csrRowPtr[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx[offset] = csrColIdx_tmp[i];
                csrVal[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;

                offset = csrRowPtr[csrColIdx_tmp[i]] + csrRowPtr_counter[csrColIdx_tmp[i]];
                csrColIdx[offset] = csrRowIdx_tmp[i];
                csrVal[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
            }
            else
            {
                Int offset = csrRowPtr[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx[offset] = csrColIdx_tmp[i];
                csrVal[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;
            }
        }
    }
    else
    {
        for (i = 0; i < nnz_mtx_report; i++)
        {
            Int offset = csrRowPtr[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
            csrColIdx[offset] = csrColIdx_tmp[i];
            csrVal[offset] = csrVal_tmp[i];
            csrRowPtr_counter[csrRowIdx_tmp[i]]++;
        }
    }

    // free tmp space
    free(csrColIdx_tmp);
    free(csrVal_tmp);
    free(csrRowIdx_tmp);
    free(csrRowPtr_counter);

    return 0;
}

void exclusive_scan(MAT_PTR_TYPE *input, Int length)
{
    if(length == 0 || length == 1)
        return;

    MAT_PTR_TYPE old_val, new_val;

    old_val = input[0];
    input[0] = 0;
    Int i;
    for (i = 1; i < length; i++)
    {
        new_val = input[i];
        input[i] = old_val + input[i-1];
        old_val = new_val;
    }
}
Int mmio_allinone_complex(Int *m, Int *n, Int *nnz, Int *isSymmetric,
                  Int **csrRowPtr, Int **csrColIdx, MAT_VAL_TYPE **csrVal_real,
                  MAT_VAL_TYPE **csrVal_virtual,
                  char *filename)
{
    Int m_tmp, n_tmp;
    MAT_PTR_TYPE nnz_tmp;

    Int ret_code;
    MM_typecode matcode;
    FILE *f;

    MAT_PTR_TYPE nnz_mtx_report;
    Int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric_tmp = 0, isComplex = 0;
    Int i;
    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }

    if ( mm_is_pattern( matcode ) )  { isPattern = 1; /*printf("type = Pattern\n");*/ }
    if ( mm_is_real ( matcode) )     { isReal = 1; /*printf("type = real\n");*/ }
    if ( mm_is_complex( matcode ) )  { isComplex = 1; /*printf("type = real\n");*/ }
    if ( mm_is_integer ( matcode ) ) { isInteger = 1; /*printf("type = integer\n");*/ }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);
    if (ret_code != 0)
        return -4;

    if ( mm_is_symmetric( matcode ) || mm_is_hermitian( matcode ) )
    {
        isSymmetric_tmp = 1;
        //printf("input matrix is symmetric = true\n");
    }
    else
    {
        //printf("input matrix is symmetric = false\n");
    }

    MAT_PTR_TYPE *csrRowPtr_counter = (MAT_PTR_TYPE *)malloc((m_tmp+1) * sizeof(MAT_PTR_TYPE));
    memset(csrRowPtr_counter, 0, (m_tmp+1) * sizeof(MAT_PTR_TYPE));

    Int *csrRowIdx_tmp = (Int *)malloc(nnz_mtx_report * sizeof(Int));
    Int *csrColIdx_tmp = (Int *)malloc(nnz_mtx_report * sizeof(Int));
    MAT_VAL_TYPE *csrVal_tmp    = (MAT_VAL_TYPE *)malloc(nnz_mtx_report * sizeof(MAT_VAL_TYPE));
    MAT_VAL_TYPE *csrVal_tmp_v    = (MAT_VAL_TYPE *)malloc(nnz_mtx_report * sizeof(MAT_VAL_TYPE));
    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%Lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (i = 0; i < nnz_mtx_report; i++)
    {
        Int idxi, idxj;
        Real fval, fval_im;
        Int ival;
        Int returnvalue;

        if (isReal)
        {
            returnvalue = fscanf(f, "%d %d %Lg\n", &idxi, &idxj, &fval);
        }
        else if (isComplex)
        {
            returnvalue = fscanf(f, "%d %d %Lg %Lg\n", &idxi, &idxj, &fval, &fval_im);
        }
        else if (isInteger)
        {
            returnvalue = fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval = ival;
        }
        else if (isPattern)
        {
            returnvalue = fscanf(f, "%d %d\n", &idxi, &idxj);
            fval = 1.0;
        }

        // adjust from 1-based to 0-based
        idxi--;
        idxj--;

        csrRowPtr_counter[idxi]++;
        csrRowIdx_tmp[i] = idxi;
        csrColIdx_tmp[i] = idxj;
        csrVal_tmp[i] = fval;
        csrVal_tmp_v[i] =fval_im;
    }

    if (f != stdin)
        fclose(f);

    if (isSymmetric_tmp)
    {
        for (i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
        }
    }

    // exclusive scan for csrRowPtr_counter
    exclusive_scan(csrRowPtr_counter, m_tmp+1);

    MAT_PTR_TYPE *csrRowPtr_alias = (MAT_PTR_TYPE *)malloc((m_tmp+1) * sizeof(MAT_PTR_TYPE));
    nnz_tmp = csrRowPtr_counter[m_tmp];
    Int *csrColIdx_alias = (Int *)malloc(nnz_tmp * sizeof(Int));
    MAT_VAL_TYPE *csrVal_alias    = (MAT_VAL_TYPE *)malloc(nnz_tmp * sizeof(MAT_VAL_TYPE));
    MAT_VAL_TYPE *csrVal_alias_v    = (MAT_VAL_TYPE *)malloc(nnz_tmp * sizeof(MAT_VAL_TYPE));
    memcpy(csrRowPtr_alias, csrRowPtr_counter, (m_tmp+1) * sizeof(MAT_PTR_TYPE));
    memset(csrRowPtr_counter, 0, (m_tmp+1) * sizeof(MAT_PTR_TYPE));

    if (isSymmetric_tmp)
    {
        for (i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
            {
                MAT_PTR_TYPE offset = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx_alias[offset] = csrColIdx_tmp[i];
                csrVal_alias[offset] = csrVal_tmp[i];
                csrVal_alias_v[offset] = csrVal_tmp_v[i]; 
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;

                offset = csrRowPtr_alias[csrColIdx_tmp[i]] + csrRowPtr_counter[csrColIdx_tmp[i]];
                csrColIdx_alias[offset] = csrRowIdx_tmp[i];
                csrVal_alias[offset] = csrVal_tmp[i];
                csrVal_alias_v[offset] = csrVal_tmp_v[i]; 
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
            }
            else
            {
                MAT_PTR_TYPE offset = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx_alias[offset] = csrColIdx_tmp[i];
                csrVal_alias[offset] = csrVal_tmp[i];
                csrVal_alias_v[offset] = csrVal_tmp_v[i]; 
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;
            }
        }
    }
    else
    {
        for (i = 0; i < nnz_mtx_report; i++)
        {
            MAT_PTR_TYPE offset = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
            csrColIdx_alias[offset] = csrColIdx_tmp[i];
            csrVal_alias[offset] = csrVal_tmp[i];
            csrVal_alias_v[offset] = csrVal_tmp_v[i]; 
            csrRowPtr_counter[csrRowIdx_tmp[i]]++;
        }
    }

    *m = m_tmp;
    *n = n_tmp;
    *nnz = nnz_tmp;
    *isSymmetric = isSymmetric_tmp;

    *csrRowPtr = csrRowPtr_alias;
    *csrColIdx = csrColIdx_alias;
    *csrVal_real = csrVal_alias;
    *csrVal_virtual = csrVal_alias_v;

    // free tmp space
    free(csrColIdx_tmp);
    free(csrVal_tmp);
    free(csrRowIdx_tmp);
    free(csrRowPtr_counter);

    return 0;
}


Int mmio_allinone(Int *m, Int *n, Int *nnz, Int *isSymmetric,
                  Int **csrRowPtr, Int **csrColIdx, MAT_VAL_TYPE **csrVal,
                  char *filename)
{
    Int m_tmp, n_tmp;
    MAT_PTR_TYPE nnz_tmp;

    Int ret_code;
    MM_typecode matcode;
    FILE *f;

    MAT_PTR_TYPE nnz_mtx_report;
    Int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric_tmp = 0, isComplex = 0;
    Int i;
    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }

    if ( mm_is_pattern( matcode ) )  { isPattern = 1; /*printf("type = Pattern\n");*/ }
    if ( mm_is_real ( matcode) )     { isReal = 1; /*printf("type = real\n");*/ }
    if ( mm_is_complex( matcode ) )  { isComplex = 1; /*printf("type = real\n");*/ }
    if ( mm_is_integer ( matcode ) ) { isInteger = 1; /*printf("type = integer\n");*/ }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);
    if (ret_code != 0)
        return -4;

    if ( mm_is_symmetric( matcode ) || mm_is_hermitian( matcode ) )
    {
        isSymmetric_tmp = 1;
        //printf("input matrix is symmetric = true\n");
    }
    else
    {
        //printf("input matrix is symmetric = false\n");
    }

    MAT_PTR_TYPE *csrRowPtr_counter = (MAT_PTR_TYPE *)malloc((m_tmp+1) * sizeof(MAT_PTR_TYPE));
    memset(csrRowPtr_counter, 0, (m_tmp+1) * sizeof(MAT_PTR_TYPE));

    Int *csrRowIdx_tmp = (Int *)malloc(nnz_mtx_report * sizeof(Int));
    Int *csrColIdx_tmp = (Int *)malloc(nnz_mtx_report * sizeof(Int));
    MAT_VAL_TYPE *csrVal_tmp    = (MAT_VAL_TYPE *)malloc(nnz_mtx_report * sizeof(MAT_VAL_TYPE));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%Lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (i = 0; i < nnz_mtx_report; i++)
    {
        Int idxi, idxj;
        Real fval, fval_im;
        Int ival;
        Int returnvalue;

        if (isReal)
        {
            returnvalue = fscanf(f, "%d %d %Lf\n", &idxi, &idxj, &fval);
        }
        else if (isComplex)
        {
            returnvalue = fscanf(f, "%d %d %Lf %Lf\n", &idxi, &idxj, &fval, &fval_im);
        }
        else if (isInteger)
        {
            returnvalue = fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval = ival;
        }
        else if (isPattern)
        {
            returnvalue = fscanf(f, "%d %d\n", &idxi, &idxj);
            fval = 1.0;
        }

        // adjust from 1-based to 0-based
        idxi--;
        idxj--;

        csrRowPtr_counter[idxi]++;
        csrRowIdx_tmp[i] = idxi;
        csrColIdx_tmp[i] = idxj;
        csrVal_tmp[i] = fval;
    }

    if (f != stdin)
        fclose(f);

    if (isSymmetric_tmp)
    {
        for (i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
        }
    }

    // exclusive scan for csrRowPtr_counter
    exclusive_scan(csrRowPtr_counter, m_tmp+1);

    MAT_PTR_TYPE *csrRowPtr_alias = (MAT_PTR_TYPE *)malloc((m_tmp+1) * sizeof(MAT_PTR_TYPE));
    nnz_tmp = csrRowPtr_counter[m_tmp];
    Int *csrColIdx_alias = (Int *)malloc(nnz_tmp * sizeof(Int));
    MAT_VAL_TYPE *csrVal_alias    = (MAT_VAL_TYPE *)malloc(nnz_tmp * sizeof(MAT_VAL_TYPE));

    memcpy(csrRowPtr_alias, csrRowPtr_counter, (m_tmp+1) * sizeof(MAT_PTR_TYPE));
    memset(csrRowPtr_counter, 0, (m_tmp+1) * sizeof(MAT_PTR_TYPE));

    if (isSymmetric_tmp)
    {
        for (i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
            {
                MAT_PTR_TYPE offset = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx_alias[offset] = csrColIdx_tmp[i];
                csrVal_alias[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;

                offset = csrRowPtr_alias[csrColIdx_tmp[i]] + csrRowPtr_counter[csrColIdx_tmp[i]];
                csrColIdx_alias[offset] = csrRowIdx_tmp[i];
                csrVal_alias[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
            }
            else
            {
                MAT_PTR_TYPE offset = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx_alias[offset] = csrColIdx_tmp[i];
                csrVal_alias[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;
            }
        }
    }
    else
    {
        for (i = 0; i < nnz_mtx_report; i++)
        {
            MAT_PTR_TYPE offset = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
            csrColIdx_alias[offset] = csrColIdx_tmp[i];
            csrVal_alias[offset] = csrVal_tmp[i];
            csrRowPtr_counter[csrRowIdx_tmp[i]]++;
        }
    }

    *m = m_tmp;
    *n = n_tmp;
    *nnz = nnz_tmp;
    *isSymmetric = isSymmetric_tmp;

    *csrRowPtr = csrRowPtr_alias;
    *csrColIdx = csrColIdx_alias;
    *csrVal = csrVal_alias;

    // free tmp space
    free(csrColIdx_tmp);
    free(csrVal_tmp);
    free(csrRowIdx_tmp);
    free(csrRowPtr_counter);

    return 0;
}
void matrix_transposition(const Int           m,
                          const Int           n,
                          const MAT_PTR_TYPE     nnz,
                          const MAT_PTR_TYPE    *csrRowPtr,
                          const Int          *csrColIdx,
                          const MAT_VAL_TYPE *csrVal,
                          Int          *cscRowIdx,
                          MAT_PTR_TYPE    *cscColPtr,
                          MAT_VAL_TYPE *cscVal)
{
    // histogram in column pointer
    memset (cscColPtr, 0, sizeof(MAT_PTR_TYPE) * (n+1));
    Int i, row, j;
    for (i = 0; i < nnz; i++)
    {
        cscColPtr[csrColIdx[i]]++;
    }

    // prefix-sum scan to get the column pointer
    exclusive_scan(cscColPtr, n + 1);

    MAT_PTR_TYPE *cscColIncr = (MAT_PTR_TYPE *)malloc(sizeof(MAT_PTR_TYPE) * (n+1));
    memcpy (cscColIncr, cscColPtr, sizeof(MAT_PTR_TYPE) * (n+1));

    // insert nnz to csc
    for (row = 0; row < m; row++)
    {
        for (j = csrRowPtr[row]; j < csrRowPtr[row+1]; j++)
        {
            Int col = csrColIdx[j];

            cscRowIdx[cscColIncr[col]] = row;
            cscVal[cscColIncr[col]] = csrVal[j];
            cscColIncr[col]++;
        }
    }

    free (cscColIncr);
}

Int mmio_save_as_bin(Int m,Int n,Int nnz,const Int *rowptr,const Int *colidx,const VALUE_TYPE* val,const char*file){
    char buffer[1024];
    char fileBuffer[1024];
    char *s;
    strcpy(buffer,"mtx_cache/");
    strcpy(fileBuffer,file);
    for(s = fileBuffer ; *s ;++s){
        if(*s=='/' || *s =='\\' || *s==' '){
            *s='_';
        }
    }
    strcat(buffer,fileBuffer);
    strcat(buffer,".bin");

    FILE *pFile = fopen(buffer, "wb");
    if(pFile) {
        fwrite(&m, sizeof(char), sizeof(Int), pFile);
        fwrite(&n, sizeof(char), sizeof(Int), pFile);
        fwrite(&nnz, sizeof(char), sizeof(Int), pFile);
        fwrite(rowptr, sizeof(char), sizeof(Int) * (m + 1), pFile);
        fwrite(colidx, sizeof(char), sizeof(Int) * (nnz), pFile);
        fwrite(val, sizeof(char), sizeof(VALUE_TYPE) * (nnz), pFile);
        return 0;
    }else return 1;
}
// Int mmio_read_from_bin(Int *m,Int *n,Int *nnz,Int **rowptr,Int **colidx,VALUE_TYPE** val,const char *file,
//         Int aligenSize
// ){
//     char buffer[1024];
//     char fileBuffer[1024];
//     strcpy(buffer,"mtx_cache/");
//     strcpy(fileBuffer,file);
//     for(char *s = fileBuffer ; *s ;++s){
//         if(*s=='/' || *s =='\\' || *s==' '){
//             *s='_';
//         }
//     }
//     strcat(buffer,fileBuffer);
//     strcat(buffer,".bin");
//     FILE *pFile = fopen(buffer, "rb");
//     if(pFile) {
//         fread(m, sizeof(char), sizeof(Int), pFile);
//         fread(n, sizeof(char), sizeof(Int), pFile);
//         fread(nnz, sizeof(char), sizeof(Int), pFile);
//         *rowptr = (Int*)aligned_alloc(aligenSize,sizeof(Int )*(*m+1));
//         *colidx = (Int*)aligned_alloc(aligenSize,sizeof(Int )*(*nnz));
//         *val = (VALUE_TYPE*)aligned_alloc(aligenSize,sizeof(VALUE_TYPE )*(*nnz));

//         fread(*rowptr, sizeof(char), sizeof(Int) * (*m + 1), pFile);
//         fread(*colidx, sizeof(char), sizeof(Int) * (*nnz), pFile);
//         fread(*val, sizeof(char), sizeof(VALUE_TYPE) * (*nnz), pFile);
//         return 0;
//     }else return 1;

// }
#endif
