//========================================================================//
//  JXFPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  seq_csr_matrix.c -- basic operations for CSR matrices.
 *  Date: 2011/09/03
 */ 
 
#include "jxf_mv.h"

/*!
 * \fn jxf_CSRMatrix *jxf_CSRMatrixCreate
 * \brief Create a CSR format matrix.
 * \param num_rows number of rows
 * \param num_cols number of columns
 * \param num_nonzeros number of nonzero entries
 * \date 2011/09/03
 */
jxf_CSRMatrix *
jxf_CSRMatrixCreate( JXF_Int num_rows, JXF_Int num_cols, JXF_Int num_nonzeros )
{
   jxf_CSRMatrix  *matrix;

   matrix = jxf_CTAlloc(jxf_CSRMatrix, 1);

   jxf_CSRMatrixData(matrix) = NULL;
   jxf_CSRMatrixI(matrix)    = NULL;
   jxf_CSRMatrixJ(matrix)    = NULL;
   jxf_CSRMatrixRownnz(matrix) = NULL;
   jxf_CSRMatrixNumRows(matrix) = num_rows;
   jxf_CSRMatrixNumCols(matrix) = num_cols;
   jxf_CSRMatrixNumNonzeros(matrix) = num_nonzeros;

   /* set defaults */
   jxf_CSRMatrixOwnsData(matrix) = 1;
   jxf_CSRMatrixNumRownnz(matrix) = num_rows;

   return matrix;
}

/*!
 * \fn JXF_Int jxf_CSRMatrixInitialize
 * \brief Initialize a CSR format matrix.
 * \param *matrix pointer to the matrix to be initialized.
 * \date 2011/09/03
 */
JXF_Int 
jxf_CSRMatrixInitialize( jxf_CSRMatrix *matrix )
{
   JXF_Int  num_rows     = jxf_CSRMatrixNumRows(matrix);
   JXF_Int  num_nonzeros = jxf_CSRMatrixNumNonzeros(matrix);

   JXF_Int  ierr = 0;

   if ( ! jxf_CSRMatrixData(matrix) && num_nonzeros )
      jxf_CSRMatrixData(matrix) = jxf_CTAlloc(JXF_Real, num_nonzeros);
   if ( ! jxf_CSRMatrixI(matrix) )
      jxf_CSRMatrixI(matrix)    = jxf_CTAlloc(JXF_Int, num_rows + 1);
   if ( ! jxf_CSRMatrixJ(matrix) && num_nonzeros )
      jxf_CSRMatrixJ(matrix)    = jxf_CTAlloc(JXF_Int, num_nonzeros);

   return ierr;
}


/*!
 * \fn JXF_Int jxf_CSRMatrixDestroy
 * \brief Destroy a CSR format matrix.
 * \param *matrix pointer to the matrix to be destroyed.
 * \date 2011/09/03
 */
JXF_Int 
jxf_CSRMatrixDestroy( jxf_CSRMatrix *matrix )
{
   JXF_Int  ierr=0;

   if (matrix)
   {
      jxf_TFree(jxf_CSRMatrixI(matrix));
      if (jxf_CSRMatrixRownnz(matrix))
         jxf_TFree(jxf_CSRMatrixRownnz(matrix));
      if ( jxf_CSRMatrixOwnsData(matrix) )
      {
         jxf_TFree(jxf_CSRMatrixData(matrix));
         jxf_TFree(jxf_CSRMatrixJ(matrix));
      }
      jxf_TFree(matrix);
   }

   return ierr;
}

/*!
 * \fn JXF_Int jxf_CSRMatrixPrint
 * \brief Print a CSR format matrix into given file.
 *        format of the file:
 *        --------------------------------------------
 *           nrows
 *           ia[i], i=0(1)nrows
 *           ja[i], i=0(1)nonzeros
 *            a[i], i=0(1)nonzeros
 *        -------------------------------------------- 
 *        There is only one number in each line, row and colum 
 *        index are of fortran style.    
 * \param *matrix pointer to the matrix to be printed.
 * \param *file_name pointer to the file name.
 * \date 2011/09/03
 */
JXF_Int
jxf_CSRMatrixPrint( jxf_CSRMatrix *matrix, char *file_name )
{
   FILE    *fp;

   JXF_Real  *matrix_data;
   JXF_Int     *matrix_i;
   JXF_Int     *matrix_j;
   JXF_Int      num_rows;
   
   JXF_Int      file_base = 1;
   
   JXF_Int      j;

   JXF_Int      ierr = 0;

  /*--------------------------------------------
   * Print the matrix data
   *-------------------------------------------*/
   
   matrix_data = jxf_CSRMatrixData(matrix);
   matrix_i    = jxf_CSRMatrixI(matrix);
   matrix_j    = jxf_CSRMatrixJ(matrix);
   num_rows    = jxf_CSRMatrixNumRows(matrix);

   fp = fopen(file_name, "w");

   jxf_fprintf(fp, "%d\n", num_rows);

   for (j = 0; j <= num_rows; j ++)
   {
      jxf_fprintf(fp, "%d\n", matrix_i[j] + file_base);
   }

   for (j = 0; j < matrix_i[num_rows]; j ++)
   {
      jxf_fprintf(fp, "%d\n", matrix_j[j] + file_base);
   }

   if (matrix_data)
   {
      for (j = 0; j < matrix_i[num_rows]; j ++)
      {
         jxf_fprintf(fp, "%.14e\n", matrix_data[j]);
      }
   }
   else
   {
      jxf_fprintf(fp, "Warning: No matrix data!\n");
   }

   fclose(fp);

   return ierr;
}

/*!
 * \fn jxf_CSRMatrix *jxf_CSRMatrixRead
 * \brief Read a CSR format matrix from given file.
 *        format of the file:
 *        --------------------------------------------
 *           nrows
 *           ia[i], i=0(1)nrows
 *           ja[i], i=0(1)nonzeros
 *            a[i], i=0(1)nonzeros
 *        -------------------------------------------- 
 *        There is only one number in each line, row and colum 
 *        index are of fortran (fb=1) or C (fb=0) style.    
 * \param *file_name pointer to the file name.
 * \date 2011/09/03
 */
jxf_CSRMatrix *
jxf_CSRMatrixRead( char *file_name, JXF_Int file_base )
{
   jxf_CSRMatrix  *matrix;

   FILE    *fp;

   JXF_Real  *matrix_data;
   JXF_Int     *matrix_i;
   JXF_Int     *matrix_j;
   JXF_Int      num_rows;
   JXF_Int      num_nonzeros;
   JXF_Int      max_col = 0;

   JXF_Int      j;

  /*------------------------------------------
   *  Read in the data
   *----------------------------------------*/

   fp = fopen(file_name, "r");

   jxf_fscanf(fp, "%d", &num_rows);

   matrix_i = jxf_CTAlloc(JXF_Int, num_rows + 1);
   for (j = 0; j < num_rows+1; j ++)
   {
      jxf_fscanf(fp, "%d", &matrix_i[j]);
      matrix_i[j] -= file_base;
   }

   num_nonzeros = matrix_i[num_rows];

   matrix = jxf_CSRMatrixCreate(num_rows, num_rows, matrix_i[num_rows]);
   jxf_CSRMatrixI(matrix) = matrix_i;
   jxf_CSRMatrixInitialize(matrix);

   matrix_j = jxf_CSRMatrixJ(matrix);
   for (j = 0; j < num_nonzeros; j ++)
   {
      jxf_fscanf(fp, "%d", &matrix_j[j]);
      matrix_j[j] -= file_base;

      if (matrix_j[j] > max_col)
      {
         max_col = matrix_j[j];
      }
   }

   matrix_data = jxf_CSRMatrixData(matrix);
   for (j = 0; j < matrix_i[num_rows]; j ++)
   {
      jxf_fscanf(fp, "%le", &matrix_data[j]);
   }

   fclose(fp);

   jxf_CSRMatrixNumNonzeros(matrix) = num_nonzeros;
   jxf_CSRMatrixNumCols(matrix) = ++ max_col;

   return matrix;
}

jxf_CSRMatrix *
jxf_CSRMatrixRead2( char *file_name, JXF_Int file_base )
{
   jxf_CSRMatrix  *matrix;

   FILE    *fp;

   JXF_Real  *matrix_data;
   JXF_Int     *matrix_i;
   JXF_Int     *matrix_j;
   JXF_Int      num_rows;
   JXF_Int      num_nonzeros;
   JXF_Int      max_col = 0;

   JXF_Int      j, tmp;

  /*------------------------------------------
   *  Read in the data
   *----------------------------------------*/

   fp = fopen(file_name, "r");

   jxf_fscanf(fp, "%d", &num_rows);
   jxf_fscanf(fp, "%d", &num_nonzeros);
   for (j = 0; j < num_rows; j ++)
   {
      jxf_fscanf(fp, "%d", &tmp);
   }

   matrix_i = jxf_CTAlloc(JXF_Int, num_rows + 1);
   for (j = 0; j < num_rows+1; j ++)
   {
      jxf_fscanf(fp, "%d", &matrix_i[j]);
      matrix_i[j] -= file_base;
   }

   jxf_assert(matrix_i[num_rows] == num_nonzeros);

   matrix = jxf_CSRMatrixCreate(num_rows, num_rows, matrix_i[num_rows]);
   jxf_CSRMatrixI(matrix) = matrix_i;
   jxf_CSRMatrixInitialize(matrix);

   matrix_j = jxf_CSRMatrixJ(matrix);
   for (j = 0; j < num_nonzeros; j ++)
   {
      jxf_fscanf(fp, "%d", &matrix_j[j]);
      matrix_j[j] -= file_base;

      if (matrix_j[j] > max_col)
      {
         max_col = matrix_j[j];
      }
   }

   matrix_data = jxf_CSRMatrixData(matrix);
   for (j = 0; j < matrix_i[num_rows]; j ++)
   {
      jxf_fscanf(fp, "%le", &matrix_data[j]);
   }

   fclose(fp);

   jxf_CSRMatrixNumNonzeros(matrix) = num_nonzeros;
   jxf_CSRMatrixNumCols(matrix) = ++ max_col;

   return matrix;
}

/*!
 * \fn JXF_Int jxf_CSRMatrixGetBandWidth
 * \brief Get the left and right bandwidth
 * \author Yue Xiaoqiang
 * \date 2012/10/12
 */
JXF_Int
jxf_CSRMatrixGetBandWidth( jxf_CSRMatrix *A, JXF_Int *nbl_ptr, JXF_Int *nbr_ptr )
{
   JXF_Int ierr = 0;
   JXF_Int *IA = jxf_CSRMatrixI(A);
   JXF_Int *JA = jxf_CSRMatrixJ(A);
   JXF_Int myid, mybegin, myend;
   JXF_Int max_l, max_r;
   JXF_Int i, end_row_A, j;
   JXF_Int num_threads;
   
   num_threads = jxf_NumThreads();
   
   // num_rows must be greater than openmp_holds
   JXF_Int *max_left_right = jxf_CTAlloc(JXF_Int, 2*num_threads);
#define JXF_SMP_PRIVATE myid,mybegin,myend,max_l,max_r,i,end_row_A,j
#include "../../include/jxf_smp_forloop.h"
   for (myid = 0; myid < num_threads; myid ++)
   {
      JXF_OMP_GET_START_END(myid, num_threads, A->num_rows, mybegin, myend);
      max_l = 0;
      max_r = 0;
      for (i = mybegin; i < myend; i ++)
      {
         end_row_A = IA[i+1];
         for (j = IA[i]; j < end_row_A; j ++)
         {
            max_l = jxf_max(i-JA[j], max_l);
            max_r = jxf_max(JA[j]-i, max_r);
         }
      }
      max_left_right[myid*2] = max_l;
      max_left_right[myid*2+1] = max_r;
   }
   max_l = max_left_right[0];
   max_r = max_left_right[1];
   for (i = 1; i < num_threads; i ++)
   {
      max_l = jxf_max(max_l, max_left_right[i*2]);
      max_r = jxf_max(max_r, max_left_right[i*2+1]);
   }
   jxf_TFree(max_left_right);
  *nbl_ptr = max_l;
  *nbr_ptr = max_r;
   
   return ierr;
}

/*!
 * \fn JXF_Int jxf_CSRMatrixSetRownnz
 * \brief function to set the substructure rownnz and num_rowsnnz inside the CSRMatrix
 * it needs the A_i substructure of CSRMatrix to find the nonzero rows.
 * It runs after the create CSR and when A_i is known..It does not check for
 * the existence of A_i or of the CSR matrix.
 * \date 2015/11/28
 */
JXF_Int
jxf_CSRMatrixSetRownnz( jxf_CSRMatrix *matrix )
{
   JXF_Int  ierr=0;
   JXF_Int  num_rows = jxf_CSRMatrixNumRows(matrix);
   JXF_Int *A_i = jxf_CSRMatrixI(matrix);
   JXF_Int *Arownnz;
   JXF_Int i, adiag;
   JXF_Int irownnz=0;

   for (i=0; i < num_rows; i++)
   {
      adiag = (A_i[i+1] - A_i[i]);
      if(adiag > 0) irownnz++;
   }
   jxf_CSRMatrixNumRownnz(matrix) = irownnz;
   if ((irownnz == 0) || (irownnz == num_rows))
   {
      jxf_CSRMatrixRownnz(matrix) = NULL;
   }
   else
   {
      Arownnz = jxf_CTAlloc(JXF_Int, irownnz);
      irownnz = 0;
      for (i=0; i < num_rows; i++)
      {
         adiag = A_i[i+1]-A_i[i];
         if(adiag > 0) Arownnz[irownnz++] = i;
      }
      jxf_CSRMatrixRownnz(matrix) = Arownnz;
   }
   return ierr;
}
