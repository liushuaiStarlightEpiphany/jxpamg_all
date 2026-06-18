//========================================================================//
//  JXFPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  seq_csr_matvec.c -- basic operations for mat-vec multiplication.
 *  Date: 2011/09/03
 */ 

#include "jxf_mv.h"

/*!
 * \fn JXF_Int jxf_CSRMatrixMatvec
 * \brief Perform y = alpha*A*x + beta*y.
 * \date 2011/09/03
 */
JXF_Int
jxf_CSRMatrixMatvec( JXF_Real        alpha,
                    jxf_CSRMatrix *A,
                    jxf_Vector    *x,
                    JXF_Real        beta,
                    jxf_Vector    *y     )
{
   JXF_Real     *A_data   = jxf_CSRMatrixData(A);
   JXF_Int        *A_i      = jxf_CSRMatrixI(A);
   JXF_Int        *A_j      = jxf_CSRMatrixJ(A);
   JXF_Int         num_rows = jxf_CSRMatrixNumRows(A);
   JXF_Int         num_cols = jxf_CSRMatrixNumCols(A);

   JXF_Int        *A_rownnz = jxf_CSRMatrixRownnz(A);
   JXF_Int         num_rownnz = jxf_CSRMatrixNumRownnz(A);

   JXF_Real     *x_data = jxf_VectorData(x);
   JXF_Real     *y_data = jxf_VectorData(y);
   JXF_Int         x_size = jxf_VectorSize(x);
   JXF_Int         y_size = jxf_VectorSize(y);
   JXF_Int         num_vectors = jxf_VectorNumVectors(x);
   JXF_Int         idxstride_y = jxf_VectorIndexStride(y);
   JXF_Int         vecstride_y = jxf_VectorVectorStride(y);
   JXF_Int         idxstride_x = jxf_VectorIndexStride(x);
   JXF_Int         vecstride_x = jxf_VectorVectorStride(x);

   JXF_Real      temp, tempx;

   JXF_Int         i, j, jj;

   JXF_Int         m;

   JXF_Real      xpar = 0.7;

   JXF_Int         ierr = 0;

   /*---------------------------------------------------------------------
    *  Check for size compatibility.  Matvec returns ierr = 1 if
    *  length of X doesn't equal the number of columns of A,
    *  ierr = 2 if the length of Y doesn't equal the number of rows
    *  of A, and ierr = 3 if both are true.
    *
    *  Because temporary vectors are often used in Matvec, none of 
    *  these conditions terminates processing, and the ierr flag
    *  is informational only.
    *--------------------------------------------------------------------*/
 
    jxf_assert( num_vectors == jxf_VectorNumVectors(y) );

    if (num_cols != x_size)
              ierr = 1;

    if (num_rows != y_size)
              ierr = 2;

    if (num_cols != x_size && num_rows != y_size)
              ierr = 3;

   /*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
    *-----------------------------------------------------------------------*/

    if (alpha == 0.0)
    {
#define JXF_SMP_PRIVATE i
#include "../../../include/jxf_smp_forloop.h"
       for (i = 0; i < num_rows*num_vectors; i ++)
          y_data[i] *= beta;

       return ierr;
    }

  /*--------------------------------------------
   * y = (beta/alpha)*y
   *------------------------------------------*/
   
   temp = beta / alpha;
   
   if (temp != 1.0)
   {
      if (temp == 0.0)
      {
#define JXF_SMP_PRIVATE i
#include "../../../include/jxf_smp_forloop.h"
	 for (i = 0; i < num_rows*num_vectors; i ++)
	    y_data[i] = 0.0;
      }
      else
      {
#define JXF_SMP_PRIVATE i
#include "../../../include/jxf_smp_forloop.h"
	 for (i = 0; i < num_rows*num_vectors; i ++)
	    y_data[i] *= temp;
      }
   }

  /*-----------------------------------------------
   * y += A*x
   *---------------------------------------------*/

   /* use rownnz pointer to do the A*x multiplication  
      when num_rownnz is smaller than num_rows */

   if (num_rownnz < xpar*(num_rows))
   {
   
#define JXF_SMP_PRIVATE i,jj,m,tempx
#include "../../../include/jxf_smp_forloop.h"

      for (i = 0; i < num_rownnz; i ++)
      {
         m = A_rownnz[i];

         if (num_vectors == 1)
         {
            tempx = y_data[m];
            for (jj = A_i[m]; jj < A_i[m+1]; jj ++) 
            {
               tempx += A_data[jj] * x_data[A_j[jj]];
            }
            y_data[m] = tempx;
         }
         else
         {
            for (j = 0; j < num_vectors; ++ j)
            {
               tempx = y_data[ j*vecstride_y + m*idxstride_y ];
               for (jj = A_i[m]; jj < A_i[m+1]; jj ++) 
               {
                  tempx +=  A_data[jj] * x_data[ j*vecstride_x + A_j[jj]*idxstride_x ];
               }
               y_data[ j*vecstride_y + m*idxstride_y] = tempx;
            }
         }
      }

   }
   else
   {
   
#define JXF_SMP_PRIVATE i,jj,temp
#include "../../../include/jxf_smp_forloop.h"
      for (i = 0; i < num_rows; i ++)
      {
         if (num_vectors == 1)
         {
            temp = y_data[i];
            for (jj = A_i[i]; jj < A_i[i+1]; jj ++)
            {
               temp += A_data[jj] * x_data[A_j[jj]];
            }
            y_data[i] = temp;
         }
         else
         {
            for (j = 0; j < num_vectors; ++ j)
            {
               temp = y_data[ j*vecstride_y + i*idxstride_y ];
               for (jj = A_i[i]; jj < A_i[i+1]; jj ++)
               {
                  temp += A_data[jj] * x_data[ j*vecstride_x + A_j[jj]*idxstride_x ];
               }
               y_data[ j*vecstride_y + i*idxstride_y ] = temp;
            }
         }
      }
   }


  /*----------------------------------------------------
   *  y = alpha*y
   *---------------------------------------------------*/

   if (alpha != 1.0)
   {
#define JXF_SMP_PRIVATE i
#include "../../../include/jxf_smp_forloop.h"
      for (i = 0; i < num_rows*num_vectors; i ++)
      {
	 y_data[i] *= alpha;
      }	 
   }

   return ierr;
}

/*!
 * \fn JXF_Int jxf_CSRMatrixMatvecT
 * \brief Perform y = alpha*A^T*x + beta*y.
 * \date 2011/09/03
 */ 
JXF_Int
jxf_CSRMatrixMatvecT( JXF_Real        alpha,
                     jxf_CSRMatrix *A,
                     jxf_Vector    *x,
                     JXF_Real        beta,
                     jxf_Vector    *y     )
{
   JXF_Real     *A_data    = jxf_CSRMatrixData(A);
   JXF_Int        *A_i       = jxf_CSRMatrixI(A);
   JXF_Int        *A_j       = jxf_CSRMatrixJ(A);
   JXF_Int         num_rows  = jxf_CSRMatrixNumRows(A);
   JXF_Int         num_cols  = jxf_CSRMatrixNumCols(A);

   JXF_Real     *x_data = jxf_VectorData(x);
   JXF_Real     *y_data = jxf_VectorData(y);
   JXF_Int         x_size = jxf_VectorSize(x);
   JXF_Int         y_size = jxf_VectorSize(y);
   JXF_Int         num_vectors = jxf_VectorNumVectors(x);
   JXF_Int         idxstride_y = jxf_VectorIndexStride(y);
   JXF_Int         vecstride_y = jxf_VectorVectorStride(y);
   JXF_Int         idxstride_x = jxf_VectorIndexStride(x);
   JXF_Int         vecstride_x = jxf_VectorVectorStride(x);

   JXF_Real      temp;

   JXF_Int         i, i1, j, jv, jj, ns, ne, size, rest;
   JXF_Int         num_threads;

   JXF_Int         ierr  = 0;

   /*---------------------------------------------------------------------
    *  Check for size compatibility.  MatvecT returns ierr = 1 if
    *  length of X doesn't equal the number of rows of A,
    *  ierr = 2 if the length of Y doesn't equal the number of 
    *  columns of A, and ierr = 3 if both are true.
    *
    *  Because temporary vectors are often used in MatvecT, none of 
    *  these conditions terminates processing, and the ierr flag
    *  is informational only.
    *--------------------------------------------------------------------*/

    jxf_assert( num_vectors == jxf_VectorNumVectors(y) );
 
    if (num_rows != x_size)
              ierr = 1;

    if (num_cols != y_size)
              ierr = 2;

    if (num_rows != x_size && num_cols != y_size)
              ierr = 3;
              
  /*--------------------------------------------------------------
   * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
   *-------------------------------------------------------------*/

   if (alpha == 0.0)
   {
#define JXF_SMP_PRIVATE i
#include "../../../include/jxf_smp_forloop.h"
      for (i = 0; i < num_cols*num_vectors; i ++)
      {
         y_data[i] *= beta;
      }	 

      return ierr;
   }

  /*------------------------------------------------
   * y = (beta/alpha)*y
   *-----------------------------------------------*/

   temp = beta / alpha;
   
   if (temp != 1.0)
   {
      if (temp == 0.0)
      {
#define JXF_SMP_PRIVATE i
#include "../../../include/jxf_smp_forloop.h"
	 for (i = 0; i < num_cols*num_vectors; i ++)
	 {
	    y_data[i] = 0.0;
	 }
      }
      else
      {
#define JXF_SMP_PRIVATE i
#include "../../../include/jxf_smp_forloop.h"
	 for (i = 0; i < num_cols*num_vectors; i ++)
	 {
	    y_data[i] *= temp;
	 }
      }
   }

   /*-----------------------------------------------------------------
    * y += A^T*x
    *-----------------------------------------------------------------*/
   num_threads = jxf_NumThreads();
   if (num_threads > 1)
   {

#define JXF_SMP_PRIVATE i, i1,jj,j,ns,ne,size,rest
#include "../../../include/jxf_smp_forloop.h"
      for (i1 = 0; i1 < num_threads; i1 ++)
      {
         size = num_cols / num_threads;
         rest = num_cols - size*num_threads;
         if (i1 < rest)
         {
            ns = i1*size + i1 - 1;
            ne = (i1 + 1)*size + i1 + 1;
         }
         else
         {
            ns = i1*size + rest - 1;
            ne = (i1 + 1)*size + rest;
         }
         
         if (num_vectors == 1)
         {
            for (i = 0; i < num_rows; i ++)
            {
               for (jj = A_i[i]; jj < A_i[i+1]; jj ++)
               {
                  j = A_j[jj];
                  if (j > ns && j < ne)
                  {
                     y_data[j] += A_data[jj] * x_data[i];
                  }
               }
            }
         }
         else
         {
            for (i = 0; i < num_rows; i ++)
            {
               for (jv = 0; jv < num_vectors; ++ jv)
               {
                  for (jj = A_i[i]; jj < A_i[i+1]; jj ++)
                  {
                     j = A_j[jj];
                     if (j > ns && j < ne)
                     {
                        y_data[ j*idxstride_y + jv*vecstride_y ] +=
                           A_data[jj] * x_data[ i*idxstride_x + jv*vecstride_x ];
                     }
                  }
               }
            }
         }

      }
   }
   else 
   {
      for (i = 0; i < num_rows; i ++)
      {
         if (num_vectors == 1)
         {
            for (jj = A_i[i]; jj < A_i[i+1]; jj ++)
            {
               j = A_j[jj];
               y_data[j] += A_data[jj] * x_data[i];
            }
         }
         else
         {
            for (jv = 0; jv < num_vectors; ++ jv)
            {
               for (jj = A_i[i]; jj < A_i[i+1]; jj ++)
               {
                  j = A_j[jj];
                  y_data[ j*idxstride_y + jv*vecstride_y ] +=
                     A_data[jj] * x_data[ i*idxstride_x + jv*vecstride_x ];
               }
            }
         }
      }
   }
   

  /*--------------------------------------------
   * y = alpha*y
   *-------------------------------------------*/

   if (alpha != 1.0)
   {
#define JXF_SMP_PRIVATE i
#include "../../../include/jxf_smp_forloop.h"
      for (i = 0; i < num_cols*num_vectors; i ++)
      {
         y_data[i] *= alpha;
      }
   }

   return ierr;
}


/*!
 * \fn JXF_Int jxf_CSRMatrixMatvecAgg
 * \brief Perform y = alpha*A*x + beta*y.(nonzeros of A = 1)
 * \date 2025/08/05
 */
JXF_Int
jxf_CSRMatrixMatvecAgg( JXF_Real        alpha,
                    jxf_CSRMatrix *A,
                    jxf_Vector    *x,
                    JXF_Real        beta,
                    jxf_Vector    *y     )
{
//    JXF_Real     *A_data   = jxf_CSRMatrixData(A);
   JXF_Int        *A_i      = jxf_CSRMatrixI(A);
   JXF_Int        *A_j      = jxf_CSRMatrixJ(A);
   JXF_Int         num_rows = jxf_CSRMatrixNumRows(A);
   JXF_Int         num_cols = jxf_CSRMatrixNumCols(A);

   JXF_Int        *A_rownnz = jxf_CSRMatrixRownnz(A);
   JXF_Int         num_rownnz = jxf_CSRMatrixNumRownnz(A);

   JXF_Real     *x_data = jxf_VectorData(x);
   JXF_Real     *y_data = jxf_VectorData(y);
   JXF_Int         x_size = jxf_VectorSize(x);
   JXF_Int         y_size = jxf_VectorSize(y);
   JXF_Int         num_vectors = jxf_VectorNumVectors(x);
   JXF_Int         idxstride_y = jxf_VectorIndexStride(y);
   JXF_Int         vecstride_y = jxf_VectorVectorStride(y);
   JXF_Int         idxstride_x = jxf_VectorIndexStride(x);
   JXF_Int         vecstride_x = jxf_VectorVectorStride(x);

   JXF_Real      temp, tempx;

   JXF_Int         i, j, jj;

   JXF_Int         m;

   JXF_Real      xpar = 0.7;

   JXF_Int         ierr = 0;

   /*---------------------------------------------------------------------
    *  Check for size compatibility.  Matvec returns ierr = 1 if
    *  length of X doesn't equal the number of columns of A,
    *  ierr = 2 if the length of Y doesn't equal the number of rows
    *  of A, and ierr = 3 if both are true.
    *
    *  Because temporary vectors are often used in Matvec, none of 
    *  these conditions terminates processing, and the ierr flag
    *  is informational only.
    *--------------------------------------------------------------------*/
 
    jxf_assert( num_vectors == jxf_VectorNumVectors(y) );

    if (num_cols != x_size)
              ierr = 1;

    if (num_rows != y_size)
              ierr = 2;

    if (num_cols != x_size && num_rows != y_size)
              ierr = 3;

   /*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
    *-----------------------------------------------------------------------*/

    if (alpha == 0.0)
    {
#define JXF_SMP_PRIVATE i
#include "../../../include/jxf_smp_forloop.h"
       for (i = 0; i < num_rows*num_vectors; i ++)
          y_data[i] *= beta;

       return ierr;
    }

  /*--------------------------------------------
   * y = (beta/alpha)*y
   *------------------------------------------*/
   
   temp = beta / alpha;
   
   if (temp != 1.0)
   {
      if (temp == 0.0)
      {
#define JXF_SMP_PRIVATE i
#include "../../../include/jxf_smp_forloop.h"
	 for (i = 0; i < num_rows*num_vectors; i ++)
	    y_data[i] = 0.0;
      }
      else
      {
#define JXF_SMP_PRIVATE i
#include "../../../include/jxf_smp_forloop.h"
	 for (i = 0; i < num_rows*num_vectors; i ++)
	    y_data[i] *= temp;
      }
   }

  /*-----------------------------------------------
   * y += A*x
   *---------------------------------------------*/

   /* use rownnz pointer to do the A*x multiplication  
      when num_rownnz is smaller than num_rows */

   if (num_rownnz < xpar*(num_rows))
   {
   
#define JXF_SMP_PRIVATE i,jj,m,tempx
#include "../../../include/jxf_smp_forloop.h"

      for (i = 0; i < num_rownnz; i ++)
      {
         m = A_rownnz[i];

         if (num_vectors == 1)
         {
            tempx = y_data[m];
            for (jj = A_i[m]; jj < A_i[m+1]; jj ++) 
            {
            //    tempx += A_data[jj] * x_data[A_j[jj]];
               tempx += x_data[A_j[jj]];
            }
            y_data[m] = tempx;
         }
         else
         {
            for (j = 0; j < num_vectors; ++ j)
            {
               tempx = y_data[ j*vecstride_y + m*idxstride_y ];
               for (jj = A_i[m]; jj < A_i[m+1]; jj ++) 
               {
                //   tempx +=  A_data[jj] * x_data[ j*vecstride_x + A_j[jj]*idxstride_x ];
                  tempx +=  x_data[ j*vecstride_x + A_j[jj]*idxstride_x ];
               }
               y_data[ j*vecstride_y + m*idxstride_y] = tempx;
            }
         }
      }

   }
   else
   {
   
#define JXF_SMP_PRIVATE i,jj,temp
#include "../../../include/jxf_smp_forloop.h"
      for (i = 0; i < num_rows; i ++)
      {
         if (num_vectors == 1)
         {
            temp = y_data[i];
            for (jj = A_i[i]; jj < A_i[i+1]; jj ++)
            {
            //    temp += A_data[jj] * x_data[A_j[jj]];
               temp += x_data[A_j[jj]];
            }
            y_data[i] = temp;
         }
         else
         {
            for (j = 0; j < num_vectors; ++ j)
            {
               temp = y_data[ j*vecstride_y + i*idxstride_y ];
               for (jj = A_i[i]; jj < A_i[i+1]; jj ++)
               {
                //   temp += A_data[jj] * x_data[ j*vecstride_x + A_j[jj]*idxstride_x ];
                  temp += x_data[ j*vecstride_x + A_j[jj]*idxstride_x ];
               }
               y_data[ j*vecstride_y + i*idxstride_y ] = temp;
            }
         }
      }
   }


  /*----------------------------------------------------
   *  y = alpha*y
   *---------------------------------------------------*/

   if (alpha != 1.0)
   {
#define JXF_SMP_PRIVATE i
#include "../../../include/jxf_smp_forloop.h"
      for (i = 0; i < num_rows*num_vectors; i ++)
      {
	 y_data[i] *= alpha;
      }	 
   }

   return ierr;
}

/*!
 * \fn JXF_Int jxf_CSRMatrixMatvecTAgg
 * \brief Perform y = alpha*A^T*x + beta*y.(nonzeros of A = 1)
 * \date 2025/08/05
 */ 
JXF_Int
jxf_CSRMatrixMatvecTAgg( JXF_Real        alpha,
                     jxf_CSRMatrix *A,
                     jxf_Vector    *x,
                     JXF_Real        beta,
                     jxf_Vector    *y     )
{
//    JXF_Real     *A_data    = jxf_CSRMatrixData(A);
   JXF_Int        *A_i       = jxf_CSRMatrixI(A);
   JXF_Int        *A_j       = jxf_CSRMatrixJ(A);
   JXF_Int         num_rows  = jxf_CSRMatrixNumRows(A);
   JXF_Int         num_cols  = jxf_CSRMatrixNumCols(A);

   JXF_Real     *x_data = jxf_VectorData(x);
   JXF_Real     *y_data = jxf_VectorData(y);
   JXF_Int         x_size = jxf_VectorSize(x);
   JXF_Int         y_size = jxf_VectorSize(y);
   JXF_Int         num_vectors = jxf_VectorNumVectors(x);
   JXF_Int         idxstride_y = jxf_VectorIndexStride(y);
   JXF_Int         vecstride_y = jxf_VectorVectorStride(y);
   JXF_Int         idxstride_x = jxf_VectorIndexStride(x);
   JXF_Int         vecstride_x = jxf_VectorVectorStride(x);

   JXF_Real      temp;

   JXF_Int         i, i1, j, jv, jj, ns, ne, size, rest;
   JXF_Int         num_threads;

   JXF_Int         ierr  = 0;

   /*---------------------------------------------------------------------
    *  Check for size compatibility.  MatvecT returns ierr = 1 if
    *  length of X doesn't equal the number of rows of A,
    *  ierr = 2 if the length of Y doesn't equal the number of 
    *  columns of A, and ierr = 3 if both are true.
    *
    *  Because temporary vectors are often used in MatvecT, none of 
    *  these conditions terminates processing, and the ierr flag
    *  is informational only.
    *--------------------------------------------------------------------*/

    jxf_assert( num_vectors == jxf_VectorNumVectors(y) );
 
    if (num_rows != x_size)
              ierr = 1;

    if (num_cols != y_size)
              ierr = 2;

    if (num_rows != x_size && num_cols != y_size)
              ierr = 3;
              
  /*--------------------------------------------------------------
   * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
   *-------------------------------------------------------------*/

   if (alpha == 0.0)
   {
#define JXF_SMP_PRIVATE i
#include "../../../include/jxf_smp_forloop.h"
      for (i = 0; i < num_cols*num_vectors; i ++)
      {
         y_data[i] *= beta;
      }	 

      return ierr;
   }

  /*------------------------------------------------
   * y = (beta/alpha)*y
   *-----------------------------------------------*/

   temp = beta / alpha;
   
   if (temp != 1.0)
   {
      if (temp == 0.0)
      {
#define JXF_SMP_PRIVATE i
#include "../../../include/jxf_smp_forloop.h"
	 for (i = 0; i < num_cols*num_vectors; i ++)
	 {
	    y_data[i] = 0.0;
	 }
      }
      else
      {
#define JXF_SMP_PRIVATE i
#include "../../../include/jxf_smp_forloop.h"
	 for (i = 0; i < num_cols*num_vectors; i ++)
	 {
	    y_data[i] *= temp;
	 }
      }
   }

   /*-----------------------------------------------------------------
    * y += A^T*x
    *-----------------------------------------------------------------*/
   num_threads = jxf_NumThreads();
   if (num_threads > 1)
   {

#define JXF_SMP_PRIVATE i, i1,jj,j,ns,ne,size,rest
#include "../../../include/jxf_smp_forloop.h"
      for (i1 = 0; i1 < num_threads; i1 ++)
      {
         size = num_cols / num_threads;
         rest = num_cols - size*num_threads;
         if (i1 < rest)
         {
            ns = i1*size + i1 - 1;
            ne = (i1 + 1)*size + i1 + 1;
         }
         else
         {
            ns = i1*size + rest - 1;
            ne = (i1 + 1)*size + rest;
         }
         
         if (num_vectors == 1)
         {
            for (i = 0; i < num_rows; i ++)
            {
               for (jj = A_i[i]; jj < A_i[i+1]; jj ++)
               {
                  j = A_j[jj];
                  if (j > ns && j < ne)
                  {
                    //  y_data[j] += A_data[jj] * x_data[i];
                     y_data[j] += x_data[i];
                  }
               }
            }
         }
         else
         {
            for (i = 0; i < num_rows; i ++)
            {
               for (jv = 0; jv < num_vectors; ++ jv)
               {
                  for (jj = A_i[i]; jj < A_i[i+1]; jj ++)
                  {
                     j = A_j[jj];
                     if (j > ns && j < ne)
                     {
                        y_data[ j*idxstride_y + jv*vecstride_y ] +=
                        //    A_data[jj] * x_data[ i*idxstride_x + jv*vecstride_x ];
                           x_data[ i*idxstride_x + jv*vecstride_x ];
                     }
                  }
               }
            }
         }

      }
   }
   else 
   {
      for (i = 0; i < num_rows; i ++)
      {
         if (num_vectors == 1)
         {
            for (jj = A_i[i]; jj < A_i[i+1]; jj ++)
            {
               j = A_j[jj];
            //    y_data[j] += A_data[jj] * x_data[i];
               y_data[j] += x_data[i];
            }
         }
         else
         {
            for (jv = 0; jv < num_vectors; ++ jv)
            {
               for (jj = A_i[i]; jj < A_i[i+1]; jj ++)
               {
                  j = A_j[jj];
                  y_data[ j*idxstride_y + jv*vecstride_y ] +=
                    //  A_data[jj] * x_data[ i*idxstride_x + jv*vecstride_x ];
                     x_data[ i*idxstride_x + jv*vecstride_x ];
               }
            }
         }
      }
   }
   

  /*--------------------------------------------
   * y = alpha*y
   *-------------------------------------------*/

   if (alpha != 1.0)
   {
#define JXF_SMP_PRIVATE i
#include "../../../include/jxf_smp_forloop.h"
      for (i = 0; i < num_cols*num_vectors; i ++)
      {
         y_data[i] *= alpha;
      }
   }

   return ierr;
}