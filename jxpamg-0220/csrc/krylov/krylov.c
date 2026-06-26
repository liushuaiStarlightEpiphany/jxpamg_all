//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  krylov.c
 *  Date: 2011/09/03
 */ 

#include "jx_pamg.h"
#include "jx_krylov.h"

/*!
 * \fn char *jx_ParKrylovCAlloc
 */ 
char *
jx_ParKrylovCAlloc( JX_Int count, JX_Int elt_size )
{
   return ( jx_CAlloc( count, elt_size ) );
}

/*!
 * \fn JX_Int jx_ParKrylovFree
 */ 
JX_Int
jx_ParKrylovFree( char *ptr )
{
   JX_Int ierr = 0;
   jx_Free( ptr );
   return ierr;
}

/*!
 * \fn JX_Int jx_ParKrylovCommInfo
 */ 
JX_Int
jx_ParKrylovCommInfo( void *A, JX_Int *my_id, JX_Int *num_procs )
{
   MPI_Comm comm = jx_ParCSRMatrixComm ((jx_ParCSRMatrix *) A);
   jx_MPI_Comm_size(comm, num_procs);
   jx_MPI_Comm_rank(comm, my_id);
   return 0;
}

/*!
 * \fn void *jx_ParKrylovCreateVector
 */ 
void *
jx_ParKrylovCreateVector( void *vvector )
{
   jx_ParVector *vector = vvector;
   jx_ParVector *new_vector;

   new_vector = jx_ParVectorCreate( jx_ParVectorComm(vector),
				    jx_ParVectorGlobalSize(vector),	
                                    jx_ParVectorPartitioning(vector) );
   jx_ParVectorSetPartitioningOwner(new_vector,0);
   jx_ParVectorInitialize(new_vector);

   return ( (void *) new_vector );
}

/*!
 * \fn void *jx_ParKrylovCreateVectorArray
 */ 
void *
jx_ParKrylovCreateVectorArray( JX_Int n, void *vvector )
{
   jx_ParVector  *vector = vvector;
   jx_ParVector **new_vector;
   JX_Int i, j;

   new_vector = jx_CTAlloc(jx_ParVector*, n);
   for (i = 0; i < n; i ++)
   {
      new_vector[i] = jx_ParVectorCreate( jx_ParVectorComm(vector),
				          jx_ParVectorGlobalSize(vector),	
                                          jx_ParVectorPartitioning(vector) );
      jx_ParVectorSetPartitioningOwner(new_vector[i], 0);
      if (jx_ParVectorInitialize(new_vector[i]) == JX_ERROR_MEMORY)  // Feng Chunsheng & Yue Xiaoqiang 2012/10/26
      {
         jx__kdim_memory_error = i - 1;
         jx_error_flag = 0;
         for (j = 0; j < i; j ++)
         {
            jx_ParVectorDestroy(new_vector[j]);
         }
         jx_TFree(new_vector);
         new_vector = NULL;
         break;
      }
   }

   return ( (void *) new_vector );
}

/*!
 * \fn JX_Int jx_ParKrylovDestroyVector
 */ 
JX_Int
jx_ParKrylovDestroyVector( void *vvector )
{
   jx_ParVector *vector = vvector;
   return( jx_ParVectorDestroy( vector ) );
}

/*!
 * \fn void *jx_ParKrylovMatvecCreate
 */ 
void *
jx_ParKrylovMatvecCreate( void *A, void *x )
{
   void *matvec_data;
   matvec_data = NULL;
   return ( matvec_data );
}

/*!
 * \fn JX_Int jx_ParKrylovMatvec
 */
JX_Int
jx_ParKrylovMatvec( void   *matvec_data,
                    JX_Real  alpha,
                    void   *A,
                    void   *x,
                    JX_Real  beta,
                    void   *y   )
{
   return ( jx_ParCSRMatrixMatvec ( alpha,
                                    (jx_ParCSRMatrix *) A,
                                    (jx_ParVector *) x,
                                    beta,
                                    (jx_ParVector *) y ) );
}

/*!
 * \fn JX_Int jx_ParKrylovMatvecDestroy
 */
JX_Int
jx_ParKrylovMatvecDestroy( void *matvec_data )
{
   return 0;
}

/*!
 * \fn JX_Real jx_ParKrylovInnerProd
 */
JX_Real
jx_ParKrylovInnerProd( void *x, void *y )
{
   return ( jx_ParVectorInnerProd( (jx_ParVector *) x, (jx_ParVector *) y ) );
}

/*!
 * \fn JX_Int jx_ParKrylovMassInnerProd
 */
JX_Int
jx_ParKrylovMassInnerProd( void *x, void **y, JX_Int k, JX_Int unroll, void  *result )
{
   return ( jx_ParVectorMassInnerProd( (jx_ParVector *) x, (jx_ParVector **) y, k, unroll, (JX_Real *)result ) );
}

/*!
 * \fn JX_Int jx_ParKrylovMassDotpTwo
 */
JX_Int
jx_ParKrylovMassDotpTwo( void *x, void *y, void **z, JX_Int k, JX_Int unroll, void *result_x, void *result_y )
{
   return ( jx_ParVectorMassDotpTwo( (jx_ParVector *) x, (jx_ParVector *) y, (jx_ParVector **) z,
              k, unroll, (JX_Real *) result_x, (JX_Real *) result_y ) );
}

/*!
 * \fn JX_Int jx_ParKrylovCopyVector
 */
JX_Int
jx_ParKrylovCopyVector( void *x, void *y )
{
   return ( jx_ParVectorCopy( (jx_ParVector *) x, (jx_ParVector *) y ) );
}

/*!
 * \fn JX_Int jx_ParKrylovClearVector
 */
JX_Int
jx_ParKrylovClearVector( void *x )
{
   return ( jx_ParVectorSetConstantValues( (jx_ParVector *) x, 0.0 ) );
}

/*!
 * \fn JX_Int jx_ParKrylovScaleVector
 */
JX_Int
jx_ParKrylovScaleVector( JX_Real alpha, void *x )
{
   return ( jx_ParVectorScale( alpha, (jx_ParVector *) x ) );
}

/*!
 * \fn JX_Int jx_ParKrylovAxpy
 */
JX_Int
jx_ParKrylovAxpy( JX_Real alpha, void *x, void *y )
{
   return ( jx_ParVectorAxpy( alpha, (jx_ParVector *) x, (jx_ParVector *) y ) );
}

/*!
 * \fn JX_Int jx_ParKrylovMassAxpy
 */
JX_Int
jx_ParKrylovMassAxpy( JX_Real *alpha, void **x, void *y, JX_Int k, JX_Int unroll )
{
   return ( jx_ParVectorMassAxpy( alpha, (jx_ParVector **) x, (jx_ParVector *) y , k, unroll ) );
}

/*!
 * \fn JX_Int jx_ParKrylovIdentitySetup
 */
JX_Int
jx_ParKrylovIdentitySetup( void *vdata, void *A )
{
   return 0;
}

/*!
 * \fn JX_Int jx_ParKrylovIdentity
 */
JX_Int
jx_ParKrylovIdentity( void *vdata, void *A, void *b, void *x )
{
   return( jx_ParKrylovCopyVector( b, x ) );
}


/* ================================================ */
/* BSR 版本的 Krylov 函数开始                        */
/* ================================================ */

/*!
 * \fn JX_Int jx_ParBSRKrylovCommInfo
 */ 
JX_Int
jx_ParBSRKrylovCommInfo( void *A, JX_Int *my_id, JX_Int *num_procs )
{
   if (A == NULL) {
       *my_id = 0;
       *num_procs = 1;
       return jx_error_flag;
   }
   
   jx_ParBSRMatrix *parbsr_A = (jx_ParBSRMatrix *)A;
   MPI_Comm comm = jx_ParBSRMatrixComm(parbsr_A);
   jx_MPI_Comm_size(comm, num_procs);
   jx_MPI_Comm_rank(comm, my_id);
   return 0;
}

/*!
 * \fn void *jx_ParBSRKrylovCreateVector
 */ 
void *
jx_ParBSRKrylovCreateVector( void *vvector )
{
   jx_ParVector *vector = (jx_ParVector *)vvector;
   jx_ParVector *new_vector;

   new_vector = jx_ParVectorCreate( jx_ParVectorComm(vector),
                                    jx_ParVectorGlobalSize(vector),    
                                    jx_ParVectorPartitioning(vector) );
   if (new_vector) {
       jx_ParVectorSetPartitioningOwner(new_vector, 0);
       jx_ParVectorInitialize(new_vector);
   }

   return ( (void *) new_vector );
}

/*!
 * \fn void *jx_ParBSRKrylovCreateVectorArray
 */ 
void *
jx_ParBSRKrylovCreateVectorArray( JX_Int n, void *vvector )
{
   jx_ParVector  *vector = (jx_ParVector *)vvector;
   jx_ParVector **new_vector;
   JX_Int i, j;

   new_vector = jx_CTAlloc(jx_ParVector*, n);
   for (i = 0; i < n; i ++)
   {
      new_vector[i] = jx_ParVectorCreate( jx_ParVectorComm(vector),
                                          jx_ParVectorGlobalSize(vector),    
                                          jx_ParVectorPartitioning(vector) );
      jx_ParVectorSetPartitioningOwner(new_vector[i], 0);
      if (jx_ParVectorInitialize(new_vector[i]) == JX_ERROR_MEMORY)  // Feng Chunsheng & Yue Xiaoqiang 2012/10/26
      {
         jx__kdim_memory_error = i - 1;
         jx_error_flag = 0;
         for (j = 0; j < i; j ++)
         {
            jx_ParVectorDestroy(new_vector[j]);
         }
         jx_TFree(new_vector);
         new_vector = NULL;
         break;
      }
   }

   return ( (void *) new_vector );
}

/*!
 * \fn JX_Int jx_ParBSRKrylovDestroyVector
 */ 
JX_Int
jx_ParBSRKrylovDestroyVector( void *vvector )
{
   jx_ParVector *vector = (jx_ParVector *)vvector;
   return( jx_ParVectorDestroy( vector ) );
}

/*!
 * \fn void *jx_ParBSRKrylovMatvecCreate
 */ 
void *
jx_ParBSRKrylovMatvecCreate( void *A, void *x )
{
   // 对于 BSR 矩阵，我们不需要额外的数据结构
   // 直接返回 NULL，因为所有信息都在矩阵 A 中
   return NULL;
}

/*!
 * \fn JX_Int jx_ParBSRKrylovMatvec
 */
JX_Int
jx_ParBSRKrylovMatvec( void   *matvec_data,
                    JX_Real  alpha,
                    void   *A,
                    void   *x,
                    JX_Real  beta,
                    void   *y   )
{
   // 忽略 matvec_data，因为 BSR 矩阵不需要它
   return ( jx_ParBSRMatrixMatvec ( alpha,
                                    (jx_ParBSRMatrix *) A,
                                    (jx_ParVector *) x,
                                    beta,
                                    (jx_ParVector *) y ) );
}

/*!
 * \fn JX_Int jx_ParBSRKrylovMatvecDestroy
 */
JX_Int
jx_ParBSRKrylovMatvecDestroy( void *matvec_data )
{
   // 无资源需要释放
   return 0;
}

/*!
 * \fn JX_Real jx_ParBSRKrylovInnerProd
 */
JX_Real
jx_ParBSRKrylovInnerProd( void *x, void *y )
{
   return ( jx_ParVectorInnerProd( (jx_ParVector *) x, (jx_ParVector *) y ) );
}

/*!
 * \fn JX_Int jx_ParBSRKrylovMassInnerProd
 */
JX_Int
jx_ParBSRKrylovMassInnerProd( void *x, void **y, JX_Int k, JX_Int unroll, void  *result )
{
   // 这个函数可能需要实现，但目前先返回简单版本
   // 注意：这里需要根据 BSR 特性进行调整
   JX_Real *res = (JX_Real *)result;
   for (JX_Int i = 0; i < k; i++) {
       res[i] = jx_ParBSRKrylovInnerProd(x, y[i]);
   }
   return 0;
}

/*!
 * \fn JX_Int jx_ParBSRKrylovMassDotpTwo
 */
JX_Int
jx_ParBSRKrylovMassDotpTwo( void *x, void *y, void **z, JX_Int k, JX_Int unroll, void *result_x, void *result_y )
{
   // 简化实现
   JX_Real *res_x = (JX_Real *)result_x;
   JX_Real *res_y = (JX_Real *)result_y;
   for (JX_Int i = 0; i < k; i++) {
       res_x[i] = jx_ParBSRKrylovInnerProd(x, z[i]);
       res_y[i] = jx_ParBSRKrylovInnerProd(y, z[i]);
   }
   return 0;
}

/*!
 * \fn JX_Int jx_ParBSRKrylovCopyVector
 */
JX_Int
jx_ParBSRKrylovCopyVector( void *x, void *y )
{
   return ( jx_ParVectorCopy( (jx_ParVector *) x, (jx_ParVector *) y ) );
}

/*!
 * \fn JX_Int jx_ParBSRKrylovClearVector
 */
JX_Int
jx_ParBSRKrylovClearVector( void *x )
{
   return ( jx_ParVectorSetConstantValues( (jx_ParVector *) x, 0.0 ) );
}

/*!
 * \fn JX_Int jx_ParBSRKrylovScaleVector
 */
JX_Int
jx_ParBSRKrylovScaleVector( JX_Real alpha, void *x )
{
   return ( jx_ParVectorScale( alpha, (jx_ParVector *) x ) );
}

/*!
 * \fn JX_Int jx_ParBSRKrylovAxpy
 */
JX_Int
jx_ParBSRKrylovAxpy( JX_Real alpha, void *x, void *y )
{
   return ( jx_ParVectorAxpy( alpha, (jx_ParVector *) x, (jx_ParVector *) y ) );
}

/*!
 * \fn JX_Int jx_ParBSRKrylovMassAxpy
 */
JX_Int
jx_ParBSRKrylovMassAxpy( JX_Real *alpha, void **x, void *y, JX_Int k, JX_Int unroll )
{
   // 简化实现
   for (JX_Int i = 0; i < k; i++) {
       jx_ParBSRKrylovAxpy(alpha[i], x[i], y);
   }
   return 0;
}

/*!
 * \fn JX_Int jx_ParBSRKrylovIdentitySetup
 */
JX_Int
jx_ParBSRKrylovIdentitySetup( void *vdata, void *A )
{
   return 0;
}

/*!
 * \fn JX_Int jx_ParBSRKrylovIdentity
 */
JX_Int
jx_ParBSRKrylovIdentity( void *vdata, void *A, void *b, void *x )
{
   return( jx_ParBSRKrylovCopyVector( b, x ) );
}