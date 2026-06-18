//========================================================================//
//  JXFPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  krylov.c
 *  Date: 2011/09/03
 */ 

#include "jxf_pamg.h"
#include "jxf_krylov.h"

/*!
 * \fn char *jxf_ParKrylovCAlloc
 */ 
char *
jxf_ParKrylovCAlloc( JXF_Int count, JXF_Int elt_size )
{
   return ( jxf_CAlloc( count, elt_size ) );
}

/*!
 * \fn JXF_Int jxf_ParKrylovFree
 */ 
JXF_Int
jxf_ParKrylovFree( char *ptr )
{
   JXF_Int ierr = 0;
   jxf_Free( ptr );
   return ierr;
}

/*!
 * \fn JXF_Int jxf_ParKrylovCommInfo
 */ 
JXF_Int
jxf_ParKrylovCommInfo( void *A, JXF_Int *my_id, JXF_Int *num_procs )
{
   MPI_Comm comm = jxf_ParCSRMatrixComm ((jxf_ParCSRMatrix *) A);
   jxf_MPI_Comm_size(comm, num_procs);
   jxf_MPI_Comm_rank(comm, my_id);
   return 0;
}

/*!
 * \fn void *jxf_ParKrylovCreateVector
 */ 
void *
jxf_ParKrylovCreateVector( void *vvector )
{
   jxf_ParVector *vector = vvector;
   jxf_ParVector *new_vector;

   new_vector = jxf_ParVectorCreate( jxf_ParVectorComm(vector),
				    jxf_ParVectorGlobalSize(vector),	
                                    jxf_ParVectorPartitioning(vector) );
   jxf_ParVectorSetPartitioningOwner(new_vector,0);
   jxf_ParVectorInitialize(new_vector);

   return ( (void *) new_vector );
}

/*!
 * \fn void *jxf_ParKrylovCreateVectorArray
 */ 
void *
jxf_ParKrylovCreateVectorArray( JXF_Int n, void *vvector )
{
   jxf_ParVector  *vector = vvector;
   jxf_ParVector **new_vector;
   JXF_Int i, j;

   new_vector = jxf_CTAlloc(jxf_ParVector*, n);
   for (i = 0; i < n; i ++)
   {
      new_vector[i] = jxf_ParVectorCreate( jxf_ParVectorComm(vector),
				          jxf_ParVectorGlobalSize(vector),	
                                          jxf_ParVectorPartitioning(vector) );
      jxf_ParVectorSetPartitioningOwner(new_vector[i], 0);
      if (jxf_ParVectorInitialize(new_vector[i]) == JXF_ERROR_MEMORY)  // Feng Chunsheng & Yue Xiaoqiang 2012/10/26
      {
         jxf__kdim_memory_error = i - 1;
         jxf_error_flag = 0;
         for (j = 0; j < i; j ++)
         {
            jxf_ParVectorDestroy(new_vector[j]);
         }
         jxf_TFree(new_vector);
         new_vector = NULL;
         break;
      }
   }

   return ( (void *) new_vector );
}

/*!
 * \fn JXF_Int jxf_ParKrylovDestroyVector
 */ 
JXF_Int
jxf_ParKrylovDestroyVector( void *vvector )
{
   jxf_ParVector *vector = vvector;
   return( jxf_ParVectorDestroy( vector ) );
}

/*!
 * \fn void *jxf_ParKrylovMatvecCreate
 */ 
void *
jxf_ParKrylovMatvecCreate( void *A, void *x )
{
   void *matvec_data;
   matvec_data = NULL;
   return ( matvec_data );
}

/*!
 * \fn JXF_Int jxf_ParKrylovMatvec
 */
JXF_Int
jxf_ParKrylovMatvec( void   *matvec_data,
                    JXF_Real  alpha,
                    void   *A,
                    void   *x,
                    JXF_Real  beta,
                    void   *y   )
{
   return ( jxf_ParCSRMatrixMatvec ( alpha,
                                    (jxf_ParCSRMatrix *) A,
                                    (jxf_ParVector *) x,
                                    beta,
                                    (jxf_ParVector *) y ) );
}

/*!
 * \fn JXF_Int jxf_ParKrylovMatvecDestroy
 */
JXF_Int
jxf_ParKrylovMatvecDestroy( void *matvec_data )
{
   return 0;
}

/*!
 * \fn JXF_Real jxf_ParKrylovInnerProd
 */
JXF_Real
jxf_ParKrylovInnerProd( void *x, void *y )
{
   return ( jxf_ParVectorInnerProd( (jxf_ParVector *) x, (jxf_ParVector *) y ) );
}

/*!
 * \fn JXF_Int jxf_ParKrylovMassInnerProd
 */
JXF_Int
jxf_ParKrylovMassInnerProd( void *x, void **y, JXF_Int k, JXF_Int unroll, void  *result )
{
   return ( jxf_ParVectorMassInnerProd( (jxf_ParVector *) x, (jxf_ParVector **) y, k, unroll, (JXF_Real *)result ) );
}

/*!
 * \fn JXF_Int jxf_ParKrylovMassDotpTwo
 */
JXF_Int
jxf_ParKrylovMassDotpTwo( void *x, void *y, void **z, JXF_Int k, JXF_Int unroll, void *result_x, void *result_y )
{
   return ( jxf_ParVectorMassDotpTwo( (jxf_ParVector *) x, (jxf_ParVector *) y, (jxf_ParVector **) z,
              k, unroll, (JXF_Real *) result_x, (JXF_Real *) result_y ) );
}

/*!
 * \fn JXF_Int jxf_ParKrylovCopyVector
 */
JXF_Int
jxf_ParKrylovCopyVector( void *x, void *y )
{
   return ( jxf_ParVectorCopy( (jxf_ParVector *) x, (jxf_ParVector *) y ) );
}

/*!
 * \fn JXF_Int jxf_ParKrylovClearVector
 */
JXF_Int
jxf_ParKrylovClearVector( void *x )
{
   return ( jxf_ParVectorSetConstantValues( (jxf_ParVector *) x, 0.0 ) );
}

/*!
 * \fn JXF_Int jxf_ParKrylovScaleVector
 */
JXF_Int
jxf_ParKrylovScaleVector( JXF_Real alpha, void *x )
{
   return ( jxf_ParVectorScale( alpha, (jxf_ParVector *) x ) );
}

/*!
 * \fn JXF_Int jxf_ParKrylovAxpy
 */
JXF_Int
jxf_ParKrylovAxpy( JXF_Real alpha, void *x, void *y )
{
   return ( jxf_ParVectorAxpy( alpha, (jxf_ParVector *) x, (jxf_ParVector *) y ) );
}

/*!
 * \fn JXF_Int jxf_ParKrylovMassAxpy
 */
JXF_Int
jxf_ParKrylovMassAxpy( JXF_Real *alpha, void **x, void *y, JXF_Int k, JXF_Int unroll )
{
   return ( jxf_ParVectorMassAxpy( alpha, (jxf_ParVector **) x, (jxf_ParVector *) y , k, unroll ) );
}

/*!
 * \fn JXF_Int jxf_ParKrylovIdentitySetup
 */
JXF_Int
jxf_ParKrylovIdentitySetup( void *vdata, void *A )
{
   return 0;
}

/*!
 * \fn JXF_Int jxf_ParKrylovIdentity
 */
JXF_Int
jxf_ParKrylovIdentity( void *vdata, void *A, void *b, void *x )
{
   return( jxf_ParKrylovCopyVector( b, x ) );
}


/* ================================================ */
/* BSR 版本的 Krylov 函数开始                        */
/* ================================================ */

/*!
 * \fn JXF_Int jxf_ParBSRKrylovCommInfo
 */ 
JXF_Int
jxf_ParBSRKrylovCommInfo( void *A, JXF_Int *my_id, JXF_Int *num_procs )
{
   if (A == NULL) {
       *my_id = 0;
       *num_procs = 1;
       return jxf_error_flag;
   }
   
   jxf_ParBSRMatrix *parbsr_A = (jxf_ParBSRMatrix *)A;
   MPI_Comm comm = jxf_ParBSRMatrixComm(parbsr_A);
   jxf_MPI_Comm_size(comm, num_procs);
   jxf_MPI_Comm_rank(comm, my_id);
   return 0;
}

/*!
 * \fn void *jxf_ParBSRKrylovCreateVector
 */ 
void *
jxf_ParBSRKrylovCreateVector( void *vvector )
{
   jxf_ParVector *vector = (jxf_ParVector *)vvector;
   jxf_ParVector *new_vector;

   new_vector = jxf_ParVectorCreate( jxf_ParVectorComm(vector),
                                    jxf_ParVectorGlobalSize(vector),    
                                    jxf_ParVectorPartitioning(vector) );
   if (new_vector) {
       jxf_ParVectorSetPartitioningOwner(new_vector, 0);
       jxf_ParVectorInitialize(new_vector);
   }

   return ( (void *) new_vector );
}

/*!
 * \fn void *jxf_ParBSRKrylovCreateVectorArray
 */ 
void *
jxf_ParBSRKrylovCreateVectorArray( JXF_Int n, void *vvector )
{
   jxf_ParVector  *vector = (jxf_ParVector *)vvector;
   jxf_ParVector **new_vector;
   JXF_Int i, j;

   new_vector = jxf_CTAlloc(jxf_ParVector*, n);
   for (i = 0; i < n; i ++)
   {
      new_vector[i] = jxf_ParVectorCreate( jxf_ParVectorComm(vector),
                                          jxf_ParVectorGlobalSize(vector),    
                                          jxf_ParVectorPartitioning(vector) );
      jxf_ParVectorSetPartitioningOwner(new_vector[i], 0);
      if (jxf_ParVectorInitialize(new_vector[i]) == JXF_ERROR_MEMORY)  // Feng Chunsheng & Yue Xiaoqiang 2012/10/26
      {
         jxf__kdim_memory_error = i - 1;
         jxf_error_flag = 0;
         for (j = 0; j < i; j ++)
         {
            jxf_ParVectorDestroy(new_vector[j]);
         }
         jxf_TFree(new_vector);
         new_vector = NULL;
         break;
      }
   }

   return ( (void *) new_vector );
}

/*!
 * \fn JXF_Int jxf_ParBSRKrylovDestroyVector
 */ 
JXF_Int
jxf_ParBSRKrylovDestroyVector( void *vvector )
{
   jxf_ParVector *vector = (jxf_ParVector *)vvector;
   return( jxf_ParVectorDestroy( vector ) );
}

/*!
 * \fn void *jxf_ParBSRKrylovMatvecCreate
 */ 
void *
jxf_ParBSRKrylovMatvecCreate( void *A, void *x )
{
   // 对于 BSR 矩阵，我们不需要额外的数据结构
   // 直接返回 NULL，因为所有信息都在矩阵 A 中
   return NULL;
}

/*!
 * \fn JXF_Int jxf_ParBSRKrylovMatvec
 */
JXF_Int
jxf_ParBSRKrylovMatvec( void   *matvec_data,
                    JXF_Real  alpha,
                    void   *A,
                    void   *x,
                    JXF_Real  beta,
                    void   *y   )
{
   // 忽略 matvec_data，因为 BSR 矩阵不需要它
   return ( jxf_ParBSRMatrixMatvec ( alpha,
                                    (jxf_ParBSRMatrix *) A,
                                    (jxf_ParVector *) x,
                                    beta,
                                    (jxf_ParVector *) y ) );
}

/*!
 * \fn JXF_Int jxf_ParBSRKrylovMatvecDestroy
 */
JXF_Int
jxf_ParBSRKrylovMatvecDestroy( void *matvec_data )
{
   // 无资源需要释放
   return 0;
}

/*!
 * \fn JXF_Real jxf_ParBSRKrylovInnerProd
 */
JXF_Real
jxf_ParBSRKrylovInnerProd( void *x, void *y )
{
   return ( jxf_ParVectorInnerProd( (jxf_ParVector *) x, (jxf_ParVector *) y ) );
}

/*!
 * \fn JXF_Int jxf_ParBSRKrylovMassInnerProd
 */
JXF_Int
jxf_ParBSRKrylovMassInnerProd( void *x, void **y, JXF_Int k, JXF_Int unroll, void  *result )
{
   // 这个函数可能需要实现，但目前先返回简单版本
   // 注意：这里需要根据 BSR 特性进行调整
   JXF_Real *res = (JXF_Real *)result;
   for (JXF_Int i = 0; i < k; i++) {
       res[i] = jxf_ParBSRKrylovInnerProd(x, y[i]);
   }
   return 0;
}

/*!
 * \fn JXF_Int jxf_ParBSRKrylovMassDotpTwo
 */
JXF_Int
jxf_ParBSRKrylovMassDotpTwo( void *x, void *y, void **z, JXF_Int k, JXF_Int unroll, void *result_x, void *result_y )
{
   // 简化实现
   JXF_Real *res_x = (JXF_Real *)result_x;
   JXF_Real *res_y = (JXF_Real *)result_y;
   for (JXF_Int i = 0; i < k; i++) {
       res_x[i] = jxf_ParBSRKrylovInnerProd(x, z[i]);
       res_y[i] = jxf_ParBSRKrylovInnerProd(y, z[i]);
   }
   return 0;
}

/*!
 * \fn JXF_Int jxf_ParBSRKrylovCopyVector
 */
JXF_Int
jxf_ParBSRKrylovCopyVector( void *x, void *y )
{
   return ( jxf_ParVectorCopy( (jxf_ParVector *) x, (jxf_ParVector *) y ) );
}

/*!
 * \fn JXF_Int jxf_ParBSRKrylovClearVector
 */
JXF_Int
jxf_ParBSRKrylovClearVector( void *x )
{
   return ( jxf_ParVectorSetConstantValues( (jxf_ParVector *) x, 0.0 ) );
}

/*!
 * \fn JXF_Int jxf_ParBSRKrylovScaleVector
 */
JXF_Int
jxf_ParBSRKrylovScaleVector( JXF_Real alpha, void *x )
{
   return ( jxf_ParVectorScale( alpha, (jxf_ParVector *) x ) );
}

/*!
 * \fn JXF_Int jxf_ParBSRKrylovAxpy
 */
JXF_Int
jxf_ParBSRKrylovAxpy( JXF_Real alpha, void *x, void *y )
{
   return ( jxf_ParVectorAxpy( alpha, (jxf_ParVector *) x, (jxf_ParVector *) y ) );
}

/*!
 * \fn JXF_Int jxf_ParBSRKrylovMassAxpy
 */
JXF_Int
jxf_ParBSRKrylovMassAxpy( JXF_Real *alpha, void **x, void *y, JXF_Int k, JXF_Int unroll )
{
   // 简化实现
   for (JXF_Int i = 0; i < k; i++) {
       jxf_ParBSRKrylovAxpy(alpha[i], x[i], y);
   }
   return 0;
}

/*!
 * \fn JXF_Int jxf_ParBSRKrylovIdentitySetup
 */
JXF_Int
jxf_ParBSRKrylovIdentitySetup( void *vdata, void *A )
{
   return 0;
}

/*!
 * \fn JXF_Int jxf_ParBSRKrylovIdentity
 */
JXF_Int
jxf_ParBSRKrylovIdentity( void *vdata, void *A, void *b, void *x )
{
   return( jxf_ParBSRKrylovCopyVector( b, x ) );
}