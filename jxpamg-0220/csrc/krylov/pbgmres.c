//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2024        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//
// pbgmres.c - BSR版本的GMRES实现
#include "jx_combined.h"
#include "jx_krylov.h"

/*!
 * \fn JX_Int JX_ParBSRGMRESCreate
 * \brief 创建BSR矩阵的GMRES求解器
 */ 
JX_Int
JX_ParBSRGMRESCreate( MPI_Comm comm, JX_Solver *solver )
{
   jx_GMRESFunctions * gmres_functions =
      jx_GMRESFunctionsCreate(
         jx_CAlloc, 
         jx_ParKrylovFree, 
         jx_ParBSRKrylovCommInfo,
         jx_ParBSRKrylovCreateVector,
         jx_ParBSRKrylovCreateVectorArray,
         jx_ParBSRKrylovDestroyVector, 
         jx_ParBSRKrylovMatvecCreate,
         jx_ParBSRKrylovMatvec, 
         jx_ParBSRKrylovMatvecDestroy,
         jx_ParBSRKrylovInnerProd, 
         jx_ParBSRKrylovCopyVector,
         jx_ParBSRKrylovClearVector,
         jx_ParBSRKrylovScaleVector, 
         jx_ParBSRKrylovAxpy,
         jx_ParBSRKrylovIdentitySetup, 
         jx_ParBSRKrylovIdentity );

   *solver = ( (JX_Solver) jx_GMRESCreate( gmres_functions ) );
   
   if (!solver) 
   {
      jx_error_in_arg(2);
   }

   return jx_error_flag;
}

/*!
 * \fn JX_Int JX_ParBSRGMRESDestroy
 */ 
JX_Int 
JX_ParBSRGMRESDestroy( JX_Solver solver )
{
   return( jx_GMRESDestroy( (void *) solver ) );
}

/*!
 * \fn void *jx_ParBSRGMRESCreate
 */
void *
jx_ParBSRGMRESCreate( MPI_Comm comm )
{
   jx_GMRESData *solver = NULL;
   
   jx_GMRESFunctions * gmres_functions =
      jx_GMRESFunctionsCreate(
         jx_CAlloc, 
         jx_ParKrylovFree, 
         jx_ParBSRKrylovCommInfo,
         jx_ParBSRKrylovCreateVector,
         jx_ParBSRKrylovCreateVectorArray,
         jx_ParBSRKrylovDestroyVector, 
         jx_ParBSRKrylovMatvecCreate,
         jx_ParBSRKrylovMatvec, 
         jx_ParBSRKrylovMatvecDestroy,
         jx_ParBSRKrylovInnerProd, 
         jx_ParBSRKrylovCopyVector,
         jx_ParBSRKrylovClearVector,
         jx_ParBSRKrylovScaleVector, 
         jx_ParBSRKrylovAxpy,
         jx_ParBSRKrylovIdentitySetup, 
         jx_ParBSRKrylovIdentity );

   solver = ( (jx_GMRESData *)jx_GMRESCreate( gmres_functions ) );
   
   if (!solver) 
   {
      jx_error_in_arg(2);
   }

   return solver;
}

/*!
 * \fn JX_Int jx_ParBSRGMRESSolve
 * \brief BSR版本的GMRES求解函数
 */
JX_Int
jx_ParBSRGMRESSolve( void *gmres_vdata, void *preOperator, void *matA, 
                    void *vecB, void *vecX )
{
   // 直接调用通用的GMRES求解函数
   return jx_GMRESSolve(gmres_vdata, preOperator, matA, vecB, vecX);
}

/*!
 * \fn JX_Int JX_ParBSRGMRESSetup
 * \brief BSR版本的GMRES设置函数
 */
JX_Int 
JX_ParBSRGMRESSetup( JX_Solver solver, JX_Matrix matA, JX_Vector vecB, JX_Vector vecX )
{
   return( jx_GMRESSetup( solver, matA, vecB, vecX ) );
}

/*!
 * \fn JX_Int JX_ParBSRGMRESSolve
 */
JX_Int 
JX_ParBSRGMRESSolve( JX_Solver solver, JX_Matrix preOperator, JX_Matrix matA, 
                     JX_Vector vecB, JX_Vector vecX )
{
   return( jx_GMRESSolve( solver, preOperator, matA, vecB, vecX ) );
}