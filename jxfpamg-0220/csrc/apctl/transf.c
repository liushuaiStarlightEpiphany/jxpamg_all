//========================================================================//
//  JXFPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  transf.c
 *  Date: 2012/03/05
 *
 *  Created by peghoty
 *
 */ 

#include "jxf_pamg.h"
#include "jxf_apctl.h"

/*!
 * \fn JXF_Int jxf_APCTLKrylovSolBack4Jasmin
 * \brief 收集解向量函数. 
 * \note 三个进程组的根进程各自转换一个并行向量. 效率比 jxf_APCTLKrylovSolBack4Jasmin2 要高. 
 * \author peghoty 
 * \date 2012/02/27
 */
JXF_Int 
jxf_APCTLKrylovSolBack4Jasmin( MPI_Comm      comm,
                              MPI_Comm      comm_x, 
                              JXF_Int           groupid_x, 
                              jxf_ParVector *par_sol, 
                              jxf_ParVector *uR_p, 
                              jxf_ParVector *uE_p, 
                              jxf_ParVector *uI_p )
{
   JXF_Int myid, nprocs;
   JXF_Int rootid_R, rootid_E, rootid_I;
   JXF_Int N = jxf_ParVectorGlobalSize(par_sol);  
   JXF_Int n = N / 3;
   JXF_Int npeachgroup;
  
   jxf_ParVector *tmpR = NULL;
   jxf_ParVector *tmpE = NULL;
   jxf_ParVector *tmpI = NULL;
   
   jxf_Vector *tmpR_s = NULL;
   jxf_Vector *tmpE_s = NULL;
   jxf_Vector *tmpI_s = NULL;   
   
   JXF_Int *partition = jxf_ParVectorPartitioning(par_sol);
   jxf_Vector *locvec = jxf_ParVectorLocalVector(par_sol);
   JXF_Real temp_adrress = 0.0;
 
   jxf_MPI_Comm_rank(comm, &myid);
   jxf_MPI_Comm_size(comm, &nprocs);

   npeachgroup = nprocs / 3;

   // 三个进程组的根进程号
   rootid_R = 0; 
   rootid_E = npeachgroup;
   rootid_I = 2*npeachgroup;
   
   if (groupid_x == 0)
   {
      // 将光子进程组上的解向量收集为 rootid_R 号进程上的一个串行向量 tmpR_s
      tmpR = jxf_ParVectorCreate(comm_x, n, partition);
      tmpR->local_vector->data = &temp_adrress;
      jxf_ParVectorInitialize(tmpR);
      jxf_ParVectorSetDataOwner(tmpR, 0);
      jxf_ParVectorSetPartitioningOwner(tmpR, 0);
      tmpR->local_vector->data = locvec->data;
      
      tmpR_s = jxf_ParVectorToVectorAll(tmpR);
   }   
   else if (groupid_x == 1)
   {
      // 将电子进程组上的解向量收集为 rootid_E 号进程上的一个串行向量 tmpE_s
      tmpE = jxf_ParVectorCreate(comm_x, n, partition);
      tmpE->local_vector->data = &temp_adrress;
      jxf_ParVectorInitialize(tmpE);
      jxf_ParVectorSetDataOwner(tmpE, 0);
      jxf_ParVectorSetPartitioningOwner(tmpE, 0);
      tmpE->local_vector->data = locvec->data;
      
      tmpE_s = jxf_ParVectorToVectorAll(tmpE);
   }
   else if (groupid_x == 2)
   {
      // 将离子进程组上的解向量收集为 rootid_I 号进程上的一个串行向量 tmpI_s
      tmpI = jxf_ParVectorCreate(comm_x, n, partition);
      tmpI->local_vector->data = &temp_adrress;
      jxf_ParVectorInitialize(tmpI);
      jxf_ParVectorSetDataOwner(tmpI, 0);
      jxf_ParVectorSetPartitioningOwner(tmpI, 0);
      tmpI->local_vector->data = locvec->data;
      
      tmpI_s = jxf_ParVectorToVectorAll(tmpI);
   }

   // 将三个串行向量转化为并行向量
   jxf_VectorToParVector_Allocated_FromGivenPro(comm, rootid_R, tmpR_s, jxf_ParVectorPartitioning(uR_p), uR_p); 
   jxf_VectorToParVector_Allocated_FromGivenPro(comm, rootid_E, tmpE_s, jxf_ParVectorPartitioning(uE_p), uE_p);
   jxf_VectorToParVector_Allocated_FromGivenPro(comm, rootid_I, tmpI_s, jxf_ParVectorPartitioning(uI_p), uI_p);
   
   // 释放三个串行向量的内存
   if (groupid_x == 0)
   {
      jxf_SeqVectorDestroy(tmpR_s);
      jxf_TFree(jxf_ParVectorLocalVector(tmpR));
      jxf_ParVectorDestroy(tmpR);      
   }
   else if (groupid_x == 1)
   {
      jxf_SeqVectorDestroy(tmpE_s);
      jxf_TFree(jxf_ParVectorLocalVector(tmpE));
      jxf_ParVectorDestroy(tmpE);      
   }
   else if (groupid_x == 2)
   {
      jxf_SeqVectorDestroy(tmpI_s);
      jxf_TFree(jxf_ParVectorLocalVector(tmpI));
      jxf_ParVectorDestroy(tmpI);      
   }

   return (0);
}

JXF_Int 
jxf_APCTLKrylovSolBack4mgJasmin( MPI_Comm      comm,
                                MPI_Comm      comm_x,
                                JXF_Int           groupid_x,
                                JXF_Int           ng,
                                jxf_ParVector  *par_sol,
                                jxf_ParVector **uR_p,
                                jxf_ParVector  *uE_p,
                                jxf_ParVector  *uI_p )
{
   JXF_Int myid, nprocs;
   JXF_Int N = jxf_ParVectorGlobalSize(par_sol);
   JXF_Int n = N / (ng + 2);
   JXF_Int npeachgroup, gidx;

   jxf_ParVector *tmpR = NULL;
   jxf_ParVector *tmpE = NULL;
   jxf_ParVector *tmpI = NULL;

   jxf_Vector *tmpR_s = NULL;
   jxf_Vector *tmpE_s = NULL;
   jxf_Vector *tmpI_s = NULL;
 
   JXF_Int *partition = jxf_ParVectorPartitioning(par_sol);
   jxf_Vector *locvec = jxf_ParVectorLocalVector(par_sol);
   JXF_Real temp_adrress = 0.0;
 
   jxf_MPI_Comm_rank(comm, &myid);
   jxf_MPI_Comm_size(comm, &nprocs);

   npeachgroup = nprocs / (ng + 2);

   if (groupid_x < ng)
   {
      tmpR = jxf_ParVectorCreate(comm_x, n, partition);
      tmpR->local_vector->data = &temp_adrress;
      jxf_ParVectorInitialize(tmpR);
      jxf_ParVectorSetDataOwner(tmpR, 0);
      jxf_ParVectorSetPartitioningOwner(tmpR, 0);
      tmpR->local_vector->data = locvec->data;
      tmpR_s = jxf_ParVectorToVectorAll(tmpR);
   }
   else if (groupid_x == ng)
   {
      tmpE = jxf_ParVectorCreate(comm_x, n, partition);
      tmpE->local_vector->data = &temp_adrress;
      jxf_ParVectorInitialize(tmpE);
      jxf_ParVectorSetDataOwner(tmpE, 0);
      jxf_ParVectorSetPartitioningOwner(tmpE, 0);
      tmpE->local_vector->data = locvec->data;
      tmpE_s = jxf_ParVectorToVectorAll(tmpE);
   }
   else if (groupid_x == ng+1)
   {
      tmpI = jxf_ParVectorCreate(comm_x, n, partition);
      tmpI->local_vector->data = &temp_adrress;
      jxf_ParVectorInitialize(tmpI);
      jxf_ParVectorSetDataOwner(tmpI, 0);
      jxf_ParVectorSetPartitioningOwner(tmpI, 0);
      tmpI->local_vector->data = locvec->data;
      tmpI_s = jxf_ParVectorToVectorAll(tmpI);
   }

   // 将三个串行向量转化为并行向量
   for (gidx = 0; gidx < ng; gidx ++)
   {
      jxf_VectorToParVector_Allocated_FromGivenPro(comm, gidx*npeachgroup, tmpR_s, jxf_ParVectorPartitioning(uR_p[gidx]), uR_p[gidx]);
   }
   jxf_VectorToParVector_Allocated_FromGivenPro(comm, ng*npeachgroup, tmpE_s, jxf_ParVectorPartitioning(uE_p), uE_p);
   jxf_VectorToParVector_Allocated_FromGivenPro(comm, (ng+1)*npeachgroup, tmpI_s, jxf_ParVectorPartitioning(uI_p), uI_p);
 
   // 释放三个串行向量的内存
   if (groupid_x < ng)
   {
      jxf_SeqVectorDestroy(tmpR_s);
      jxf_TFree(jxf_ParVectorLocalVector(tmpR));
      jxf_ParVectorDestroy(tmpR);
   }
   else if (groupid_x == ng)
   {
      jxf_SeqVectorDestroy(tmpE_s);
      jxf_TFree(jxf_ParVectorLocalVector(tmpE));
      jxf_ParVectorDestroy(tmpE);
   }
   else if (groupid_x == ng+1)
   {
      jxf_SeqVectorDestroy(tmpI_s);
      jxf_TFree(jxf_ParVectorLocalVector(tmpI));
      jxf_ParVectorDestroy(tmpI);
   }

   return (0);
}

/*!
 * \fn JXF_Int jxf_APCTLKrylovSolBack4Jasmin2
 * \brief 收集解向量函数. 
 * \note 从 0 号进程分别转换三个并行向量. 效率不如 jxf_APCTLKrylovSolBack4Jasmin 高.
 * \author peghoty 
 * \date 2012/02/27
 */
JXF_Int 
jxf_APCTLKrylovSolBack4Jasmin2( MPI_Comm      comm,
                               jxf_ParVector *par_sol, 
                               jxf_ParVector *uR_p, 
                               jxf_ParVector *uE_p, 
                               jxf_ParVector *uI_p )
{
   JXF_Int N = jxf_ParVectorGlobalSize(par_sol);  
   JXF_Int n = N / 3;
   
   JXF_Real temp_adrress = 0.0;
   
   jxf_Vector *tmpR_s = NULL;
   jxf_Vector *tmpE_s = NULL;
   jxf_Vector *tmpI_s = NULL;  
   
   jxf_Vector *sol = NULL;
   
   sol = jxf_ParVectorToVectorAll(par_sol);
   
   tmpR_s = jxf_SeqVectorCreate(n);
   tmpR_s->data = &temp_adrress;
   jxf_SeqVectorInitialize(tmpR_s);
   tmpR_s->data = sol->data;
   
   tmpE_s = jxf_SeqVectorCreate(n);
   tmpE_s->data = &temp_adrress;
   jxf_SeqVectorInitialize(tmpE_s);
   tmpE_s->data = sol->data + n;
   
   tmpI_s = jxf_SeqVectorCreate(n);
   tmpI_s->data = &temp_adrress;
   jxf_SeqVectorInitialize(tmpI_s);
   tmpI_s->data = sol->data + 2*n;
   
   jxf_VectorToParVector_Allocated2(comm, tmpR_s, jxf_ParVectorPartitioning(uR_p), uR_p); 
   jxf_VectorToParVector_Allocated2(comm, tmpE_s, jxf_ParVectorPartitioning(uE_p), uE_p);
   jxf_VectorToParVector_Allocated2(comm, tmpI_s, jxf_ParVectorPartitioning(uI_p), uI_p);

   jxf_TFree(tmpR_s);
   jxf_TFree(tmpE_s);
   jxf_TFree(tmpI_s);
   jxf_SeqVectorDestroy(sol);
   
   return (0);
} 

/*!
 * \fn JXF_Int jxf_MatVecGroupR
 * \brief 利用并行矩阵 A(G) 生成 ARR(GR), VRE(GR).
 * \note 要求进程数为 3 的倍数，且分划为自己指定的匹配分划.
 * \date 2012/02/23
 */ 
JXF_Int 
jxf_MatVecGroupR( MPI_Comm           comm_x,
                 jxf_ParCSRMatrix   *A, 
                 jxf_ParCSRMatrix  **ARR_ptr, 
                 jxf_ParVector     **VRE_ptr )
{
   JXF_Int myid_x, nprocs_x;

   jxf_ParCSRMatrix  *ARR = NULL; 
   jxf_ParVector     *VRE = NULL;
   
   jxf_CSRMatrix	*local_ARR = NULL;
     
   // 并行矩阵 A 中的部分成员           
   jxf_CSRMatrix	*diag = jxf_ParCSRMatrixDiag(A);
   jxf_CSRMatrix	*offd = jxf_ParCSRMatrixOffd(A);
   JXF_Int *col_map_offd  = jxf_ParCSRMatrixColMapOffd(A);    
   JXF_Int *row_starts    = jxf_ParCSRMatrixRowStarts(A); 
   JXF_Int first_col_diag = jxf_ParCSRMatrixFirstColDiag(A);
   JXF_Int N = jxf_ParCSRMatrixGlobalNumRows(A);

   JXF_Int *ia_diag = jxf_CSRMatrixI(diag);
   JXF_Int *ja_diag = jxf_CSRMatrixJ(diag);
   JXF_Real *aa_diag = jxf_CSRMatrixData(diag);
    
   JXF_Int *ia_offd = jxf_CSRMatrixI(offd);
   JXF_Int *ja_offd = jxf_CSRMatrixJ(offd);
   JXF_Real *aa_offd = jxf_CSRMatrixData(offd);

   JXF_Int *row_starts_R = NULL; 
   JXF_Int *col_starts_R = NULL;
      
   JXF_Int *ia = NULL;
   JXF_Int *ja = NULL;
   JXF_Real *aa = NULL;   
   JXF_Real *vre_data = NULL;
   
   JXF_Int i, j, k, m;
   JXF_Int vre_cnt, nzcnt;
   JXF_Int local_size;
   JXF_Int global_num_rows;
   JXF_Int global_num_cols; 
   
   JXF_Int num_rows;
   JXF_Int num_cols;
   JXF_Int num_nonzeros;  
   
   JXF_Int first_col_diag_R;
   JXF_Int last_col_diag_R;         
   
   jxf_MPI_Comm_rank(comm_x, &myid_x);
   jxf_MPI_Comm_size(comm_x, &nprocs_x); 
              
   local_size = jxf_CSRMatrixNumRows(diag);

   //------------------------------------------------------------------
   //  生成 row_starts_R, col_starts_R 数组                             
   //------------------------------------------------------------------ 
   row_starts_R = jxf_CTAlloc(JXF_Int, nprocs_x + 1);
   for (i = 0; i <= nprocs_x; i ++)
   {
      row_starts_R[i] = row_starts[i];
   }
   col_starts_R = row_starts_R;
   
   //------------------------------------------------------------------
   //  创建一个并行矩阵 ARR                                              
   //------------------------------------------------------------------    
   global_num_rows = N / 3;
   global_num_cols = global_num_rows;   
   ARR = jxf_ParCSRMatrixCreate(comm_x, global_num_rows, global_num_cols, 
                               row_starts_R, col_starts_R, 0, 0, 0);

   //------------------------------------------------------------------
   //  创建一个并行向量 VRE                         
   //------------------------------------------------------------------ 
   VRE = jxf_ParVectorCreate(comm_x, global_num_rows, jxf_ParCSRMatrixRowStarts(ARR));
   jxf_ParVectorSetPartitioningOwner(VRE, 0);
   jxf_ParVectorInitialize(VRE);
   vre_data = jxf_VectorData(jxf_ParVectorLocalVector(VRE));
     
   //-------------------------------------------------------------------------------------------
   //  生成并行矩阵 ARR 和并行向量 VRE
   //-------------------------------------------------------------------------------------------
   /* 创建并初始化串行矩阵 local_ARR */
   num_rows = local_size;
   num_cols = global_num_cols;
   num_nonzeros = jxf_CSRMatrixNumNonzeros(diag) + jxf_CSRMatrixNumNonzeros(offd) - local_size;
   local_ARR = jxf_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
   jxf_CSRMatrixInitialize(local_ARR);
   ia = jxf_CSRMatrixI(local_ARR);
   ja = jxf_CSRMatrixJ(local_ARR);
   aa = jxf_CSRMatrixData(local_ARR);  
   
   /* 对 diag 和 offd 的所有非零元素循环，从中抽出子向量 VRE 和长方阵 local_ARR */
   nzcnt = 0;
   vre_cnt = 0;
   ia[0] = 0;
   for (i = 0; i < local_size; i ++)
   {
      // 处理对角块
      for (j = ia_diag[i]; j < ia_diag[i+1]; j ++)
      {
         ja[nzcnt] = ja_diag[j] + first_col_diag;
         aa[nzcnt] = aa_diag[j];
         nzcnt ++;   
      }
      
      // 处理非对角块
      for (j = ia_offd[i]; j < ia_offd[i+1]; j ++)
      {
         k = ja_offd[j];      // 局部列号
         m = col_map_offd[k]; // 整体列号
         if (m >= global_num_rows)
         {
            vre_data[vre_cnt ++] = aa_offd[j];
         }
         else
         {
            ja[nzcnt] = m; ///
            aa[nzcnt] = aa_offd[j];
            nzcnt ++;   
         }
      }
      ia[i+1] = nzcnt;      
   }
   
   /* 生成 ARR 的对角块和非对角块，以及 col_map_offd 映射数组 */
   first_col_diag_R = col_starts_R[myid_x];
   last_col_diag_R  = col_starts_R[myid_x+1] - 1;
   jxf_GenerateDiagAndOffd(local_ARR, ARR, first_col_diag_R, last_col_diag_R);
   
   jxf_CSRMatrixDestroy(local_ARR);
   
   *ARR_ptr = ARR;
   *VRE_ptr = VRE;

   return (0);   
}

/*!
 * \fn JXF_Int jxf_MatGroupR
 * \brief 利用并行矩阵 A(G) 生成 ARR(GR).
 * \note 要求进程数为 3 的倍数，且分划为自己指定的匹配分划.
 * \date 2012/02/23
 */ 
JXF_Int 
jxf_MatGroupR( MPI_Comm           comm_x,
              jxf_ParCSRMatrix   *A, 
              jxf_ParCSRMatrix  **ARR_ptr )
{
   JXF_Int myid_x, nprocs_x;

   jxf_ParCSRMatrix *ARR = NULL;    
   jxf_CSRMatrix	   *local_ARR = NULL;
     
   // 并行矩阵 A 中的部分成员           
   jxf_CSRMatrix	*diag = jxf_ParCSRMatrixDiag(A);
   jxf_CSRMatrix	*offd = jxf_ParCSRMatrixOffd(A);
   JXF_Int *col_map_offd  = jxf_ParCSRMatrixColMapOffd(A);    
   JXF_Int *row_starts    = jxf_ParCSRMatrixRowStarts(A); 
   JXF_Int first_col_diag = jxf_ParCSRMatrixFirstColDiag(A);
   JXF_Int N = jxf_ParCSRMatrixGlobalNumRows(A);

   JXF_Int *ia_diag = jxf_CSRMatrixI(diag);
   JXF_Int *ja_diag = jxf_CSRMatrixJ(diag);
   JXF_Real *aa_diag = jxf_CSRMatrixData(diag);
    
   JXF_Int *ia_offd = jxf_CSRMatrixI(offd);
   JXF_Int *ja_offd = jxf_CSRMatrixJ(offd);
   JXF_Real *aa_offd = jxf_CSRMatrixData(offd);

   JXF_Int *row_starts_R = NULL; 
   JXF_Int *col_starts_R = NULL;
      
   JXF_Int *ia = NULL;
   JXF_Int *ja = NULL;
   JXF_Real *aa = NULL;   
   
   JXF_Int i, j, k, m;
   JXF_Int nzcnt;
   JXF_Int local_size;
   JXF_Int global_num_rows;
   JXF_Int global_num_cols; 
   
   JXF_Int num_rows;
   JXF_Int num_cols;
   JXF_Int num_nonzeros;  
   
   JXF_Int first_col_diag_R;
   JXF_Int last_col_diag_R;         
   
   jxf_MPI_Comm_rank(comm_x, &myid_x);
   jxf_MPI_Comm_size(comm_x, &nprocs_x); 
              
   local_size = jxf_CSRMatrixNumRows(diag);

   //------------------------------------------------------------------
   //  生成 row_starts_R, col_starts_R 数组                             
   //------------------------------------------------------------------ 
   row_starts_R = jxf_CTAlloc(JXF_Int, nprocs_x + 1);
   for (i = 0; i <= nprocs_x; i ++)
   {
      row_starts_R[i] = row_starts[i];
   }
   col_starts_R = row_starts_R;
   
   //------------------------------------------------------------------
   //  创建一个并行矩阵 ARR                                              
   //------------------------------------------------------------------    
   global_num_rows = N / 3;
   global_num_cols = global_num_rows;   
   ARR = jxf_ParCSRMatrixCreate(comm_x, global_num_rows, global_num_cols, 
                               row_starts_R, col_starts_R, 0, 0, 0);
     
   //-------------------------------------------------------------------------------------------
   //  生成并行矩阵 ARR
   //-------------------------------------------------------------------------------------------
   /* 创建并初始化串行矩阵 local_ARR */
   num_rows = local_size;
   num_cols = global_num_cols;
   num_nonzeros = jxf_CSRMatrixNumNonzeros(diag) + jxf_CSRMatrixNumNonzeros(offd) - local_size;
   local_ARR = jxf_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
   jxf_CSRMatrixInitialize(local_ARR);
   ia = jxf_CSRMatrixI(local_ARR);
   ja = jxf_CSRMatrixJ(local_ARR);
   aa = jxf_CSRMatrixData(local_ARR);  
   
   /* 对 diag 和 offd 的所有非零元素循环，从中抽出长方阵 local_ARR */
   nzcnt = 0;
   ia[0] = 0;
   for (i = 0; i < local_size; i ++)
   {
      // 处理对角块
      for (j = ia_diag[i]; j < ia_diag[i+1]; j ++)
      {
         ja[nzcnt] = ja_diag[j] + first_col_diag;
         aa[nzcnt] = aa_diag[j];
         nzcnt ++;   
      }
      
      // 处理非对角块
      for (j = ia_offd[i]; j < ia_offd[i+1]; j ++)
      {
         k = ja_offd[j];      // 局部列号
         m = col_map_offd[k]; // 整体列号
         if (m < global_num_rows)
         {
            ja[nzcnt] = m; ///
            aa[nzcnt] = aa_offd[j];
            nzcnt ++;   
         }
      }
      ia[i+1] = nzcnt;      
   }
   
   /* 生成 ARR 的对角块和非对角块，以及 col_map_offd 映射数组 */
   first_col_diag_R = col_starts_R[myid_x];
   last_col_diag_R  = col_starts_R[myid_x+1] - 1;
   jxf_GenerateDiagAndOffd(local_ARR, ARR, first_col_diag_R, last_col_diag_R);
   
   jxf_CSRMatrixDestroy(local_ARR);
   
   *ARR_ptr = ARR;

   return (0);   
}


/*!
 * \fn JXF_Int jxf_MatVecGroupI
 * \brief 利用并行矩阵 A(G) 生成 AII(GI), VIE(GI).
 * \note 要求进程数为 3 的倍数，且分划为自己指定的匹配分划.
 * \date 2012/02/23
 */ 
JXF_Int 
jxf_MatVecGroupI( MPI_Comm           comm_x,
                 jxf_ParCSRMatrix   *A, 
                 jxf_ParCSRMatrix  **AII_ptr, 
                 jxf_ParVector     **VIE_ptr )
{
   JXF_Int myid_x, nprocs_x;
   JXF_Int nprocs;

   jxf_ParCSRMatrix  *AII = NULL; 
   jxf_ParVector     *VIE = NULL;
   
   jxf_CSRMatrix	*local_AII = NULL;
     
   // 并行矩阵 A 中的部分成员           
   jxf_CSRMatrix	*diag = jxf_ParCSRMatrixDiag(A);
   jxf_CSRMatrix	*offd = jxf_ParCSRMatrixOffd(A);
   JXF_Int *col_map_offd  = jxf_ParCSRMatrixColMapOffd(A);    
   JXF_Int *row_starts    = jxf_ParCSRMatrixRowStarts(A); 
   JXF_Int first_col_diag = jxf_ParCSRMatrixFirstColDiag(A);
   JXF_Int N = jxf_ParCSRMatrixGlobalNumRows(A);

   JXF_Int *ia_diag = jxf_CSRMatrixI(diag);
   JXF_Int *ja_diag = jxf_CSRMatrixJ(diag);
   JXF_Real *aa_diag = jxf_CSRMatrixData(diag);
    
   JXF_Int *ia_offd = jxf_CSRMatrixI(offd);
   JXF_Int *ja_offd = jxf_CSRMatrixJ(offd);
   JXF_Real *aa_offd = jxf_CSRMatrixData(offd);

   JXF_Int *row_starts_I = NULL; 
   JXF_Int *col_starts_I = NULL;
      
   JXF_Int *ia = NULL;
   JXF_Int *ja = NULL;
   JXF_Real *aa = NULL;   
   JXF_Real *vie_data = NULL;
   
   JXF_Int i, j, k, m;
   JXF_Int vie_cnt, nzcnt;
   JXF_Int local_size;
   JXF_Int global_num_rows;
   JXF_Int global_num_cols; 
   
   JXF_Int num_rows;
   JXF_Int num_cols;
   JXF_Int num_nonzeros;  
   
   JXF_Int first_col_diag_I;
   JXF_Int last_col_diag_I; 
   
   JXF_Int npstart;  
   JXF_Int disp; 
   JXF_Int global_num_rows2;     
   
   jxf_MPI_Comm_rank(comm_x, &myid_x);
   jxf_MPI_Comm_size(comm_x, &nprocs_x); 
   jxf_MPI_Comm_size(jxf_ParCSRMatrixComm(A), &nprocs);
    
   npstart = (2*nprocs) / 3;           
   local_size = jxf_CSRMatrixNumRows(diag);

   //------------------------------------------------------------------
   //  生成 row_starts_I, col_starts_I 数组                             
   //------------------------------------------------------------------ 
   row_starts_I = jxf_CTAlloc(JXF_Int, nprocs_x + 1);
   for (i = 0; i <= nprocs_x; i ++)
   {
      row_starts_I[i] = row_starts[npstart+i] - row_starts[npstart];
   }
   col_starts_I = row_starts_I;


   //------------------------------------------------------------------
   //  创建一个并行矩阵 AII                                              
   //------------------------------------------------------------------    
   global_num_rows = N / 3;
   global_num_cols = global_num_rows;   
   AII = jxf_ParCSRMatrixCreate(comm_x, global_num_rows, global_num_cols, 
                               row_starts_I, col_starts_I, 0, 0, 0);

   //------------------------------------------------------------------
   //  创建一个并行向量 VIE                         
   //------------------------------------------------------------------ 
   VIE = jxf_ParVectorCreate(comm_x, global_num_rows, jxf_ParCSRMatrixRowStarts(AII));
   jxf_ParVectorSetPartitioningOwner(VIE, 0);
   jxf_ParVectorInitialize(VIE);
   vie_data = jxf_VectorData(jxf_ParVectorLocalVector(VIE));
     
   //-------------------------------------------------------------------------------------------
   //  生成并行矩阵 AII 和并行向量 VIE
   //-------------------------------------------------------------------------------------------
   /* 创建并初始化串行矩阵 local_AII */
   num_rows = local_size;
   num_cols = global_num_cols;
   num_nonzeros = jxf_CSRMatrixNumNonzeros(diag) + jxf_CSRMatrixNumNonzeros(offd) - local_size;
   local_AII = jxf_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
   jxf_CSRMatrixInitialize(local_AII);
   ia = jxf_CSRMatrixI(local_AII);
   ja = jxf_CSRMatrixJ(local_AII);
   aa = jxf_CSRMatrixData(local_AII);  

   /* 对 diag 和 offd 的所有非零元素循环，从中抽出子向量 VIE 和长方阵 local_AII */
   nzcnt = 0;
   vie_cnt = 0;
   ia[0] = 0;
   global_num_rows2 = 2*global_num_rows;
   disp = first_col_diag - global_num_rows2;
   for (i = 0; i < local_size; i ++)
   {
      // 处理对角块
      for (j = ia_diag[i]; j < ia_diag[i+1]; j ++)
      {
         ja[nzcnt] = ja_diag[j] + disp;
         aa[nzcnt] = aa_diag[j];
         nzcnt ++;   
      }
      
      // 处理非对角块
      for (j = ia_offd[i]; j < ia_offd[i+1]; j ++)
      {
         k = ja_offd[j];      // 局部列号
         m = col_map_offd[k]; // 整体列号
         if (m < global_num_rows2)
         {
            vie_data[vie_cnt ++] = aa_offd[j];
         }
         else
         {
            ja[nzcnt] = m - global_num_rows2; ///
            aa[nzcnt] = aa_offd[j];
            nzcnt ++;   
         }
      }
      ia[i+1] = nzcnt;      
   }
    
   /* 生成 AII 的对角块和非对角块，以及 col_map_offd 映射数组 */
   first_col_diag_I = col_starts_I[myid_x];
   last_col_diag_I  = col_starts_I[myid_x+1] - 1;
   jxf_GenerateDiagAndOffd(local_AII, AII, first_col_diag_I, last_col_diag_I);
   
   jxf_CSRMatrixDestroy(local_AII);
   
   *AII_ptr = AII;
   *VIE_ptr = VIE;

   return (0);   
}

/*!
 * \fn JXF_Int jxf_MatGroupI
 * \brief 利用并行矩阵 A(G) 生成 AII(GI).
 * \note 要求进程数为 3 的倍数，且分划为自己指定的匹配分划.
 * \date 2012/02/23
 */ 
JXF_Int 
jxf_MatGroupI( MPI_Comm           comm_x,
              jxf_ParCSRMatrix   *A, 
              jxf_ParCSRMatrix  **AII_ptr )
{
   JXF_Int myid_x, nprocs_x;
   JXF_Int nprocs;

   jxf_ParCSRMatrix *AII = NULL; 
   jxf_CSRMatrix	   *local_AII = NULL;
     
   // 并行矩阵 A 中的部分成员           
   jxf_CSRMatrix	*diag = jxf_ParCSRMatrixDiag(A);
   jxf_CSRMatrix	*offd = jxf_ParCSRMatrixOffd(A);
   JXF_Int *col_map_offd  = jxf_ParCSRMatrixColMapOffd(A);    
   JXF_Int *row_starts    = jxf_ParCSRMatrixRowStarts(A); 
   JXF_Int first_col_diag = jxf_ParCSRMatrixFirstColDiag(A);
   JXF_Int N = jxf_ParCSRMatrixGlobalNumRows(A);

   JXF_Int *ia_diag = jxf_CSRMatrixI(diag);
   JXF_Int *ja_diag = jxf_CSRMatrixJ(diag);
   JXF_Real *aa_diag = jxf_CSRMatrixData(diag);
    
   JXF_Int *ia_offd = jxf_CSRMatrixI(offd);
   JXF_Int *ja_offd = jxf_CSRMatrixJ(offd);
   JXF_Real *aa_offd = jxf_CSRMatrixData(offd);

   JXF_Int *row_starts_I = NULL; 
   JXF_Int *col_starts_I = NULL;
      
   JXF_Int *ia = NULL;
   JXF_Int *ja = NULL;
   JXF_Real *aa = NULL;   

   JXF_Int i, j, k, m;
   JXF_Int nzcnt;
   JXF_Int local_size;
   JXF_Int global_num_rows;
   JXF_Int global_num_cols; 
   
   JXF_Int num_rows;
   JXF_Int num_cols;
   JXF_Int num_nonzeros;  
   
   JXF_Int first_col_diag_I;
   JXF_Int last_col_diag_I; 
   
   JXF_Int npstart;  
   JXF_Int disp; 
   JXF_Int global_num_rows2;     
   
   jxf_MPI_Comm_rank(comm_x, &myid_x);
   jxf_MPI_Comm_size(comm_x, &nprocs_x); 
   jxf_MPI_Comm_size(jxf_ParCSRMatrixComm(A), &nprocs);
    
   npstart = (2*nprocs) / 3;           
   local_size = jxf_CSRMatrixNumRows(diag);

   //------------------------------------------------------------------
   //  生成 row_starts_I, col_starts_I 数组                             
   //------------------------------------------------------------------ 
   row_starts_I = jxf_CTAlloc(JXF_Int, nprocs_x + 1);
   for (i = 0; i <= nprocs_x; i ++)
   {
      row_starts_I[i] = row_starts[npstart+i] - row_starts[npstart];
   }
   col_starts_I = row_starts_I;


   //------------------------------------------------------------------
   //  创建一个并行矩阵 AII                                              
   //------------------------------------------------------------------    
   global_num_rows = N / 3;
   global_num_cols = global_num_rows;   
   AII = jxf_ParCSRMatrixCreate(comm_x, global_num_rows, global_num_cols, 
                               row_starts_I, col_starts_I, 0, 0, 0);

   //-------------------------------------------------------------------------------------------
   //  生成并行矩阵 AII
   //-------------------------------------------------------------------------------------------
   /* 创建并初始化串行矩阵 local_AII */
   num_rows = local_size;
   num_cols = global_num_cols;
   num_nonzeros = jxf_CSRMatrixNumNonzeros(diag) + jxf_CSRMatrixNumNonzeros(offd) - local_size;
   local_AII = jxf_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
   jxf_CSRMatrixInitialize(local_AII);
   ia = jxf_CSRMatrixI(local_AII);
   ja = jxf_CSRMatrixJ(local_AII);
   aa = jxf_CSRMatrixData(local_AII);  

   /* 对 diag 和 offd 的所有非零元素循环，从中抽出长方阵 local_AII */
   nzcnt = 0;
   ia[0] = 0;
   global_num_rows2 = 2*global_num_rows;
   disp = first_col_diag - global_num_rows2;
   for (i = 0; i < local_size; i ++)
   {
      // 处理对角块
      for (j = ia_diag[i]; j < ia_diag[i+1]; j ++)
      {
         ja[nzcnt] = ja_diag[j] + disp;
         aa[nzcnt] = aa_diag[j];
         nzcnt ++;   
      }
      
      // 处理非对角块
      for (j = ia_offd[i]; j < ia_offd[i+1]; j ++)
      {
         k = ja_offd[j];      // 局部列号
         m = col_map_offd[k]; // 整体列号
         if (m >= global_num_rows2)
         {
            ja[nzcnt] = m - global_num_rows2; ///
            aa[nzcnt] = aa_offd[j];
            nzcnt ++;   
         }
      }
      ia[i+1] = nzcnt;      
   }
    
   /* 生成 AII 的对角块和非对角块，以及 col_map_offd 映射数组 */
   first_col_diag_I = col_starts_I[myid_x];
   last_col_diag_I  = col_starts_I[myid_x+1] - 1;
   jxf_GenerateDiagAndOffd(local_AII, AII, first_col_diag_I, last_col_diag_I);
   
   jxf_CSRMatrixDestroy(local_AII);
   
   *AII_ptr = AII;

   return (0);   
}

/*!
 * \fn JXF_Int jxf_MatVecGroupE
 * \brief 利用并行矩阵 A(G) 生成 AEE(GE), VER(GE), VEI(GE).
 * \note 要求进程数为 3 的倍数，且分划为自己指定的匹配分划.
 * \date 2012/02/23
 */ 
JXF_Int 
jxf_MatVecGroupE( MPI_Comm           comm_x,
                 jxf_ParCSRMatrix   *A, 
                 jxf_ParCSRMatrix  **AEE_ptr, 
                 jxf_ParVector     **VER_ptr,
                 jxf_ParVector     **VEI_ptr )
{
   JXF_Int myid_x, nprocs_x;
   JXF_Int nprocs;

   jxf_ParCSRMatrix  *AEE = NULL; 
   jxf_ParVector     *VER = NULL;
   jxf_ParVector     *VEI = NULL;
   
   jxf_CSRMatrix	*local_AEE = NULL;
     
   // 并行矩阵 A 中的部分成员           
   jxf_CSRMatrix	*diag = jxf_ParCSRMatrixDiag(A);
   jxf_CSRMatrix	*offd = jxf_ParCSRMatrixOffd(A);
   JXF_Int *col_map_offd  = jxf_ParCSRMatrixColMapOffd(A);    
   JXF_Int *row_starts    = jxf_ParCSRMatrixRowStarts(A); 
   JXF_Int first_col_diag = jxf_ParCSRMatrixFirstColDiag(A);
   JXF_Int N = jxf_ParCSRMatrixGlobalNumRows(A);

   JXF_Int *ia_diag = jxf_CSRMatrixI(diag);
   JXF_Int *ja_diag = jxf_CSRMatrixJ(diag);
   JXF_Real *aa_diag = jxf_CSRMatrixData(diag);
    
   JXF_Int *ia_offd = jxf_CSRMatrixI(offd);
   JXF_Int *ja_offd = jxf_CSRMatrixJ(offd);
   JXF_Real *aa_offd = jxf_CSRMatrixData(offd);

   JXF_Int *row_starts_E = NULL; 
   JXF_Int *col_starts_E = NULL;
      
   JXF_Int *ia = NULL;
   JXF_Int *ja = NULL;
   JXF_Real *aa = NULL;   
   JXF_Real *ver_data = NULL;
   JXF_Real *vei_data = NULL;
   
   JXF_Int i, j, k, m;
   JXF_Int ver_cnt, vei_cnt, nzcnt;
   JXF_Int local_size;
   JXF_Int global_num_rows;
   JXF_Int global_num_cols; 
   
   JXF_Int num_rows;
   JXF_Int num_cols;
   JXF_Int num_nonzeros;  
   
   JXF_Int first_col_diag_E;
   JXF_Int last_col_diag_E; 
   
   JXF_Int npstart;  
   JXF_Int disp; 
   JXF_Int global_num_rows2;     
   
   jxf_MPI_Comm_rank(comm_x, &myid_x);
   jxf_MPI_Comm_size(comm_x, &nprocs_x); 
   jxf_MPI_Comm_size(jxf_ParCSRMatrixComm(A), &nprocs);
    
   npstart = nprocs / 3;           
   local_size = jxf_CSRMatrixNumRows(diag);

   //------------------------------------------------------------------
   //  生成 row_starts_E, col_starts_E 数组                             
   //------------------------------------------------------------------ 
   row_starts_E = jxf_CTAlloc(JXF_Int, nprocs_x + 1);
   for (i = 0; i <= nprocs_x; i ++)
   {
      row_starts_E[i] = row_starts[npstart+i] - row_starts[npstart];
   }
   col_starts_E = row_starts_E;

   //------------------------------------------------------------------
   //  创建一个并行矩阵 AEE                                              
   //------------------------------------------------------------------    
   global_num_rows = N / 3;
   global_num_cols = global_num_rows;   
   AEE = jxf_ParCSRMatrixCreate(comm_x, global_num_rows, global_num_cols, 
                               row_starts_E, col_starts_E, 0, 0, 0);

   //------------------------------------------------------------------
   //  创建两个并行向量 VER, VEI                         
   //------------------------------------------------------------------ 
   VER = jxf_ParVectorCreate(comm_x, global_num_rows, jxf_ParCSRMatrixRowStarts(AEE));
   jxf_ParVectorSetPartitioningOwner(VER, 0);
   jxf_ParVectorInitialize(VER);
   ver_data = jxf_VectorData(jxf_ParVectorLocalVector(VER));
   
   VEI = jxf_ParVectorCreate(comm_x, global_num_rows, jxf_ParCSRMatrixRowStarts(AEE));
   jxf_ParVectorSetPartitioningOwner(VEI, 0);
   jxf_ParVectorInitialize(VEI);
   vei_data = jxf_VectorData(jxf_ParVectorLocalVector(VEI));   
    
   //-------------------------------------------------------------------------------------------
   //  生成并行矩阵 AEE 和并行向量 VER, VEI
   //-------------------------------------------------------------------------------------------
   /* 创建并初始化串行矩阵 local_AEE */
   num_rows = local_size;
   num_cols = global_num_cols;
   num_nonzeros = jxf_CSRMatrixNumNonzeros(diag) + jxf_CSRMatrixNumNonzeros(offd) - 2*local_size;
   local_AEE = jxf_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
   jxf_CSRMatrixInitialize(local_AEE);
   ia = jxf_CSRMatrixI(local_AEE);
   ja = jxf_CSRMatrixJ(local_AEE);
   aa = jxf_CSRMatrixData(local_AEE);  

   /* 对 diag 和 offd 的所有非零元素循环，从中抽出子向量 VER, VEI 和长方阵 local_AEE */
   nzcnt = 0;
   ver_cnt = 0;
   vei_cnt = 0;
   ia[0] = 0;
   global_num_rows2 = 2*global_num_rows;
   disp = first_col_diag - global_num_rows;
   for (i = 0; i < local_size; i ++)
   {
      // 处理对角块
      for (j = ia_diag[i]; j < ia_diag[i+1]; j ++)
      {
         ja[nzcnt] = ja_diag[j] + disp;
         aa[nzcnt] = aa_diag[j];
         nzcnt ++;   
      }
      
      // 处理非对角块
      for (j = ia_offd[i]; j < ia_offd[i+1]; j ++)
      {
         k = ja_offd[j];      // 局部列号
         m = col_map_offd[k]; // 整体列号
         if (m < global_num_rows)
         {
            ver_data[ver_cnt ++] = aa_offd[j];
         }
         else if (m >= global_num_rows2)
         {
            vei_data[vei_cnt ++] = aa_offd[j];
         }
         else
         {
            ja[nzcnt] = m - global_num_rows;
            aa[nzcnt] = aa_offd[j];
            nzcnt ++;   
         }
      }
      ia[i+1] = nzcnt;      
   }
     
   /* 生成 AEE 的对角块和非对角块，以及 col_map_offd 映射数组 */
   first_col_diag_E = col_starts_E[myid_x];
   last_col_diag_E  = col_starts_E[myid_x+1] - 1;
   jxf_GenerateDiagAndOffd(local_AEE, AEE, first_col_diag_E, last_col_diag_E);
   
   jxf_CSRMatrixDestroy(local_AEE);
   
   *AEE_ptr = AEE;
   *VER_ptr = VER;
   *VEI_ptr = VEI;
   
   return (0);   
}

/*!
 * \fn JXF_Int jxf_DataCombine4ApctlKrylov
 * \brief 将子块数据合并为整体数据(单进程情形).
 * \author peghoty
 * \date 2012/02/25 
 */
JXF_Int 
jxf_DataCombine4ApctlKrylov( // input:
                            jxf_ParCSRMatrix   *ARR_p, 
                            jxf_ParCSRMatrix   *AEE_p, 
                            jxf_ParCSRMatrix   *AII_p, 
                            jxf_ParVector      *VRE_p, 
                            jxf_ParVector      *VER_p, 
                            jxf_ParVector      *VEI_p, 
                            jxf_ParVector      *VIE_p, 
                            jxf_ParVector      *fR_p, 
                            jxf_ParVector      *fE_p, 
                            jxf_ParVector      *fI_p,
                            jxf_ParVector      *uR_p, 
                            jxf_ParVector      *uE_p, 
                            jxf_ParVector      *uI_p,
                            // output:    
                            jxf_ParCSRMatrix  **A_ptr,  
                            jxf_ParVector     **f_ptr, 
                            jxf_ParVector     **u_ptr )
{
   MPI_Comm comm = jxf_ParCSRMatrixComm(ARR_p);

   jxf_ParCSRMatrix *A = NULL;  
   jxf_ParVector    *f = NULL; 
   jxf_ParVector    *u = NULL;
   
   jxf_CSRMatrix *A_s = NULL;  
   jxf_Vector    *f_s = NULL; 
   jxf_Vector    *u_s = NULL;
      
   jxf_CSRMatrix *ARR = jxf_ParCSRMatrixDiag(ARR_p);
   jxf_CSRMatrix *AEE = jxf_ParCSRMatrixDiag(AEE_p);
   jxf_CSRMatrix *AII = jxf_ParCSRMatrixDiag(AII_p);
   
   jxf_Vector *VRE = jxf_ParVectorLocalVector(VRE_p);
   jxf_Vector *VER = jxf_ParVectorLocalVector(VER_p);
   jxf_Vector *VEI = jxf_ParVectorLocalVector(VEI_p);
   jxf_Vector *VIE = jxf_ParVectorLocalVector(VIE_p);
   jxf_Vector *fR  = jxf_ParVectorLocalVector(fR_p);
   jxf_Vector *fE  = jxf_ParVectorLocalVector(fE_p);
   jxf_Vector *fI  = jxf_ParVectorLocalVector(fI_p);
   jxf_Vector *uR  = jxf_ParVectorLocalVector(uR_p);
   jxf_Vector *uE  = jxf_ParVectorLocalVector(uE_p);
   jxf_Vector *uI  = jxf_ParVectorLocalVector(uI_p);   

   //------------------------------------------------------------------
   // 生成串行矩阵 A_s, 以及串行向量 f_s, u_s
   //------------------------------------------------------------------
    
   jxf_3tGlobalSystem( ARR, AEE, AII, VRE, VER, VEI, VIE, 
                      fR, fE, fI, uR, uE, uI, &A_s, &f_s, &u_s ); 

   //------------------------------------------------------------------
   // 由串行矩阵 A_s, 串行向量 f_s, u_s 转成并行矩阵 A 和并行向量 f, u 
   //------------------------------------------------------------------
   
   A = jxf_CSRMatrixToParCSRMatrix_sp(comm, A_s);
   f = jxf_VectorToParVector_sp(comm, f_s);
   u = jxf_VectorToParVector_sp(comm, u_s);
  
   //------------------------------------------------------------------
   // 释放内存，注意由于 A_s, f_s, u_s 的数据部分包含在 A, f, u 中，所有这里
   // 只需释放结构体，而不需要调用 XXDestroy.
   //------------------------------------------------------------------   
   jxf_TFree(A_s);
   jxf_TFree(f_s);
   jxf_TFree(u_s);
   
   *A_ptr = A;
   *f_ptr = f;
   *u_ptr = u;

   return (0);
}                              

/*!
 * \fn JXF_Int jxf_3tGlobalSystem
 * \brief Form the global linear system.
 * \author peghoty
 * \date 2011/10/18 
 */
JXF_Int 
jxf_3tGlobalSystem( jxf_CSRMatrix  *ARR, 
                   jxf_CSRMatrix  *AEE, 
                   jxf_CSRMatrix  *AII, 
                   jxf_Vector     *VRE, 
                   jxf_Vector     *VER, 
                   jxf_Vector     *VEI, 
                   jxf_Vector     *VIE, 
                   jxf_Vector     *fR, 
                   jxf_Vector     *fE, 
                   jxf_Vector     *fI,
                   jxf_Vector     *uR, 
                   jxf_Vector     *uE, 
                   jxf_Vector     *uI,                       
                   jxf_CSRMatrix **A_ptr, 
                   jxf_Vector    **f_ptr,
                   jxf_Vector    **u_ptr )
{
   jxf_CSRMatrix *A = NULL; 
   jxf_Vector    *f = NULL;
   jxf_Vector    *u = NULL;

   JXF_Int n  = jxf_CSRMatrixNumRows(ARR);
   JXF_Int nz = jxf_CSRMatrixNumNonzeros(ARR);
   JXF_Int nplusn = 2*n;   
   JXF_Int N  = 3*n;
   JXF_Int NZ = 3*nz + 4*n;
   
   JXF_Real *aa = NULL;
   JXF_Int    *ia = NULL;
   JXF_Int    *ja = NULL;
   JXF_Real *f_data = NULL;
   JXF_Real *u_data = NULL;
    
   JXF_Real *aa_R = jxf_CSRMatrixData(ARR);
   JXF_Int    *ia_R = jxf_CSRMatrixI(ARR);
   JXF_Int    *ja_R = jxf_CSRMatrixJ(ARR);
   
   JXF_Real *aa_E = jxf_CSRMatrixData(AEE);
   JXF_Int    *ia_E = jxf_CSRMatrixI(AEE);
   JXF_Int    *ja_E = jxf_CSRMatrixJ(AEE);

   JXF_Real *aa_I = jxf_CSRMatrixData(AII);
   JXF_Int    *ia_I = jxf_CSRMatrixI(AII);
   JXF_Int    *ja_I = jxf_CSRMatrixJ(AII);
   
   JXF_Real *vre_data = jxf_VectorData(VRE);
   JXF_Real *ver_data = jxf_VectorData(VER);
   JXF_Real *vei_data = jxf_VectorData(VEI);
   JXF_Real *vie_data = jxf_VectorData(VIE);
   
   JXF_Real *fR_data = jxf_VectorData(fR);
   JXF_Real *fE_data = jxf_VectorData(fE);
   JXF_Real *fI_data = jxf_VectorData(fI);
   
   JXF_Real *uR_data = jxf_VectorData(uR);
   JXF_Real *uE_data = jxf_VectorData(uE);
   JXF_Real *uI_data = jxf_VectorData(uI);   
   
   /* local variables */
   JXF_Int i, i1, i2, j;
   JXF_Int cnt;

   //---------------------------------------------------
   //  Generate the CSR matrix A
   //---------------------------------------------------
   
   /* create a CSR matrix */
   A = jxf_CSRMatrixCreate(N, N, NZ);
   jxf_CSRMatrixInitialize(A);
   aa = jxf_CSRMatrixData(A);
   ia = jxf_CSRMatrixI(A);
   ja = jxf_CSRMatrixJ(A);
   
   /* initialize the counter */
   cnt = 0;
   ia[0] = 0;   

   /* deal with ARR and VRE */
   for (i = 0; i < n; i ++)
   {
      for (j = ia_R[i]; j < ia_R[i+1]; j ++)
      {
         aa[cnt] = aa_R[j];
         ja[cnt] = ja_R[j];
         cnt ++;
      }
      aa[cnt] = vre_data[i];
      ja[cnt] = i + n;
      cnt ++;
      
      ia[i+1] = cnt;
   }

   /* deal with AEE and VER, VEI */
   for (i = 0; i < n; i ++)
   {
      for (j = ia_E[i]; j < ia_E[i+1]; j ++)
      {
         aa[cnt] = aa_E[j];
         ja[cnt] = ja_E[j] + n;
         cnt ++;
      }
      aa[cnt] = ver_data[i];
      ja[cnt] = i;
      cnt ++;

      aa[cnt] = vei_data[i];
      ja[cnt] = i + nplusn;
      cnt ++;
      
      ia[i+n+1] = cnt;
   }

   /* deal with AII and VIE */
   for (i = 0; i < n; i ++)
   {
      for (j = ia_I[i]; j < ia_I[i+1]; j ++)
      {
         aa[cnt] = aa_I[j];
         ja[cnt] = ja_I[j] + nplusn;
         cnt ++;
      }
      aa[cnt] = vie_data[i];
      ja[cnt] = i + n;
      cnt ++;
      
      ia[i+nplusn+1] = cnt;
   }

   //---------------------------------------------------
   //  Generate the vector f and u
   //---------------------------------------------------

   /* create f and u  */
   f = jxf_SeqVectorCreate(N);
   jxf_SeqVectorInitialize(f);
   f_data = jxf_VectorData(f);

   u = jxf_SeqVectorCreate(N);
   jxf_SeqVectorInitialize(u);
   u_data = jxf_VectorData(u);
  
   for (i = 0; i < n; i ++)
   {  
      i1 = i + n;
      i2 = i + nplusn;
      f_data[i]  = fR_data[i];
      f_data[i1] = fE_data[i];
      f_data[i2] = fI_data[i];
      u_data[i]  = uR_data[i];
      u_data[i1] = uE_data[i];
      u_data[i2] = uI_data[i];      
   }

   *A_ptr = A;
   *f_ptr = f;
   *u_ptr = u;
   
   return (0);
}

/*!
 * \fn JXF_Int jxf_ParaDataTrans4ApctlKrylov
 * \brief Data transferring for Apctl-Krylov method.
 * \author peghoty 
 * \date 2012/02/25
 */
JXF_Int 
jxf_ParaDataTrans4ApctlKrylov( // input:
                              MPI_Comm           comm, 
                              MPI_Comm           comm_x, 
                              JXF_Int                groupid_x,
                              jxf_ParCSRMatrix   *ARR_p, 
                              jxf_ParCSRMatrix   *AEE_p, 
                              jxf_ParCSRMatrix   *AII_p, 
                              jxf_ParVector      *VRE_p, 
                              jxf_ParVector      *VER_p, 
                              jxf_ParVector      *VEI_p, 
                              jxf_ParVector      *VIE_p, 
                              jxf_ParVector      *fR_p, 
                              jxf_ParVector      *fE_p, 
                              jxf_ParVector      *fI_p,
                              jxf_ParVector      *uR_p, 
                              jxf_ParVector      *uE_p, 
                              jxf_ParVector      *uI_p,
                              // output:    
                              jxf_ParCSRMatrix  **ARR_ptr,
                              jxf_ParCSRMatrix  **AEE_ptr,
                              jxf_ParCSRMatrix  **AII_ptr, 
                              jxf_ParVector     **VRE_ptr, 
                              jxf_ParVector     **VER_ptr, 
                              jxf_ParVector     **VEI_ptr, 
                              jxf_ParVector     **VIE_ptr,
                              jxf_ParCSRMatrix  **A_ptr,  
                              jxf_ParVector     **f_ptr, 
                              jxf_ParVector     **u_ptr )                    
{
   jxf_ParCSRMatrix  *ARR = NULL;
   jxf_ParCSRMatrix  *AEE = NULL;
   jxf_ParCSRMatrix  *AII = NULL; 
   jxf_ParVector     *VRE = NULL; 
   jxf_ParVector     *VER = NULL; 
   jxf_ParVector     *VEI = NULL; 
   jxf_ParVector     *VIE = NULL;

   jxf_ParCSRMatrix  *A = NULL;  
   jxf_ParVector     *f = NULL; 
   jxf_ParVector     *u = NULL;
   
   jxf_Vector *VRE_loc = NULL; 
   jxf_Vector *VER_loc = NULL; 
   jxf_Vector *VEI_loc = NULL; 
   jxf_Vector *VIE_loc = NULL;

   jxf_Vector *f_loc = NULL; 
   jxf_Vector *u_loc = NULL; 
   
   JXF_Int *row_starts_loc = NULL;     
   JXF_Int *col_starts_loc = NULL;
   
   JXF_Int *row_starts_glo = NULL;     
   JXF_Int *col_starts_glo = NULL;   
   
   JXF_Int *row_starts = jxf_ParCSRMatrixRowStarts(ARR_p);
   
   JXF_Int n = jxf_ParCSRMatrixGlobalNumRows(ARR_p);
   JXF_Int N = 3*n;
   JXF_Int n2 = 2*n;
   
   JXF_Int first_col_diag;
   JXF_Int last_col_diag;   
                          
   JXF_Int myid, myid_x, nprocs;
 
   jxf_CSRMatrix *ARR_comb = NULL;
   JXF_Real *aa_R = NULL;
   JXF_Int    *ia_R = NULL;
   JXF_Int    *ja_R = NULL;

   jxf_CSRMatrix *AEE_comb = NULL;    
   JXF_Real *aa_E = NULL;
   JXF_Int    *ia_E = NULL;
   JXF_Int    *ja_E = NULL;

   jxf_CSRMatrix *AII_comb = NULL;  
   JXF_Real *aa_I = NULL;
   JXF_Int    *ia_I = NULL;
   JXF_Int    *ja_I = NULL;
   
   JXF_Real *vre_data = jxf_VectorData(jxf_ParVectorLocalVector(VRE_p));
   JXF_Real *ver_data = jxf_VectorData(jxf_ParVectorLocalVector(VER_p));
   JXF_Real *vei_data = jxf_VectorData(jxf_ParVectorLocalVector(VEI_p));
   JXF_Real *vie_data = jxf_VectorData(jxf_ParVectorLocalVector(VIE_p));
   
   JXF_Real *fr_data = jxf_VectorData(jxf_ParVectorLocalVector(fR_p));
   JXF_Real *fe_data = jxf_VectorData(jxf_ParVectorLocalVector(fE_p));
   JXF_Real *fi_data = jxf_VectorData(jxf_ParVectorLocalVector(fI_p));
   
   JXF_Real *ur_data = jxf_VectorData(jxf_ParVectorLocalVector(uR_p));
   JXF_Real *ue_data = jxf_VectorData(jxf_ParVectorLocalVector(uE_p));
   JXF_Real *ui_data = jxf_VectorData(jxf_ParVectorLocalVector(uI_p));

   jxf_CSRMatrix *A_loc = NULL; 
   JXF_Int num_rows, num_cols, num_nonzeros;
   JXF_Real *aa = NULL;
   JXF_Int    *ia = NULL;
   JXF_Int    *ja = NULL;
   
   jxf_CSRMatrix *A_rec = NULL;
   JXF_Real *aarec = NULL;
   JXF_Int    *iarec = NULL;
   JXF_Int    *jarec = NULL;         
   
   JXF_Int  num_sends;                 // 发送进程个数
   JXF_Int *send_procs_dbl  = NULL;    // 发送进程编号
   JXF_Int *send_procs_int  = NULL;    // 发送进程编号
   JXF_Int *send_starts_dbl = NULL;    // 实型数据发送缓存区的管理数组
   JXF_Int *send_starts_int = NULL;    // 整型发送缓存区的管理数组

   JXF_Int  num_recvs;                 // 接收进程个数
   JXF_Int *recv_procs_dbl  = NULL;    // 接收进程编号
   JXF_Int *recv_procs_int  = NULL;    // 接收进程编号
   JXF_Int *recv_starts_dbl = NULL;    // 实型数据接收缓存区的管理数组
   JXF_Int *recv_starts_int = NULL;    // 整型数据接收缓存区的管理数组    

   JXF_Real *send_buf_dbl = NULL;
   JXF_Real *recv_buf_dbl = NULL;
   
   JXF_Int    *send_buf_int = NULL;
   JXF_Int    *recv_buf_int = NULL;

   jxf_ParVecCommPkg    *comm_pkg_dbl    = NULL;
   jxf_ParVecCommPkg    *comm_pkg_int    = NULL;
   jxf_ParVecCommHandle *comm_handle_dbl = NULL;
   jxf_ParVecCommHandle *comm_handle_int = NULL;   
         
   JXF_Int num_proc_eachgroup;          
   JXF_Int num_proc_eachgroup2;         
   JXF_Int local_row;                   // 行数
   JXF_Int local_col;                   // 列数
   JXF_Int local_nnz;                   // 非零元素个数 
   
   JXF_Int *local_nnz_array = NULL;      
   JXF_Int *local_row_array = NULL;     
   
   JXF_Int size_base_int;
   JXF_Int size0, size1, size2;
   JXF_Int rnp0, rnp1, rnp2;
     
   JXF_Int i, j, k;
   JXF_Int row_cnt, ja_cnt, aa_cnt; 
   JXF_Int f_cnt, u_cnt; 
   JXF_Int vre_cnt, ver_cnt, vei_cnt, vie_cnt;  
   JXF_Int start;
   JXF_Int loc_row_0, loc_row_1, loc_row_2;
   JXF_Int loc_nnz_0, loc_nnz_1, loc_nnz_2, loc_nnz_01;  
     
   jxf_MPI_Comm_rank(comm, &myid);
   jxf_MPI_Comm_rank(comm_x, &myid_x);
   jxf_MPI_Comm_size(comm, &nprocs); 

   //====================================================================================
   // Step 1: 数据通信
   //====================================================================================
   
   ARR_comb = jxf_MergeDiagAndOffd(ARR_p);
   AEE_comb = jxf_MergeDiagAndOffd(AEE_p);
   AII_comb = jxf_MergeDiagAndOffd(AII_p);  
   
   local_nnz = jxf_CSRMatrixNumNonzeros(ARR_comb);
   local_row = jxf_CSRMatrixNumRows(ARR_comb); 
   local_col = jxf_CSRMatrixNumCols(ARR_comb);
 
   aa_R = jxf_CSRMatrixData(ARR_comb);
   ia_R = jxf_CSRMatrixI(ARR_comb);
   ja_R = jxf_CSRMatrixJ(ARR_comb);
   
   aa_E = jxf_CSRMatrixData(AEE_comb);
   ia_E = jxf_CSRMatrixI(AEE_comb);
   ja_E = jxf_CSRMatrixJ(AEE_comb);
   
   aa_I = jxf_CSRMatrixData(AII_comb);
   ia_I = jxf_CSRMatrixI(AII_comb);
   ja_I = jxf_CSRMatrixJ(AII_comb);   

   num_proc_eachgroup  = nprocs / 3;
   num_proc_eachgroup2 = 2*num_proc_eachgroup;
   
   /* num_sends */
   num_sends = 3;
   
   send_procs_dbl  = jxf_CTAlloc(JXF_Int, num_sends);
   send_procs_int  = jxf_CTAlloc(JXF_Int, num_sends);
   send_starts_dbl = jxf_CTAlloc(JXF_Int, num_sends + 1);
   send_starts_int = jxf_CTAlloc(JXF_Int, num_sends + 1);
 
   /* send_procs_dbl */
   send_procs_dbl[0] = myid / 3;
   send_procs_dbl[1] = send_procs_dbl[0] + num_proc_eachgroup;
   send_procs_dbl[2] = send_procs_dbl[1] + num_proc_eachgroup;
 
   /* send_procs_int */
   send_procs_int[0] = send_procs_dbl[0];
   send_procs_int[1] = send_procs_dbl[1];
   send_procs_int[2] = send_procs_dbl[2];
     
   /* send_starts_dbl */
   size0 = local_nnz + 3*local_row;
   size1 = local_nnz + 4*local_row;
   size2 = local_nnz + 3*local_row;
   send_starts_dbl[0] = 0;
   send_starts_dbl[1] = size0;               
   send_starts_dbl[2] = send_starts_dbl[1] + size1; 
   send_starts_dbl[3] = send_starts_dbl[2] + size2; 
    
   /* send_starts_int */     
   size_base_int = local_nnz + (local_row + 1);   
   send_starts_int[0] = 0;
   send_starts_int[1] = size_base_int;   // ARR
   send_starts_int[2] = 2*size_base_int; // AEE
   send_starts_int[3] = 3*size_base_int; // AII

   /* num_recvs */
   num_recvs = 3;
   
   recv_procs_dbl  = jxf_CTAlloc(JXF_Int, num_recvs);
   recv_procs_int  = jxf_CTAlloc(JXF_Int, num_recvs);
   recv_starts_dbl = jxf_CTAlloc(JXF_Int, num_recvs + 1);
   recv_starts_int = jxf_CTAlloc(JXF_Int, num_recvs + 1);
   
   /* recv_procs_dbl */
   recv_procs_dbl[0] = (myid % num_proc_eachgroup)*3;
   recv_procs_dbl[1] = recv_procs_dbl[0] + 1;
   recv_procs_dbl[2] = recv_procs_dbl[1] + 1;
  
   /* recv_procs_int */
   recv_procs_int[0] = recv_procs_dbl[0];
   recv_procs_int[1] = recv_procs_dbl[1];
   recv_procs_int[2] = recv_procs_dbl[2];
   
   /* 消息全收集 */
   local_nnz_array = jxf_CTAlloc(JXF_Int, nprocs);
   local_row_array = jxf_CTAlloc(JXF_Int, nprocs);
   jxf_MPI_Allgather(&local_nnz, 1, JXF_MPI_INT, local_nnz_array, 1, JXF_MPI_INT, comm);
   jxf_MPI_Allgather(&local_row, 1, JXF_MPI_INT, local_row_array, 1, JXF_MPI_INT, comm);
 
   /* recv_starts_dbl */
   
   rnp0 = recv_procs_dbl[0];
   rnp1 = recv_procs_dbl[1];
   rnp2 = recv_procs_dbl[2]; 
   
   loc_row_0 = local_row_array[rnp0];
   loc_row_1 = local_row_array[rnp1];
   loc_row_2 = local_row_array[rnp2];
  
   loc_nnz_0 = local_nnz_array[rnp0];
   loc_nnz_1 = local_nnz_array[rnp1];
   loc_nnz_2 = local_nnz_array[rnp2];
        
   if (groupid_x == 1)
   {
      size0 = loc_nnz_0 + 4*loc_row_0;
      size1 = loc_nnz_1 + 4*loc_row_1;
      size2 = loc_nnz_2 + 4*loc_row_2;
   }
   else
   {  
      size0 = loc_nnz_0 + 3*loc_row_0;
      size1 = loc_nnz_1 + 3*loc_row_1;
      size2 = loc_nnz_2 + 3*loc_row_2;
   }
   recv_starts_dbl[0] = 0;
   recv_starts_dbl[1] = size0;
   recv_starts_dbl[2] = recv_starts_dbl[1] + size1;
   recv_starts_dbl[3] = recv_starts_dbl[2] + size2;  
   
   size0 = loc_nnz_0 + (loc_row_0 + 1);
   size1 = loc_nnz_1 + (loc_row_1 + 1);
   size2 = loc_nnz_2 + (loc_row_2 + 1);
   recv_starts_int[0] = 0;
   recv_starts_int[1] = size0;
   recv_starts_int[2] = recv_starts_int[1] + size1;
   recv_starts_int[3] = recv_starts_int[2] + size2;
   
  
   //-------------------------------------------------------------------------------
   //  开设发送和接收缓存区空间 send_buf 和 recv_buf
   //-------------------------------------------------------------------------------   
   
   send_buf_dbl = jxf_CTAlloc(JXF_Real, send_starts_dbl[num_sends]);
   recv_buf_dbl = jxf_CTAlloc(JXF_Real, recv_starts_dbl[num_recvs]); 
   
   send_buf_int = jxf_CTAlloc(JXF_Int, send_starts_int[num_sends]);
   recv_buf_int = jxf_CTAlloc(JXF_Int, recv_starts_int[num_recvs]);   

  
   //-------------------------------------------
   //  填充发送缓存区空间 send_buf_dbl 
   //-------------------------------------------
   
   k = 0; 
   
   for (i = 0; i < local_nnz; i ++)
   {
      send_buf_dbl[k++] = aa_R[i]; 
   }
   for (i = 0; i < local_row; i ++)
   {
      send_buf_dbl[k++] = vre_data[i];
   }
   for (i = 0; i < local_row; i ++)
   {
      send_buf_dbl[k++] = fr_data[i];
   }
   for (i = 0; i < local_row; i ++)
   {
      send_buf_dbl[k++] = ur_data[i];
   }

   for (i = 0; i < local_nnz; i ++)
   {
      send_buf_dbl[k++] = aa_E[i]; 
   }
   for (i = 0; i < local_row; i ++)
   {
      send_buf_dbl[k++] = ver_data[i];
   }
   for (i = 0; i < local_row; i ++)
   {
      send_buf_dbl[k++] = vei_data[i];
   }
   for (i = 0; i < local_row; i ++)
   {
      send_buf_dbl[k++] = fe_data[i];
   }
   for (i = 0; i < local_row; i ++)
   {
      send_buf_dbl[k++] = ue_data[i];
   }
     
   for (i = 0; i < local_nnz; i ++)
   {
      send_buf_dbl[k++] = aa_I[i]; 
   }
   for (i = 0; i < local_row; i ++)
   {
      send_buf_dbl[k++] = vie_data[i];
   }
   for (i = 0; i < local_row; i ++)
   {
      send_buf_dbl[k++] = fi_data[i];
   }
   for (i = 0; i < local_row; i ++)
   {
      send_buf_dbl[k++] = ui_data[i];
   }
   //jxf_printf(" myid = %d send_buf_dbl_size = %d\n", myid, k);
   
    
   //-------------------------------------------------------------------
   //  填充发送缓存区空间 send_buf_int
   //--------------------------------------------------------------------
      
   k = 0; 
   
   for (i = 0; i < local_row+1; i ++)
   {
      send_buf_int[k++] = ia_R[i]; 
   }
   for (i = 0; i < local_nnz; i ++)
   {
      send_buf_int[k++] = ja_R[i]; 
   }
   
   for (i = 0; i < local_row+1; i ++)
   {
      send_buf_int[k++] = ia_E[i]; 
   }
   for (i = 0; i < local_nnz; i ++)
   {
      send_buf_int[k++] = ja_E[i]; 
   }
   
   for (i = 0; i < local_row+1; i ++)
   {
      send_buf_int[k++] = ia_I[i]; 
   }
   for (i = 0; i < local_nnz; i ++)
   {
      send_buf_int[k++] = ja_I[i]; 
   }
   //jxf_printf(" myid = %d send_buf_int_size = %d\n", myid, k);

    
   //-------------------------------------------------------
   //  创建通信包 comm_pkg_dbl，并填充其成员
   //-------------------------------------------------------
   
   comm_pkg_dbl = jxf_CTAlloc(jxf_ParVecCommPkg, 1);

   jxf_ParVecCommPkgComm(comm_pkg_dbl)       = comm;                                   
   jxf_ParVecCommPkgNumSends(comm_pkg_dbl)   = num_sends;
   jxf_ParVecCommPkgSendProcs(comm_pkg_dbl)  = send_procs_dbl;
   jxf_ParVecCommPkgSendStarts(comm_pkg_dbl) = send_starts_dbl;
   jxf_ParVecCommPkgNumRecvs(comm_pkg_dbl)   = num_recvs;
   jxf_ParVecCommPkgRecvProcs(comm_pkg_dbl)  = recv_procs_dbl;
   jxf_ParVecCommPkgRecvStarts(comm_pkg_dbl) = recv_starts_dbl; 

   //-------------------------------------------------------
   //  创建通信包 comm_pkg_int，并填充其成员
   //-------------------------------------------------------
   
   comm_pkg_int = jxf_CTAlloc(jxf_ParVecCommPkg, 1);

   jxf_ParVecCommPkgComm(comm_pkg_int)       = comm;                                   
   jxf_ParVecCommPkgNumSends(comm_pkg_int)   = num_sends;
   jxf_ParVecCommPkgSendProcs(comm_pkg_int)  = send_procs_int;
   jxf_ParVecCommPkgSendStarts(comm_pkg_int) = send_starts_int;
   jxf_ParVecCommPkgNumRecvs(comm_pkg_int)   = num_recvs;
   jxf_ParVecCommPkgRecvProcs(comm_pkg_int)  = recv_procs_int;
   jxf_ParVecCommPkgRecvStarts(comm_pkg_int) = recv_starts_int; 
   

   //-------------------------------------------------------
   //  通信: 发送和接收数据
   //-------------------------------------------------------
   
   comm_handle_dbl = jxf_ParVecCommHandleCreate(1, comm_pkg_dbl, send_buf_dbl, recv_buf_dbl);
   comm_handle_int = jxf_ParVecCommHandleCreate(11, comm_pkg_int, send_buf_int, recv_buf_int);
   
   /* ... some computational work can be done here to ovelap the communication! */
   
   jxf_ParVecCommHandleDestroy(comm_handle_dbl);
   jxf_ParVecCommHandleDestroy(comm_handle_int);


   //====================================================================================
   // Step 2: 将接收缓存区中的数据转化为本地串行矩阵和串行向量的形式，
   //         为进一步生成并行数据作准备.
   //====================================================================================

   num_rows = 0;
   num_nonzeros = 0;
   for (i = 0; i < num_recvs; i ++)
   {
      num_rows += local_row_array[recv_procs_int[i]];
      num_nonzeros += local_nnz_array[recv_procs_int[i]];
   }
   num_cols = local_col;

   A_loc = jxf_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
   jxf_CSRMatrixInitialize(A_loc);
   aa = jxf_CSRMatrixData(A_loc);
   ia = jxf_CSRMatrixI(A_loc);
   ja = jxf_CSRMatrixJ(A_loc);

   k = 0;       // 累加接收缓存区中的数据
   row_cnt = 0; // 累加矩阵 A 的行数 
   ja_cnt  = 0; // 累加 ja 数组中的元素

   loc_nnz_01 = loc_nnz_0 + loc_nnz_1;
   
   for (i = 0; i <= loc_row_0; i ++)
   {
      ia[row_cnt ++] = recv_buf_int[k ++];
   }
   for (i = 0; i < loc_nnz_0; i ++)
   {
      ja[ja_cnt ++] = recv_buf_int[k ++];
   }
   row_cnt --; // be careful here!
   
   for (i = 0; i <= loc_row_1; i ++)
   {
      ia[row_cnt ++] = recv_buf_int[k ++] + loc_nnz_0;
   }
   for (i = 0; i < loc_nnz_1; i ++)
   {
      ja[ja_cnt ++] = recv_buf_int[k ++];
   }
   row_cnt --; // be careful here!
   
   for (i = 0; i <= loc_row_2; i ++)
   {
      ia[row_cnt++] = recv_buf_int[k ++] + loc_nnz_01;
   }
   for (i = 0; i < loc_nnz_2; i ++)
   {
      ja[ja_cnt ++] = recv_buf_int[k ++];
   }
   
   /* 生成 aa 数组 */
   aa_cnt = 0;  // 累加 aa 数组中的元素
   
   start = recv_starts_dbl[0]; 
   for (i = 0; i < loc_nnz_0; i ++)
   {
      aa[aa_cnt ++] = recv_buf_dbl[start + i];
   }
   
   start = recv_starts_dbl[1]; 
   for (i = 0; i < loc_nnz_1; i ++)
   {
      aa[aa_cnt ++] = recv_buf_dbl[start + i];
   } 
   
   start = recv_starts_dbl[2]; 
   for (i = 0; i < loc_nnz_2; i ++)
   {
      aa[aa_cnt ++] = recv_buf_dbl[start + i];
   } 

   //jxf_printf(" myid = %d aa_cnt = %d num_nonzeros = %d\n", myid, aa_cnt, num_nonzeros);  

   f_loc = jxf_SeqVectorCreate(num_rows);
   u_loc = jxf_SeqVectorCreate(num_rows);
   jxf_SeqVectorInitialize(f_loc);
   jxf_SeqVectorInitialize(u_loc);
      
   if (groupid_x == 0) // 光子进程组
   {
      VRE_loc = jxf_SeqVectorCreate(num_rows);
      jxf_SeqVectorInitialize(VRE_loc);
      
      vre_cnt = 0;
      f_cnt  = 0;
      u_cnt  = 0;
      
      start = recv_starts_dbl[0] + loc_nnz_0; 
      for (i = 0; i < loc_row_0; i ++)
      {
         jxf_VectorData(VRE_loc)[vre_cnt ++] = recv_buf_dbl[start + i];
      }
      start += loc_row_0;
      for (i = 0; i < loc_row_0; i ++)
      {
         jxf_VectorData(f_loc)[f_cnt ++] = recv_buf_dbl[start + i];
      }  
      start += loc_row_0;
      for (i = 0; i < loc_row_0; i ++)
      {
         jxf_VectorData(u_loc)[u_cnt ++] = recv_buf_dbl[start + i];
      }
      
      start = recv_starts_dbl[1] + loc_nnz_1; 
      for (i = 0; i < loc_row_1; i ++)
      {
         jxf_VectorData(VRE_loc)[vre_cnt ++] = recv_buf_dbl[start + i];
      }
      start += loc_row_1;
      for (i = 0; i < loc_row_1; i ++)
      {
         jxf_VectorData(f_loc)[f_cnt ++] = recv_buf_dbl[start + i];
      }  
      start += loc_row_1;
      for (i = 0; i < loc_row_1; i ++)
      {
         jxf_VectorData(u_loc)[u_cnt ++] = recv_buf_dbl[start + i];
      }
      
      start = recv_starts_dbl[2] + loc_nnz_2; 
      for (i = 0; i < loc_row_2; i ++)
      {
         jxf_VectorData(VRE_loc)[vre_cnt ++] = recv_buf_dbl[start + i];
      }
      start += loc_row_2;
      for (i = 0; i < loc_row_2; i ++)
      {
         jxf_VectorData(f_loc)[f_cnt ++] = recv_buf_dbl[start + i];
      }  
      start += loc_row_2;
      for (i = 0; i < loc_row_2; i ++)
      {
         jxf_VectorData(u_loc)[u_cnt ++] = recv_buf_dbl[start + i];
      }                           
   }
   else if (groupid_x == 1) // 电子进程组
   {
      VER_loc = jxf_SeqVectorCreate(num_rows);
      VEI_loc = jxf_SeqVectorCreate(num_rows);
      jxf_SeqVectorInitialize(VER_loc);
      jxf_SeqVectorInitialize(VEI_loc);
      
      ver_cnt = 0;
      vei_cnt = 0;
      f_cnt  = 0;
      u_cnt  = 0;
      
      start = recv_starts_dbl[0] + loc_nnz_0; 
      for (i = 0; i < loc_row_0; i ++)
      {
         jxf_VectorData(VER_loc)[ver_cnt ++] = recv_buf_dbl[start + i];
      }
      start += loc_row_0;
      for (i = 0; i < loc_row_0; i ++)
      {
         jxf_VectorData(VEI_loc)[vei_cnt ++] = recv_buf_dbl[start + i];
      }        
      start += loc_row_0;
      for (i = 0; i < loc_row_0; i ++)
      {
         jxf_VectorData(f_loc)[f_cnt ++] = recv_buf_dbl[start + i];
      }  
      start += loc_row_0;
      for (i = 0; i < loc_row_0; i ++)
      {
         jxf_VectorData(u_loc)[u_cnt ++] = recv_buf_dbl[start + i];
      }
      
      start = recv_starts_dbl[1] + loc_nnz_1; 
      for (i = 0; i < loc_row_1; i ++)
      {
         jxf_VectorData(VER_loc)[ver_cnt ++] = recv_buf_dbl[start + i];
      }
      start += loc_row_1;
      for (i = 0; i < loc_row_1; i ++)
      {
         jxf_VectorData(VEI_loc)[vei_cnt ++] = recv_buf_dbl[start + i];
      }        
      start += loc_row_1;
      for (i = 0; i < loc_row_1; i ++)
      {
         jxf_VectorData(f_loc)[f_cnt ++] = recv_buf_dbl[start + i];
      }  
      start += loc_row_1;
      for (i = 0; i < loc_row_1; i ++)
      {
         jxf_VectorData(u_loc)[u_cnt ++] = recv_buf_dbl[start + i];
      }
      
      start = recv_starts_dbl[2] + loc_nnz_2; 
      for (i = 0; i < loc_row_2; i ++)
      {
         jxf_VectorData(VER_loc)[ver_cnt ++] = recv_buf_dbl[start + i];
      }
      start += loc_row_2;
      for (i = 0; i < loc_row_2; i ++)
      {
         jxf_VectorData(VEI_loc)[vei_cnt ++] = recv_buf_dbl[start + i];
      }       
      start += loc_row_2;
      for (i = 0; i < loc_row_2; i ++)
      {
         jxf_VectorData(f_loc)[f_cnt ++] = recv_buf_dbl[start + i];
      }  
      start += loc_row_2;
      for (i = 0; i < loc_row_2; i ++)
      {
         jxf_VectorData(u_loc)[u_cnt ++] = recv_buf_dbl[start + i];
      }     
   }
   else if (groupid_x == 2) // 离子进程组
   {
      VIE_loc = jxf_SeqVectorCreate(num_rows);
      jxf_SeqVectorInitialize(VIE_loc);
      
      vie_cnt = 0;
      f_cnt  = 0;
      u_cnt  = 0;
      
      start = recv_starts_dbl[0] + loc_nnz_0; 
      for (i = 0; i < loc_row_0; i ++)
      {
         jxf_VectorData(VIE_loc)[vie_cnt ++] = recv_buf_dbl[start + i];
      }
      start += loc_row_0;
      for (i = 0; i < loc_row_0; i ++)
      {
         jxf_VectorData(f_loc)[f_cnt ++] = recv_buf_dbl[start + i];
      }  
      start += loc_row_0;
      for (i = 0; i < loc_row_0; i ++)
      {
         jxf_VectorData(u_loc)[u_cnt ++] = recv_buf_dbl[start + i];
      }
      
      start = recv_starts_dbl[1] + loc_nnz_1; 
      for (i = 0; i < loc_row_1; i ++)
      {
         jxf_VectorData(VIE_loc)[vie_cnt ++] = recv_buf_dbl[start + i];
      }
      start += loc_row_1;
      for (i = 0; i < loc_row_1; i ++)
      {
         jxf_VectorData(f_loc)[f_cnt ++] = recv_buf_dbl[start + i];
      }  
      start += loc_row_1;
      for (i = 0; i < loc_row_1; i ++)
      {
         jxf_VectorData(u_loc)[u_cnt ++] = recv_buf_dbl[start + i];
      }
      
      start = recv_starts_dbl[2] + loc_nnz_2; 
      for (i = 0; i < loc_row_2; i ++)
      {
         jxf_VectorData(VIE_loc)[vie_cnt ++] = recv_buf_dbl[start + i];
      }
      start += loc_row_2;
      for (i = 0; i < loc_row_2; i ++)
      {
         jxf_VectorData(f_loc)[f_cnt ++] = recv_buf_dbl[start + i];
      }  
      start += loc_row_2;
      for (i = 0; i < loc_row_2; i ++)
      {
         jxf_VectorData(u_loc)[u_cnt ++] = recv_buf_dbl[start + i];
      }            
   }

   row_starts_loc = jxf_CTAlloc(JXF_Int, num_proc_eachgroup + 1);     
   col_starts_loc = row_starts_loc;
   for (i = 0; i <= num_proc_eachgroup; i ++)
   {
      row_starts_loc[i] = row_starts[3*i];
   } 

      
   if (groupid_x == 0) // 光子进程组
   {
      ARR = jxf_ParCSRMatrixCreate(comm_x, n, n, row_starts_loc, col_starts_loc, 0, 0, 0);
      first_col_diag = col_starts_loc[myid_x];
      last_col_diag  = col_starts_loc[myid_x+1] - 1;
      jxf_GenerateDiagAndOffd(A_loc, ARR, first_col_diag, last_col_diag);
      
      VRE = jxf_ParVectorCreate(comm_x, n, row_starts_loc);
      jxf_ParVectorSetPartitioningOwner(VRE, 0);
      jxf_TFree(jxf_ParVectorLocalVector(VRE));
      jxf_ParVectorLocalVector(VRE) = VRE_loc;                           
   }
   else if (groupid_x == 1) // 电子进程组
   {
      AEE = jxf_ParCSRMatrixCreate(comm_x, n, n, row_starts_loc, col_starts_loc, 0, 0, 0);
      first_col_diag = col_starts_loc[myid_x];
      last_col_diag  = col_starts_loc[myid_x+1] - 1;
      jxf_GenerateDiagAndOffd(A_loc, AEE, first_col_diag, last_col_diag);   
          
      VER = jxf_ParVectorCreate(comm_x, n, row_starts_loc);
      jxf_ParVectorSetPartitioningOwner(VER, 0);
      jxf_TFree(jxf_ParVectorLocalVector(VER));
      jxf_ParVectorLocalVector(VER) = VER_loc;
   
      VEI = jxf_ParVectorCreate(comm_x, n, row_starts_loc);
      jxf_ParVectorSetPartitioningOwner(VEI, 0);
      jxf_TFree(jxf_ParVectorLocalVector(VEI));
      jxf_ParVectorLocalVector(VEI) = VEI_loc;
   }
   else if (groupid_x == 2) // 离子进程组
   {
      AII = jxf_ParCSRMatrixCreate(comm_x, n, n, row_starts_loc, col_starts_loc, 0, 0, 0);
      first_col_diag = col_starts_loc[myid_x];
      last_col_diag  = col_starts_loc[myid_x+1] - 1;
      jxf_GenerateDiagAndOffd(A_loc, AII, first_col_diag, last_col_diag);

      VIE = jxf_ParVectorCreate(comm_x, n, row_starts_loc);
      jxf_ParVectorSetPartitioningOwner(VIE, 0);
      jxf_TFree(jxf_ParVectorLocalVector(VIE));
      jxf_ParVectorLocalVector(VIE) = VIE_loc;               
   }


   row_starts_glo = jxf_CTAlloc(JXF_Int, nprocs + 1);     
   col_starts_glo = row_starts_glo;
   for (i = 0; i < num_proc_eachgroup; i ++)
   {
      row_starts_glo[i] = row_starts_loc[i];
      row_starts_glo[i + num_proc_eachgroup]  = row_starts_loc[i] + n;
      row_starts_glo[i + num_proc_eachgroup2] = row_starts_loc[i] + n2;
   } 
   row_starts_glo[nprocs] = N;


   if (groupid_x == 1)
   {
      num_nonzeros = jxf_CSRMatrixNumNonzeros(A_loc) + 2*num_rows;
   }
   else
   {
      num_nonzeros = jxf_CSRMatrixNumNonzeros(A_loc) + num_rows;
   }
   A_rec = jxf_CSRMatrixCreate(num_rows, N, num_nonzeros);
   jxf_CSRMatrixInitialize(A_rec);
   aarec = jxf_CSRMatrixData(A_rec);
   iarec = jxf_CSRMatrixI(A_rec);
   jarec = jxf_CSRMatrixJ(A_rec);
   
   if (groupid_x == 0) // 光子进程组
   {
      k = 0;
      iarec[0] = 0;
      start = row_starts_glo[myid];
      for (i = 0; i < num_rows; i ++)
      {
         for (j = ia[i]; j < ia[i+1]; j ++)
         {
            jarec[k] = ja[j];
            aarec[k] = aa[j];
            k ++;
         }
         jarec[k] = start + i + n;
         aarec[k] = jxf_VectorData(VRE_loc)[i];
         k ++;
         
         iarec[i + 1] = k;
      }
   }
   else if (groupid_x == 1) // 电子进程组
   {
      k = 0;
      iarec[0] = 0;
      start = row_starts_glo[myid];
      for (i = 0; i < num_rows; i ++)
      {
         for (j = ia[i]; j < ia[i+1]; j ++)
         {
            jarec[k] = ja[j] + n;
            aarec[k] = aa[j];
            k ++;
         }
         jarec[k] = start + i - n;
         aarec[k] = jxf_VectorData(VER_loc)[i];
         k ++;
         
         jarec[k] = start + i + n;
         aarec[k] = jxf_VectorData(VEI_loc)[i];
         k ++;
         
         iarec[i + 1] = k;
      }
   }
   else if (groupid_x == 2) // 离子进程组
   {
      k = 0;
      iarec[0] = 0;
      start = row_starts_glo[myid];
      for (i = 0; i < num_rows; i ++)
      {
         for (j = ia[i]; j < ia[i+1]; j ++)
         {
            jarec[k] = ja[j] + n2;
            aarec[k] = aa[j];
            k ++;
         }
         jarec[k] = start + i - n;
         aarec[k] = jxf_VectorData(VIE_loc)[i];
         k ++;
         
         iarec[i + 1] = k;
      }
   }
    
   A = jxf_ParCSRMatrixCreate(comm, N, N, row_starts_glo, col_starts_glo, 0, 0, 0); 
   first_col_diag = col_starts_glo[myid];
   last_col_diag  = col_starts_glo[myid+1] - 1;
   jxf_GenerateDiagAndOffd(A_rec, A, first_col_diag, last_col_diag);

   f = jxf_ParVectorCreate(comm, N, row_starts_glo);
   jxf_ParVectorSetPartitioningOwner(f, 0);
   jxf_TFree(jxf_ParVectorLocalVector(f));
   jxf_ParVectorLocalVector(f) = f_loc;   

   u = jxf_ParVectorCreate(comm, N, row_starts_glo);
   jxf_ParVectorSetPartitioningOwner(u, 0);
   jxf_TFree(jxf_ParVectorLocalVector(u));
   jxf_ParVectorLocalVector(u) = u_loc; 
    
   jxf_TFree(send_buf_dbl);
   jxf_TFree(recv_buf_dbl);
   jxf_TFree(send_buf_int);
   jxf_TFree(recv_buf_int);
   jxf_TFree(local_nnz_array);
   jxf_TFree(local_row_array);
   
   jxf_ParVecCommPkgDestroy(comm_pkg_dbl);
   jxf_ParVecCommPkgDestroy(comm_pkg_int);
   
   jxf_CSRMatrixDestroy(ARR_comb);
   jxf_CSRMatrixDestroy(AEE_comb);
   jxf_CSRMatrixDestroy(AII_comb);
   
   jxf_CSRMatrixDestroy(A_loc);
   jxf_CSRMatrixDestroy(A_rec);
      
   *ARR_ptr = ARR;
   *AEE_ptr = AEE;
   *AII_ptr = AII;
   *VRE_ptr = VRE; 
   *VER_ptr = VER; 
   *VEI_ptr = VEI; 
   *VIE_ptr = VIE;  
   
   *A_ptr = A;  
   *f_ptr = f;
   *u_ptr = u;     
      
   return (0);
}

JXF_Int 
jxf_ParaDataTrans4ApctlmgKrylov( MPI_Comm           comm,
                                MPI_Comm           comm_x,
                                JXF_Int                groupid_x,
                                jxf_ParCSRMatrix  **ARR_p,
                                jxf_ParCSRMatrix   *AEE_p,
                                jxf_ParCSRMatrix   *AII_p,
                                jxf_ParVector     **VRE_p,
                                jxf_ParVector     **VER_p,
                                jxf_ParVector      *VEI_p,
                                jxf_ParVector      *VIE_p,
                                jxf_ParVector     **fR_p,
                                jxf_ParVector      *fE_p,
                                jxf_ParVector      *fI_p,
                                jxf_ParVector     **uR_p,
                                jxf_ParVector      *uE_p,
                                jxf_ParVector      *uI_p,
                                JXF_Int             ng,
                                jxf_ParCSRMatrix  **ARR_ptr,
                                jxf_ParCSRMatrix  **AEE_ptr,
                                jxf_ParCSRMatrix  **AII_ptr,
                                jxf_ParVector     **VRE_ptr,
                                jxf_ParVector    ***VER_ptr,
                                jxf_ParVector     **VEI_ptr,
                                jxf_ParVector     **VIE_ptr,
                                jxf_ParCSRMatrix  **A_ptr,
                                jxf_ParVector     **f_ptr,
                                jxf_ParVector     **u_ptr )
{
   jxf_ParCSRMatrix  *ARR = NULL;
   jxf_ParCSRMatrix  *AEE = NULL;
   jxf_ParCSRMatrix  *AII = NULL;
   jxf_ParVector     *VRE = NULL;
   jxf_ParVector    **VER = NULL;
   jxf_ParVector     *VEI = NULL;
   jxf_ParVector     *VIE = NULL;

   jxf_ParCSRMatrix  *A = NULL;
   jxf_ParVector     *f = NULL;
   jxf_ParVector     *u = NULL;

   jxf_Vector  *VRE_loc = NULL;
   jxf_Vector **VER_loc = NULL;
   jxf_Vector  *VEI_loc = NULL;
   jxf_Vector  *VIE_loc = NULL;

   jxf_Vector *f_loc = NULL;
   jxf_Vector *u_loc = NULL;

   JXF_Int *row_starts_loc = NULL;
   JXF_Int *col_starts_loc = NULL;

   JXF_Int *row_starts_glo = NULL;
   JXF_Int *col_starts_glo = NULL;

   JXF_Int *row_starts = jxf_ParCSRMatrixRowStarts(ARR_p[0]);

   JXF_Int n = jxf_ParCSRMatrixGlobalNumRows(ARR_p[0]);
   JXF_Int N = (ng + 2) * n;
   JXF_Int e_pos = ng * n;
   JXF_Int i_pos = e_pos + n;

   JXF_Int first_col_diag;
   JXF_Int last_col_diag;

   JXF_Int myid, myid_x, nprocs;

   jxf_CSRMatrix **ARR_comb = jxf_CTAlloc(jxf_CSRMatrix *, ng);
   JXF_Real *aa_R = NULL;
   JXF_Int    *ia_R = NULL;
   JXF_Int    *ja_R = NULL;

   jxf_CSRMatrix *AEE_comb = NULL;
   JXF_Real *aa_E = NULL;
   JXF_Int    *ia_E = NULL;
   JXF_Int    *ja_E = NULL;

   jxf_CSRMatrix *AII_comb = NULL;
   JXF_Real *aa_I = NULL;
   JXF_Int    *ia_I = NULL;
   JXF_Int    *ja_I = NULL;

   JXF_Real *vre_data = NULL;
   JXF_Real *ver_data = NULL;
   JXF_Real *vei_data = jxf_VectorData(jxf_ParVectorLocalVector(VEI_p));
   JXF_Real *vie_data = jxf_VectorData(jxf_ParVectorLocalVector(VIE_p));

   JXF_Real *fr_data = NULL;
   JXF_Real *fe_data = jxf_VectorData(jxf_ParVectorLocalVector(fE_p));
   JXF_Real *fi_data = jxf_VectorData(jxf_ParVectorLocalVector(fI_p));

   JXF_Real *ur_data = NULL;
   JXF_Real *ue_data = jxf_VectorData(jxf_ParVectorLocalVector(uE_p));
   JXF_Real *ui_data = jxf_VectorData(jxf_ParVectorLocalVector(uI_p));

   jxf_CSRMatrix *A_loc = NULL;
   JXF_Int num_rows, num_nonzeros;
   JXF_Real *aa = NULL;
   JXF_Int    *ia = NULL;
   JXF_Int    *ja = NULL;

   jxf_CSRMatrix *A_rec = NULL;
   JXF_Real *aarec = NULL;
   JXF_Int    *iarec = NULL;
   JXF_Int    *jarec = NULL;

   JXF_Int  num_sends;                 // 发送进程个数
   JXF_Int *send_procs_dbl  = NULL;    // 发送进程编号
   JXF_Int *send_procs_int  = NULL;    // 发送进程编号
   JXF_Int *send_starts_dbl = NULL;    // 实型数据发送缓存区的管理数组
   JXF_Int *send_starts_int = NULL;    // 整型发送缓存区的管理数组

   JXF_Int  num_recvs;                 // 接收进程个数
   JXF_Int *recv_procs_dbl  = NULL;    // 接收进程编号
   JXF_Int *recv_procs_int  = NULL;    // 接收进程编号
   JXF_Int *recv_starts_dbl = NULL;    // 实型数据接收缓存区的管理数组
   JXF_Int *recv_starts_int = NULL;    // 整型数据接收缓存区的管理数组

   JXF_Real *send_buf_dbl = NULL;
   JXF_Real *recv_buf_dbl = NULL;

   JXF_Int *send_buf_int = NULL;
   JXF_Int *recv_buf_int = NULL;

   jxf_ParVecCommPkg    *comm_pkg_dbl    = NULL;
   jxf_ParVecCommPkg    *comm_pkg_int    = NULL;
   jxf_ParVecCommHandle *comm_handle_dbl = NULL;
   jxf_ParVecCommHandle *comm_handle_int = NULL;

   JXF_Int num_proc_eachgroup;
   JXF_Int local_row;                   // 行数
   JXF_Int local_nnz;                   // 非零元素个数

   JXF_Int *local_nnz_array = NULL;
   JXF_Int *local_row_array = NULL;

   JXF_Int *size_recv = NULL;
   JXF_Int *ver_cnt = NULL;

   JXF_Int size_base_int;
   JXF_Int size0, size1;

   JXF_Int i, j, k, gidx, incmt;
   JXF_Int row_cnt, ja_cnt, aa_cnt;
   JXF_Int f_cnt, u_cnt;
   JXF_Int vre_cnt, vei_cnt, vie_cnt;
   JXF_Int start, scaa, scbb;

   jxf_MPI_Comm_rank(comm, &myid);
   jxf_MPI_Comm_rank(comm_x, &myid_x);
   jxf_MPI_Comm_size(comm, &nprocs);

   //====================================================================================
   // Step 1: 数据通信
   //====================================================================================

   for (gidx = 0; gidx < ng; gidx ++) ARR_comb[gidx] = jxf_MergeDiagAndOffd(ARR_p[gidx]);
   AEE_comb = jxf_MergeDiagAndOffd(AEE_p);
   AII_comb = jxf_MergeDiagAndOffd(AII_p);

   local_nnz = jxf_CSRMatrixNumNonzeros(ARR_comb[0]);
   local_row = jxf_CSRMatrixNumRows(ARR_comb[0]);

   aa_E = jxf_CSRMatrixData(AEE_comb);
   ia_E = jxf_CSRMatrixI(AEE_comb);
   ja_E = jxf_CSRMatrixJ(AEE_comb);

   aa_I = jxf_CSRMatrixData(AII_comb);
   ia_I = jxf_CSRMatrixI(AII_comb);
   ja_I = jxf_CSRMatrixJ(AII_comb);

   num_proc_eachgroup = nprocs / (ng + 2);

   /* num_sends */
   num_sends = ng + 2;

   send_procs_dbl  = jxf_CTAlloc(JXF_Int, num_sends);
   send_procs_int  = jxf_CTAlloc(JXF_Int, num_sends);
   send_starts_dbl = jxf_CTAlloc(JXF_Int, num_sends+1);
   send_starts_int = jxf_CTAlloc(JXF_Int, num_sends+1);

   /* send_procs_dbl */
   send_procs_dbl[0] = myid / num_sends;
   for (gidx = 1; gidx < num_sends; gidx ++) send_procs_dbl[gidx] = send_procs_dbl[gidx-1] + num_proc_eachgroup;

   /* send_procs_int */
   for (gidx = 0; gidx < num_sends; gidx ++) send_procs_int[gidx] = send_procs_dbl[gidx];

   /* send_starts_dbl */
   size0 = local_nnz + 3 * local_row; // aa, vre, f, u
   size1 = size0 + ng * local_row; // aa, ver, vei, f, u
   send_starts_dbl[0] = 0;
   for (gidx = 0; gidx < ng; gidx ++) send_starts_dbl[gidx+1] = send_starts_dbl[gidx] + size0;
   send_starts_dbl[ng+1] = send_starts_dbl[ng] + size1;
   send_starts_dbl[ng+2] = send_starts_dbl[ng+1] + size0;

   /* send_starts_int */
   size_base_int = local_nnz + local_row + 1; // ia, ja
   send_starts_int[0] = 0;
   for (gidx = 0; gidx < num_sends; gidx ++) send_starts_int[gidx+1] = send_starts_int[gidx] + size_base_int;

   /* num_recvs */
   num_recvs = ng + 2;

   recv_procs_dbl  = jxf_CTAlloc(JXF_Int, num_recvs);
   recv_procs_int  = jxf_CTAlloc(JXF_Int, num_recvs);
   recv_starts_dbl = jxf_CTAlloc(JXF_Int, num_recvs+1);
   recv_starts_int = jxf_CTAlloc(JXF_Int, num_recvs+1);

   /* recv_procs_dbl */
   recv_procs_dbl[0] = (myid % num_proc_eachgroup) * num_recvs;
   for (gidx = 1; gidx < num_recvs; gidx ++) recv_procs_dbl[gidx] = recv_procs_dbl[gidx-1] + 1;

   /* recv_procs_int */
   for (gidx = 0; gidx < num_recvs; gidx ++) recv_procs_int[gidx] = recv_procs_dbl[gidx];

   /* 消息全收集 */
   local_nnz_array = jxf_CTAlloc(JXF_Int, nprocs);
   local_row_array = jxf_CTAlloc(JXF_Int, nprocs);
   jxf_MPI_Allgather(&local_nnz, 1, JXF_MPI_INT, local_nnz_array, 1, JXF_MPI_INT, comm);
   jxf_MPI_Allgather(&local_row, 1, JXF_MPI_INT, local_row_array, 1, JXF_MPI_INT, comm);

   /* recv_starts_dbl */
   size_recv = jxf_CTAlloc(JXF_Int, num_recvs);
   if (groupid_x == ng)
   {
      for (gidx = 0; gidx < num_recvs; gidx ++)
      {
         size_recv[gidx] = local_nnz_array[recv_procs_dbl[gidx]] + (ng + 3) * local_row_array[recv_procs_dbl[gidx]];
      }
   }
   else
   {
      for (gidx = 0; gidx < num_recvs; gidx ++)
      {
         size_recv[gidx] = local_nnz_array[recv_procs_dbl[gidx]] + 3 * local_row_array[recv_procs_dbl[gidx]];
      }
   }
   recv_starts_dbl[0] = 0;
   for (gidx = 0; gidx < num_recvs; gidx ++) recv_starts_dbl[gidx+1] = recv_starts_dbl[gidx] + size_recv[gidx];

   /* recv_starts_int */
   for (gidx = 0; gidx < num_recvs; gidx ++)
   {
      size_recv[gidx] = local_nnz_array[recv_procs_dbl[gidx]] + local_row_array[recv_procs_dbl[gidx]] + 1;
   }
   recv_starts_int[0] = 0;
   for (gidx = 0; gidx < num_recvs; gidx ++) recv_starts_int[gidx+1] = recv_starts_int[gidx] + size_recv[gidx];

   //-------------------------------------------------------------------------------
   //  开设发送和接收缓存区空间 send_buf 和 recv_buf
   //-------------------------------------------------------------------------------

   send_buf_dbl = jxf_CTAlloc(JXF_Real, send_starts_dbl[num_sends]);
   recv_buf_dbl = jxf_CTAlloc(JXF_Real, recv_starts_dbl[num_recvs]);
   send_buf_int = jxf_CTAlloc(JXF_Int, send_starts_int[num_sends]);
   recv_buf_int = jxf_CTAlloc(JXF_Int, recv_starts_int[num_recvs]);

   //-------------------------------------------
   //  填充发送缓存区空间 send_buf_dbl
   //-------------------------------------------

   k = 0;
   for (gidx = 0; gidx < ng; gidx ++)
   {
      aa_R = jxf_CSRMatrixData(ARR_comb[gidx]);
      vre_data = jxf_VectorData(jxf_ParVectorLocalVector(VRE_p[gidx]));
      fr_data = jxf_VectorData(jxf_ParVectorLocalVector(fR_p[gidx]));
      ur_data = jxf_VectorData(jxf_ParVectorLocalVector(uR_p[gidx]));
      for (i = 0; i < local_nnz; i ++)
      {
         send_buf_dbl[k++] = aa_R[i];
      }
      for (i = 0; i < local_row; i ++)
      {
         send_buf_dbl[k++] = vre_data[i];
      }
      for (i = 0; i < local_row; i ++)
      {
         send_buf_dbl[k++] = fr_data[i];
      }
      for (i = 0; i < local_row; i ++)
      {
         send_buf_dbl[k++] = ur_data[i];
      }
   }
   for (i = 0; i < local_nnz; i ++)
   {
      send_buf_dbl[k++] = aa_E[i];
   }
   for (gidx = 0; gidx < ng; gidx ++)
   {
      ver_data = jxf_VectorData(jxf_ParVectorLocalVector(VER_p[gidx]));
      for (i = 0; i < local_row; i ++)
      {
         send_buf_dbl[k++] = ver_data[i];
      }
   }
   for (i = 0; i < local_row; i ++)
   {
      send_buf_dbl[k++] = vei_data[i];
   }
   for (i = 0; i < local_row; i ++)
   {
      send_buf_dbl[k++] = fe_data[i];
   }
   for (i = 0; i < local_row; i ++)
   {
      send_buf_dbl[k++] = ue_data[i];
   }
   for (i = 0; i < local_nnz; i ++)
   {
      send_buf_dbl[k++] = aa_I[i];
   }
   for (i = 0; i < local_row; i ++)
   {
      send_buf_dbl[k++] = vie_data[i];
   }
   for (i = 0; i < local_row; i ++)
   {
      send_buf_dbl[k++] = fi_data[i];
   }
   for (i = 0; i < local_row; i ++)
   {
      send_buf_dbl[k++] = ui_data[i];
   }

   //-------------------------------------------------------------------
   //  填充发送缓存区空间 send_buf_int
   //-------------------------------------------------------------------

   k = 0;
   for (gidx = 0; gidx < ng; gidx ++)
   {
      ia_R = jxf_CSRMatrixI(ARR_comb[gidx]);
      ja_R = jxf_CSRMatrixJ(ARR_comb[gidx]);
      for (i = 0; i < local_row; i ++)
      {
         send_buf_int[k++] = ia_R[i];
      }
      send_buf_int[k++] = ia_R[local_row];
      for (i = 0; i < local_nnz; i ++)
      {
         send_buf_int[k++] = ja_R[i];
      }
   }
   for (i = 0; i < local_row; i ++)
   {
      send_buf_int[k++] = ia_E[i];
   }
   send_buf_int[k++] = ia_E[local_row];
   for (i = 0; i < local_nnz; i ++)
   {
      send_buf_int[k++] = ja_E[i];
   }
   for (i = 0; i < local_row; i ++)
   {
      send_buf_int[k++] = ia_I[i];
   }
   send_buf_int[k++] = ia_I[local_row];
   for (i = 0; i < local_nnz; i ++)
   {
      send_buf_int[k++] = ja_I[i];
   }

   //-------------------------------------------------------
   //  创建通信包 comm_pkg_dbl，并填充其成员
   //-------------------------------------------------------

   comm_pkg_dbl = jxf_CTAlloc(jxf_ParVecCommPkg, 1);
   jxf_ParVecCommPkgComm(comm_pkg_dbl)       = comm;
   jxf_ParVecCommPkgNumSends(comm_pkg_dbl)   = num_sends;
   jxf_ParVecCommPkgSendProcs(comm_pkg_dbl)  = send_procs_dbl;
   jxf_ParVecCommPkgSendStarts(comm_pkg_dbl) = send_starts_dbl;
   jxf_ParVecCommPkgNumRecvs(comm_pkg_dbl)   = num_recvs;
   jxf_ParVecCommPkgRecvProcs(comm_pkg_dbl)  = recv_procs_dbl;
   jxf_ParVecCommPkgRecvStarts(comm_pkg_dbl) = recv_starts_dbl;

   //-------------------------------------------------------
   //  创建通信包 comm_pkg_int，并填充其成员
   //-------------------------------------------------------

   comm_pkg_int = jxf_CTAlloc(jxf_ParVecCommPkg, 1);
   jxf_ParVecCommPkgComm(comm_pkg_int)       = comm;
   jxf_ParVecCommPkgNumSends(comm_pkg_int)   = num_sends;
   jxf_ParVecCommPkgSendProcs(comm_pkg_int)  = send_procs_int;
   jxf_ParVecCommPkgSendStarts(comm_pkg_int) = send_starts_int;
   jxf_ParVecCommPkgNumRecvs(comm_pkg_int)   = num_recvs;
   jxf_ParVecCommPkgRecvProcs(comm_pkg_int)  = recv_procs_int;
   jxf_ParVecCommPkgRecvStarts(comm_pkg_int) = recv_starts_int;

   //-------------------------------------------------------
   //  通信: 发送和接收数据
   //-------------------------------------------------------
 
   comm_handle_dbl = jxf_ParVecCommHandleCreate(1, comm_pkg_dbl, send_buf_dbl, recv_buf_dbl);
   comm_handle_int = jxf_ParVecCommHandleCreate(11, comm_pkg_int, send_buf_int, recv_buf_int);

   /* some computational work can be done here to ovelap the communication! */
 
   jxf_ParVecCommHandleDestroy(comm_handle_dbl);
   jxf_ParVecCommHandleDestroy(comm_handle_int);

   //====================================================================================
   // Step 2: 将接收缓存区中的数据转化为本地串行矩阵和串行向量的形式,
   //         为进一步生成并行数据作准备.
   //====================================================================================

   num_rows = 0;
   num_nonzeros = 0;
   for (i = 0; i < num_recvs; i ++)
   {
      num_rows += local_row_array[recv_procs_int[i]];
      num_nonzeros += local_nnz_array[recv_procs_int[i]];
   }

   A_loc = jxf_CSRMatrixCreate(num_rows, jxf_CSRMatrixNumCols(ARR_comb[0]), num_nonzeros);
   jxf_CSRMatrixInitialize(A_loc);
   aa = jxf_CSRMatrixData(A_loc);
   ia = jxf_CSRMatrixI(A_loc);
   ja = jxf_CSRMatrixJ(A_loc);

   k = 0;       // 累加接收缓存区中的数据
   row_cnt = 0; // 累加矩阵 A 的行数
   ja_cnt  = 0; // 累加 ja 数组中的元素
   aa_cnt  = 0;  // 累加 aa 数组中的元素
   incmt   = 0;
   for (gidx = 0; gidx < num_recvs; gidx ++)
   {
      scaa = local_row_array[recv_procs_dbl[gidx]] + 1;
      for (i = 0; i < scaa; i ++)
      {
         ia[row_cnt++] = recv_buf_int[k++] + incmt;
      }
      scbb = local_nnz_array[recv_procs_dbl[gidx]];
      for (i = 0; i < scbb; i ++)
      {
         ja[ja_cnt++] = recv_buf_int[k++];
      }
      incmt += scbb;
      row_cnt --; // be careful here!
      start = recv_starts_dbl[gidx];
      for (i = 0; i < scbb; i ++)
      {
         aa[aa_cnt++] = recv_buf_dbl[start+i];
      }
   }

   f_loc = jxf_SeqVectorCreate(num_rows);
   jxf_SeqVectorInitialize(f_loc);
   u_loc = jxf_SeqVectorCreate(num_rows);
   jxf_SeqVectorInitialize(u_loc);

   if (groupid_x < ng) // 光子进程组
   {
      VRE_loc = jxf_SeqVectorCreate(num_rows);
      jxf_SeqVectorInitialize(VRE_loc);
      vre_cnt = 0;
      f_cnt   = 0;
      u_cnt   = 0;
      for (gidx = 0; gidx < num_recvs; gidx ++)
      {
         start = recv_starts_dbl[gidx] + local_nnz_array[recv_procs_dbl[gidx]];
         scaa = local_row_array[recv_procs_dbl[gidx]];
         for (i = 0; i < scaa; i ++)
         {
            jxf_VectorData(VRE_loc)[vre_cnt++] = recv_buf_dbl[start+i];
         }
         start += scaa;
         for (i = 0; i < scaa; i ++)
         {
            jxf_VectorData(f_loc)[f_cnt++] = recv_buf_dbl[start+i];
         }
         start += scaa;
         for (i = 0; i < scaa; i ++)
         {
            jxf_VectorData(u_loc)[u_cnt++] = recv_buf_dbl[start+i];
         }
      }
   }
   else if (groupid_x == ng) // 电子进程组
   {
      VER_loc = jxf_CTAlloc(jxf_Vector *, ng);
      for (gidx = 0; gidx < ng; gidx ++)
      {
         VER_loc[gidx] = jxf_SeqVectorCreate(num_rows);
         jxf_SeqVectorInitialize(VER_loc[gidx]);
      }
      VEI_loc = jxf_SeqVectorCreate(num_rows);
      jxf_SeqVectorInitialize(VEI_loc);
      ver_cnt = jxf_CTAlloc(JXF_Int, ng);
      vei_cnt = 0;
      f_cnt   = 0;
      u_cnt   = 0;
      for (gidx = 0; gidx < num_recvs; gidx ++)
      {
         start = recv_starts_dbl[gidx] + local_nnz_array[recv_procs_dbl[gidx]];
         scaa = local_row_array[recv_procs_dbl[gidx]];
         for (j = 0; j < ng; j ++)
         {
            for (i = 0; i < scaa; i ++)
            {
               jxf_VectorData(VER_loc[j])[ver_cnt[j]++] = recv_buf_dbl[start+i];
            }
            start += scaa;
         }
         for (i = 0; i < scaa; i ++)
         {
            jxf_VectorData(VEI_loc)[vei_cnt++] = recv_buf_dbl[start+i];
         }
         start += scaa;
         for (i = 0; i < scaa; i ++)
         {
            jxf_VectorData(f_loc)[f_cnt++] = recv_buf_dbl[start+i];
         }
         start += scaa;
         for (i = 0; i < scaa; i ++)
         {
            jxf_VectorData(u_loc)[u_cnt++] = recv_buf_dbl[start+i];
         }
      }
   }
   else if (groupid_x == ng+1) // 离子进程组
   {
      VIE_loc = jxf_SeqVectorCreate(num_rows);
      jxf_SeqVectorInitialize(VIE_loc);
      vie_cnt = 0;
      f_cnt   = 0;
      u_cnt   = 0;
      for (gidx = 0; gidx < num_recvs; gidx ++)
      {
         start = recv_starts_dbl[gidx] + local_nnz_array[recv_procs_dbl[gidx]];
         scaa = local_row_array[recv_procs_dbl[gidx]];
         for (i = 0; i < scaa; i ++)
         {
            jxf_VectorData(VIE_loc)[vie_cnt++] = recv_buf_dbl[start+i];
         }
         start += scaa;
         for (i = 0; i < scaa; i ++)
         {
            jxf_VectorData(f_loc)[f_cnt++] = recv_buf_dbl[start+i];
         }
         start += scaa;
         for (i = 0; i < scaa; i ++)
         {
            jxf_VectorData(u_loc)[u_cnt++] = recv_buf_dbl[start+i];
         }
      }
   }

   row_starts_loc = jxf_CTAlloc(JXF_Int, num_proc_eachgroup+1);
   for (i = 0; i <= num_proc_eachgroup; i ++)
   {
      row_starts_loc[i] = row_starts[(ng+2)*i];
   }
   col_starts_loc = row_starts_loc;

   if (groupid_x < ng) // 光子进程组
   {
      ARR = jxf_ParCSRMatrixCreate(comm_x, n, n, row_starts_loc, col_starts_loc, 0, 0, 0);
      first_col_diag = col_starts_loc[myid_x];
      last_col_diag  = col_starts_loc[myid_x+1] - 1;
      jxf_GenerateDiagAndOffd(A_loc, ARR, first_col_diag, last_col_diag);
      VRE = jxf_ParVectorCreate(comm_x, n, row_starts_loc);
      jxf_ParVectorSetPartitioningOwner(VRE, 0);
      jxf_TFree(jxf_ParVectorLocalVector(VRE));
      jxf_ParVectorLocalVector(VRE) = VRE_loc;
   }
   else if (groupid_x == ng) // 电子进程组
   {
      AEE = jxf_ParCSRMatrixCreate(comm_x, n, n, row_starts_loc, col_starts_loc, 0, 0, 0);
      first_col_diag = col_starts_loc[myid_x];
      last_col_diag  = col_starts_loc[myid_x+1] - 1;
      jxf_GenerateDiagAndOffd(A_loc, AEE, first_col_diag, last_col_diag);
      VER = jxf_CTAlloc(jxf_ParVector *, ng);
      for (gidx = 0; gidx < ng; gidx ++)
      {
         VER[gidx] = jxf_ParVectorCreate(comm_x, n, row_starts_loc);
         jxf_ParVectorSetPartitioningOwner(VER[gidx], 0);
         jxf_TFree(jxf_ParVectorLocalVector(VER[gidx]));
         jxf_ParVectorLocalVector(VER[gidx]) = VER_loc[gidx];
      }
      VEI = jxf_ParVectorCreate(comm_x, n, row_starts_loc);
      jxf_ParVectorSetPartitioningOwner(VEI, 0);
      jxf_TFree(jxf_ParVectorLocalVector(VEI));
      jxf_ParVectorLocalVector(VEI) = VEI_loc;
   }
   else if (groupid_x == ng+1) // 离子进程组
   {
      AII = jxf_ParCSRMatrixCreate(comm_x, n, n, row_starts_loc, col_starts_loc, 0, 0, 0);
      first_col_diag = col_starts_loc[myid_x];
      last_col_diag  = col_starts_loc[myid_x+1] - 1;
      jxf_GenerateDiagAndOffd(A_loc, AII, first_col_diag, last_col_diag);
      VIE = jxf_ParVectorCreate(comm_x, n, row_starts_loc);
      jxf_ParVectorSetPartitioningOwner(VIE, 0);
      jxf_TFree(jxf_ParVectorLocalVector(VIE));
      jxf_ParVectorLocalVector(VIE) = VIE_loc;
   }

   row_starts_glo = jxf_CTAlloc(JXF_Int, nprocs+1);
   scaa = 0;
   scbb = 0;
   for (j = 0; j < ng; j ++)
   {
      for (i = 0; i < num_proc_eachgroup; i ++)
      {
         row_starts_glo[i+scbb] = row_starts_loc[i] + scaa;
      }
      scaa += n;
      scbb += num_proc_eachgroup;
   }
   for (i = 0; i < num_proc_eachgroup; i ++)
   {
      row_starts_glo[i+scbb] = row_starts_loc[i] + scaa;
   }
   scaa += n;
   scbb += num_proc_eachgroup;
   for (i = 0; i < num_proc_eachgroup; i ++)
   {
      row_starts_glo[i+scbb] = row_starts_loc[i] + scaa;
   }
   row_starts_glo[nprocs] = N;
   col_starts_glo = row_starts_glo;

   if (groupid_x == ng)
   {
      num_nonzeros = jxf_CSRMatrixNumNonzeros(A_loc) + (ng + 1) * num_rows;
   }
   else
   {
      num_nonzeros = jxf_CSRMatrixNumNonzeros(A_loc) + num_rows;
   }
   A_rec = jxf_CSRMatrixCreate(num_rows, N, num_nonzeros);
   jxf_CSRMatrixInitialize(A_rec);
   aarec = jxf_CSRMatrixData(A_rec);
   iarec = jxf_CSRMatrixI(A_rec);
   jarec = jxf_CSRMatrixJ(A_rec);

   if (groupid_x < ng) // 光子进程组
   {
      k = 0;
      iarec[0] = 0;
      start = row_starts_glo[myid];
      scaa = groupid_x * n;
      for (i = 0; i < num_rows; i ++)
      {
         scbb = ia[i+1];
         for (j = ia[i]; j < scbb; j ++)
         {
            jarec[k] = ja[j] + scaa;
            aarec[k] = aa[j];
            k ++;
         }
         jarec[k] = start + i + e_pos - scaa;
         aarec[k] = jxf_VectorData(VRE_loc)[i];
         k ++;
         iarec[i+1] = k;
      }
   }
   else if (groupid_x == ng) // 电子进程组
   {
      k = 0;
      iarec[0] = 0;
      start = row_starts_glo[myid];
      for (i = 0; i < num_rows; i ++)
      {
         scbb = ia[i+1];
         for (j = ia[i]; j < scbb; j ++)
         {
            jarec[k] = ja[j] + e_pos;
            aarec[k] = aa[j];
            k ++;
         }
         scaa = start + i - e_pos;
         for (j = 0; j < ng; j ++)
         {
            jarec[k] = scaa + j * n;
            aarec[k] = jxf_VectorData(VER_loc[j])[i];
            k ++;
         }
         jarec[k] = start + i + n;
         aarec[k] = jxf_VectorData(VEI_loc)[i];
         k ++;
         iarec[i+1] = k;
      }
   }
   else if (groupid_x == ng+1) // 离子进程组
   {
      k = 0;
      iarec[0] = 0;
      start = row_starts_glo[myid];
      for (i = 0; i < num_rows; i ++)
      {
         scbb = ia[i+1];
         for (j = ia[i]; j < scbb; j ++)
         {
            jarec[k] = ja[j] + i_pos;
            aarec[k] = aa[j];
            k ++;
         }
         jarec[k] = start + i - n;
         aarec[k] = jxf_VectorData(VIE_loc)[i];
         k ++;
         iarec[i+1] = k;
      }
   }

   A = jxf_ParCSRMatrixCreate(comm, N, N, row_starts_glo, col_starts_glo, 0, 0, 0);
   first_col_diag = col_starts_glo[myid];
   last_col_diag  = col_starts_glo[myid+1] - 1;
   jxf_GenerateDiagAndOffd(A_rec, A, first_col_diag, last_col_diag);

   f = jxf_ParVectorCreate(comm, N, row_starts_glo);
   jxf_ParVectorSetPartitioningOwner(f, 0);
   jxf_TFree(jxf_ParVectorLocalVector(f));
   jxf_ParVectorLocalVector(f) = f_loc;

   u = jxf_ParVectorCreate(comm, N, row_starts_glo);
   jxf_ParVectorSetPartitioningOwner(u, 0);
   jxf_TFree(jxf_ParVectorLocalVector(u));
   jxf_ParVectorLocalVector(u) = u_loc;

   jxf_TFree(send_buf_dbl);
   jxf_TFree(recv_buf_dbl);
   jxf_TFree(send_buf_int);
   jxf_TFree(recv_buf_int);
   jxf_TFree(local_nnz_array);
   jxf_TFree(local_row_array);

   jxf_ParVecCommPkgDestroy(comm_pkg_dbl);
   jxf_ParVecCommPkgDestroy(comm_pkg_int);

   jxf_TFree(size_recv);
   if (ver_cnt) jxf_TFree(ver_cnt);

   for (gidx = 0; gidx < ng; gidx ++) jxf_CSRMatrixDestroy(ARR_comb[gidx]);
   jxf_TFree(ARR_comb);
   jxf_CSRMatrixDestroy(AEE_comb);
   jxf_CSRMatrixDestroy(AII_comb);

   jxf_CSRMatrixDestroy(A_loc);
   jxf_CSRMatrixDestroy(A_rec);

  *ARR_ptr = ARR;
  *AEE_ptr = AEE;
  *AII_ptr = AII;
  *VRE_ptr = VRE;
  *VER_ptr = VER;
  *VEI_ptr = VEI;
  *VIE_ptr = VIE;

  *A_ptr = A;
  *f_ptr = f;
  *u_ptr = u;

   return (0);
}

/*!
 * \fn JXF_Int jxf_MatVecGroup2All
 * \brief 将基于进程分组的并行矩阵向量转换为基于所有进程的并行矩阵向量,
 *        其并行分划均采用默认的负载平衡的分划.
 * \author peghoty 
 * \date 2012/03/02
 */
JXF_Int 
jxf_MatVecGroup2All( MPI_Comm          comm, 
                    JXF_Int               groupid_x,
                    jxf_ParCSRMatrix  *ARR, 
                    jxf_ParCSRMatrix  *AEE, 
                    jxf_ParCSRMatrix  *AII, 
                    jxf_ParVector     *VRE, 
                    jxf_ParVector     *VER, 
                    jxf_ParVector     *VEI, 
                    jxf_ParVector     *VIE,
                    jxf_ParCSRMatrix **ARR_all_ptr, 
                    jxf_ParCSRMatrix **AEE_all_ptr, 
                    jxf_ParCSRMatrix **AII_all_ptr, 
                    jxf_ParVector    **VRE_all_ptr, 
                    jxf_ParVector    **VER_all_ptr, 
                    jxf_ParVector    **VEI_all_ptr, 
                    jxf_ParVector    **VIE_all_ptr )
{
   JXF_Int myid, nprocs;
   JXF_Int npeachgroup;
   JXF_Int rootid_R, rootid_E, rootid_I;

   jxf_CSRMatrix *ARR_s = NULL;
   jxf_CSRMatrix *AEE_s = NULL;
   jxf_CSRMatrix *AII_s = NULL;
   jxf_Vector    *VRE_s = NULL;
   jxf_Vector    *VER_s = NULL;
   jxf_Vector    *VEI_s = NULL;
   jxf_Vector    *VIE_s = NULL;

   jxf_ParCSRMatrix *ARR_all = NULL; 
   jxf_ParCSRMatrix *AEE_all = NULL;
   jxf_ParCSRMatrix *AII_all = NULL; 
   jxf_ParVector    *VRE_all = NULL; 
   jxf_ParVector    *VER_all = NULL; 
   jxf_ParVector    *VEI_all = NULL; 
   jxf_ParVector    *VIE_all = NULL;
                    
   jxf_MPI_Comm_rank(comm, &myid);
   jxf_MPI_Comm_size(comm, &nprocs);

   // 三个进程组的根进程号
   npeachgroup = nprocs / 3;
   rootid_R = 0;
   rootid_E = npeachgroup; 
   rootid_I = 2*npeachgroup;
       
   if (groupid_x == 0)
   {
      ARR_s = jxf_ParCSRMatrixToCSRMatrixAll(ARR);
      VRE_s = jxf_ParVectorToVectorAll(VRE);
   }
   else if (groupid_x == 1)
   {
      AEE_s = jxf_ParCSRMatrixToCSRMatrixAll(AEE);
      VER_s = jxf_ParVectorToVectorAll(VER);
      VEI_s = jxf_ParVectorToVectorAll(VEI);
   }
   else if (groupid_x == 2)
   {
      AII_s = jxf_ParCSRMatrixToCSRMatrixAll(AII);
      VIE_s = jxf_ParVectorToVectorAll(VIE); 
   }
   
   /* (从指定进程)串行矩阵转并行矩阵 */ 
   ARR_all = jxf_CSRMatrixToParCSRMatrix_FromGivenPro(comm, rootid_R, ARR_s, NULL, NULL); 
   AEE_all = jxf_CSRMatrixToParCSRMatrix_FromGivenPro(comm, rootid_E, AEE_s, NULL, NULL);
   AII_all = jxf_CSRMatrixToParCSRMatrix_FromGivenPro(comm, rootid_I, AII_s, NULL, NULL);
   
   /* (从指定进程)串行向量转并行向量 */ 
   VRE_all = jxf_VectorToParVector_FromGivenPro(comm, rootid_R, VRE_s, NULL);
   VER_all = jxf_VectorToParVector_FromGivenPro(comm, rootid_E, VER_s, NULL);
   VEI_all = jxf_VectorToParVector_FromGivenPro(comm, rootid_E, VEI_s, NULL);
   VIE_all = jxf_VectorToParVector_FromGivenPro(comm, rootid_I, VIE_s, NULL);
   
   if (groupid_x == 0)
   {
      jxf_CSRMatrixDestroy(ARR_s);
      jxf_SeqVectorDestroy(VRE_s);
   }
   else if (groupid_x == 1)
   {
      jxf_CSRMatrixDestroy(AEE_s);
      jxf_SeqVectorDestroy(VER_s);
      jxf_SeqVectorDestroy(VEI_s);
   }
   else if (groupid_x == 2)
   {
      jxf_CSRMatrixDestroy(AII_s);
      jxf_SeqVectorDestroy(VIE_s);
   }
 
   *ARR_all_ptr = ARR_all;
   *AEE_all_ptr = AEE_all;
   *AII_all_ptr = AII_all;
   *VRE_all_ptr = VRE_all;
   *VER_all_ptr = VER_all;
   *VEI_all_ptr = VEI_all;
   *VIE_all_ptr = VIE_all;

   return (0);
}

/*!
 * \fn JXF_Int jxf_3tDataTransFromSeq2SubPara
 * \brief 将串行矩阵 A_s 和串行向量 f_s，u_s 转化为子块的并行数据,
 *        其并行分划均采用默认的负载平衡的分划.
 * \author peghoty 
 * \date 2012/03/08
 */
JXF_Int
jxf_3tDataTransFromSeq2SubPara( JXF_Int               iniguess,
                               MPI_Comm          comm,
                               jxf_CSRMatrix     *A_s, 
                               jxf_Vector        *f_s, 
                               jxf_Vector        *u_s, 
                               jxf_ParCSRMatrix **ARR_p_ptr, 
                               jxf_ParCSRMatrix **AEE_p_ptr, 
                               jxf_ParCSRMatrix **AII_p_ptr, 
                               jxf_ParVector    **VRE_p_ptr, 
                               jxf_ParVector    **VER_p_ptr, 
                               jxf_ParVector    **VEI_p_ptr, 
                               jxf_ParVector    **VIE_p_ptr, 
                               jxf_ParVector    **fR_p_ptr, 
                               jxf_ParVector    **fE_p_ptr, 
                               jxf_ParVector    **fI_p_ptr,
                               jxf_ParVector    **uR_p_ptr, 
                               jxf_ParVector    **uE_p_ptr, 
                               jxf_ParVector    **uI_p_ptr )
{
   /* 目标并行数据 */
   jxf_ParCSRMatrix *ARR_p = NULL; 
   jxf_ParCSRMatrix *AEE_p = NULL; 
   jxf_ParCSRMatrix *AII_p = NULL; 
   jxf_ParVector    *VRE_p = NULL; 
   jxf_ParVector    *VER_p = NULL; 
   jxf_ParVector    *VEI_p = NULL; 
   jxf_ParVector    *VIE_p = NULL; 
   
   jxf_ParVector    *fR_p  = NULL; 
   jxf_ParVector    *fE_p  = NULL; 
   jxf_ParVector    *fI_p  = NULL;
   
   jxf_ParVector    *uR_p  = NULL; 
   jxf_ParVector    *uE_p  = NULL;  
   jxf_ParVector    *uI_p  = NULL;
   
   /* 辅助串行矩阵和向量 */
   jxf_CSRMatrix    *ARR_s = NULL;
   jxf_CSRMatrix    *AEE_s = NULL;
   jxf_CSRMatrix    *AII_s = NULL;
   jxf_Vector       *VRE_s = NULL;
   jxf_Vector       *VER_s = NULL;
   jxf_Vector       *VEI_s = NULL;
   jxf_Vector       *VIE_s = NULL;

   jxf_Vector       *fR_s  = NULL; 
   jxf_Vector       *fE_s  = NULL; 
   jxf_Vector       *fI_s  = NULL;
   
   jxf_Vector       *uR_s  = NULL; 
   jxf_Vector       *uE_s  = NULL;  
   jxf_Vector       *uI_s  = NULL;
    
   JXF_Int myid, nprocs;
   JXF_Int n = 0;

   jxf_MPI_Comm_rank(comm, &myid);
   jxf_MPI_Comm_size(comm, &nprocs);

   if (nprocs == 1)
   {
      //-------------------------------------------------------------------------------------
      //  抽取子矩阵和子向量
      //-------------------------------------------------------------------------------------   
      jxf_3tGetSubBlocks_REIV(A_s, &ARR_s, &AEE_s, &AII_s, &VRE_s, &VER_s, &VEI_s, &VIE_s); 
      jxf_3tGetSubVecs(f_s, &fR_s, &fE_s, &fI_s);
      if (iniguess) 
      {
         jxf_3tGetSubVecs(u_s, &uR_s, &uE_s, &uI_s);
      }
      else
      { 
         n = jxf_CSRMatrixNumRows(A_s) / 3; // 子矩阵的规模
      }

      //---------------------------------------------------------
      //  将串行矩阵和向量转换成并行矩阵和并行向量
      //---------------------------------------------------------      
      ARR_p = jxf_CSRMatrixToParCSRMatrix_sp(comm, ARR_s);
      AEE_p = jxf_CSRMatrixToParCSRMatrix_sp(comm, AEE_s);
      AII_p = jxf_CSRMatrixToParCSRMatrix_sp(comm, AII_s);
      VRE_p = jxf_VectorToParVector_sp(comm, VRE_s);
      VER_p = jxf_VectorToParVector_sp(comm, VER_s);
      VEI_p = jxf_VectorToParVector_sp(comm, VEI_s);
      VIE_p = jxf_VectorToParVector_sp(comm, VIE_s);
      
      fR_p = jxf_VectorToParVector_sp(comm, fR_s);
      fE_p = jxf_VectorToParVector_sp(comm, fE_s);
      fI_p = jxf_VectorToParVector_sp(comm, fI_s);     

      if (iniguess) 
      {
         uR_p = jxf_VectorToParVector_sp(comm, uR_s);
         uE_p = jxf_VectorToParVector_sp(comm, uE_s);
         uI_p = jxf_VectorToParVector_sp(comm, uI_s);    
      }
      else
      {
         uR_p = jxf_ParVectorCreate(comm, n, NULL);
         jxf_ParVectorInitialize(uR_p);
         uE_p = jxf_ParVectorCreate(comm, n, NULL);
         jxf_ParVectorInitialize(uE_p);
         uI_p = jxf_ParVectorCreate(comm, n, NULL);
         jxf_ParVectorInitialize(uI_p);
      }

      //---------------------------------------------------------
      //  释放辅助矩阵和向量(注意不要释放数据部分!)
      //---------------------------------------------------------        
      jxf_TFree(ARR_s);
      jxf_TFree(AEE_s);
      jxf_TFree(AII_s);
      jxf_TFree(VRE_s);
      jxf_TFree(VER_s);
      jxf_TFree(VEI_s);
      jxf_TFree(VIE_s );
      jxf_TFree(fR_s);
      jxf_TFree(fE_s);
      jxf_TFree(fI_s);
      
      if (iniguess) 
      {
         jxf_TFree(uR_s);
         jxf_TFree(uE_s);
         jxf_TFree(uI_s);   
      }
   }
   else if (nprocs > 1)
   {    
      
      /* 前三个进程分别从串行矩阵 A_s 中抽取子块 */   
      if (myid == 0)
      {
         jxf_3tGetSubBlocks_RV(A_s, &ARR_s, &VRE_s);
         jxf_3tGetSubVecs(f_s, &fR_s, &fE_s, &fI_s);
      }

      if (myid == 1)
      {
         jxf_3tGetSubBlocks_EV(A_s, &AEE_s, &VER_s, &VEI_s);
      }
   
      if (myid == 2)
      {
         jxf_3tGetSubBlocks_IV(A_s, &AII_s, &VIE_s);
         if (iniguess) 
         {
            jxf_3tGetSubVecs(u_s, &uR_s, &uE_s, &uI_s);
         }
         else
         {
            n = jxf_CSRMatrixNumRows(A_s) / 3; 
         } 
      }
   
      /* 将子块转为并行矩阵和并行向量 */
      
      // 从 0 号进程转换 
      ARR_p = jxf_CSRMatrixToParCSRMatrix(comm, ARR_s, NULL, NULL); 
      VRE_p = jxf_VectorToParVector(comm, VRE_s, NULL);
      fR_p = jxf_VectorToParVector(comm, fR_s, NULL);
      fE_p = jxf_VectorToParVector(comm, fE_s, NULL);
      fI_p = jxf_VectorToParVector(comm, fI_s, NULL);

      // 从 1 号进程转换 
      AEE_p = jxf_CSRMatrixToParCSRMatrix_FromGivenPro(comm, 1, AEE_s, NULL, NULL); 
      VER_p = jxf_VectorToParVector_FromGivenPro(comm, 1, VER_s, NULL);
      VEI_p = jxf_VectorToParVector_FromGivenPro(comm, 1, VEI_s, NULL);

      // 从 2 号进程转换
      AII_p = jxf_CSRMatrixToParCSRMatrix_FromGivenPro(comm, 2, AII_s, NULL, NULL); 
      VIE_p = jxf_VectorToParVector_FromGivenPro(comm, 2, VIE_s, NULL);
      if (iniguess)
      {
         uR_p = jxf_VectorToParVector_FromGivenPro(comm, 2, uR_s, NULL);
         uE_p = jxf_VectorToParVector_FromGivenPro(comm, 2, uE_s, NULL);
         uI_p = jxf_VectorToParVector_FromGivenPro(comm, 2, uI_s, NULL);
      }
      else
      {
         jxf_MPI_Bcast(&n, 1, JXF_MPI_INT, 2, comm);
         uR_p = jxf_ParVectorCreate(comm, n, NULL);
         jxf_ParVectorInitialize(uR_p);
         uE_p = jxf_ParVectorCreate(comm, n, NULL);
         jxf_ParVectorInitialize(uE_p);
         uI_p = jxf_ParVectorCreate(comm, n, NULL);
         jxf_ParVectorInitialize(uI_p);            
      }
   
      /* 释放辅助矩阵和向量 */   
      if (myid == 0)
      {
         jxf_CSRMatrixDestroy(ARR_s);
         jxf_SeqVectorDestroy(VRE_s);
         jxf_SeqVectorDestroy(fR_s);
         jxf_SeqVectorDestroy(fE_s);
         jxf_SeqVectorDestroy(fI_s);    
      }

      if (myid == 1)
      {
         jxf_CSRMatrixDestroy(AEE_s);
         jxf_SeqVectorDestroy(VER_s); 
         jxf_SeqVectorDestroy(VEI_s);        
      }
   
      if (myid == 2)
      {
         jxf_CSRMatrixDestroy(AII_s);
         jxf_SeqVectorDestroy(VIE_s);

         if (iniguess)
         {
            jxf_SeqVectorDestroy(uR_s);
            jxf_SeqVectorDestroy(uE_s);
            jxf_SeqVectorDestroy(uI_s);
         }
      }
      
   } // end if (nprocs > 1)

   
   //------------------------------------
   //  返回并行数据
   //------------------------------------ 
   *ARR_p_ptr = ARR_p; 
   *AEE_p_ptr = AEE_p; 
   *AII_p_ptr = AII_p; 
   *VRE_p_ptr = VRE_p; 
   *VER_p_ptr = VER_p;
   *VEI_p_ptr = VEI_p; 
   *VIE_p_ptr = VIE_p; 
   *fR_p_ptr  = fR_p; 
   *fE_p_ptr  = fE_p; 
   *fI_p_ptr  = fI_p;
   *uR_p_ptr  = uR_p; 
   *uE_p_ptr  = uE_p; 
   *uI_p_ptr  = uI_p;
   
   return 0;    
}

/*!
 * \fn JXF_Int jxf_3tDataTransFromSeq2SubPar0
 * \brief 将零号进程上的串行矩阵 A_s 和串行向量 f_s，u_s 转化为子块的并行数据,
 *        其并行分划均采用默认的负载平衡的分划.
 * \author peghoty 
 * \date 2012/03/08
 */
JXF_Int
jxf_3tDataTransFromSeq2SubPar0( JXF_Int               iniguess,
                               MPI_Comm          comm,
                               jxf_CSRMatrix     *A_s, 
                               jxf_Vector        *f_s, 
                               jxf_Vector        *u_s, 
                               jxf_ParCSRMatrix **ARR_p_ptr, 
                               jxf_ParCSRMatrix **AEE_p_ptr, 
                               jxf_ParCSRMatrix **AII_p_ptr, 
                               jxf_ParVector    **VRE_p_ptr, 
                               jxf_ParVector    **VER_p_ptr, 
                               jxf_ParVector    **VEI_p_ptr, 
                               jxf_ParVector    **VIE_p_ptr, 
                               jxf_ParVector    **fR_p_ptr, 
                               jxf_ParVector    **fE_p_ptr, 
                               jxf_ParVector    **fI_p_ptr,
                               jxf_ParVector    **uR_p_ptr, 
                               jxf_ParVector    **uE_p_ptr, 
                               jxf_ParVector    **uI_p_ptr )
{
   /* 目标并行数据 */
   jxf_ParCSRMatrix *ARR_p = NULL; 
   jxf_ParCSRMatrix *AEE_p = NULL; 
   jxf_ParCSRMatrix *AII_p = NULL; 
   jxf_ParVector    *VRE_p = NULL; 
   jxf_ParVector    *VER_p = NULL; 
   jxf_ParVector    *VEI_p = NULL; 
   jxf_ParVector    *VIE_p = NULL; 
   
   jxf_ParVector    *fR_p  = NULL; 
   jxf_ParVector    *fE_p  = NULL; 
   jxf_ParVector    *fI_p  = NULL;
   
   jxf_ParVector    *uR_p  = NULL; 
   jxf_ParVector    *uE_p  = NULL;  
   jxf_ParVector    *uI_p  = NULL;
   
   /* 辅助串行矩阵和向量 */
   jxf_CSRMatrix    *ARR_s = NULL;
   jxf_CSRMatrix    *AEE_s = NULL;
   jxf_CSRMatrix    *AII_s = NULL;
   jxf_Vector       *VRE_s = NULL;
   jxf_Vector       *VER_s = NULL;
   jxf_Vector       *VEI_s = NULL;
   jxf_Vector       *VIE_s = NULL;

   jxf_Vector       *fR_s  = NULL; 
   jxf_Vector       *fE_s  = NULL; 
   jxf_Vector       *fI_s  = NULL;
   
   jxf_Vector       *uR_s  = NULL; 
   jxf_Vector       *uE_s  = NULL;  
   jxf_Vector       *uI_s  = NULL;
    
   JXF_Int myid, nprocs;
   JXF_Int global_size = 0;
   JXF_Int n;

   jxf_MPI_Comm_rank(comm, &myid);
   jxf_MPI_Comm_size(comm, &nprocs);

   if (nprocs == 1)
   {
      //-------------------------------------------------------------------------------------
      //  抽取子矩阵和子向量
      //-------------------------------------------------------------------------------------   
      jxf_3tGetSubBlocks_REIV(A_s, &ARR_s, &AEE_s, &AII_s, &VRE_s, &VER_s, &VEI_s, &VIE_s); 
      jxf_3tGetSubVecs(f_s, &fR_s, &fE_s, &fI_s);
      if (iniguess) 
      {
         jxf_3tGetSubVecs(u_s, &uR_s, &uE_s, &uI_s);
      }
      else
      { 
         global_size = jxf_CSRMatrixNumRows(A_s);
         n = global_size / 3;
      }

      //---------------------------------------------------------
      //  将串行矩阵和向量转换成并行矩阵和并行向量
      //---------------------------------------------------------      
      ARR_p = jxf_CSRMatrixToParCSRMatrix_sp(comm, ARR_s);
      AEE_p = jxf_CSRMatrixToParCSRMatrix_sp(comm, AEE_s);
      AII_p = jxf_CSRMatrixToParCSRMatrix_sp(comm, AII_s);
      VRE_p = jxf_VectorToParVector_sp(comm, VRE_s);
      VER_p = jxf_VectorToParVector_sp(comm, VER_s);
      VEI_p = jxf_VectorToParVector_sp(comm, VEI_s);
      VIE_p = jxf_VectorToParVector_sp(comm, VIE_s);
      
      fR_p = jxf_VectorToParVector_sp(comm, fR_s);
      fE_p = jxf_VectorToParVector_sp(comm, fE_s);
      fI_p = jxf_VectorToParVector_sp(comm, fI_s);     

      if (iniguess) 
      {
         uR_p = jxf_VectorToParVector_sp(comm, uR_s);
         uE_p = jxf_VectorToParVector_sp(comm, uE_s);
         uI_p = jxf_VectorToParVector_sp(comm, uI_s);    
      }
      else
      {
         uR_p = jxf_ParVectorCreate(comm, n, NULL);
         jxf_ParVectorInitialize(uR_p);
         uE_p = jxf_ParVectorCreate(comm, n, NULL);
         jxf_ParVectorInitialize(uE_p);
         uI_p = jxf_ParVectorCreate(comm, n, NULL);
         jxf_ParVectorInitialize(uI_p);
      }

      //---------------------------------------------------------
      //  释放辅助矩阵和向量(注意不要释放数据部分!)
      //---------------------------------------------------------        
      jxf_TFree(ARR_s);
      jxf_TFree(AEE_s);
      jxf_TFree(AII_s);
      jxf_TFree(VRE_s);
      jxf_TFree(VER_s);
      jxf_TFree(VEI_s);
      jxf_TFree(VIE_s );
      jxf_TFree(fR_s);
      jxf_TFree(fE_s);
      jxf_TFree(fI_s);
      
      if (iniguess) 
      {
         jxf_TFree(uR_s);
         jxf_TFree(uE_s);
         jxf_TFree(uI_s);   
      }
   }
   else if (nprocs > 1)
   {    
      //----------------------------------------------------------------------------------------
      //  在 0 号进程，抽取子矩阵和子向量
      //----------------------------------------------------------------------------------------
      if (myid == 0)
      {
         jxf_3tGetSubBlocks_REIV(A_s, &ARR_s, &AEE_s, &AII_s, &VRE_s, &VER_s, &VEI_s, &VIE_s); 
         jxf_3tGetSubVecs(f_s, &fR_s, &fE_s, &fI_s);
         if (iniguess) 
         {
            jxf_3tGetSubVecs(u_s, &uR_s, &uE_s, &uI_s);
         }
         else
         { 
            global_size = jxf_CSRMatrixNumRows(A_s);
            n = global_size / 3;
         }
      }

      //-------------------------------------------------------------------------
      //  将串行矩阵和向量转换成并行矩阵和并行向量
      //-------------------------------------------------------------------------    
      ARR_p = jxf_CSRMatrixToParCSRMatrix(comm, ARR_s, NULL, NULL);
      AEE_p = jxf_CSRMatrixToParCSRMatrix(comm, AEE_s, NULL, NULL);
      AII_p = jxf_CSRMatrixToParCSRMatrix(comm, AII_s, NULL, NULL);
      VRE_p = jxf_VectorToParVector(comm, VRE_s, NULL);
      VER_p = jxf_VectorToParVector(comm, VER_s, NULL);
      VEI_p = jxf_VectorToParVector(comm, VEI_s, NULL);
      VIE_p = jxf_VectorToParVector(comm, VIE_s, NULL);

      fR_p = jxf_VectorToParVector(comm, fR_s, NULL);
      fE_p = jxf_VectorToParVector(comm, fE_s, NULL);
      fI_p = jxf_VectorToParVector(comm, fI_s, NULL);
   
      if (iniguess) 
      {
         uR_p = jxf_VectorToParVector(comm, uR_s, NULL);
         uE_p = jxf_VectorToParVector(comm, uE_s, NULL);
         uI_p = jxf_VectorToParVector(comm, uI_s, NULL);
      }
      else
      {
         jxf_MPI_Bcast(&n, 1, JXF_MPI_INT, 0, comm);
         uR_p = jxf_ParVectorCreate(comm, n, NULL);
         jxf_ParVectorInitialize(uR_p);
         uE_p = jxf_ParVectorCreate(comm, n, NULL);
         jxf_ParVectorInitialize(uE_p);
         uI_p = jxf_ParVectorCreate(comm, n, NULL);
         jxf_ParVectorInitialize(uI_p);
      } 
   
      //--------------------------------------------
      //  释放辅助矩阵和向量
      //--------------------------------------------
      if (myid == 0)
      {
         jxf_CSRMatrixDestroy(ARR_s);
         jxf_CSRMatrixDestroy(AEE_s);
         jxf_CSRMatrixDestroy(AII_s);
         jxf_SeqVectorDestroy(VRE_s);
         jxf_SeqVectorDestroy(VER_s); 
         jxf_SeqVectorDestroy(VEI_s);
         jxf_SeqVectorDestroy(VIE_s); 
      
         jxf_SeqVectorDestroy(fR_s);
         jxf_SeqVectorDestroy(fE_s);
         jxf_SeqVectorDestroy(fI_s);

         if (iniguess)
         {
            jxf_SeqVectorDestroy(uR_s);
            jxf_SeqVectorDestroy(uE_s);
            jxf_SeqVectorDestroy(uI_s);
         }
      }
      
   }  // end if (nprocs > 1)
   

   //------------------------------------
   //  返回并行数据
   //------------------------------------        
   *ARR_p_ptr = ARR_p; 
   *AEE_p_ptr = AEE_p; 
   *AII_p_ptr = AII_p; 
   *VRE_p_ptr = VRE_p; 
   *VER_p_ptr = VER_p;
   *VEI_p_ptr = VEI_p; 
   *VIE_p_ptr = VIE_p; 
   *fR_p_ptr  = fR_p; 
   *fE_p_ptr  = fE_p; 
   *fI_p_ptr  = fI_p;
   *uR_p_ptr  = uR_p; 
   *uE_p_ptr  = uE_p; 
   *uI_p_ptr  = uI_p;
   
   return 0;    
}

/*!
 * \fn JXF_Int jxf_ParaDataTransSEQIF_mp
 * \brief 将 0 号进程上的串行矩阵 A_s 和串行向量 f_s，u_s 转化为以下三类并行数据:
 *  (1) 整体离散系统基于整个进程组的并行数据，且具有匹配的并行分划(用于 GMRES); 
 *  (2) 子块基于进程分组的并行数据(用于 apctl 迭代);
 *  (3) 子块基于整体进程的并行数据(用于生成 pctl 算法中的粗矩阵 Ac).
 * \author peghoty 
 * \date 2012/03/09
 */
JXF_Int  
jxf_ParaDataTransSEQIF_mp( // input:
                          MPI_Comm           comm, 
                          MPI_Comm           comm_x, 
                          JXF_Int                groupid_x,
                          JXF_Real             theta_wc_E, 
                          JXF_Real             threshold_wc_E,
                          /* 串行离散系统数据 */
                          jxf_CSRMatrix      *A_s, 
                          jxf_Vector         *f_s, 
                          jxf_Vector         *u_s, 
                          // output: 
                          /* 系数矩阵各子块在整个进程组上的并行数据 */
                          jxf_ParCSRMatrix  **ARR_p_ptr, 
                          jxf_ParCSRMatrix  **AEE_p_ptr, 
                          jxf_ParCSRMatrix  **AII_p_ptr, 
                          jxf_ParVector     **VRE_p_ptr, 
                          jxf_ParVector     **VER_p_ptr, 
                          jxf_ParVector     **VEI_p_ptr, 
                          jxf_ParVector     **VIE_p_ptr, 
                          /* 系数矩阵各子块基于进程组的并行数据 */                         
                          jxf_ParCSRMatrix  **ARR_ptr,
                          jxf_ParCSRMatrix  **AEE_ptr,
                          jxf_ParCSRMatrix  **AII_ptr, 
                          jxf_ParVector     **VRE_ptr, 
                          jxf_ParVector     **VER_ptr, 
                          jxf_ParVector     **VEI_ptr, 
                          jxf_ParVector     **VIE_ptr,
                          /* 并行离散系统数据 */
                          jxf_ParCSRMatrix  **A_ptr,  
                          jxf_ParVector     **f_ptr, 
                          jxf_ParVector     **u_ptr,
                          /* 是否需要粗网格校正的标志变量 */
                          JXF_Int               *Need_CC_ptr )
{
   JXF_Int myid, nprocs;
   JXF_Int np_R, np_E, np_I;
   JXF_Int rootid_E;
   JXF_Int N;
   JXF_Int *mypartition = NULL;
   JXF_Int *A_partition = NULL;
       
   /* 系数矩阵各子块在整个进程组上的并行数据 */
   jxf_ParCSRMatrix  *ARR_p = NULL; 
   jxf_ParCSRMatrix  *AEE_p = NULL; 
   jxf_ParCSRMatrix  *AII_p = NULL; 
   jxf_ParVector     *VRE_p = NULL; 
   jxf_ParVector     *VER_p = NULL; 
   jxf_ParVector     *VEI_p = NULL; 
   jxf_ParVector     *VIE_p = NULL; 
   
   /* 系数矩阵各子块基于进程组的并行数据 */     
   jxf_ParCSRMatrix  *ARR = NULL;
   jxf_ParCSRMatrix  *AEE = NULL;
   jxf_ParCSRMatrix  *AII = NULL; 
   jxf_ParVector     *VRE = NULL; 
   jxf_ParVector     *VER = NULL; 
   jxf_ParVector     *VEI = NULL; 
   jxf_ParVector     *VIE = NULL;
   
   /* 并行离散系统数据 */
   jxf_ParCSRMatrix  *A = NULL;  
   jxf_ParVector     *f = NULL;
   jxf_ParVector     *u = NULL;
   
   /* 是否需要粗网格校正的标志变量 */
   JXF_Int Need_CC;
   
   jxf_MPI_Comm_rank(comm, &myid);
   jxf_MPI_Comm_size(comm, &nprocs);
   
   np_R = nprocs / 3;
   np_E = np_R; 
   np_I = np_R;
   rootid_E = np_R;
   
   if (myid == 0)
   {
      N = jxf_CSRMatrixNumRows(A_s);        
   }
   
   //================================================================
   //  根据变量 np_R, np_E, np_I，生成基于进程分组的（向量）分划数组
   //  mypartition, 该数组将用来作为矩阵 A 的行列分划数组.
   //  Note: 这样可以提高块磨光时的并行效率!
   //  Note: 当 np = 1 时，该模块生成长度为 2 的一般分划数组.
   //================================================================

   if (myid == 0) 
   {
      jxf_3tGetMyPartition(nprocs, np_R, np_E, np_I, N, &mypartition);
   }
 
   
   //==============================================================================
   //  生成 (1) 整体离散系统基于整个进程组的并行数据，且具有匹配的并行分划(用于 GMRES)
   //==============================================================================
         
   A = jxf_CSRMatrixToParCSRMatrix(comm, A_s, mypartition, mypartition);
   A_partition = jxf_ParCSRMatrixRowStarts(A);
   f = jxf_VectorToParVector(comm, f_s, A_partition);
   jxf_ParVectorSetPartitioningOwner(f, 0); 
   u = jxf_VectorToParVector(comm, u_s, A_partition);
   jxf_ParVectorSetPartitioningOwner(u, 0); 


   //==============================================================================
   //  生成 (2) 子块基于进程分组的并行数据(用于 apctl 迭代)
   //==============================================================================
     
   if (groupid_x == 0)
   {  
      jxf_MatVecGroupR(comm_x, A, &ARR, &VRE);
   }
   else if (groupid_x == 1)
   {
      jxf_MatVecGroupE(comm_x, A, &AEE, &VER, &VEI);
      Need_CC = jxf_3tAPCTLWeakCouplingE(theta_wc_E, threshold_wc_E, AEE, VER, VEI);  
   }
   else if (groupid_x == 2)
   {
      jxf_MatVecGroupI(comm_x, A, &AII, &VIE);
   }   
   jxf_MPI_Bcast(&Need_CC, 1, JXF_MPI_INT, rootid_E, comm); /* 将标志变量广播给所有进程 */


   //==============================================================================
   //  生成 (3) 子块基于整体进程的并行数据(用于生成 pctl 算法中的粗矩阵 Ac).
   //==============================================================================

   if (Need_CC == 1)
   {   
      /* 将基于进程分组的并行矩阵向量转换为基于所有进程的并行矩阵向量，为生成粗矩阵作数据准备 */
      jxf_MatVecGroup2All( comm, groupid_x, 
                          ARR, AEE, AII, VRE, VER, VEI, VIE,
                          &ARR_p, &AEE_p, &AII_p, 
                          &VRE_p, &VER_p, &VEI_p, &VIE_p );  
   }
   
   
   //==========================================================
   //  返回并行数据和标志变量
   //==========================================================
   
   *ARR_p_ptr = ARR_p;
   *AEE_p_ptr = AEE_p;
   *AII_p_ptr = AII_p; 
   *VRE_p_ptr = VRE_p; 
   *VER_p_ptr = VER_p; 
   *VEI_p_ptr = VEI_p; 
   *VIE_p_ptr = VIE_p;   
   *ARR_ptr = ARR;
   *AEE_ptr = AEE;
   *AII_ptr = AII;
   *VRE_ptr = VRE;
   *VER_ptr = VER;
   *VEI_ptr = VEI;
   *VIE_ptr = VIE;
   *A_ptr = A;
   *f_ptr = f;
   *u_ptr = u;
   *Need_CC_ptr = Need_CC;
   
   return (0);
}                                        

/*!
 * \fn JXF_Int jxf_ParaDataTransSEQIF_sp
 * \brief 将串行矩阵 A_s 和串行向量 f_s，u_s 转化为两类并行数据.
 * \author peghoty 
 * \date 2012/03/10
 */
JXF_Int  
jxf_ParaDataTransSEQIF_sp( // input:
                          MPI_Comm           comm, 
                          /* 串行离散系统数据 */
                          jxf_CSRMatrix      *A_s, 
                          jxf_Vector         *f_s, 
                          jxf_Vector         *u_s, 
                          // output:
                          /* 系数矩阵各子块基于进程组的并行数据 */                         
                          jxf_ParCSRMatrix  **ARR_ptr,
                          jxf_ParCSRMatrix  **AEE_ptr,
                          jxf_ParCSRMatrix  **AII_ptr, 
                          jxf_ParVector     **VRE_ptr, 
                          jxf_ParVector     **VER_ptr, 
                          jxf_ParVector     **VEI_ptr, 
                          jxf_ParVector     **VIE_ptr,
                          /* 并行离散系统数据 */
                          jxf_ParCSRMatrix  **A_ptr,  
                          jxf_ParVector     **f_ptr, 
                          jxf_ParVector     **u_ptr )                    
{
   /* 系数矩阵各子块基于进程组的并行数据 */     
   jxf_ParCSRMatrix  *ARR = NULL;
   jxf_ParCSRMatrix  *AEE = NULL;
   jxf_ParCSRMatrix  *AII = NULL; 
   jxf_ParVector     *VRE = NULL; 
   jxf_ParVector     *VER = NULL; 
   jxf_ParVector     *VEI = NULL; 
   jxf_ParVector     *VIE = NULL;

   /* 系数矩阵各子块 */     
   jxf_CSRMatrix  *ARR_s = NULL;
   jxf_CSRMatrix  *AEE_s = NULL;
   jxf_CSRMatrix  *AII_s = NULL; 
   jxf_Vector     *VRE_s = NULL; 
   jxf_Vector     *VER_s = NULL; 
   jxf_Vector     *VEI_s = NULL; 
   jxf_Vector     *VIE_s = NULL;
   
   /* 并行离散系统数据 */
   jxf_ParCSRMatrix  *A = NULL;  
   jxf_ParVector     *f = NULL;
   jxf_ParVector     *u = NULL;

   //-------------------------------------------------------------------
   // 生成并行离散系统
   //-------------------------------------------------------------------
   A = jxf_CSRMatrixToParCSRMatrix(comm, A_s, NULL, NULL);
   f = jxf_VectorToParVector(comm, f_s, NULL);
   u = jxf_VectorToParVector(comm, u_s, NULL);

   //------------------------------------------------------------------------------------
   // 生成子块的并行数据
   //------------------------------------------------------------------------------------
   jxf_3tGetSubBlocks_REIV(A_s, &ARR_s, &AEE_s, &AII_s, &VRE_s, &VER_s, &VEI_s, &VIE_s);
   
   ARR = jxf_CSRMatrixToParCSRMatrix_sp(comm, ARR_s);
   AEE = jxf_CSRMatrixToParCSRMatrix_sp(comm, AEE_s);
   AII = jxf_CSRMatrixToParCSRMatrix_sp(comm, AII_s);
   
   VRE = jxf_VectorToParVector_sp(comm, VRE_s);
   VER = jxf_VectorToParVector_sp(comm, VER_s);
   VEI = jxf_VectorToParVector_sp(comm, VEI_s);
   VIE = jxf_VectorToParVector_sp(comm, VIE_s);
   
   /* 释放串行数据(注意不要释放数据部分!) */
   jxf_TFree(ARR_s);
   jxf_TFree(AEE_s);
   jxf_TFree(AII_s);
   jxf_TFree(VRE_s);
   jxf_TFree(VER_s);
   jxf_TFree(VEI_s);
   jxf_TFree(VIE_s);
   
   //----------------------------------
   // 返回并行数据
   //----------------------------------
   *ARR_ptr = ARR;
   *AEE_ptr = AEE;
   *AII_ptr = AII;
   *VRE_ptr = VRE;
   *VER_ptr = VER;
   *VEI_ptr = VEI;
   *VIE_ptr = VIE;
   *A_ptr = A;
   *f_ptr = f;
   *u_ptr = u;

   return (0);
}            

/*!
 * \fn JXF_Int jxf_APCTLKrylovSolGather
 * \brief 收集解向量，将并行向量收集到 0 号进程已经开设空间的串行向量上.
 * \author peghoty 
 * \date 2012/03/10
 */
JXF_Int 
jxf_APCTLKrylovSolGather( jxf_ParVector *par_sol, jxf_Vector *u_s )
{
   MPI_Comm  comm = jxf_ParVectorComm(par_sol);

   JXF_Int *partitioning = jxf_ParVectorPartitioning(par_sol);
   JXF_Int *recvcounts = NULL;   
   JXF_Real *sendbuf = NULL;
   JXF_Real *recvbuf = NULL;
   
   JXF_Int i, sendcount;
   JXF_Int myid, nprocs;

   jxf_MPI_Comm_rank(comm, &myid);
   jxf_MPI_Comm_size(comm, &nprocs);

   /* 在 0 号进程生成辅助数组 recvcounts */
   if (myid == 0)
   {
      recvcounts = jxf_CTAlloc(JXF_Int, nprocs);
      for (i = 0; i < nprocs; i ++)
      {
         recvcounts[i] = partitioning[i+1] - partitioning[i];
      }    
   }

   /* 确定 0 号进程需从各进程接收数据的个数 */
   sendcount = jxf_VectorSize(jxf_ParVectorLocalVector(par_sol));

   /* 将各进程上的解向量分量收集到 0 号进程的串行解向量中 */
   sendbuf = jxf_VectorData(jxf_ParVectorLocalVector(par_sol));
   if (myid == 0) recvbuf = jxf_VectorData(u_s);   
   jxf_MPI_Gatherv(sendbuf, sendcount, JXF_MPI_REAL, recvbuf, recvcounts, partitioning, JXF_MPI_REAL, 0, comm); 

   /* 释放数组 recvcounts */    
   if (myid == 0)
   {
      jxf_TFree(recvcounts);
   }
   
   return (0);
}

JXF_Int 
jxf_mgGenerateSubBlocks( MPI_Comm           comm,
                        MPI_Comm           comm_x,
                        JXF_Int             groupid_x,
                        JXF_Int             ng,
                        jxf_ParCSRMatrix   *par_mat,
                        jxf_ParCSRMatrix  **ARR_ptr,
                        jxf_ParCSRMatrix  **AEE_ptr,
                        jxf_ParCSRMatrix  **AII_ptr,
                        jxf_ParVector     **VRE_ptr,
                        jxf_ParVector    ***VER_ptr,
                        jxf_ParVector     **VEI_ptr,
                        jxf_ParVector     **VIE_ptr )
{
   jxf_ParCSRMatrix  *ARR = NULL;
   jxf_ParCSRMatrix  *AEE = NULL;
   jxf_ParCSRMatrix  *AII = NULL;
   jxf_ParVector     *VRE = NULL;
   jxf_ParVector    **VER = NULL;
   jxf_ParVector     *VEI = NULL;
   jxf_ParVector     *VIE = NULL;

   jxf_CSRMatrix *A_comb  = NULL;
   jxf_CSRMatrix *A_loc   = NULL;
   jxf_Vector    *VRE_loc = NULL;
   jxf_Vector   **VER_loc = NULL;
   jxf_Vector    *VEI_loc = NULL;
   jxf_Vector    *VIE_loc = NULL;

   JXF_Real *aa_comb = NULL;
   JXF_Int  *ia_comb = NULL;
   JXF_Int  *ja_comb = NULL;
   JXF_Real *aa_loc = NULL;
   JXF_Int  *ia_loc = NULL;
   JXF_Int  *ja_loc = NULL;

   JXF_Int *row_starts_glo = jxf_ParCSRMatrixRowStarts(par_mat);
   JXF_Int *ercnt = jxf_CTAlloc(JXF_Int, ng);
   JXF_Int *row_starts_loc = NULL;
   JXF_Int *col_starts_loc = NULL;

   JXF_Int N = jxf_ParCSRMatrixGlobalNumRows(par_mat);
   JXF_Int n = N / (ng + 2);

   JXF_Int myid, myid_x, nprocs, num_proc_pg;
   JXF_Int incre, decre, row, j, col, first_col_diag, last_col_diag;
   JXF_Int srt_row, end_row, eee_row, iii_row;
   JXF_Int nz_rr, nz_re, nz_ee, nz_ei, nz_ie, nz_ii;

   jxf_MPI_Comm_rank(comm, &myid);
   jxf_MPI_Comm_rank(comm_x, &myid_x);
   jxf_MPI_Comm_size(comm, &nprocs);

   num_proc_pg = nprocs / (ng + 2);
   incre = groupid_x * num_proc_pg;
   row_starts_loc = jxf_CTAlloc(JXF_Int, num_proc_pg+1);
   for (j = 0; j <= num_proc_pg; j ++) row_starts_loc[j] = row_starts_glo[incre+j] - row_starts_glo[incre];
   col_starts_loc = row_starts_loc;

   A_comb = jxf_MergeDiagAndOffd(par_mat);
   aa_comb = jxf_CSRMatrixData(A_comb);
   ia_comb = jxf_CSRMatrixI(A_comb);
   ja_comb = jxf_CSRMatrixJ(A_comb);
   if (groupid_x == ng)
   {
      decre = (ng + 1) * jxf_CSRMatrixNumRows(A_comb);
   }
   else
   {
      decre = jxf_CSRMatrixNumRows(A_comb);
   }
   A_loc = jxf_CSRMatrixCreate(jxf_CSRMatrixNumRows(A_comb), n, jxf_CSRMatrixNumNonzeros(A_comb)-decre);
   jxf_CSRMatrixInitialize(A_loc);
   aa_loc = jxf_CSRMatrixData(A_loc);
   ia_loc = jxf_CSRMatrixI(A_loc);
   ja_loc = jxf_CSRMatrixJ(A_loc);

   if (groupid_x < ng) // 光子进程组
   {
      VRE_loc = jxf_SeqVectorCreate(jxf_CSRMatrixNumRows(A_comb));
      jxf_SeqVectorInitialize(VRE_loc);
   }
   else if (groupid_x == ng) // 电子进程组
   {
      VER_loc = jxf_CTAlloc(jxf_Vector *, ng);
      for (j = 0; j < ng; j ++)
      {
         VER_loc[j] = jxf_SeqVectorCreate(jxf_CSRMatrixNumRows(A_comb));
         jxf_SeqVectorInitialize(VER_loc[j]);
      }
      VEI_loc = jxf_SeqVectorCreate(jxf_CSRMatrixNumRows(A_comb));
      jxf_SeqVectorInitialize(VEI_loc);
   }
   else if (groupid_x == ng+1) // 离子进程组
   {
      VIE_loc = jxf_SeqVectorCreate(jxf_CSRMatrixNumRows(A_comb));
      jxf_SeqVectorInitialize(VIE_loc);
   }

   eee_row = ng * n;
   iii_row = eee_row + n;
   if (groupid_x < ng) // 光子进程组
   {
      srt_row = groupid_x * n;
      end_row = srt_row + n;
      nz_rr = 0;
      nz_re = 0;
	  ia_loc[0] = 0;
      for (row = 0; row < jxf_CSRMatrixNumRows(A_comb); row ++)
      {
         for (j = ia_comb[row]; j < ia_comb[row+1]; j ++)
         {
            col = ja_comb[j];
            if ((col >= srt_row) && (col < end_row))
            {
               ja_loc[nz_rr] = col - srt_row;
               aa_loc[nz_rr] = aa_comb[j];
               nz_rr ++;
            }
            else if ((col >= eee_row) && (col < iii_row))
            {
               jxf_VectorData(VRE_loc)[nz_re++] = aa_comb[j];
            }
            else if (col >= iii_row)
            {
               jxf_printf("\033[31m >>> Ar[%d]i != 0\033[00m\n", groupid_x);
               exit(0);
            }
            else
            {
               jxf_printf("\033[31m >>> Ar[%d]r[%d]-[%d][%d] != 0\033[00m\n", groupid_x, col/n, row, col);
               exit(0);
            }
         }
         ia_loc[row+1] = nz_rr;
      }
      ARR = jxf_ParCSRMatrixCreate(comm_x, n, n, row_starts_loc, col_starts_loc, 0, 0, 0);
      first_col_diag = col_starts_loc[myid_x];
      last_col_diag  = col_starts_loc[myid_x+1] - 1;
      jxf_GenerateDiagAndOffd(A_loc, ARR, first_col_diag, last_col_diag);
      VRE = jxf_ParVectorCreate(comm_x, n, row_starts_loc);
      jxf_ParVectorSetPartitioningOwner(VRE, 0);
      jxf_TFree(jxf_ParVectorLocalVector(VRE));
      jxf_ParVectorLocalVector(VRE) = VRE_loc;
   }
   else if (groupid_x == ng) // 电子进程组
   {
      nz_ee = 0;
      nz_ei = 0;
      ia_loc[0] = 0;
      for (row = 0; row < jxf_CSRMatrixNumRows(A_comb); row ++)
      {
         for (j = ia_comb[row]; j < ia_comb[row+1]; j ++)
         {
            col = ja_comb[j];
            if (col < eee_row)
            {
               jxf_VectorData(VER_loc[col/n])[ercnt[col/n]++] = aa_comb[j];
            }
            else  if ((col >= eee_row) && (col < iii_row))
            {
               ja_loc[nz_ee] = col - eee_row;
               aa_loc[nz_ee] = aa_comb[j];
               nz_ee ++;
            }
            else
            {
               jxf_VectorData(VEI_loc)[nz_ei++] = aa_comb[j];
            }
         }
         ia_loc[row+1] = nz_ee;
      }
      AEE = jxf_ParCSRMatrixCreate(comm_x, n, n, row_starts_loc, col_starts_loc, 0, 0, 0);
      first_col_diag = col_starts_loc[myid_x];
      last_col_diag  = col_starts_loc[myid_x+1] - 1;
      jxf_GenerateDiagAndOffd(A_loc, AEE, first_col_diag, last_col_diag);
      VER = jxf_CTAlloc(jxf_ParVector *, ng);
      for (j = 0; j < ng; j ++)
      {
         VER[j] = jxf_ParVectorCreate(comm_x, n, row_starts_loc);
         jxf_ParVectorSetPartitioningOwner(VER[j], 0);
         jxf_TFree(jxf_ParVectorLocalVector(VER[j]));
         jxf_ParVectorLocalVector(VER[j]) = VER_loc[j];
      }
      VEI = jxf_ParVectorCreate(comm_x, n, row_starts_loc);
      jxf_ParVectorSetPartitioningOwner(VEI, 0);
      jxf_TFree(jxf_ParVectorLocalVector(VEI));
      jxf_ParVectorLocalVector(VEI) = VEI_loc;
   }
   else if (groupid_x == ng+1) // 离子进程组
   {
      nz_ii = 0;
      nz_ie = 0;
      ia_loc[0] = 0;
      for (row = 0; row < jxf_CSRMatrixNumRows(A_comb); row ++)
      {
         for (j = ia_comb[row]; j < ia_comb[row+1]; j ++)
         {
            col = ja_comb[j];
            if (col < eee_row)
            {
               jxf_printf("\033[31m >>> Air[%d] != 0\033[00m\n", col/n);
               exit(0);
            }
            else if (col < iii_row)
            {
               jxf_VectorData(VIE_loc)[nz_ie++] = aa_comb[j];
            }
            else
            {
               ja_loc[nz_ii] = col - iii_row;
               aa_loc[nz_ii] = aa_comb[j];
               nz_ii ++;
            }
         }
         ia_loc[row+1] = nz_ii;
      }
      AII = jxf_ParCSRMatrixCreate(comm_x, n, n, row_starts_loc, col_starts_loc, 0, 0, 0);
      first_col_diag = col_starts_loc[myid_x];
      last_col_diag  = col_starts_loc[myid_x+1] - 1;
      jxf_GenerateDiagAndOffd(A_loc, AII, first_col_diag, last_col_diag);
      VIE = jxf_ParVectorCreate(comm_x, n, row_starts_loc);
      jxf_ParVectorSetPartitioningOwner(VIE, 0);
      jxf_TFree(jxf_ParVectorLocalVector(VIE));
      jxf_ParVectorLocalVector(VIE) = VIE_loc;
   }

   jxf_CSRMatrixDestroy(A_comb);
   jxf_CSRMatrixDestroy(A_loc);
   jxf_TFree(ercnt);

  *ARR_ptr = ARR;
  *AEE_ptr = AEE;
  *AII_ptr = AII;
  *VRE_ptr = VRE;
  *VER_ptr = VER;
  *VEI_ptr = VEI;
  *VIE_ptr = VIE;

   return (0);
}
