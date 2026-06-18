
//========================================================================//
//  JXFPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2021        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 * jxfpamg_interface.c -- this is an interface function that calls jxfpamg's
   Krylov iteration method to solve a given linear system.
 *
 * One of the following solvers can be used by
 * assigning parameter 'solver_id'
 *
 *  solver_id = 0:  PAMG
 *  solver_id = 11: CG
 *  solver_id = 12: PAMG-CG
 *  solver_id = 13: DS-CG
 *  solver_id = 14: Euclid-CG
 *  solver_id = 21: GMRES
 *  solver_id = 22: PAMG-GMRES
 *  solver_id = 23: DS-GMRES
 *  solver_id = 24: Euclid-GMRES
 *  solver_id = 25: ILU-GMRES
 *  solver_id = 26: ILU-AdaptiveGMRES
 *  solver_id = 28: Euclid2-AdaptiveGMRES
 *  solver_id = 29: Euclid3-AdaptiveGMRES
 *  solver_id = 31: BiCGSTAB
 *  solver_id = 32: PAMG-BiCGSTAB
 *  solver_id = 33: DS-BiCGSTAB
 *  solver_id = 34: Euclid-BiCGSTAB
 *  solver_id = 42: PAMG-FlexGMRES
 *  solver_id = 62: PAMG-COGMRES
 *
 * combined preconditioners
 *
 *  solver_id = 17: ILU-PAMG-ILU-CG
 *  solver_id = 18: PAMG-ILU-PAMG-CG
 *  solver_id = 51: PAMG-Euclid-PAMG-GMRES
 *  solver_id = 52: Euclid-PAMG-GMRES
 *  solver_id = 53: PAMG-Euclid-GMRES
 *  solver_id = 54: Euclid-PAMG-Euclid-GMRES
 *  solver_id = 56: ILU-PAMG-GMRES
 *  solver_id = 57: ILU-PAMG-ILU-GMRES
 *
 *  solver_id = 101: PARDISO
 *
 *
 *  Modified by Zhao li, Yue Xiaoqiang 2021/05/29
 *
 *
 *  Xiangtan University
 *  yuexq1111@163.com
 *
 */

#include "jxf_pamg.h"
#include "jxf_ilu.h"
#include "jxf_krylov.h"
#include "jxf_diagscale.h"
#include "jxf_euclid.h"
#include "jxf_combined.h"
// #define JXF_USING_BIG_DOUBLE 1
#define JXF_LOW_REAL float

#define PRINT_MIN  0
#define DEBUG_MODE 0

// #include "jxf_pamg.h"
// #include "jxf_apctl.h"
// #include "jxf_euclid.h"
// #include "jxf_multils.h"
// #include "jxf_smp_forloop.h"
// #include "jxf_combined.h"
// #include "jxf_ilu.h"
// #include "jxf_mv.h"
// #include "jxf_util.h"
// #include "jxf_diagscale.h"
// #include "jxf_krylov.h"

#if 0
JXF_Int
jxf_LU( jxf_ParCSRMatrix *par_matrix,
               jxf_ParVector    *par_rhs,
               jxf_ParVector    *par_app);
               JXF_Int
jxf_gselim_lu( JXF_LOW_REAL *A_matrix, JXF_LOW_REAL *x_vector, JXF_Int size );
JXF_Int jxf_Pre_Pardiso(Pardiso_data *pdata,jxf_ParCSRMatrix *par_matrix,
               jxf_ParVector    *par_rhs,
               jxf_ParVector    *par_app);
JXF_Int
jxf_pardiso_double(jxf_CSRMatrix *AA, jxf_Vector *bb, jxf_Vector *xx);

JXF_Int
jxf_pardiso_float(jxf_CSRMatrix *AA, jxf_Vector *bb, jxf_Vector *xx);
#endif
JXF_Int jxf_CSRMatrixFilter( jxf_CSRMatrix *matrix, JXF_Real eps, jxf_CSRMatrix *matrixOut);
JXF_Int JXFPAMG_Solver(JXF_Int                 argc,
                     char                *argv[],
                     jxf_CSRMatrix        *As,
                     MPI_Comm            comm,
                     jxf_ParCSRMatrix     *par_matrix,
                     jxf_ParVector        *par_rhs,
                     jxf_ParVector        *par_sol,
                     JXF_Real              tol,
                     JXF_Int                 num_f,
                     JXF_Int                 solver_id,
                     JXF_Int                 *iter);

// 先几何节点后物理量(3个量)，转化为先物理量后几何节点
JXF_Int TranMatrixOrder( JXF_Int n, JXF_Int nnz, JXF_Int *ia, JXF_Int *ja, JXF_Real *aa , JXF_Real *b);
JXF_Int TranMatrixOrder_new( JXF_Int n, JXF_Int nnz, JXF_Int *ia, JXF_Int *ja, JXF_Real *aa , JXF_Real *b);





JXF_Int JXFPAMG_Solver_Interface(JXF_Int argc, char *argv[],MPI_Comm comm,JXF_Int row, JXF_Int *ia, JXF_Int  *ja, JXF_Real *a,
                               JXF_Real *ser_x, JXF_Real *ser_b, JXF_Real tol, JXF_Int solver_id, JXF_Int num_functions, JXF_Int *iter)
{
   JXF_Int i,j;
   int myid,num_procs;

   JXF_Int n = row;
   JXF_Int nnz;
   JXF_Real startwtime, endwtime, t1, t2;

   jxf_CSRMatrix    *A_Ser;
   jxf_Vector       *b_Ser;
   jxf_Vector       *x_Ser;

   jxf_ParCSRMatrix  *A_Par;
   jxf_ParVector     *b_Par;
   jxf_ParVector     *x_Par;

   JXF_Int       *row_part=NULL;
   JXF_Int       *col_part=NULL;
   // MPI_Comm comm = MPI_COMM_WORLD;
   MPI_Comm_rank(comm, &myid );
   MPI_Comm_size(comm, &num_procs);

   // 若编号从1开始，则减1
   if (myid ==0 ){
      if (ia[0] == 1) {
         nnz = ia[n]-1;
         for (i=0; i<=n; i++) ia[i]--;
         for (i=0; i<nnz; i++) ja[i]--;
      } else
      nnz = ia[n];
   }

   // if (myid == 0 && num_functions > 1){
   //    t1 = jxf_MPI_Wtime();
   //    TranMatrixOrder( n, nnz, ia, ja, a, ser_b);
   //    t2 = jxf_MPI_Wtime();
   //    jxf_printf("\n >>> 先几何节点后物理量(3个量)转化为先物理量后几何节点的时间:  %.4f(s)\n\n", t2 - t1);
   // }
   if (myid == 0) t1 = jxf_MPI_Wtime();
   // 创建串行矩阵，并划分矩阵
   if (myid ==0 ){
      // TranMatrixOrder_new(n,nnz,ia,ja,a,ser_b);
      A_Ser = jxf_CSRMatrixCreate(n, n, nnz);
      jxf_CSRMatrixI(A_Ser) = ia;
      jxf_CSRMatrixJ(A_Ser) = ja;
      jxf_CSRMatrixData(A_Ser) = a;
      jxf_CSRMatrixInitialize(A_Ser);
      jxf_CSRMatrixReorder( (jxf_CSRMatrix *)A_Ser);

      b_Ser = jxf_SeqVectorCreate(n);
      jxf_VectorData(b_Ser)= ser_b;
      jxf_SeqVectorInitialize(b_Ser);

      x_Ser = jxf_SeqVectorCreate(n);
      jxf_VectorData(x_Ser)=ser_x;
      jxf_SeqVectorInitialize(x_Ser);

      if (num_functions > 1)
      {
         JXF_Int num_nodes,size,rest;
         num_nodes = n/num_functions;

         if (n != num_functions*num_nodes)
         {
            row_part = NULL;
            col_part = NULL;
         }
         else
         {
            row_part = jxf_CTAlloc(JXF_Int, num_procs+1);
            row_part[0] = 0;
            size = num_nodes/num_procs;
            rest = num_nodes-size*num_procs;
            for (i=0; i < num_procs; i++)	 {
               row_part[i+1] = row_part[i]+size*num_functions;
               if (i < rest) row_part[i+1] += num_functions;
            }
            col_part = row_part;
         }
      }
   }

   //	printf("%s  %s : %d  \n",__FILE__,  __FUNCTION__,__LINE__ );
   A_Par = jxf_CSRMatrixToParCSRMatrix(comm, A_Ser, row_part, col_part);

   int *partitioning;
   jxf_ParCSRMatrixGetRowPartitioning(A_Par, &partitioning);
   b_Par = jxf_VectorToParVector(comm, b_Ser, partitioning);
   x_Par = jxf_ParVectorCreate(comm, jxf_ParVectorGlobalSize(b_Par), partitioning);
   jxf_ParVectorSetPartitioningOwner(x_Par, 0);
   jxf_ParVectorInitialize(x_Par);
   jxf_ParVectorSetConstantValues(x_Par, 0.0);
   if (myid == 0){
      t2 = jxf_MPI_Wtime();
      printf("\n >>> 串行矩阵转并行矩阵的时间 :  %.4f(s)\n\n", t2 - t1);
   }

   /*--------------------------------------------------------------------------
   * Solving the system using boomer AMG as follows
   *--------------------------------------------------------------------------*/
   if (myid == 0) startwtime = jxf_MPI_Wtime();
   // mumps_Interface_float(comm,row, nnz, ia,ja, a, ser_x, ser_b);
   JXFPAMG_Solver(argc, argv, A_Ser, comm, A_Par, b_Par, x_Par, tol, solver_id, num_functions, iter);


   if (myid == 0)
   {
      endwtime = jxf_MPI_Wtime();
      printf(" >>>Time of JXFPAMG_Solver :  %.4f(s)\n", endwtime-startwtime);
   }

   if (myid == 0) t1 = jxf_MPI_Wtime();
   //printf("%s  %s : %d  \n",__FILE__,  __FUNCTION__,__LINE__ );
   x_Ser = jxf_ParVectorToVectorAll(x_Par);

   /// output
   if (myid == 0){
      for (i = 0; i < n; i++)  ser_x [i] = x_Ser->data[i];
   }
   if (myid == 0){
      t2 = jxf_MPI_Wtime();
      printf("\n >>> 并行向量转串行向量的时间 :  %.4f(s)\n", t2 - t1);
   }


   // if (myid == 0){
   jxf_ParCSRMatrixDestroy(A_Par);
   A_Par = NULL;
   jxf_ParVectorDestroy(b_Par);
   b_Par = NULL;
   jxf_ParVectorDestroy(x_Par);
   x_Par = NULL;
   jxf_SeqVectorDestroy(x_Ser);
   x_Ser =NULL;
   // }
   ///
   return (0);
}


JXF_Int JXFPAMG_Solver_Interface_omp(JXF_Int argc, char *argv[], JXF_Int row, JXF_Int *ia, JXF_Int  *ja, JXF_Real *a,
                               JXF_Real *ser_x, JXF_Real *ser_b, JXF_Real tol, JXF_Int solver_id, JXF_Int num_functions, JXF_Int *iter)
{
   JXF_Int i,j;
   int myid,num_procs;

   JXF_Int n = row;
   JXF_Int nnz;
   JXF_Real startwtime, endwtime, t1, t2;

   jxf_CSRMatrix    *A_Ser;
   jxf_Vector       *b_Ser;
   jxf_Vector       *x_Ser;

   jxf_ParCSRMatrix  *A_Par;
   jxf_ParVector     *b_Par;
   jxf_ParVector     *x_Par;

   JXF_Int       *row_part=NULL;
   JXF_Int       *col_part=NULL;
   MPI_Comm comm = MPI_COMM_WORLD;
   MPI_Comm_rank(comm, &myid );
   MPI_Comm_size(comm, &num_procs);

   // 若编号从1开始，则减1
   if (myid ==0 ){
      if (ia[0] == 1) {
         nnz = ia[n]-1;
         for (i=0; i<=n; i++) ia[i]--;
         for (i=0; i<nnz; i++) ja[i]--;
      } else
      nnz = ia[n];
   }

   // if (myid == 0 && num_functions > 1){
   //    t1 = jxf_MPI_Wtime();
   //    TranMatrixOrder( n, nnz, ia, ja, a, ser_b);
   //    t2 = jxf_MPI_Wtime();
   //    jxf_printf("\n >>> 先几何节点后物理量(3个量)转化为先物理量后几何节点的时间:  %.4f(s)\n\n", t2 - t1);
   // }
   if (myid == 0) t1 = jxf_MPI_Wtime();
   // 创建串行矩阵，并划分矩阵
   if (myid ==0 ){
      A_Ser = jxf_CSRMatrixCreate(n, n, nnz);
      jxf_CSRMatrixI(A_Ser) = ia;
      jxf_CSRMatrixJ(A_Ser) = ja;
      jxf_CSRMatrixData(A_Ser) = a;
      jxf_CSRMatrixInitialize(A_Ser);
      jxf_CSRMatrixReorder( (jxf_CSRMatrix *)A_Ser);

      b_Ser = jxf_SeqVectorCreate(n);
      jxf_VectorData(b_Ser)= ser_b;
      jxf_SeqVectorInitialize(b_Ser);

      x_Ser = jxf_SeqVectorCreate(n);
      jxf_VectorData(x_Ser)=ser_x;
      jxf_SeqVectorInitialize(x_Ser);

      if (num_functions > 1)
      {
         JXF_Int num_nodes,size,rest;
         num_nodes = n/num_functions;

         if (n != num_functions*num_nodes)
         {
            row_part = NULL;
            col_part = NULL;
         }
         else
         {
            row_part = jxf_CTAlloc(JXF_Int, num_procs+1);
            row_part[0] = 0;
            size = num_nodes/num_procs;
            rest = num_nodes-size*num_procs;
            for (i=0; i < num_procs; i++)	 {
               row_part[i+1] = row_part[i]+size*num_functions;
               if (i < rest) row_part[i+1] += num_functions;
            }
            col_part = row_part;
         }
      }
   }

   A_Par = jxf_CSRMatrixToParCSRMatrix_sp(comm, A_Ser);
   b_Par = jxf_VectorToParVector_sp(comm, b_Ser);
   x_Par = jxf_VectorToParVector_sp(comm, x_Ser);


   /*--------------------------------------------------------------------------
   * Solving the system using boomer AMG as follows
   *--------------------------------------------------------------------------*/
   if (myid == 0) startwtime = jxf_MPI_Wtime();
   JXFPAMG_Solver(argc, argv, A_Ser, comm, A_Par, b_Par, x_Par, tol, solver_id, num_functions, iter);

   if (myid == 0)
   {
      endwtime = jxf_MPI_Wtime();
      jxf_printf(" >>>Time of JXFPAMG_Solver :  %.4f(s)\n", endwtime-startwtime);
   }

   ///
   return (0);
}







JXF_Int JXFPAMG_Solver(JXF_Int              argc,
                  char                *argv[],
                  jxf_CSRMatrix        *As,
                  MPI_Comm            comm,
                  jxf_ParCSRMatrix     *par_matrix,
                  jxf_ParVector        *par_rhs,
                  jxf_ParVector        *par_sol,
                  JXF_Real              tol,
                  JXF_Int               solver_id,
                  JXF_Int               num_functions,
                  JXF_Int               *iter)
{
   // MPI_Comm  comm = MPI_COMM_WORLD;
   JXF_Int       myid, nprocs;
   JXF_Int       arg_index   = 0;
   JXF_Int       print_usage = 0;

#if JXF_USING_OPENMP || defined (JXF_USING_PGCC_SMP)
   JXF_Int       nthreads;
#endif

   JXF_Real starttime, endtime;
   JXF_Real starttimeT, endtimeT;

   char *MatFile = NULL;
   char *RhsFile = NULL;
   char *GusFile = NULL;
   char  AppFile[120];

   // jxf_ParCSRMatrix *par_matrix = NULL;
   // jxf_ParVector    *par_rhs    = NULL;
   // jxf_ParVector    *par_sol    = NULL;
   jxf_CSRMatrix    *ser_matrix = NULL, *As_hat = NULL;
   jxf_ParCSRMatrix   *Ap_hat = NULL;

   JXF_Int *partitioning = NULL;

   JXF_Int **grid_relax_points = NULL;
   JXF_Int *num_grid_sweeps = NULL;

   /* DiagScale Precond */
   JXF_Solver ds_solver;

   /* ILU Precond */
   JXF_Solver ilu_solver;
   JXF_Real drop_tol;

   /* Euclid solver */
   JXF_Solver euclid_solver;
   JXF_Int euclid_level;
   JXF_Int euclid_bj;

   /* Combined */
   JXF_Solver combined_solver;
   JXF_Int theta_psi;
   JXF_Int theta_rho;
   JXF_Int theta_phi;
   JXF_Real theta_dis;

   /* JXFPAMG solver */
   JXF_Solver   amg_solver;
   JXF_Int       max_levels;
   JXF_Int       cycle_type;
   JXF_Int       relax_type;
   JXF_Int       measure_type;
   JXF_Int       rap2;
   // JXF_Int       num_functions;
   JXF_Int       ns_down;
   JXF_Int       ns_up;
   JXF_Int       ns_coarse;
   JXF_Int       restri_type;
   JXF_Int       keepTranspose;
   JXF_Int       coarsen_type;
   JXF_Int       coarse_solver;
   JXF_Int       interp_type;
   JXF_Int       P_max_elmts;
   JXF_Int       agg_num_levels;
   JXF_Int       ai_measure_type;
   JXF_Int       ai_relax_type;
   JXF_Real    strong_threshold;
   JXF_Real    max_row_sum;

   JXF_Real    relax_wt;
   JXF_Real    outer_wt;

   JXF_Real     S_commpkg_switch;
   JXF_Real     AIR_strong_th;

   JXF_Int       coarse_threshold;
   JXF_Real     coarse_ratio;
   JXF_Int       coarsestsolverid;
   JXF_Int       conv_criteria;
   JXF_Int       amg_print_level;
   JXF_Int       CF;

   /* iterative method */
   JXF_Solver   solver;
   JXF_Real     resdown_0_threshold;
   JXF_Real     convfac_threshold_2;
   JXF_Int       max_iter;
   JXF_Int       k_dim;
   JXF_Int       is_check_restarted;  /* peghoty, 2011/11/08 */
   JXF_Int       twonorm;
   JXF_Int       problem_id;
   JXF_Int       file_base;
   JXF_Int       print_level;
   JXF_Int       keepsol;
   JXF_Int       TTest;
   JXF_Int       cgs;
   JXF_Int       unroll;

   /* other variables */
   JXF_Int       lu_length;
   JXF_Int       glosize, i;
   JXF_Int       last_precond_type;
   JXF_Int       initguess = 0;
   JXF_Int       num_iterations;
   JXF_Real      final_res_norm;
   JXF_Real      norm;
   JXF_Real      eps = 1e-8;
   JXF_Real      trunc_factor;
   //--------------------------
   // 获取进程数和进程编号
   //--------------------------
   jxf_MPI_Comm_rank(comm, &myid);
   jxf_MPI_Comm_size(comm, &nprocs);

   //-----------------------
   //  参数设置
   //-----------------------
   max_levels       = 25;       /* 最大网格层数 */
   cycle_type       = 1;        /* Cycle 类型  1: V_Cycle; 2：W_Cycle */
   relax_type       = 3;        /* Relax 类型  3: hGS; 6：hSGS */
   coarse_solver    = 9;        /* coarse_solver 粗空间解法器类型  9: GE; 10: Pardiso */
   measure_type     = 0;        /* 影响值的计算方式 0：局部；1：全局 */
   rap2             = 0;        /* RAP计算方式  0：RAP；1：先算Q=AP，再算RQ */
   // num_functions    = 1;     /* 一个节点自由度的个数 */
   ns_down          = 1;
   ns_up            = 1;
   ns_coarse        = 1;
   restri_type      = 0;        /* 限制算子类型 0: P^T, 1: AIR, 2: AIR-2 */
   keepTranspose    = 0;        /* 存放限制算子  0：no；1：yes */
   CF               = -1;
                           // zhaoli,2021.06.29,
                           // CF=-1: 缺省值（ CF = 1），
                           // CF=0: 自然序，
                           // CF=1: down cycle 先 C 后 F, up cycle 先 F 后 C
                           // CF=2: down cycle 先 F 后 C, up cycle 先 C 后 F


   coarsen_type     = 10;        /* 粗化策略 */
   // coarsen_type = 0: CLJP
   // coarsen_type = 1: Ruge
   // coarsen_type = 11: Ruge 1st pass only
   // coarsen_type = 2: Ruge2B
   // coarsen_type = 3: Ruge3
   // coarsen_type = 4: Ruge3c
   // coarsen_type = 5: Ruge relax special points
   // coarsen_type = 6: Falgout
   // coarsen_type = 8: PMIS
   // coarsen_type = 10: HMIS
   // coarsen_type = 90: RCLJP
   // coarsen_type = 91: RRS0
   // coarsen_type = 990: CLJP_AI
   // coarsen_type = 991: Ruge_AI
   // coarsen_type = 993: Ruge3_AI
   // coarsen_type = 96: Falgout_AI
   // coarsen_type = 98: PMIS_AI
   // coarsen_type = 910: HMIS_AI
   // coarsen_type = 908, 918, 928, 938, 968: AI-TYPE.

   interp_type      = 0;        /* 插值策略 */
   // interp_type = 0: modified classical interpolation
   // interp_type = 3: direct interpolation (with separation of weights)
   // interp_type = 4: multipass interpolation
   // interp_type = 5: multipass interpolation (with separation of weights)
   // interp_type = 6: extended classical modified interpolation
   // interp_type = 7: extended (if no common C neighbor) classical modified interpolation
   // interp_type = 8: standard interpolation
   // interp_type = 9: standard interpolation (with separation of weights)
   trunc_factor     = 0;        /* 插值矩阵截断因子 */
   P_max_elmts      = 0;        /* 插值算子每行最大非零元个数 */
   agg_num_levels   = 0;        /* Aggressive粗化的层数 */
   ai_measure_type  = 0;        /* AI-策略  0: no; 1: yes */
   ai_relax_type    = 0;        /* AI-磨光  0: no; 1: yes */
   amg_print_level  = 0;        /* work only when AMG as preconditioner */
   //strong_threshold = 0.1;     /* 强弱连通参数, 0.25 for 2D, 0.5 for 3D is recommended */
   strong_threshold = 0.1;
   max_row_sum      = 0.9;      /* 行和参数 */
   relax_wt         = 1.0;
   outer_wt         = 1.0;
   S_commpkg_switch = 1.0;
   AIR_strong_th    = 0.25;

   coarse_threshold = 25;      /* 最粗网格层上网格节点个数的最大值 */
   coarse_ratio     = 0.75;     /* 相邻两个网格层的粗点个数超过细点个数的 coarse_ratio, 则换成 CLJP 粗化 */
   coarsestsolverid = 9;        /* 最粗网格层解法器 */
   conv_criteria    = 0;        /* 收敛准则类型 */

   theta_psi = 4;               /* 控制多尺度强度的度量 */
   theta_rho = 3;               /* 控制多尺度分布的第一个度量 */
   theta_phi = 3;               /* 控制多尺度分布的第二个度量 */
   theta_dis = 1.0e-3;          /* 控制多尺度分布的个数 */

   drop_tol         = 0.0;      /* Drop-tolerance for ILU(0) factorization */

   euclid_level     = 1;        /* level of fill-in */
   euclid_bj        = 0;        /* Select PILU (0) or Block Jacobi ILU (1) */

   //tol                 = 1.0e-7; /* 控制精度 */
   resdown_0_threshold = 1.0e-4; /* 第一次残量相对初始残差的下降阈值 */
   convfac_threshold_2 = 0.1;    /* 相继两次残量下降阈值 */
   max_iter            = 1000;   /* 迭代法最大迭代次数 */
   k_dim               = 30;     /* 回头数 */
   is_check_restarted  = 1;      /* peghoty, 2011/11/08 */
   twonorm             = 1;      /* PCG 法中的范数控制类型，0: B 范数; 1: l2 范数 */
   print_level         = 0;      /* 0: 关闭；1：Setup参数；2：Solve参数；3：Setup+Solve参数 */
   keepsol             = 0;      /* 是否保存解向量 */
   TTest               = 1;      /* 是否测试时间 */
   cgs                 = 1;      /* COGMRES: if 2 performs reorthogonalization */
   unroll              = 0;      /* COGMRES: Set number of unrolling in mass funcyions, can be 4 or 8. Default: no unrolling */
   // solver_id           = 22;
   problem_id          = 1;
   file_base           = 1;
#if JXF_USING_OPENMP || defined (JXF_USING_PGCC_SMP)
   nthreads            = 1;      /* 线程数 */
#endif



   //-----------------------
   //  命令行修改参数
   //-----------------------
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-sid") == 0 )
      {
         arg_index ++;
         solver_id = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-pid") == 0 )
      {
         arg_index ++;
         problem_id = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-maxiter") == 0 )
      {
         arg_index ++;
         max_iter = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-cyt") == 0 )
      {
         arg_index ++;
         cycle_type = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mxl") == 0 )
      {
         arg_index ++;
         max_levels = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-cf") == 0 )
      {
         arg_index ++;
         CF = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-cs") == 0 )
      {
         arg_index ++;
         coarse_solver = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ns_down") == 0 )
      {
         arg_index ++;
         ns_down = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ns_coarse") == 0 )
      {
         arg_index ++;
         ns_coarse = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ns_up") == 0 )
      {
         arg_index ++;
         ns_up = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-eps") == 0 )
      {
         arg_index ++;
         eps = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-wt") == 0 )
      {
         arg_index ++;
         relax_wt = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-cgs") == 0 )
      {
         arg_index ++;
         cgs = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-unroll") == 0 )
      {
         arg_index ++;
         unroll = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-fb") == 0 )
      {
         arg_index ++;
         file_base = atoi(argv[arg_index++]);
      }
      // else if ( strcmp(argv[arg_index], "-nf") == 0 )
      // {
      //    arg_index ++;
      //    num_functions = atoi(argv[arg_index++]);
      // }
#if JXF_USING_OPENMP || defined (JXF_USING_PGCC_SMP)
      else if ( strcmp(argv[arg_index], "-nts") == 0 )
      {
         arg_index ++;
         nthreads = atoi(argv[arg_index++]);
      }
#endif
      else if ( strcmp(argv[arg_index], "-rap2") == 0 )
      {
         arg_index ++;
         rap2 = 1;
      }
      else if ( strcmp(argv[arg_index], "-air") == 0 )
      {
         arg_index ++;
         restri_type = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-kt") == 0 )
      {
         arg_index ++;
         keepTranspose = 1;
      }
      else if (strcmp(argv[arg_index], "-Pmx") == 0)
      {
         arg_index ++;
         P_max_elmts = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-agg_nl") == 0 )
      {
         arg_index ++;
         agg_num_levels = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-tnrm") == 0 )
      {
         arg_index ++;
         twonorm = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-kdim") == 0 )
      {
         arg_index ++;
         k_dim = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-dtol") == 0 )
      {
         arg_index ++;
         drop_tol = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-tr") == 0 )
      {
         arg_index++;
         trunc_factor  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-euc_lvl") == 0 )
      {
         arg_index ++;
         euclid_level = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-euc_bj") == 0 )
      {
         arg_index ++;
         euclid_bj = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-rlx") == 0 )
      {
         arg_index ++;
         relax_type = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ai_rlx") == 0 )
      {
         arg_index ++;
         ai_relax_type = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ai_mt") == 0 )
      {
         arg_index ++;
         ai_measure_type = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ct") == 0 )
      {
         arg_index ++;
         coarsen_type = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ipt") == 0 )
      {
         arg_index ++;
         interp_type = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mxct") == 0 )
      {
         arg_index ++;
         coarse_threshold = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-str") == 0 )
      {
         arg_index ++;
         strong_threshold = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mxrs") == 0 )
      {
         arg_index ++;
         max_row_sum = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-scs") == 0 )
      {
         arg_index ++;
         S_commpkg_switch = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-airst") == 0 )
      {
         arg_index ++;
         AIR_strong_th = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-amg_ptlv") == 0 )
      {
         arg_index ++;
         amg_print_level = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-theta_ms") == 0 )
      {
         arg_index ++;
         theta_psi = atoi(argv[arg_index++]);
         theta_rho = atoi(argv[arg_index++]);
         theta_phi = atoi(argv[arg_index++]);
         theta_dis = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ptlv") == 0 )
      {
         arg_index ++;
         print_level = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-help") == 0 )
      {
         print_usage = 1;
         break;
      }
      else
      {
         arg_index ++;
      }
   }
   if (print_usage)
   {
      jxf_printf("\n");
      jxf_printf("  Usage: %s [<options>]\n", argv[0]);
      jxf_printf("\n");
      jxf_printf("    -eps <val>   : threshold for filtering out small elements, zhao li\n");
      jxf_printf("    -sid <val>   : solver id\n");
      jxf_printf("    -pid <val>   : problem id\n");
      jxf_printf("    -nf  <val>   : number of functions\n");
      jxf_printf("    -nts <val>   : threads number\n");
      jxf_printf("    -rap2        : 2nd implementation of RAP\n");
      jxf_printf("    -kt          : keep transpose\n");
      jxf_printf("    -Pmx <val>   : maximal number of elements per row for interpolation\n");
      jxf_printf(" -agg_nl <val>   : num_levels for aggressive coarsening\n");
      jxf_printf("   -tnrm <val>   : two_norm in PCG\n");
      jxf_printf("   -kdim <val>   : krylov dimension\n");
      jxf_printf("   -dtol <val>   : Drop-tolerance for ILU(0) factorization\n");
      jxf_printf("-euc_lvl <val>   : level of fill-in for Euclid\n");
      jxf_printf(" -euc_bj <val>   : PILU or Block Jacobi ILU\n");
      jxf_printf("  -rlx <val>     : relaxation type\n");
      jxf_printf("  -ai_rlx <val>  : AI relaxation type\n");
      jxf_printf("  -ai_mt  <val>  : AI measure type\n");
      jxf_printf("     -ct <val>   : coarsening type\n");
      jxf_printf("    -ipt <val>   : interpolation type\n");
      jxf_printf("   -mxct <val>   : max. size on coarsest grid\n");
      jxf_printf("    -str <val>   : AMG strength threshold\n");
      jxf_printf("   -mxrs <val>   : maximum row sum threshold for dependency weakening\n");
      jxf_printf("-amg_ptlv <val>  : print_level of AMG when AMG as preconditioner\n");
      jxf_printf("-theta_ms <iiiv> : threshold for multi-scale judgement\n");
      jxf_printf("   -ptlv <val>   : print_level\n");
      jxf_printf("   -help         : using help message\n\n");
      exit(1);
   }


   if (myid == 0)
   {
#if JXF_USING_BIG_INT
      jxf_printf(" Using BIG_INT,");
#endif
#if JXF_USING_BIG_DOUBLE
      jxf_printf(" BIG_DOUBLE,");
#endif
#if JXF_USING_OPENMP || defined (JXF_USING_PGCC_SMP)
      jxf_printf(" With OpenMP using %d threads,", nthreads);
#endif
      jxf_printf(" JXFPAMG MPI using %d processors +++++++++++++++++++++\n\n", nprocs);
   }

   //----------------------------------------------------------------
   // 设定线程数
   //----------------------------------------------------------------
#if JXF_USING_OPENMP || defined (JXF_USING_PGCC_SMP)
   omp_set_num_threads(nthreads);
#endif

   if (restri_type) /* Set Restriction to be AIR */
   {
      interp_type = 100; /* 1-pt Interp */
      relax_type = 3;
      ns_down = 3;
      ns_up = 3;
      grid_relax_points = jxf_CTAlloc(JXF_Int *, 4);
      grid_relax_points[0] = NULL;
      grid_relax_points[1] = jxf_CTAlloc(JXF_Int, ns_down);
      grid_relax_points[2] = jxf_CTAlloc(JXF_Int, ns_up);
      grid_relax_points[3] = jxf_CTAlloc(JXF_Int, ns_coarse);
      for (i = 0; i < ns_down; i ++) grid_relax_points[1][i] = 0; /* down cycle */
      if (ns_up == 3) /* up cycle */
      {
         grid_relax_points[2][0] = -1; // F
         grid_relax_points[2][1] = -1; // F
         grid_relax_points[2][2] = 1;  // C
      }
      else if (ns_up == 2)
      {
         grid_relax_points[2][0] = -1;
         grid_relax_points[2][1] = -1;
      }
      for (i = 0; i < ns_coarse; i ++) grid_relax_points[3][i] = 0; /* coarse: all */
      coarse_threshold = 100;
      agg_num_levels = 0; /* does not support aggressive coarsening */
   }


   // zhaoli,2021.06.29,
   // CF=0: 自然序，
   // CF=1: down cycle 先 C 后 F, up cycle 先 F 后 C
   // CF=2: down cycle 先 F 后 C, up cycle 先 C 后 F
   if (CF==0)
   {
      grid_relax_points = jxf_CTAlloc(JXF_Int *, 4);
      grid_relax_points[0] = jxf_CTAlloc(JXF_Int, ns_down);   // 最细网格层
      grid_relax_points[1] = jxf_CTAlloc(JXF_Int, ns_down);   // 前磨光
      grid_relax_points[2] = jxf_CTAlloc(JXF_Int, ns_up);     // 后磨光
      grid_relax_points[3] = jxf_CTAlloc(JXF_Int, ns_coarse); // 粗空间

      num_grid_sweeps   = jxf_CTAlloc(JXF_Int,4);
      num_grid_sweeps[0] = ns_down;
      num_grid_sweeps[1] = ns_down;
      num_grid_sweeps[2] = ns_up;
      num_grid_sweeps[3] = ns_coarse;


      /* fine grid  所有点 */
      for (i = 0; i < ns_down; i++){
         grid_relax_points[0][i]   = 0;
      }
      /* down cycle  所有点 */
      for (i = 0; i < ns_down; i++){
         grid_relax_points[1][i]   = 0;
      }
      /* up cycle  所有点 */
      for (i = 0; i < ns_up; i++){
         grid_relax_points[2][i]   = 0;
      }
      for (i = 0; i < ns_coarse; i ++) grid_relax_points[3][i] = 0; /* coarse: all */
   }
   else if( CF==1 ){
      ns_down *= 2;
      ns_up   *= 2;

      grid_relax_points = jxf_CTAlloc(JXF_Int *, 4);
      grid_relax_points[0] = jxf_CTAlloc(JXF_Int, ns_down);   // 最细网格层
      grid_relax_points[1] = jxf_CTAlloc(JXF_Int, ns_down);   // 前磨光
      grid_relax_points[2] = jxf_CTAlloc(JXF_Int, ns_up);     // 后磨光
      grid_relax_points[3] = jxf_CTAlloc(JXF_Int, ns_coarse); // 粗空间

      num_grid_sweeps   = jxf_CTAlloc(JXF_Int,4);
      num_grid_sweeps[0] = ns_down;
      num_grid_sweeps[1] = ns_down;
      num_grid_sweeps[2] = ns_up;
      num_grid_sweeps[3] = ns_coarse;


      /* fine grid  先 C 后 F*/
      for (i = 0; i < ns_down; i+=2){
         grid_relax_points[0][i]   = 1;
         grid_relax_points[0][i+1] = -1;
      }
      /* down cycle 先 C 后 F*/
      for (i = 0; i < ns_down; i+=2){
         grid_relax_points[1][i]   = 1;
         grid_relax_points[1][i+1] = -1;
      }
      /* up cycle 先 F 后 C*/
      for (i = 0; i < ns_up; i+=2){
         grid_relax_points[2][i]   = -1;
         grid_relax_points[2][i+1] = 1;
      }
      for (i = 0; i < ns_coarse; i ++) grid_relax_points[3][i] = 0; /* coarse: all */
   }
   else if( CF==2 ){
      ns_down *= 2;
      ns_up   *= 2;

      grid_relax_points = jxf_CTAlloc(JXF_Int *, 4);
      grid_relax_points[0] = jxf_CTAlloc(JXF_Int, ns_down);   // 最细网格层
      grid_relax_points[1] = jxf_CTAlloc(JXF_Int, ns_down);   // 前磨光
      grid_relax_points[2] = jxf_CTAlloc(JXF_Int, ns_up);     // 后磨光
      grid_relax_points[3] = jxf_CTAlloc(JXF_Int, ns_coarse); // 粗空间

      num_grid_sweeps   = jxf_CTAlloc(JXF_Int,4);
      num_grid_sweeps[0] = ns_down;
      num_grid_sweeps[1] = ns_down;
      num_grid_sweeps[2] = ns_up;
      num_grid_sweeps[3] = ns_coarse;


      /* fine grid  先 F 后 C*/
      for (i = 0; i < ns_down; i+=2){
         grid_relax_points[0][i]   = -1;
         grid_relax_points[0][i+1] = 1;
      }
      /* down cycle 先 F 后 C*/
      for (i = 0; i < ns_down; i+=2){
         grid_relax_points[1][i]   = -1;
         grid_relax_points[1][i+1] = 1;
      }
      /* up cycle 先 C 后 F*/
      for (i = 0; i < ns_up; i+=2){
         grid_relax_points[2][i]   = 1;
         grid_relax_points[2][i+1] = -1;
      }
      for (i = 0; i < ns_coarse; i ++) grid_relax_points[3][i] = 0; /* coarse: all */
   }






#if 0
   //--------------------------------------------------------------------
   //  利用文件创建并行矩阵和并行右端, 以及并行初始迭代向量(零向量或随机向量)
   //--------------------------------------------------------------------
   if (TTest) starttime = jxf_MPI_Wtime();

   par_matrix = jxf_BuildMatParFromOneFile(MatFile, 1, file_base);
   par_rhs    = jxf_BuildRhsParFromOneFile(RhsFile, par_matrix);

   if (!GusFile)
   {
      glosize      = jxf_ParVectorGlobalSize(par_rhs);
      partitioning = jxf_ParVectorPartitioning(par_rhs);
      par_sol      = jxf_ParVectorCreate(comm, glosize, partitioning);
      jxf_ParVectorSetPartitioningOwner(par_sol, 0);
      jxf_ParVectorInitialize(par_sol);
      if (initguess == 0)
      {
         jxf_ParVectorSetConstantValues(par_sol, 0.0);
      }
      else
      {
         jxf_ParVectorSetRandomValues(par_sol, 22775);
         norm = jxf_ParVectorInnerProd(par_sol, par_sol);
         norm = 1./sqrt(norm);
         jxf_ParVectorScale(norm, par_sol);
      }
   }
   else
   {
      par_sol = jxf_BuildRhsParFromOneFile(GusFile, par_matrix);
   }

   endtime = jxf_MPI_Wtime();
   jxf_GetWallTime(comm, "BuildParLinearSystem", starttime, endtime, 0, 2);
#endif

   //-----------------------------------------------------------
   //  求解线性代数系统
   //-----------------------------------------------------------
   starttimeT = jxf_MPI_Wtime();

   switch (solver_id)
   {
      #if 0
      case 11011:  /* 直接法 */
	   {
         jxf_DMUMPS_data *ddata;
         ddata = jxf_CTAlloc(jxf_DMUMPS_data, 1);


         jxf_Mumps_double_setup(ddata,par_matrix);
         jxf_Mumps_double_solve(ddata,par_matrix,par_rhs,par_sol);

         ddata->id.job = -2;
         dmumps_c(&ddata->id); /* Terminate instance */
      }
      break;

      case 201:  /* 直接法 */
	   {
         // jxf_Mumps_float(NULL,par_matrix, par_rhs, par_sol);
         JXF_Real *ADATA = jxf_CSRMatrixData(As);
         JXF_Int *AIA = jxf_CSRMatrixI(As);
         JXF_Int *AJA = jxf_CSRMatrixJ(As);
         JXF_Int num_rows = jxf_CSRMatrixNumRows(As);
         JXF_Int num_nonzeros = jxf_CSRMatrixNumNonzeros(As);
         jxf_Vector *b = jxf_ParVectorLocalVector(par_rhs);
         JXF_Real *bDATA = jxf_VectorData(b);
         jxf_Vector *x = jxf_ParVectorLocalVector(par_sol);
         JXF_Real *xDATA = jxf_VectorData(x);

         mumps_Interface_float(comm, num_rows,num_nonzeros, AIA, AJA,ADATA,xDATA, bDATA);

         // jxf_SeqVectorDestroy(b);
         // jxf_SeqVectorDestroy(x);

      }
      break;

      case 101:  /* 直接法 */
	   {
         //  jxf_DMUMPS_data *ddata;
         //  ddata = jxf_CTAlloc(jxf_DMUMPS_data, 1);


         //  jxf_Mumps_double_setup(ddata,par_matrix);
         //  jxf_Mumps_double_solve(ddata,par_matrix,par_rhs,par_sol);
          jxf_SMUMPS_data *ddata;
          ddata = jxf_CTAlloc(jxf_SMUMPS_data, 1);


          jxf_Mumps_float_setup(ddata,par_matrix);
          jxf_Mumps_float_solve(ddata,par_matrix,par_rhs,par_sol);
         //  jxf_Mumps_float_solve(ddata,par_matrix,par_rhs,par_sol);

         ddata->id.job = -2;
         smumps_c(&ddata->id); /* Terminate instance */


		//   jxf_pardiso(par_matrix, par_rhs, par_sol);
		//   jxf_PAMGRelax10(par_matrix,par_rhs,NULL,0,0.0,0.0,par_sol,NULL);
      //   Pardiso_data *pdata;
      //   jxf_Pre_Pardiso(pdata,par_matrix, par_rhs, par_sol);
      //   jxf_Mumps(NULL,par_matrix, par_rhs, par_sol);
      //   jxf_Mumps_double(NULL,par_matrix, par_rhs, par_sol);
         //   jxf_Mumps_float(NULL,par_matrix, par_rhs, par_sol);
      //  jxf_LU(par_matrix,par_rhs,par_sol);
      //   jxf_printf("1046\n");
	   }
      break;
      case 1011:  /* MUMPS-F-GMRES */
	   {
         if (myid == 0) jxf_printf("\n >>> Solver: MUMPS-GMRES(%d) \n\n", k_dim);

         if (TTest) starttime = jxf_MPI_Wtime();

          jxf_SMUMPS_data *ddata;
          ddata = jxf_CTAlloc(jxf_SMUMPS_data, 1);

         JXF_ParCSRGMRESCreate(comm, &solver);
         JXF_GMRESSetKDim(solver, k_dim);
         JXF_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JXF_GMRESSetMaxIter(solver, max_iter);
         JXF_GMRESSetTol(solver, tol);
         JXF_GMRESSetLogging(solver, 1);
         JXF_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */

         JXF_GMRESSetPrecond(solver, (JXF_PtrToSolverFcn)jxf_Mumps_float_solve,
                                    (JXF_PtrToSolverFcn)jxf_Mumps_float_setup, (JXF_Solver)ddata);
         jxf_Mumps_float_setup(ddata,par_matrix);

         JXF_GMRESSetup(solver, (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "PAMG-GMRES Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_GMRESSolve(solver, (JXF_Matrix)par_matrix, // preOperater
                               (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "PAMG-GMRES Solve", starttime, endtime, 0, 2);
         }

         JXF_GMRESGetNumIterations(solver, &num_iterations);
         JXF_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jxf_printf(" >>> num_iterations = %d\n", num_iterations);
            jxf_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         ddata->id.job = -2;
         smumps_c(&ddata->id); /* Terminate instance */
         JXF_ParCSRGMRESDestroy(solver);
	   }
      break;

      case 102:  /* PARDISO-GMRES(%d) */
	   {

        if (myid == 0) jxf_printf("\n >>> Solver: PARDISO-GMRES(%d) \n\n", k_dim);

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_ParCSRGMRESCreate(comm, &solver);
         JXF_GMRESSetKDim(solver, k_dim);
         JXF_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JXF_GMRESSetMaxIter(solver, max_iter);
         JXF_GMRESSetTol(solver, tol);
         JXF_GMRESSetLogging(solver, 1);
         JXF_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */

         JXF_GMRESSetPrecond(solver, (JXF_PtrToSolverFcn)jxf_Pre_Pardiso,
                                       NULL, NULL);


         JXF_GMRESSetup(solver, (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "PAMG-GMRES Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_GMRESSolve(solver, (JXF_Matrix)par_matrix, // preOperater
                               (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "PAMG-GMRES Solve", starttime, endtime, 0, 2);
         }

         JXF_GMRESGetNumIterations(solver, &num_iterations);
         JXF_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jxf_printf(" >>> num_iterations = %d\n", num_iterations);
            jxf_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JXF_ParCSRGMRESDestroy(solver);
	   }
      break;

      case 103:  /* PARDISO-FlexGMRES(%d)  */
	   {
       if (myid == 0) jxf_printf("\n >>> Solver: PARDISO-FlexGMRES(%d) \n\n", k_dim);

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_ParCSRFlexGMRESCreate(comm, &solver);
         JXF_FlexGMRESSetKDim(solver, k_dim);
         JXF_FlexGMRESSetIsCheckRestarted(solver, is_check_restarted);
         JXF_FlexGMRESSetMaxIter(solver, max_iter);
         JXF_FlexGMRESSetTol(solver, tol);
         JXF_FlexGMRESSetLogging(solver, 1);
         JXF_FlexGMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */

         JXF_FlexGMRESSetPrecond(solver, (JXF_PtrToSolverFcn)jxf_Pre_Pardiso,
                                       NULL, NULL);

         /* this is optional - could be a user defined one instead */
         JXF_FlexGMRESSetModifyPC(solver, (JXF_PtrToModifyPCFcn)jxf_FlexGMRESModifyPCDefault);

         JXF_FlexGMRESSetup(solver, (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "PAMG-FlexGMRES Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_FlexGMRESSolve(solver, (JXF_Matrix)par_matrix, // preOperater
                               (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "PAMG-FlexGMRES Solve", starttime, endtime, 0, 2);
         }

         JXF_FlexGMRESGetNumIterations(solver, &num_iterations);
         JXF_FlexGMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jxf_printf(" >>> num_iterations = %d\n", num_iterations);
            jxf_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JXF_PAMGDestroy(amg_solver);
         JXF_ParCSRFlexGMRESDestroy(solver);
      }
      break;

      case 3021:  /* Mumps-BiCGSTAB */
      {
         if (myid == 0) jxf_printf("\n >>> Solver: Mumps-BiCGSTAB(%d) \n\n");

         if (TTest) starttime = jxf_MPI_Wtime();

         jxf_SMUMPS_data *ddata;
         ddata = jxf_CTAlloc(jxf_SMUMPS_data, 1);

         JXF_ParCSRBiCGSTABCreate(comm, &solver);
         JXF_BiCGSTABSetMaxIter(solver, max_iter);
         JXF_BiCGSTABSetTol(solver, tol);
         JXF_BiCGSTABSetAbsoluteTol(solver, 0.0);
         JXF_BiCGSTABSetConvCriteria(solver, 0);
         JXF_BiCGSTABSetLogging(solver, 1);
         JXF_BiCGSTABSetPrintLevel(solver, print_level);

         JXF_BiCGSTABSetPrecond(solver, (JXF_PtrToSolverFcn)jxf_Mumps_float_solve,
                                    (JXF_PtrToSolverFcn)jxf_Mumps_float_setup, (JXF_Solver)ddata);
         jxf_Mumps_float_setup(ddata,par_matrix);

         JXF_BiCGSTABSetup(solver, (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "PAMG-BiCGSTAB Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_BiCGSTABSolve(solver, (JXF_Matrix)par_matrix, // preOperater
                                  (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "PAMG-BiCGSTAB Solve", starttime, endtime, 0, 2);
         }

         JXF_BiCGSTABGetNumIterations(solver, &num_iterations);
         JXF_BiCGSTABGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jxf_printf(" >>> num_iterations = %d\n", num_iterations);
            jxf_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         ddata->id.job = -2;
         smumps_c(&ddata->id); /* Terminate instance */
         JXF_ParCSRBiCGSTABDestroy(solver);
      }
      break;

      case 3022:  /* PARDISO-FlexGMRES(%d)  */
	   {
       if (myid == 0) jxf_printf("\n >>> Solver: PARDISO-FlexGMRES(%d) \n\n", k_dim);

         if (TTest) starttime = jxf_MPI_Wtime();

         jxf_SMUMPS_data *ddata;
         ddata = jxf_CTAlloc(jxf_SMUMPS_data, 1);

         JXF_ParCSRFlexGMRESCreate(comm, &solver);
         JXF_FlexGMRESSetKDim(solver, k_dim);
         JXF_FlexGMRESSetIsCheckRestarted(solver, is_check_restarted);
         JXF_FlexGMRESSetMaxIter(solver, max_iter);
         JXF_FlexGMRESSetTol(solver, tol);
         JXF_FlexGMRESSetLogging(solver, 1);
         JXF_FlexGMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */

         JXF_FlexGMRESSetPrecond(solver, (JXF_PtrToSolverFcn)jxf_Mumps_float_solve,
                                    (JXF_PtrToSolverFcn)jxf_Mumps_float_setup, (JXF_Solver)ddata);
         jxf_Mumps_float_setup(ddata,par_matrix);

         /* this is optional - could be a user defined one instead */
         JXF_FlexGMRESSetModifyPC(solver, (JXF_PtrToModifyPCFcn)jxf_FlexGMRESModifyPCDefault);

         JXF_FlexGMRESSetup(solver, (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "PAMG-FlexGMRES Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_FlexGMRESSolve(solver, (JXF_Matrix)par_matrix, // preOperater
                               (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "PAMG-FlexGMRES Solve", starttime, endtime, 0, 2);
         }

         JXF_FlexGMRESGetNumIterations(solver, &num_iterations);
         JXF_FlexGMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jxf_printf(" >>> num_iterations = %d\n", num_iterations);
            jxf_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         ddata->id.job = -2;
         smumps_c(&ddata->id); /* Terminate instance */
         JXF_ParCSRFlexGMRESDestroy(solver);
      }
      break;


      case 302:  /* PARDISO-BiCGSTAB */
      {
         if (myid == 0) jxf_printf("\n >>> Solver: PARDISO-BiCGSTAB(%d) \n\n");

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_ParCSRBiCGSTABCreate(comm, &solver);
         JXF_BiCGSTABSetMaxIter(solver, max_iter);
         JXF_BiCGSTABSetTol(solver, tol);
         JXF_BiCGSTABSetAbsoluteTol(solver, 0.0);
         JXF_BiCGSTABSetConvCriteria(solver, 0);
         JXF_BiCGSTABSetLogging(solver, 1);
         JXF_BiCGSTABSetPrintLevel(solver, print_level);

         JXF_BiCGSTABSetPrecond(solver, (JXF_PtrToSolverFcn)jxf_Pre_Pardiso,
                                       NULL, NULL);

         JXF_BiCGSTABSetup(solver, (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "PAMG-BiCGSTAB Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_BiCGSTABSolve(solver, (JXF_Matrix)par_matrix, // preOperater
                                  (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "PAMG-BiCGSTAB Solve", starttime, endtime, 0, 2);
         }

         JXF_BiCGSTABGetNumIterations(solver, &num_iterations);
         JXF_BiCGSTABGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jxf_printf(" >>> num_iterations = %d\n", num_iterations);
            jxf_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JXF_ParCSRBiCGSTABDestroy(solver);
      }
      break;
#endif
      case 0:  /* PAMG */
      {
         if (myid == 0) jxf_printf("\n >>> Solver: PAMG \n\n");

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_PAMGCreate(&amg_solver);
         if (restri_type)
         {
            jxf_assert(restri_type >= 0);
            JXF_PAMGSetRestriction(amg_solver, restri_type);
            JXF_PAMGSetGridRelaxPoints(amg_solver, grid_relax_points);
         }
         JXF_PAMGSetMaxLevels(amg_solver, max_levels);
         JXF_PAMGSetMaxIter(amg_solver, max_iter);
         JXF_PAMGSetNumFunctions(amg_solver, num_functions);
         JXF_PAMGSetCycleType(amg_solver, cycle_type);
         JXF_PAMGSetMeasureType(amg_solver, measure_type);
         JXF_PAMGSetRAP2(amg_solver, rap2);
         JXF_PAMGSetKeepTranspose(amg_solver, keepTranspose);
         JXF_PAMGSetTol(amg_solver, tol);
         JXF_PAMGSetConvCriteria(amg_solver, conv_criteria);
         JXF_PAMGSetCoarsenType(amg_solver, coarsen_type);
         JXF_PAMGSetInterpType(amg_solver, interp_type);
         JXF_PAMGSetPMaxElmts(amg_solver, P_max_elmts);
         JXF_PAMGSetAggNumLevels(amg_solver, agg_num_levels);
         JXF_PAMGSetAIMeasureType(amg_solver, ai_measure_type);
         JXF_PAMGSetAIRelaxType(amg_solver, ai_relax_type);
         JXF_PAMGSetStrongThreshold(amg_solver, strong_threshold);
         JXF_PAMGSetMaxRowSum(amg_solver, max_row_sum);
         JXF_PAMGSetPrintLevel(amg_solver, print_level);
         JXF_PAMGSetCoarsestSolverID(amg_solver, coarsestsolverid);
         JXF_PAMGSetCoarseThreshold(amg_solver, coarse_threshold);
         JXF_PAMGSetCoarseRatio(amg_solver, coarse_ratio);
         JXF_PAMGSetRelaxWt(amg_solver, relax_wt);
         JXF_PAMGSetOuterWt(amg_solver, outer_wt);
         if (ns_down > -1) JXF_PAMGSetCycleNumSweeps(amg_solver, ns_down, 1);           /* sweep for "down" */
         if (ns_up > -1) JXF_PAMGSetCycleNumSweeps(amg_solver, ns_up, 2);           /* sweep for "up" */
         JXF_PAMGSetCycleNumSweeps(amg_solver, ns_coarse, 3);           /* sweep for "coarsest" */
         JXF_PAMGSetCycleRelaxType(amg_solver, relax_type, 1);  /* relax_type for "down" */
         JXF_PAMGSetCycleRelaxType(amg_solver, relax_type, 2);  /* relax_type for "up" */
         JXF_PAMGSetCycleRelaxType(amg_solver, coarse_solver, 3);           /* relax_type for "coarsest" */

         //------------------------------------------------------------
         //    JXF_PAMG Setup
         //------------------------------------------------------------
         if (max_levels != 1)
         {
            JXF_PAMGSetup(amg_solver, (JXF_ParCSRMatrix) par_matrix);
         }

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "PAMG Setup", starttime, endtime, 0, 2);
         }

         //------------------------------------------------------------
         //    JXF_PAMG Solve
         //------------------------------------------------------------
         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_PAMGSolve(amg_solver, (JXF_ParCSRMatrix)par_matrix, (JXF_ParVector)par_rhs, (JXF_ParVector)par_sol);

         JXF_PAMGGetNumIterations(amg_solver, &num_iterations);
         JXF_PAMGGetFinalRelativeResidualNorm(amg_solver, &final_res_norm);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "PAMG Solve", starttime, endtime, 0, 2);
         }

         if (print_level == 0 && myid == 0)
         {
            jxf_printf(" >>> num_iterations = %d\n", num_iterations);
            jxf_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JXF_PAMGDestroy(amg_solver);
      }
      break;

      case 11:  /* CG */
      {
         if (myid == 0) jxf_printf("\n >>> Solver: CG \n\n");

         JXF_ParCSRPCGCreate(comm, &solver);

         JXF_PCGSetMaxIter(solver, max_iter);
         JXF_PCGSetTol(solver, tol);
         JXF_PCGSetTwoNorm(solver, twonorm);  // 0: B 范数； 1：l2 范数
         JXF_PCGSetLogging(solver, 1);
         JXF_PCGSetPrintLevel(solver, print_level);

         JXF_PCGSetup(solver, (JXF_Matrix) par_matrix, (JXF_Vector) par_rhs, (JXF_Vector) par_sol);

         JXF_PCGSolve(solver, (JXF_Matrix)par_matrix, // preOperater
                             (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         JXF_PCGGetNumIterations(solver, &num_iterations);
         JXF_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);

         JXF_ParCSRPCGDestroy(solver);
      }
      break;

      case 12:  /* PAMG-CG */
      {
         if (myid == 0) jxf_printf("\n >>> Solver: PAMG-CG \n\n");

         starttime = jxf_MPI_Wtime();

         JXF_PAMGCreate(&amg_solver);
         JXF_PAMGSetMaxLevels(amg_solver, max_levels);
         JXF_PAMGSetMaxIter(amg_solver, 1);
         JXF_PAMGSetCycleType(amg_solver, cycle_type);
         JXF_PAMGSetMeasureType(amg_solver, measure_type);
         JXF_PAMGSetRAP2(amg_solver, rap2);
         JXF_PAMGSetKeepTranspose(amg_solver, keepTranspose);
         JXF_PAMGSetCoarsenType(amg_solver, coarsen_type);
         JXF_PAMGSetCoarsestSolverID(amg_solver, coarsestsolverid);
         JXF_PAMGSetInterpType(amg_solver, interp_type);
         JXF_PAMGSetPMaxElmts(amg_solver, P_max_elmts);
         JXF_PAMGSetTruncFactor(amg_solver, trunc_factor);//zhaoli 2021.06.28
         JXF_PAMGSetAggNumLevels(amg_solver, agg_num_levels);
         JXF_PAMGSetAIMeasureType(amg_solver, ai_measure_type);
         JXF_PAMGSetAIRelaxType(amg_solver, ai_relax_type);
         JXF_PAMGSetStrongThreshold(amg_solver, strong_threshold);
         JXF_PAMGSetMaxRowSum(amg_solver, max_row_sum);
         JXF_PAMGSetPrintLevel(amg_solver, amg_print_level);
         JXF_PAMGSetCoarseThreshold(amg_solver, coarse_threshold);
         JXF_PAMGSetRelaxWt(amg_solver, relax_wt);
         JXF_PAMGSetOuterWt(amg_solver, outer_wt);

         if (CF != -1) {
            JXF_PAMGSetNumGridSweeps(amg_solver, num_grid_sweeps);       // zhaoli
            JXF_PAMGSetGridRelaxPoints(amg_solver, grid_relax_points);   // zhaoli
		   }
         // JXF_Int smooth_num_sweeps = 2;
         // JXF_PAMGSetSmoothNumSweeps(amg_solver, smooth_num_sweeps);
         if (ns_down > -1) JXF_PAMGSetCycleNumSweeps(amg_solver, ns_down, 1);           /* sweep for "down" */
         if (ns_up > -1) JXF_PAMGSetCycleNumSweeps(amg_solver, ns_up, 2);           /* sweep for "up" */
         JXF_PAMGSetCycleNumSweeps(amg_solver, ns_coarse, 3);           /* sweep for "coarsest" */
         JXF_PAMGSetCycleRelaxType(amg_solver, relax_type, 1);  /* relax_type for "down" */
         JXF_PAMGSetCycleRelaxType(amg_solver, relax_type, 2);  /* relax_type for "up" */
         JXF_PAMGSetCycleRelaxType(amg_solver, coarse_solver, 3);           /* relax_type for "coarsest" */

         JXF_ParCSRPCGCreate(comm, &solver);

         JXF_PCGSetMaxIter(solver, max_iter);
         JXF_PCGSetTol(solver, tol);
         JXF_PCGSetTwoNorm(solver, twonorm);  // 0: B 范数； 1：l2 范数
         JXF_PCGSetLogging(solver, 1);
         JXF_PCGSetPrintLevel(solver, print_level);

         JXF_PCGSetPrecond(solver, (JXF_PtrToSolverFcn)JXF_PAMGPrecond, (JXF_PtrToSolverFcn)JXF_PAMGSetup, amg_solver);

         JXF_PAMGSetup(amg_solver, (JXF_ParCSRMatrix)par_matrix);

         JXF_PCGSetup(solver, (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         endtime = jxf_MPI_Wtime();
         jxf_GetWallTime(comm, "PAMG-CG Setup", starttime, endtime, 0, 2);

         starttime = jxf_MPI_Wtime();

         JXF_PCGSolve(solver, (JXF_Matrix)par_matrix, // preOperater
                             (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         endtime = jxf_MPI_Wtime();
         jxf_GetWallTime(comm, "PAMG-CG Solve", starttime, endtime, 0, 2);


         // jxf_ParCSRMatrixMatvec(-1, par_matrix, par_sol, 1, par_rhs);
         // JXF_Real yL2 = jxf_ParVectorInnerProd( par_rhs, par_rhs );
         // printf(" >>>   AMG-CG :  ||b-Ax||_2 = %e\n", sqrt(yL2));


         JXF_PCGGetNumIterations(solver, &num_iterations);
         JXF_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);

		 //jxf_printf("vvvv %e\n", final_res_norm);

         JXF_PAMGDestroy(amg_solver);
         JXF_ParCSRPCGDestroy(solver);
      }
      break;

      // 改进 P \approx \hat{A}^{-1}, zhaoli,2021.06.24
      case 112:  /* PAMG-CG-zl */
      {
         if (myid == 0) jxf_printf("\n >>> Solver: PAMG-CG-zl \n\n");

         starttime = jxf_MPI_Wtime();

         // zhaoli
         if (myid == 0)
         {
            jxf_printf("eps = %e, zhaoli 2021.06.24!\n", eps);
            As_hat = jxf_CSRMatrixCreate(As->num_rows, As->num_cols, As->num_nonzeros);
            jxf_CSRMatrixInitialize(As_hat);
            jxf_CSRMatrixNumNonzeros(As_hat) = As->num_nonzeros;
            jxf_CSRMatrixNumCols(As_hat) = As->num_cols;
            // jxf_printf("1\n");
            jxf_CSRMatrixFilter(As, eps, As_hat);
            // jxf_printf("2 As_hat->num_cols = %d\n", As_hat->num_cols);
         }
         Ap_hat = jxf_CSRMatrixToParCSRMatrix(comm, As_hat, par_matrix->row_starts, par_matrix->col_starts);
         jxf_CSRMatrixDestroy(As_hat);

         JXF_PAMGCreate(&amg_solver);
         JXF_PAMGSetMaxLevels(amg_solver, max_levels);
         JXF_PAMGSetMaxIter(amg_solver, 1);
         JXF_PAMGSetCycleType(amg_solver, cycle_type);
         JXF_PAMGSetMeasureType(amg_solver, measure_type);
         JXF_PAMGSetRAP2(amg_solver, rap2);
         JXF_PAMGSetKeepTranspose(amg_solver, keepTranspose);
         JXF_PAMGSetCoarsenType(amg_solver, coarsen_type);
         JXF_PAMGSetInterpType(amg_solver, interp_type);
         JXF_PAMGSetPMaxElmts(amg_solver, P_max_elmts);
         JXF_PAMGSetAggNumLevels(amg_solver, agg_num_levels);
         JXF_PAMGSetAIMeasureType(amg_solver, ai_measure_type);
         JXF_PAMGSetAIRelaxType(amg_solver, ai_relax_type);
         JXF_PAMGSetStrongThreshold(amg_solver, strong_threshold);
         JXF_PAMGSetMaxRowSum(amg_solver, max_row_sum);
         JXF_PAMGSetPrintLevel(amg_solver, amg_print_level);
         JXF_PAMGSetCoarseThreshold(amg_solver, coarse_threshold);
         JXF_PAMGSetRelaxWt(amg_solver, relax_wt);
         JXF_PAMGSetOuterWt(amg_solver, outer_wt);
         if (ns_down > -1) JXF_PAMGSetCycleNumSweeps(amg_solver, ns_down, 1);           /* sweep for "down" */
         if (ns_up > -1) JXF_PAMGSetCycleNumSweeps(amg_solver, ns_up, 2);           /* sweep for "up" */
         JXF_PAMGSetCycleNumSweeps(amg_solver, ns_coarse, 3);           /* sweep for "coarsest" */
         JXF_PAMGSetCycleRelaxType(amg_solver, relax_type, 1);  /* relax_type for "down" */
         JXF_PAMGSetCycleRelaxType(amg_solver, relax_type, 2);  /* relax_type for "up" */
         JXF_PAMGSetCycleRelaxType(amg_solver, coarse_solver, 3);           /* relax_type for "coarsest" */

         JXF_ParCSRPCGCreate(comm, &solver);

         JXF_PCGSetMaxIter(solver, max_iter);
         JXF_PCGSetTol(solver, tol);
         JXF_PCGSetTwoNorm(solver, twonorm);  // 0: B 范数； 1：l2 范数
         JXF_PCGSetLogging(solver, 1);
         JXF_PCGSetPrintLevel(solver, print_level);

         JXF_PCGSetPrecond(solver, (JXF_PtrToSolverFcn)JXF_PAMGPrecond, (JXF_PtrToSolverFcn)JXF_PAMGSetup, amg_solver);

         // JXF_PAMGSetup(amg_solver, (JXF_ParCSRMatrix)par_matrix);
         JXF_PAMGSetup(amg_solver, (JXF_ParCSRMatrix)Ap_hat);

         JXF_PCGSetup(solver, (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         endtime = jxf_MPI_Wtime();
         jxf_GetWallTime(comm, "PAMG-CG Setup", starttime, endtime, 0, 2);

         starttime = jxf_MPI_Wtime();

         // system("free -h");
         // JXF_PCGSolve(solver, (JXF_Matrix)par_matrix, // preOperater
         JXF_PCGSolve(solver, (JXF_Matrix)Ap_hat, // preOperater
                             (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);
         // system("free -h");
         endtime = jxf_MPI_Wtime();
         jxf_GetWallTime(comm, "PAMG-CG Solve", starttime, endtime, 0, 2);

         JXF_PCGGetNumIterations(solver, &num_iterations);
         JXF_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);

         JXF_PAMGDestroy(amg_solver);
         JXF_ParCSRPCGDestroy(solver);
         // jxf_ParCSRMatrixDestroy(Ap_hat);
      }
      break;

      case 13:  /* DS-CG */
      {
         if (myid == 0) jxf_printf("\n >>> Solver: DS-CG \n\n");

         starttime = jxf_MPI_Wtime();

         ds_solver = NULL;

         JXF_ParCSRPCGCreate(comm, &solver);
         JXF_PCGSetMaxIter(solver, max_iter);
         JXF_PCGSetTol(solver, tol);
         JXF_PCGSetTwoNorm(solver, twonorm);  // 0: B 范数； 1：l2 范数
         JXF_PCGSetLogging(solver, 1);
         JXF_PCGSetPrintLevel(solver, print_level);

         JXF_PCGSetPrecond(solver, (JXF_PtrToSolverFcn)JXF_DiagScalePrecond,
                                  (JXF_PtrToSolverFcn)JXF_DiagScaleSetup, ds_solver);

         JXF_DiagScaleSetup(ds_solver, (JXF_ParCSRMatrix)par_matrix);

         JXF_PCGSetup(solver, (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         endtime = jxf_MPI_Wtime();
         jxf_GetWallTime(comm, "DS-CG Setup", starttime, endtime, 0, 2);

         starttime = jxf_MPI_Wtime();

         JXF_PCGSolve(solver, (JXF_Matrix)par_matrix, // preOperater
                             (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         endtime = jxf_MPI_Wtime();
         jxf_GetWallTime(comm, "DS-CG Solve", starttime, endtime, 0, 2);

         JXF_PCGGetNumIterations(solver, &num_iterations);
         JXF_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);

         JXF_ParCSRPCGDestroy(solver);
      }
      break;

      case 14:  /* Euclid-CG */
      {
         if (myid == 0) jxf_printf("\n >>> Solver: Euclid-CG \n\n");

         starttime = jxf_MPI_Wtime();

         JXF_EuclidCreate(comm, &euclid_solver);
         //JXF_EuclidSetParams(euclid_solver, argc, argv);
         JXF_EuclidSetLevel(euclid_solver, euclid_level);
         JXF_EuclidSetBJ(euclid_solver, euclid_bj);

         JXF_ParCSRPCGCreate(comm, &solver);
         JXF_PCGSetMaxIter(solver, max_iter);
         JXF_PCGSetTol(solver, tol);
         JXF_PCGSetTwoNorm(solver, twonorm);  // 0: B 范数； 1：l2 范数
         JXF_PCGSetLogging(solver, 1);
         JXF_PCGSetPrintLevel(solver, print_level);

         JXF_PCGSetPrecond(solver, (JXF_PtrToSolverFcn)JXF_EuclidSolve,
                                  (JXF_PtrToSolverFcn)JXF_EuclidSetup, euclid_solver);

         JXF_EuclidSetup(euclid_solver, (JXF_ParCSRMatrix)par_matrix);

         JXF_PCGSetup(solver, (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         endtime = jxf_MPI_Wtime();
         jxf_GetWallTime(comm, "Euclid-CG Setup", starttime, endtime, 0, 2);

         starttime = jxf_MPI_Wtime();

         JXF_PCGSolve(solver, (JXF_Matrix)par_matrix, // preOperater
                             (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         endtime = jxf_MPI_Wtime();
         jxf_GetWallTime(comm, "Euclid-CG Solve", starttime, endtime, 0, 2);

         JXF_PCGGetNumIterations(solver, &num_iterations);
         JXF_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);

         JXF_EuclidDestroy(euclid_solver);
         JXF_ParCSRPCGDestroy(solver);
      }
      break;

      case 17:  /* ILU-PAMG-ILU-CG */
      {
         if (myid == 0) jxf_printf("\n >>> Solver: ILU(0)-PAMG-ILU(0)-CG \n\n");

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_CombinedPrecondDataCreate(&combined_solver, comm);
         JXF_CombinedPrecondDataSetPreID(combined_solver, 17);
         JXF_CombinedPrecondDataSetDropTol(combined_solver, drop_tol);
         JXF_CombinedPrecondDataSetInterpType(combined_solver, interp_type);
         JXF_CombinedPrecondDataSetCoarsenType(combined_solver, coarsen_type);
         JXF_CombinedPrecondDataSetCycleRelaxType(combined_solver, relax_type);
         JXF_CombinedPrecondDataSetStrongThreshold(combined_solver, strong_threshold);
         JXF_CombinedPrecondDataInitialize(combined_solver, euclid_level); /* Yue Xiaoqiang 2013/12/11 */
         //JXF_CombinedPrecondDataInitializeP(combined_solver, argc, argv); /* Yue Xiaoqiang 2014/05/24 */

         JXF_ParCSRPCGCreate(comm, &solver);
         JXF_PCGSetMaxIter(solver, max_iter);
         JXF_PCGSetTol(solver, tol);
         JXF_PCGSetTwoNorm(solver, twonorm);  // 0: B 范数； 1：l2 范数
         JXF_PCGSetLogging(solver, 1);
         JXF_PCGSetPrintLevel(solver, print_level);

         JXF_PCGSetPrecond(solver, (JXF_PtrToSolverFcn)JXF_CombinedPrecondDataSolve,
                                  (JXF_PtrToSolverFcn)JXF_CombinedPrecondDataSetup, combined_solver);

         JXF_CombinedPrecondDataSetup(combined_solver, (JXF_ParCSRMatrix)par_matrix);

         JXF_PCGSetup(solver, (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "ILU-PAMG-ILU(0)-CG Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_PCGSolve(solver, (JXF_Matrix)par_matrix, // preOperater
                             (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "ILU-PAMG-ILU(0)-CG Solve", starttime, endtime, 0, 2);
         }

         JXF_CombinedPrecondDataGetLULength(combined_solver, &lu_length);
         if (myid == 0)
         {
            jxf_printf(" >>> Memory Complexity of ILU(0) = %d\n", lu_length);
         }

         JXF_PCGGetNumIterations(solver, &num_iterations);
         JXF_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jxf_printf(" >>> num_iterations = %d\n", num_iterations);
            jxf_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JXF_CombinedPrecondDataDestroy(combined_solver);
         JXF_ParCSRPCGDestroy(solver);
      }
      break;

      case 18:  /* PAMG-ILU-PAMG-CG */
      {
         if (myid == 0) jxf_printf("\n >>> Solver: PAMG-ILU(0)-PAMG-CG \n\n");

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_CombinedPrecondDataCreate(&combined_solver, comm);
         JXF_CombinedPrecondDataSetPreID(combined_solver, 19);
         JXF_CombinedPrecondDataSetDropTol(combined_solver, drop_tol);
         JXF_CombinedPrecondDataSetInterpType(combined_solver, interp_type);
         JXF_CombinedPrecondDataSetCoarsenType(combined_solver, coarsen_type);
         JXF_CombinedPrecondDataSetCycleRelaxType(combined_solver, relax_type);
         JXF_CombinedPrecondDataSetStrongThreshold(combined_solver, strong_threshold);
         JXF_CombinedPrecondDataInitialize(combined_solver, euclid_level); /* Yue Xiaoqiang 2013/12/11 */
         //JXF_CombinedPrecondDataInitializeP(combined_solver, argc, argv); /* Yue Xiaoqiang 2014/05/24 */

         JXF_ParCSRPCGCreate(comm, &solver);
         JXF_PCGSetMaxIter(solver, max_iter);
         JXF_PCGSetTol(solver, tol);
         JXF_PCGSetTwoNorm(solver, twonorm);  // 0: B 范数； 1：l2 范数
         JXF_PCGSetLogging(solver, 1);
         JXF_PCGSetPrintLevel(solver, print_level);


         JXF_PCGSetPrecond(solver, (JXF_PtrToSolverFcn)JXF_CombinedPrecondDataSolve,
                                  (JXF_PtrToSolverFcn)JXF_CombinedPrecondDataSetup, combined_solver);

         JXF_CombinedPrecondDataSetup(combined_solver, (JXF_ParCSRMatrix)par_matrix);

         JXF_PCGSetup(solver, (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "PAMG-ILU(0)-PAMG-CG Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_PCGSolve(solver, (JXF_Matrix)par_matrix, // preOperater
                             (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "PAMG-ILU(0)-PAMG-CG Solve", starttime, endtime, 0, 2);
         }

         JXF_CombinedPrecondDataGetLULength(combined_solver, &lu_length);
         if (myid == 0)
         {
            jxf_printf(" >>> Memory Complexity of ILU(0) = %d\n", lu_length);
         }

         JXF_PCGGetNumIterations(solver, &num_iterations);
         JXF_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jxf_printf(" >>> num_iterations = %d\n", num_iterations);
            jxf_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JXF_CombinedPrecondDataDestroy(combined_solver);
         JXF_ParCSRPCGDestroy(solver);
      }
      break;

      case 21:  /* GMRES */
      {
         if (myid == 0) jxf_printf("\n >>> Solver: GMRES(%d) \n\n", k_dim);

         JXF_ParCSRGMRESCreate(comm, &solver);
         JXF_GMRESSetKDim(solver, k_dim);
         JXF_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JXF_GMRESSetMaxIter(solver, max_iter);
         JXF_GMRESSetTol(solver, tol);
         JXF_GMRESSetLogging(solver, 1);
         JXF_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */

         JXF_GMRESSetup(solver, (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         JXF_GMRESSolve(solver, (JXF_Matrix)par_matrix, // preOperater
                               (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         JXF_GMRESGetNumIterations(solver, &num_iterations);
         JXF_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         JXF_ParCSRGMRESDestroy(solver);
      }
      break;

      case 220:  /* CPR-GMRES */
      {
         if (myid == 0) jxf_printf("\n >>> Solver: CPR-GMRES(%d) \n\n", k_dim);
         
         if (TTest) starttime = jxf_MPI_Wtime();
         
         // 创建CPR预条件器
         jxf_CPRPrecond *cpr_solver = JXF_CPRCreate(comm);
         
         // 设置CPR参数
         JXF_CPRSetParameter(cpr_solver, "pressure_index", 0);
         JXF_CPRSetParameter(cpr_solver, "stage1_maxit", 1);
         JXF_CPRSetParameter(cpr_solver, "stage2_maxit", 1);
         JXF_CPRSetParameter(cpr_solver, "stage1_solver_type", 1);  // AMG
         JXF_CPRSetParameter(cpr_solver, "stage2_solver_type", 2);  // BGS
         JXF_CPRSetRealParameter(cpr_solver, "threshold", 1e-15);
         
         // 创建GMRES求解器
         JXF_ParCSRGMRESCreate(comm, &solver);
         JXF_GMRESSetKDim(solver, k_dim);
         JXF_GMRESSetIsCheckRestarted(solver, is_check_restarted);
         JXF_GMRESSetMaxIter(solver, max_iter);
         JXF_GMRESSetTol(solver, tol);
         JXF_GMRESSetLogging(solver, 1);
         JXF_GMRESSetPrintLevel(solver, print_level);
         
         // 设置预条件器为CPR
         JXF_GMRESSetPrecond(solver, 
                           (JXF_PtrToSolverFcn)JXF_CPRPrecond,
                           (JXF_PtrToSolverFcn)JXF_CPRSetup,
                           cpr_solver);
         
         // 设置CPR预条件器
         JXF_CPRSetup(cpr_solver, (jxf_ParBSRMatrix)par_matrix);
         
         // 设置GMRES求解器
         JXF_GMRESSetup(solver, (JXF_Matrix)par_matrix, 
                        (JXF_Vector)par_rhs, (JXF_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "CPR-GMRES Setup", starttime, endtime, 0, 2);
         }
         
         if (TTest) starttime = jxf_MPI_Wtime();
         
         // 求解
         JXF_GMRESSolve(solver, (JXF_Matrix)par_matrix,
                              (JXF_Matrix)par_matrix,
                              (JXF_Vector)par_rhs,
                              (JXF_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "CPR-GMRES Solve", starttime, endtime, 0, 2);
         }
         
         JXF_GMRESGetNumIterations(solver, &num_iterations);
         JXF_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
         
         if (print_level == 0 && myid == 0)
         {
            jxf_printf(" >>> num_iterations = %d\n", num_iterations);
            jxf_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }
         
         // 打印CPR统计信息
         JXF_CPRPrint(cpr_solver, 2);
         
         // 销毁
         JXF_CPRDestroy(&cpr_solver);
         JXF_ParCSRGMRESDestroy(solver);

         break;
      }
      case 22:  /* PAMG-GMRES */
      {
         if (myid == 0) jxf_printf("\n >>> Solver: PAMG-GMRES(%d) \n\n", k_dim);

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_PAMGCreate(&amg_solver);
         if (restri_type)
         {
            jxf_assert(restri_type >= 0);
            JXF_PAMGSetRestriction(amg_solver, restri_type);
            JXF_PAMGSetGridRelaxPoints(amg_solver, grid_relax_points);
         }
         JXF_PAMGSetMaxLevels(amg_solver, max_levels);
         JXF_PAMGSetMaxIter(amg_solver, ns_up);
         JXF_PAMGSetCycleType(amg_solver, cycle_type);
         JXF_PAMGSetMeasureType(amg_solver, measure_type);
         JXF_PAMGSetRAP2(amg_solver, rap2);
         JXF_PAMGSetKeepTranspose(amg_solver, keepTranspose);
         JXF_PAMGSetCoarsenType(amg_solver, coarsen_type);
         JXF_PAMGSetInterpType(amg_solver, interp_type);
         JXF_PAMGSetPMaxElmts(amg_solver, P_max_elmts);
         JXF_PAMGSetAggNumLevels(amg_solver, agg_num_levels);
         JXF_PAMGSetAIMeasureType(amg_solver, ai_measure_type);
         JXF_PAMGSetAIRelaxType(amg_solver, ai_relax_type);
         JXF_PAMGSetStrongThreshold(amg_solver, strong_threshold);
         JXF_PAMGSetMaxRowSum(amg_solver, max_row_sum);
         JXF_PAMGSetPrintLevel(amg_solver, amg_print_level);
         JXF_PAMGSetCoarseThreshold(amg_solver, coarse_threshold);
         JXF_PAMGSetRelaxWt(amg_solver, relax_wt);
         JXF_PAMGSetOuterWt(amg_solver, outer_wt);
         JXF_PAMGSetSCommPkgSwitch(amg_solver, S_commpkg_switch);
         JXF_PAMGSetAIRStrongTh(amg_solver, AIR_strong_th);
         if (ns_down > -1) JXF_PAMGSetCycleNumSweeps(amg_solver, ns_down, 1);           /* sweep for "down" */
         if (ns_up > -1) JXF_PAMGSetCycleNumSweeps(amg_solver, ns_up, 2);           /* sweep for "up" */
         JXF_PAMGSetCycleNumSweeps(amg_solver, ns_coarse, 3);           /* sweep for "coarsest" */
         JXF_PAMGSetCycleRelaxType(amg_solver, relax_type, 1);  /* relax_type for "down" */
         JXF_PAMGSetCycleRelaxType(amg_solver, relax_type, 2);  /* relax_type for "up" */
         JXF_PAMGSetCycleRelaxType(amg_solver, coarse_solver, 3);           /* relax_type for "coarsest" */

         // printf("relax_type:%d \n\n",relax_type);

         JXF_ParCSRGMRESCreate(comm, &solver);
         JXF_GMRESSetKDim(solver, k_dim);
         JXF_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JXF_GMRESSetMaxIter(solver, max_iter);
         JXF_GMRESSetTol(solver, tol);
         JXF_GMRESSetLogging(solver, 1);
         JXF_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */

         JXF_GMRESSetPrecond(solver, (JXF_PtrToSolverFcn)JXF_PAMGPrecond,
                                    (JXF_PtrToSolverFcn)JXF_PAMGSetup, amg_solver);

         JXF_PAMGSetup(amg_solver, (JXF_ParCSRMatrix)par_matrix);

         JXF_GMRESSetup(solver, (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "PAMG-GMRES Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_GMRESSolve(solver, (JXF_Matrix)par_matrix, // preOperater
                               (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "PAMG-GMRES Solve", starttime, endtime, 0, 2);
         }

         JXF_GMRESGetNumIterations(solver, &num_iterations);

         JXF_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         // if (print_level == 0 && myid == 0)
         if (myid == 0)
         {
            jxf_printf(" >>> num_iterations = %d\n", num_iterations);
            jxf_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }
         JXF_PAMGDestroy(amg_solver);
         JXF_ParCSRGMRESDestroy(solver);
      }
      break;

      case 23:  /* DS-GMRES */
      {
         if (myid == 0) jxf_printf("\n >>> Solver: DS-GMRES(%d) \n\n", k_dim);

         if (TTest) starttime = jxf_MPI_Wtime();

         ds_solver = NULL;

         JXF_ParCSRGMRESCreate(comm, &solver);
         JXF_GMRESSetKDim(solver, k_dim);
         JXF_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JXF_GMRESSetMaxIter(solver, max_iter);
         JXF_GMRESSetTol(solver, tol);
         JXF_GMRESSetLogging(solver, 1);
         JXF_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */

         JXF_GMRESSetPrecond(solver, (JXF_PtrToSolverFcn)JXF_DiagScalePrecond,
                                    (JXF_PtrToSolverFcn)JXF_DiagScaleSetup, ds_solver);

         JXF_DiagScaleSetup(ds_solver, (JXF_ParCSRMatrix)par_matrix);

         JXF_GMRESSetup(solver, (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "DS-GMRES Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_GMRESSolve(solver, (JXF_Matrix)par_matrix, // preOperater
                               (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "DS-GMRES Solve", starttime, endtime, 0, 2);
         }

         JXF_GMRESGetNumIterations(solver, &num_iterations);
         JXF_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jxf_printf(" >>> num_iterations = %d\n", num_iterations);
            jxf_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JXF_ParCSRGMRESDestroy(solver);
      }
      break;

      case 24:  /* Euclid-GMRES */
      {
         if (myid == 0) jxf_printf("\n >>> Solver: Euclid-GMRES(%d) \n\n", k_dim);

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_EuclidCreate(comm, &euclid_solver);
         //JXF_EuclidSetParams(euclid_solver, argc, argv);
         JXF_EuclidSetLevel(euclid_solver, euclid_level);
         JXF_EuclidSetBJ(euclid_solver, euclid_bj);

         JXF_ParCSRGMRESCreate(comm, &solver);
         JXF_GMRESSetKDim(solver, k_dim);
         JXF_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JXF_GMRESSetMaxIter(solver, max_iter);
         JXF_GMRESSetTol(solver, tol);
         JXF_GMRESSetLogging(solver, 1);
         JXF_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */

         JXF_GMRESSetPrecond(solver, (JXF_PtrToSolverFcn)JXF_EuclidSolve,
                                    (JXF_PtrToSolverFcn)JXF_EuclidSetup, euclid_solver);

         JXF_EuclidSetup(euclid_solver, (JXF_ParCSRMatrix)par_matrix);

         JXF_GMRESSetup(solver, (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "Euclid-GMRES Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_GMRESSolve(solver, (JXF_Matrix)par_matrix, // preOperater
                               (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "Euclid-GMRES Solve", starttime, endtime, 0, 2);
         }

         JXF_GMRESGetNumIterations(solver, &num_iterations);
         JXF_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jxf_printf(" >>> num_iterations = %d\n", num_iterations);
            jxf_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JXF_EuclidDestroy(euclid_solver);
         JXF_ParCSRGMRESDestroy(solver);
      }
      break;

      case 25:  /* ILU-GMRES */
      {
         if (myid == 0) jxf_printf("\n >>> Solver: ILU(0)-GMRES(%d) \n\n", k_dim);

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_ILUZeroFactorDataCreate(&ilu_solver, comm);
         JXF_ILUZeroFactorDataSetMaxIter(ilu_solver, 1);
         JXF_ILUZeroFactorDataSetDropTol(ilu_solver, drop_tol);

         JXF_ParCSRGMRESCreate(comm, &solver);
         JXF_GMRESSetKDim(solver, k_dim);
         JXF_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JXF_GMRESSetMaxIter(solver, max_iter);
         JXF_GMRESSetTol(solver, tol);
         JXF_GMRESSetLogging(solver, 1);
         JXF_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */

         JXF_GMRESSetPrecond(solver, (JXF_PtrToSolverFcn)JXF_ILUZeroFactorDataPrecond,
                                    (JXF_PtrToSolverFcn)JXF_ILUZeroFactorDataSetup, ilu_solver);

         JXF_ILUZeroFactorDataSetup(ilu_solver, (JXF_ParCSRMatrix)par_matrix);

         JXF_GMRESSetup(solver, (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "ILU(0)-GMRES Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_GMRESSolve(solver, (JXF_Matrix)par_matrix, // preOperater
                               (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "ILU(0)-GMRES Solve", starttime, endtime, 0, 2);
         }

         JXF_ILUZeroFactorDataGetLULength(ilu_solver, &lu_length);
         if (myid == 0)
         {
            jxf_printf(" >>> Memory Complexity of ILU(0) = %d\n", lu_length);
         }

         JXF_GMRESGetNumIterations(solver, &num_iterations);
         JXF_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jxf_printf(" >>> num_iterations = %d\n", num_iterations);
            jxf_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JXF_ILUZeroFactorDataDestroy(ilu_solver);
         JXF_ParCSRGMRESDestroy(solver);
      }
      break;

      case 26:  /* ILU-AdaptiveGMRES */
      {
         if (myid == 0) jxf_printf("\n >>> Solver: ILU(0)-AdaptiveGMRES(%d) \n\n", k_dim);

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_ILUZeroFactorDataCreate(&ilu_solver, comm);
         JXF_ILUZeroFactorDataSetMaxIter(ilu_solver, 1);
         JXF_ILUZeroFactorDataSetDropTol(ilu_solver, drop_tol);

         JXF_ParCSRGMRESCreate(comm, &solver);
         JXF_GMRESSetKDim(solver, k_dim);
         JXF_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JXF_GMRESSetMaxIter(solver, max_iter);
         JXF_GMRESSetTol(solver, tol);
         JXF_GMRESSetLogging(solver, 1);
         JXF_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */
         JXF_GMRESSetResDownZeroThreshold(solver, resdown_0_threshold);
         JXF_GMRESSetConvFacThresholdTwo(solver, convfac_threshold_2);

         JXF_GMRESSetPrecond(solver, (JXF_PtrToSolverFcn)JXF_ILUZeroFactorDataPrecond,
                                    (JXF_PtrToSolverFcn)JXF_ILUZeroFactorDataSetup, ilu_solver);

         JXF_ILUZeroFactorDataSetup(ilu_solver, (JXF_ParCSRMatrix)par_matrix);

         JXF_GMRESSetup(solver, (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "ILU(0)-AdaptiveGMRES Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_GMRESAdaptiveSolve(solver, (JXF_Matrix)par_matrix, // preOperater
                                       (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "ILU(0)-AdaptiveGMRES Solve", starttime, endtime, 0, 2);
         }

         JXF_ILUZeroFactorDataGetLULength(ilu_solver, &lu_length);
         if (myid == 0)
         {
            jxf_printf(" >>> Memory Complexity of ILU(0) = %d\n", lu_length);
         }

         JXF_GMRESGetNumIterations(solver, &num_iterations);
         JXF_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jxf_printf(" >>> num_iterations = %d\n", num_iterations);
            jxf_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JXF_ILUZeroFactorDataDestroy(ilu_solver);
         JXF_ParCSRGMRESDestroy(solver);
      }
      break;

      case 28:  /* Euclid2-AdaptiveGMRES */
      {
         if (myid == 0) jxf_printf("\n >>> Solver: Euclid2-AdaptiveGMRES(%d) \n\n", k_dim);

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_CombinedPrecondDataCreate(&combined_solver, comm);
         JXF_CombinedPrecondDataSetPreID(combined_solver, 1);
         JXF_CombinedPrecondDataSetEuclidLevel(combined_solver, euclid_level);
         JXF_CombinedPrecondDataSetDropTol(combined_solver, drop_tol);
         JXF_CombinedPrecondDataSetAMGParameters(combined_solver, max_levels, relax_type, amg_print_level,
                                                interp_type, P_max_elmts, measure_type, coarsen_type,
                                                agg_num_levels, coarse_threshold, strong_threshold);

         JXF_ParCSRGMRESCreate(comm, &solver);
         JXF_GMRESSetKDim(solver, k_dim);
         JXF_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JXF_GMRESSetMaxIter(solver, max_iter);
         JXF_GMRESSetTol(solver, tol);
         JXF_GMRESSetLogging(solver, 1);
         JXF_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */
         JXF_GMRESSetResDownZeroThreshold(solver, resdown_0_threshold);
         JXF_GMRESSetConvFacThresholdTwo(solver, convfac_threshold_2);

         JXF_GMRESSetPrecond(solver, (JXF_PtrToSolverFcn)JXF_CombinedPrecondDataAdaptiveSolve,
                                    (JXF_PtrToSolverFcn)JXF_CombinedPrecondDataAdaptiveSetup2, combined_solver);

         JXF_CombinedPrecondDataAdaptiveSetup2(combined_solver, (JXF_ParCSRMatrix)par_matrix);

         JXF_GMRESSetup(solver, (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "Euclid2-AdaptiveGMRES Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_GMRESAdaptiveSolve(solver, (JXF_Matrix)par_matrix, // preOperater
                                       (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "Euclid2-AdaptiveGMRES Solve", starttime, endtime, 0, 2);
         }

         JXF_GMRESGetNumIterations(solver, &num_iterations);
         JXF_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jxf_printf(" >>> num_iterations = %d\n", num_iterations);
            jxf_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JXF_GMRESGetLastPrecondType(solver, &last_precond_type);
         if (myid == 0)
         {
            if (last_precond_type == 1)
            {
               jxf_printf("\n >>> Last Precond --- Euclid(%d) with Drop-Tol = %.6lf \n\n", euclid_level, drop_tol);
            }
            else if (last_precond_type == 21)
            {
               jxf_printf("\n >>> Last Precond --- Euclid(%d)-PAMG with Drop-Tol = %.6lf \n\n", euclid_level, drop_tol);
            }
         }

         JXF_CombinedPrecondDataDestroy(combined_solver);
         JXF_ParCSRGMRESDestroy(solver);
      }
      break;

      case 29:  /* Euclid3-AdaptiveGMRES */
      {
         if (myid == 0) jxf_printf("\n >>> Solver: Euclid3-AdaptiveGMRES(%d) \n\n", k_dim);

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_CombinedPrecondDataCreate(&combined_solver, comm);
         JXF_CombinedPrecondDataSetPreID(combined_solver, 1);
         JXF_CombinedPrecondDataSetEuclidLevel(combined_solver, euclid_level);
         JXF_CombinedPrecondDataSetDropTol(combined_solver, drop_tol);
         JXF_CombinedPrecondDataSetThetaPsiRhoPhi(combined_solver, theta_psi, theta_rho, theta_phi, theta_dis);
         JXF_CombinedPrecondDataSetAMGParameters(combined_solver, max_levels, relax_type, amg_print_level,
                                                interp_type, P_max_elmts, measure_type, coarsen_type,
                                                agg_num_levels, coarse_threshold, strong_threshold);

         JXF_ParCSRGMRESCreate(comm, &solver);
         JXF_GMRESSetKDim(solver, k_dim);
         JXF_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JXF_GMRESSetMaxIter(solver, max_iter);
         JXF_GMRESSetTol(solver, tol);
         JXF_GMRESSetLogging(solver, 1);
         JXF_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */
         JXF_GMRESSetResDownZeroThreshold(solver, resdown_0_threshold);
         JXF_GMRESSetConvFacThresholdTwo(solver, convfac_threshold_2);

         JXF_GMRESSetPrecond(solver, (JXF_PtrToSolverFcn)JXF_CombinedPrecondDataAdaptiveSolve,
                                    (JXF_PtrToSolverFcn)JXF_CombinedPrecondDataAdaptiveSetup3, combined_solver);

         JXF_CombinedPrecondDataAdaptiveSetup3(combined_solver, (JXF_ParCSRMatrix)par_matrix);

         JXF_GMRESSetup(solver, (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "Euclid3-AdaptiveGMRES Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_GMRESAdaptiveSolve(solver, (JXF_Matrix)par_matrix, // preOperater
                                       (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "Euclid3-AdaptiveGMRES Solve", starttime, endtime, 0, 2);
         }

         JXF_GMRESGetNumIterations(solver, &num_iterations);
         JXF_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jxf_printf(" >>> num_iterations = %d\n", num_iterations);
            jxf_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JXF_GMRESGetLastPrecondType(solver, &last_precond_type);
         if (myid == 0)
         {
            if (last_precond_type == 1)
            {
               jxf_printf("\n >>> Last Precond --- Euclid(%d) with Drop-Tol = %.6lf \n\n", euclid_level, drop_tol);
            }
            else if (last_precond_type == 3)
            {
               jxf_printf("\n >>> Last Precond --- PAMG \n\n");
            }
            else if (last_precond_type == 21)
            {
               jxf_printf("\n >>> Last Precond --- Euclid(%d)-PAMG with Drop-Tol = %.6lf \n\n", euclid_level, drop_tol);
            }
         }

         JXF_CombinedPrecondDataDestroy(combined_solver);
         JXF_ParCSRGMRESDestroy(solver);
      }
      break;

      case 51:  /* PAMG-Euclid-PAMG-GMRES */
      {
         if (myid == 0) jxf_printf("\n >>> Solver: PAMG-Euclid-PAMG-GMRES(%d) \n\n", k_dim);

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_CombinedPrecondDataCreate(&combined_solver, comm);
         JXF_CombinedPrecondDataSetPreID(combined_solver, 11);
         JXF_CombinedPrecondDataInitialize(combined_solver, euclid_level); /* Yue Xiaoqiang 2013/12/11 */
         //JXF_CombinedPrecondDataInitializeP(combined_solver, argc, argv); /* Yue Xiaoqiang 2014/05/24 */

         JXF_ParCSRGMRESCreate(comm, &solver);
         JXF_GMRESSetKDim(solver, k_dim);
         JXF_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JXF_GMRESSetMaxIter(solver, max_iter);
         JXF_GMRESSetTol(solver, tol);
         JXF_GMRESSetLogging(solver, 1);
         JXF_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */

         JXF_GMRESSetPrecond(solver, (JXF_PtrToSolverFcn)JXF_CombinedPrecondDataSolve,
                                    (JXF_PtrToSolverFcn)JXF_CombinedPrecondDataSetup, combined_solver);

         JXF_CombinedPrecondDataSetup(combined_solver, (JXF_ParCSRMatrix)par_matrix);

         JXF_GMRESSetup(solver, (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "PAMG-Euclid-PAMG-GMRES Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_GMRESSolve(solver, (JXF_Matrix)par_matrix, // preOperater
                               (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "PAMG-Euclid-PAMG-GMRES Solve", starttime, endtime, 0, 2);
         }

         JXF_GMRESGetNumIterations(solver, &num_iterations);
         JXF_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jxf_printf(" >>> num_iterations = %d\n", num_iterations);
            jxf_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JXF_CombinedPrecondDataDestroy(combined_solver);
         JXF_ParCSRGMRESDestroy(solver);
      }
      break;

      case 52:  /* Euclid-PAMG-GMRES */
      {
         if (myid == 0) jxf_printf("\n >>> Solver: Euclid-PAMG-GMRES(%d) \n\n", k_dim);

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_CombinedPrecondDataCreate(&combined_solver, comm);
         JXF_CombinedPrecondDataSetPreID(combined_solver, 12);
         JXF_CombinedPrecondDataInitialize(combined_solver, euclid_level); /* Yue Xiaoqiang 2013/12/11 */
         //JXF_CombinedPrecondDataInitializeP(combined_solver, argc, argv); /* Yue Xiaoqiang 2014/05/24 */

         JXF_ParCSRGMRESCreate(comm, &solver);
         JXF_GMRESSetKDim(solver, k_dim);
         JXF_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JXF_GMRESSetMaxIter(solver, max_iter);
         JXF_GMRESSetTol(solver, tol);
         JXF_GMRESSetLogging(solver, 1);
         JXF_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */

         JXF_GMRESSetPrecond(solver, (JXF_PtrToSolverFcn)JXF_CombinedPrecondDataSolve,
                                    (JXF_PtrToSolverFcn)JXF_CombinedPrecondDataSetup, combined_solver);

         JXF_CombinedPrecondDataSetup(combined_solver, (JXF_ParCSRMatrix)par_matrix);

         JXF_GMRESSetup(solver, (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "Euclid-PAMG-GMRES Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_GMRESSolve(solver, (JXF_Matrix)par_matrix, // preOperater
                               (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "Euclid-PAMG-GMRES Solve", starttime, endtime, 0, 2);
         }

         JXF_GMRESGetNumIterations(solver, &num_iterations);
         JXF_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jxf_printf(" >>> num_iterations = %d\n", num_iterations);
            jxf_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JXF_CombinedPrecondDataDestroy(combined_solver);
         JXF_ParCSRGMRESDestroy(solver);
      }
      break;

      case 53:  /* PAMG-Euclid-GMRES */
      {
         if (myid == 0) jxf_printf("\n >>> Solver: PAMG-Euclid-GMRES(%d) \n\n", k_dim);

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_CombinedPrecondDataCreate(&combined_solver, comm);
         JXF_CombinedPrecondDataSetPreID(combined_solver, 13);
         JXF_CombinedPrecondDataInitialize(combined_solver, euclid_level); /* Yue Xiaoqiang 2013/12/11 */
         //JXF_CombinedPrecondDataInitializeP(combined_solver, argc, argv); /* Yue Xiaoqiang 2014/05/24 */

         JXF_ParCSRGMRESCreate(comm, &solver);
         JXF_GMRESSetKDim(solver, k_dim);
         JXF_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JXF_GMRESSetMaxIter(solver, max_iter);
         JXF_GMRESSetTol(solver, tol);
         JXF_GMRESSetLogging(solver, 1);
         JXF_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */

         JXF_GMRESSetPrecond(solver, (JXF_PtrToSolverFcn)JXF_CombinedPrecondDataSolve,
                                    (JXF_PtrToSolverFcn)JXF_CombinedPrecondDataSetup, combined_solver);

         JXF_CombinedPrecondDataSetup(combined_solver, (JXF_ParCSRMatrix)par_matrix);

         JXF_GMRESSetup(solver, (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "PAMG-Euclid-GMRES Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_GMRESSolve(solver, (JXF_Matrix)par_matrix, // preOperater
                               (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "PAMG-Euclid-GMRES Solve", starttime, endtime, 0, 2);
         }

         JXF_GMRESGetNumIterations(solver, &num_iterations);
         JXF_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jxf_printf(" >>> num_iterations = %d\n", num_iterations);
            jxf_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JXF_CombinedPrecondDataDestroy(combined_solver);
         JXF_ParCSRGMRESDestroy(solver);
      }
      break;

      case 54:  /* Euclid-PAMG-Euclid-GMRES */
      {
         if (myid == 0) jxf_printf("\n >>> Solver: Euclid-PAMG-Euclid-GMRES(%d) \n\n", k_dim);

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_CombinedPrecondDataCreate(&combined_solver, comm);
         JXF_CombinedPrecondDataSetPreID(combined_solver, 14);
         JXF_CombinedPrecondDataInitialize(combined_solver, euclid_level); /* Yue Xiaoqiang 2013/12/11 */
         //JXF_CombinedPrecondDataInitializeP(combined_solver, argc, argv); /* Yue Xiaoqiang 2014/05/24 */

         JXF_ParCSRGMRESCreate(comm, &solver);
         JXF_GMRESSetKDim(solver, k_dim);
         JXF_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JXF_GMRESSetMaxIter(solver, max_iter);
         JXF_GMRESSetTol(solver, tol);
         JXF_GMRESSetLogging(solver, 1);
         JXF_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */

         JXF_GMRESSetPrecond(solver, (JXF_PtrToSolverFcn)JXF_CombinedPrecondDataSolve,
                                    (JXF_PtrToSolverFcn)JXF_CombinedPrecondDataSetup, combined_solver);

         JXF_CombinedPrecondDataSetup(combined_solver, (JXF_ParCSRMatrix)par_matrix);

         JXF_GMRESSetup(solver, (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "Euclid-PAMG-Euclid-GMRES Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_GMRESSolve(solver, (JXF_Matrix)par_matrix, // preOperater
                               (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "Euclid-PAMG-Euclid-GMRES Solve", starttime, endtime, 0, 2);
         }

         JXF_GMRESGetNumIterations(solver, &num_iterations);
         JXF_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jxf_printf(" >>> num_iterations = %d\n", num_iterations);
            jxf_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JXF_CombinedPrecondDataDestroy(combined_solver);
         JXF_ParCSRGMRESDestroy(solver);
      }
      break;

      case 56:  /* ILU-PAMG-GMRES */
      {
         if (myid == 0) jxf_printf("\n >>> Solver: ILU(0)-PAMG-GMRES(%d) \n\n", k_dim);

         if ((nprocs > 1) && (myid == 0))
         {
            ser_matrix = jxf_CSRMatrixRead(MatFile, file_base);
            jxf_CSRMatrixReorder(ser_matrix);
         }

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_CombinedPrecondDataCreate(&combined_solver, comm);
         JXF_CombinedPrecondDataSetPreID(combined_solver, 16);
         JXF_CombinedPrecondDataSetDropTol(combined_solver, drop_tol);
         JXF_CombinedPrecondDataInitialize(combined_solver, euclid_level); /* Yue Xiaoqiang 2013/12/11 */
         //JXF_CombinedPrecondDataInitializeP(combined_solver, argc, argv); /* Yue Xiaoqiang 2014/05/24 */
         JXF_CombinedPrecondDataSetILUMatA(combined_solver, ser_matrix); /* 0号进程上的矩阵 */

         JXF_ParCSRGMRESCreate(comm, &solver);
         JXF_GMRESSetKDim(solver, k_dim);
         JXF_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JXF_GMRESSetMaxIter(solver, max_iter);
         JXF_GMRESSetTol(solver, tol);
         JXF_GMRESSetLogging(solver, 1);
         JXF_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */

         JXF_GMRESSetPrecond(solver, (JXF_PtrToSolverFcn)JXF_CombinedPrecondDataSolve,
                                    (JXF_PtrToSolverFcn)JXF_CombinedPrecondDataSetup, combined_solver);

         JXF_CombinedPrecondDataSetup(combined_solver, (JXF_ParCSRMatrix)par_matrix);

         JXF_GMRESSetup(solver, (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "ILU-PAMG-GMRES Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_GMRESSolve(solver, (JXF_Matrix)par_matrix, // preOperater
                               (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "ILU-PAMG-GMRES Solve", starttime, endtime, 0, 2);
         }

         JXF_CombinedPrecondDataGetLULength(combined_solver, &lu_length);
         if (myid == 0)
         {
            jxf_printf(" >>> Memory Complexity of ILU(0) = %d\n", lu_length);
         }

         JXF_GMRESGetNumIterations(solver, &num_iterations);
         JXF_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jxf_printf(" >>> num_iterations = %d\n", num_iterations);
            jxf_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         if ((nprocs > 1) && (myid == 0))
         {
            jxf_CSRMatrixDestroy(ser_matrix);
         }
         JXF_CombinedPrecondDataDestroy(combined_solver);
         JXF_ParCSRGMRESDestroy(solver);
      }
      break;

      case 57:  /* ILU-PAMG-ILU-GMRES */
      {
         if (myid == 0) jxf_printf("\n >>> Solver: ILU(0)-PAMG-ILU(0)-GMRES(%d) \n\n", k_dim);

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_CombinedPrecondDataCreate(&combined_solver, comm);
         JXF_CombinedPrecondDataSetPreID(combined_solver, 17);
         JXF_CombinedPrecondDataSetDropTol(combined_solver, drop_tol);
         JXF_CombinedPrecondDataInitialize(combined_solver, euclid_level); /* Yue Xiaoqiang 2013/12/11 */
         //JXF_CombinedPrecondDataInitializeP(combined_solver, argc, argv); /* Yue Xiaoqiang 2014/05/24 */

         JXF_ParCSRGMRESCreate(comm, &solver);
         JXF_GMRESSetKDim(solver, k_dim);
         JXF_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JXF_GMRESSetMaxIter(solver, max_iter);
         JXF_GMRESSetTol(solver, tol);
         JXF_GMRESSetLogging(solver, 1);
         JXF_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */

         JXF_GMRESSetPrecond(solver, (JXF_PtrToSolverFcn)JXF_CombinedPrecondDataSolve,
                                    (JXF_PtrToSolverFcn)JXF_CombinedPrecondDataSetup, combined_solver);

         JXF_CombinedPrecondDataSetup(combined_solver, (JXF_ParCSRMatrix)par_matrix);

         JXF_GMRESSetup(solver, (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "ILU-PAMG-ILU(0)-GMRES Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_GMRESSolve(solver, (JXF_Matrix)par_matrix, // preOperater
                               (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "ILU-PAMG-ILU(0)-GMRES Solve", starttime, endtime, 0, 2);
         }

         JXF_CombinedPrecondDataGetLULength(combined_solver, &lu_length);
         if (myid == 0)
         {
            jxf_printf(" >>> Memory Complexity of ILU(0) = %d\n", lu_length);
         }

         JXF_GMRESGetNumIterations(solver, &num_iterations);
         JXF_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jxf_printf(" >>> num_iterations = %d\n", num_iterations);
            jxf_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JXF_CombinedPrecondDataDestroy(combined_solver);
         JXF_ParCSRGMRESDestroy(solver);
      }
      break;

      case 31:  /* BiCGSTAB */
      {
         if (myid == 0) jxf_printf("\n >>> Solver: BiCGSTAB \n\n");

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_ParCSRBiCGSTABCreate(comm, &solver);
         JXF_BiCGSTABSetMaxIter(solver, max_iter);
         JXF_BiCGSTABSetTol(solver, tol);
         JXF_BiCGSTABSetAbsoluteTol(solver, 0.0);
         JXF_BiCGSTABSetConvCriteria(solver, 0);
         JXF_BiCGSTABSetLogging(solver, 1);
         JXF_BiCGSTABSetPrintLevel(solver, print_level);

         JXF_BiCGSTABSetup(solver, (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "BiCGSTAB Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_BiCGSTABSolve(solver, (JXF_Matrix)par_matrix, // preOperater
                                  (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "BiCGSTAB Solve", starttime, endtime, 0, 2);
         }

         JXF_BiCGSTABGetNumIterations(solver, &num_iterations);
         JXF_BiCGSTABGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jxf_printf(" >>> num_iterations = %d\n", num_iterations);
            jxf_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JXF_ParCSRBiCGSTABDestroy(solver);
      }
      break;

      case 32:  /* PAMG-BiCGSTAB */
      {
         if (myid == 0) jxf_printf("\n >>> Solver: PAMG-BiCGSTAB \n\n");

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_PAMGCreate(&amg_solver);
         JXF_PAMGSetMaxLevels(amg_solver, max_levels);
         JXF_PAMGSetMaxIter(amg_solver, 1);
         JXF_PAMGSetCycleType(amg_solver, cycle_type);
         JXF_PAMGSetMeasureType(amg_solver, measure_type);
         JXF_PAMGSetRAP2(amg_solver, rap2);
         JXF_PAMGSetKeepTranspose(amg_solver, keepTranspose);
         JXF_PAMGSetCoarsenType(amg_solver, coarsen_type);
         JXF_PAMGSetInterpType(amg_solver, interp_type);
         JXF_PAMGSetPMaxElmts(amg_solver, P_max_elmts);
         JXF_PAMGSetAggNumLevels(amg_solver, agg_num_levels);
         JXF_PAMGSetAIMeasureType(amg_solver, ai_measure_type);
         JXF_PAMGSetAIRelaxType(amg_solver, ai_relax_type);
         JXF_PAMGSetStrongThreshold(amg_solver, strong_threshold);
         JXF_PAMGSetMaxRowSum(amg_solver, max_row_sum);
         JXF_PAMGSetPrintLevel(amg_solver, amg_print_level);
         JXF_PAMGSetCoarseThreshold(amg_solver, coarse_threshold);
         JXF_PAMGSetRelaxWt(amg_solver, relax_wt);
         JXF_PAMGSetOuterWt(amg_solver, outer_wt);
         if (ns_down > -1) JXF_PAMGSetCycleNumSweeps(amg_solver, ns_down, 1);           /* sweep for "down" */
         if (ns_up > -1) JXF_PAMGSetCycleNumSweeps(amg_solver, ns_up, 2);           /* sweep for "up" */
         JXF_PAMGSetCycleNumSweeps(amg_solver, ns_coarse, 3);           /* sweep for "coarsest" */
         JXF_PAMGSetCycleRelaxType(amg_solver, relax_type, 1);  /* relax_type for "down" */
         JXF_PAMGSetCycleRelaxType(amg_solver, relax_type, 2);  /* relax_type for "up" */
         JXF_PAMGSetCycleRelaxType(amg_solver, coarse_solver, 3);           /* relax_type for "coarsest" */

         JXF_ParCSRBiCGSTABCreate(comm, &solver);
         JXF_BiCGSTABSetMaxIter(solver, max_iter);
         JXF_BiCGSTABSetTol(solver, tol);
         JXF_BiCGSTABSetAbsoluteTol(solver, 0.0);
         JXF_BiCGSTABSetConvCriteria(solver, 0);
         JXF_BiCGSTABSetLogging(solver, 1);
         JXF_BiCGSTABSetPrintLevel(solver, print_level);

         JXF_BiCGSTABSetPrecond(solver, (JXF_PtrToSolverFcn)JXF_PAMGPrecond,
                                       (JXF_PtrToSolverFcn)JXF_PAMGSetup, amg_solver);

         JXF_PAMGSetup(amg_solver, (JXF_ParCSRMatrix)par_matrix);

         JXF_BiCGSTABSetup(solver, (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "PAMG-BiCGSTAB Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_BiCGSTABSolve(solver, (JXF_Matrix)par_matrix, // preOperater
                                  (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "PAMG-BiCGSTAB Solve", starttime, endtime, 0, 2);
         }

         JXF_BiCGSTABGetNumIterations(solver, &num_iterations);
         JXF_BiCGSTABGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jxf_printf(" >>> num_iterations = %d\n", num_iterations);
            jxf_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JXF_PAMGDestroy(amg_solver);
         JXF_ParCSRBiCGSTABDestroy(solver);
      }
      break;

      case 33:  /* DS-BiCGSTAB */
      {
         if (myid == 0) jxf_printf("\n >>> Solver: DS-BiCGSTAB \n\n");

         if (TTest) starttime = jxf_MPI_Wtime();

         ds_solver = NULL;

         JXF_ParCSRBiCGSTABCreate(comm, &solver);
         JXF_BiCGSTABSetMaxIter(solver, max_iter);
         JXF_BiCGSTABSetTol(solver, tol);
         JXF_BiCGSTABSetAbsoluteTol(solver, 0.0);
         JXF_BiCGSTABSetConvCriteria(solver, 0);
         JXF_BiCGSTABSetLogging(solver, 1);
         JXF_BiCGSTABSetPrintLevel(solver, print_level);

         JXF_BiCGSTABSetPrecond(solver, (JXF_PtrToSolverFcn)JXF_DiagScalePrecond,
                                       (JXF_PtrToSolverFcn)JXF_DiagScaleSetup, ds_solver);

         JXF_DiagScaleSetup(ds_solver, (JXF_ParCSRMatrix)par_matrix);

         JXF_BiCGSTABSetup(solver, (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "DS-BiCGSTAB Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_BiCGSTABSolve(solver, (JXF_Matrix)par_matrix, // preOperater
                                  (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "DS-BiCGSTAB Solve", starttime, endtime, 0, 2);
         }

         JXF_BiCGSTABGetNumIterations(solver, &num_iterations);
         JXF_BiCGSTABGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jxf_printf(" >>> num_iterations = %d\n", num_iterations);
            jxf_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JXF_ParCSRBiCGSTABDestroy(solver);
      }
      break;

      case 34:  /* Euclid-BiCGSTAB */
      {
         if (myid == 0) jxf_printf("\n >>> Solver: Euclid-BiCGSTAB \n\n");

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_EuclidCreate(comm, &euclid_solver);
         //JXF_EuclidSetParams(euclid_solver, argc, argv);
         JXF_EuclidSetLevel(euclid_solver, euclid_level);
         JXF_EuclidSetBJ(euclid_solver, euclid_bj);

         JXF_ParCSRBiCGSTABCreate(comm, &solver);
         JXF_BiCGSTABSetMaxIter(solver, max_iter);
         JXF_BiCGSTABSetTol(solver, tol);
         JXF_BiCGSTABSetAbsoluteTol(solver, 0.0);
         JXF_BiCGSTABSetConvCriteria(solver, 0);
         JXF_BiCGSTABSetLogging(solver, 1);
         JXF_BiCGSTABSetPrintLevel(solver, print_level);

         JXF_BiCGSTABSetPrecond(solver, (JXF_PtrToSolverFcn)JXF_EuclidSolve,
                                       (JXF_PtrToSolverFcn)JXF_EuclidSetup, euclid_solver);

         JXF_EuclidSetup(euclid_solver, (JXF_ParCSRMatrix)par_matrix);

         JXF_BiCGSTABSetup(solver, (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "Euclid-BiCGSTAB Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_BiCGSTABSolve(solver, (JXF_Matrix)par_matrix, // preOperater
                                  (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "Euclid-BiCGSTAB Solve", starttime, endtime, 0, 2);
         }

         JXF_BiCGSTABGetNumIterations(solver, &num_iterations);
         JXF_BiCGSTABGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jxf_printf(" >>> num_iterations = %d\n", num_iterations);
            jxf_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JXF_EuclidDestroy(euclid_solver);
         JXF_ParCSRBiCGSTABDestroy(solver);
      }
      break;

      case 42:  /* PAMG-FlexGMRES */
      {
         if (myid == 0) jxf_printf("\n >>> Solver: PAMG-FlexGMRES(%d) \n\n", k_dim);

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_PAMGCreate(&amg_solver);
         if (restri_type)
         {
            jxf_assert(restri_type >= 0);
            JXF_PAMGSetRestriction(amg_solver, restri_type);
            JXF_PAMGSetGridRelaxPoints(amg_solver, grid_relax_points);
         }
         JXF_PAMGSetMaxLevels(amg_solver, max_levels);
         JXF_PAMGSetMaxIter(amg_solver, 1);
         JXF_PAMGSetCycleType(amg_solver, cycle_type);
         JXF_PAMGSetMeasureType(amg_solver, measure_type);
         JXF_PAMGSetRAP2(amg_solver, rap2);
         JXF_PAMGSetKeepTranspose(amg_solver, keepTranspose);
         JXF_PAMGSetCoarsenType(amg_solver, coarsen_type);
         JXF_PAMGSetInterpType(amg_solver, interp_type);
         JXF_PAMGSetPMaxElmts(amg_solver, P_max_elmts);
         JXF_PAMGSetAggNumLevels(amg_solver, agg_num_levels);
         JXF_PAMGSetAIMeasureType(amg_solver, ai_measure_type);
         JXF_PAMGSetAIRelaxType(amg_solver, ai_relax_type);
         JXF_PAMGSetStrongThreshold(amg_solver, strong_threshold);
         JXF_PAMGSetMaxRowSum(amg_solver, max_row_sum);
         JXF_PAMGSetPrintLevel(amg_solver, amg_print_level);
         JXF_PAMGSetCoarseThreshold(amg_solver, coarse_threshold);
         JXF_PAMGSetRelaxWt(amg_solver, relax_wt);
         JXF_PAMGSetOuterWt(amg_solver, outer_wt);
         JXF_PAMGSetSCommPkgSwitch(amg_solver, S_commpkg_switch);
         JXF_PAMGSetAIRStrongTh(amg_solver, AIR_strong_th);
         if (ns_down > -1) JXF_PAMGSetCycleNumSweeps(amg_solver, ns_down, 1);           /* sweep for "down" */
         if (ns_up > -1) JXF_PAMGSetCycleNumSweeps(amg_solver, ns_up, 2);           /* sweep for "up" */
         JXF_PAMGSetCycleNumSweeps(amg_solver, ns_coarse, 3);           /* sweep for "coarsest" */
         JXF_PAMGSetCycleRelaxType(amg_solver, relax_type, 1);  /* relax_type for "down" */
         JXF_PAMGSetCycleRelaxType(amg_solver, relax_type, 2);  /* relax_type for "up" */
         JXF_PAMGSetCycleRelaxType(amg_solver, coarse_solver, 3);           /* relax_type for "coarsest" */

         JXF_ParCSRFlexGMRESCreate(comm, &solver);
         JXF_FlexGMRESSetKDim(solver, k_dim);
         JXF_FlexGMRESSetIsCheckRestarted(solver, is_check_restarted);
         JXF_FlexGMRESSetMaxIter(solver, max_iter);
         JXF_FlexGMRESSetTol(solver, tol);
         JXF_FlexGMRESSetLogging(solver, 1);
         JXF_FlexGMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */

         JXF_FlexGMRESSetPrecond(solver, (JXF_PtrToSolverFcn)JXF_PAMGPrecond,
                                    (JXF_PtrToSolverFcn)JXF_PAMGSetup, amg_solver);

         JXF_PAMGSetup(amg_solver, (JXF_ParCSRMatrix)par_matrix);

         /* this is optional - could be a user defined one instead */
         JXF_FlexGMRESSetModifyPC(solver, (JXF_PtrToModifyPCFcn)jxf_FlexGMRESModifyPCDefault);

         JXF_FlexGMRESSetup(solver, (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "PAMG-FlexGMRES Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_FlexGMRESSolve(solver, (JXF_Matrix)par_matrix, // preOperater
                               (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "PAMG-FlexGMRES Solve", starttime, endtime, 0, 2);
         }

         JXF_FlexGMRESGetNumIterations(solver, &num_iterations);
         JXF_FlexGMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jxf_printf(" >>> num_iterations = %d\n", num_iterations);
            jxf_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JXF_PAMGDestroy(amg_solver);
         JXF_ParCSRFlexGMRESDestroy(solver);
      }
      break;

      case 62:  /* PAMG-COGMRES */
      {
         if (myid == 0) jxf_printf("\n >>> Solver: PAMG-COGMRES(%d) \n\n", k_dim);

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_PAMGCreate(&amg_solver);
         if (restri_type)
         {
            jxf_assert(restri_type >= 0);
            JXF_PAMGSetRestriction(amg_solver, restri_type);
            JXF_PAMGSetGridRelaxPoints(amg_solver, grid_relax_points);
         }
         JXF_PAMGSetMaxLevels(amg_solver, max_levels);
         JXF_PAMGSetMaxIter(amg_solver, 1);
         JXF_PAMGSetCycleType(amg_solver, cycle_type);
         JXF_PAMGSetMeasureType(amg_solver, measure_type);
         JXF_PAMGSetRAP2(amg_solver, rap2);
         JXF_PAMGSetKeepTranspose(amg_solver, keepTranspose);
         JXF_PAMGSetCoarsenType(amg_solver, coarsen_type);
         JXF_PAMGSetInterpType(amg_solver, interp_type);
         JXF_PAMGSetPMaxElmts(amg_solver, P_max_elmts);
         JXF_PAMGSetAggNumLevels(amg_solver, agg_num_levels);
         JXF_PAMGSetAIMeasureType(amg_solver, ai_measure_type);
         JXF_PAMGSetAIRelaxType(amg_solver, ai_relax_type);
         JXF_PAMGSetStrongThreshold(amg_solver, strong_threshold);
         JXF_PAMGSetMaxRowSum(amg_solver, max_row_sum);
         JXF_PAMGSetPrintLevel(amg_solver, amg_print_level);
         JXF_PAMGSetCoarseThreshold(amg_solver, coarse_threshold);
         JXF_PAMGSetRelaxWt(amg_solver, relax_wt);
         JXF_PAMGSetOuterWt(amg_solver, outer_wt);
         JXF_PAMGSetSCommPkgSwitch(amg_solver, S_commpkg_switch);
         JXF_PAMGSetAIRStrongTh(amg_solver, AIR_strong_th);
         if (ns_down > -1) JXF_PAMGSetCycleNumSweeps(amg_solver, ns_down, 1);           /* sweep for "down" */
         if (ns_up > -1) JXF_PAMGSetCycleNumSweeps(amg_solver, ns_up, 2);           /* sweep for "up" */
         JXF_PAMGSetCycleNumSweeps(amg_solver, ns_coarse, 3);           /* sweep for "coarsest" */
         JXF_PAMGSetCycleRelaxType(amg_solver, relax_type, 1);  /* relax_type for "down" */
         JXF_PAMGSetCycleRelaxType(amg_solver, relax_type, 2);  /* relax_type for "up" */
         JXF_PAMGSetCycleRelaxType(amg_solver, coarse_solver, 3);           /* relax_type for "coarsest" */

         JXF_ParCSRCOGMRESCreate(comm, &solver);
         JXF_COGMRESSetKDim(solver, k_dim);
         JXF_COGMRESSetUnroll(solver, unroll);
         JXF_COGMRESSetCGS(solver, cgs);
         JXF_COGMRESSetIsCheckRestarted(solver, is_check_restarted);
         JXF_COGMRESSetMaxIter(solver, max_iter);
         JXF_COGMRESSetTol(solver, tol);
         JXF_COGMRESSetLogging(solver, 1);
         JXF_COGMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */

         JXF_COGMRESSetPrecond(solver, (JXF_PtrToSolverFcn)JXF_PAMGPrecond,
                                    (JXF_PtrToSolverFcn)JXF_PAMGSetup, amg_solver);

         JXF_PAMGSetup(amg_solver, (JXF_ParCSRMatrix)par_matrix);

         JXF_COGMRESSetup(solver, (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "PAMG-COGMRES Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jxf_MPI_Wtime();

         JXF_COGMRESSolve(solver, (JXF_Matrix)par_matrix, // preOperater
                               (JXF_Matrix)par_matrix, (JXF_Vector)par_rhs, (JXF_Vector)par_sol);

         if (TTest)
         {
            endtime = jxf_MPI_Wtime();
            jxf_GetWallTime(comm, "PAMG-COGMRES Solve", starttime, endtime, 0, 2);
         }

         JXF_COGMRESGetNumIterations(solver, &num_iterations);
         JXF_COGMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jxf_printf(" >>> num_iterations = %d\n", num_iterations);
            jxf_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JXF_PAMGDestroy(amg_solver);
         JXF_ParCSRCOGMRESDestroy(solver);
      }
      break;
   }

   if (TTest)
   {
      endtimeT = jxf_MPI_Wtime();
      jxf_GetWallTime(comm, "Total Sove Time", starttimeT, endtimeT, 0, 2);
   }

   *iter = num_iterations;
   return 0;
}



#define Debug_zl 0
JXF_Int
jxf_CSRMatrixFilter( jxf_CSRMatrix *matrix, JXF_Real eps, jxf_CSRMatrix *matrixOut)
{
   JXF_Real    *matrix_data, val;
   JXF_Int     *matrix_i;
   JXF_Int     *matrix_j;
   JXF_Int      num_rows;
   JXF_Int      i, j, istart, iend;

   matrix_data = jxf_CSRMatrixData(matrix);
   matrix_i    = jxf_CSRMatrixI(matrix);
   matrix_j    = jxf_CSRMatrixJ(matrix);
   num_rows    = jxf_CSRMatrixNumRows(matrix);


   for (j = 0; j <= num_rows; j ++)
   {
      matrixOut->i[j] = matrix_i[j];
   }

   // for (j = 0; j < matrix_i[num_rows]; j ++)
   // {
   //    matrixOut->j[j] = matrix_j[j];
   //    val = matrix_data[j];
   //    if (fabs(val) <= eps)
   //    {
   //       matrixOut->data[j] = 0;
   //       printf("1 val[%d]=%e\n",j, val);
   //    }else{
   //       matrixOut->data[j] = val;
   //       if (j<matrix_i[1])
   //       {
   //          printf("2 val[%d]=%e\n",j, val);
   //       }
   //    }
   // }
   JXF_Real max, th;
   JXF_Int count = 0;
   for ( i = 0; i < num_rows; i++)
   {
      istart = matrix_i[i]; iend = matrix_i[i+1];
      max = matrix_data[istart];
      th = eps*max;
      for (j = istart; j < iend; j ++)
      {
         matrixOut->j[j] = matrix_j[j];
         val = matrix_data[j];

         if (fabs(val) <= fabs(th))
         {
            matrixOut->data[j] = 0;
            count++;
         }else{
            matrixOut->data[j] = val;
         }

#if Debug_zl
         if (i<10)
         {
            printf("{%d, %d, %e} ",i, matrix_j[j], val);
         }
#endif
      }
#if Debug_zl
      if (i<10)
      {
         printf("\n");
      }
#endif

   }
    printf("%s, Filter out the number of small elements: %d\n", __FUNCTION__, count);
   // matrixOut->num_rows = num_rows;
   // matrixOut->num_cols = matrix->num_cols;
   // matrixOut->num_nonzeros = matrix->num_nonzeros;

   return 1;
}





// 先几何节点后物理量(3个量)，转化为先物理量后几何节点
JXF_Int TranMatrixOrder( JXF_Int n, JXF_Int nnz, JXF_Int *ia, JXF_Int *ja, JXF_Real *aa , JXF_Real *b)
{
    JXF_Int i, k, j, ik;
    JXF_Int istart, iend;
    JXF_Real val;
    JXF_Int *ia0, *ja0;
    JXF_Real *aa0, *b0;
    JXF_Int nnm = n/3;

    if (n%3 != 0)
    {
        printf("\033[31mERROR:\033[00m the number of rows in a matrix cannot be divided by 3!!!\n");
        exit(-1);
    }

    ia0 = (JXF_Int *)malloc(sizeof(JXF_Int) * (n+1));
    ja0 = (JXF_Int *)malloc(sizeof(JXF_Int) * nnz);
    aa0 = (JXF_Real *)malloc(sizeof(JXF_Real) * nnz);
    b0  = (JXF_Real *)malloc(sizeof(JXF_Real) * n);


    ia0[0] = 0;
    // generate u-direction row address ia0(i),i=1(1)nnm
    // ia(ik+1)-ia(ik)=the numbers of non-zero element
    // in the ith row about u-direction
    for ( i = 0; i < nnm; i++)
    {
        ik=3*i;
	    ia0[i+1] = ia0[i] + ia[ik+1] - ia[ik];
    }

    // generate v-direction row address ia0(i),i=nnm+1(1)2*nnm
    // ia(ik+1)-ia(ik)=the numbers of non-zero element
    // in the (nnm+i)th row about v-direction
    for ( i = 0; i < nnm; i++)
    {
        ik=3*i+1;
	    ia0[nnm+i+1] = ia0[nnm+i] + ia[ik+1] - ia[ik];
    }

    // generate w-direction row address ia0(i),i=2*nnm+1(1)3*nnm
    // ia(ik+1)-ia(ik)=the numbers of non-zero element
    // in the (nnm+i)th row about v-direction
    for ( i = 0; i < nnm; i++)
    {
        ik=3*i+2;
	    ia0[2*nnm+i+1] = ia0[2*nnm+i] + ia[ik+1] - ia[ik];
    }


    JXF_Int li, lj;
    // generate u-direction collumn address ja0(k),
    // k=ia0(i)(1)ia0(i+1)-1,i=1(1)nnm.
    for ( i = 0; i < nnm; i++)
    {
        li = ia0[i];
        lj = 0;
        ik = 3*i;
        istart = ia[ik];
        iend = ia[ik+1];
        for ( k = istart; k < iend; k++)
        {
            j = ja[k];

            if (j % 3 == 0)
            {
                ja0[li+lj] = j / 3;
                aa0[li+lj] = aa[k];
                lj++;
            }else if (j % 3 == 1)
            {
                ja0[li+lj] = nnm + j / 3;
                aa0[li+lj] = aa[k];
                lj++;
            }else{
                ja0[li+lj] = 2*nnm + j / 3;
                aa0[li+lj] = aa[k];
                lj++;
            }
        }
    }


    // generate v-direction collumn address ja0(k),
    // k=ia0(nnm+i)(1)ia0(nnm+i+1)-1,i=1(1)nnm.
    for ( i = 0; i < nnm; i++)
    {
        li = ia0[nnm+i];
        lj = 0;
        ik = 3*i+1;
        istart = ia[ik];
        iend = ia[ik+1];
        for ( k = istart; k < iend; k++)
        {
            j = ja[k];

            if (j % 3 == 0)
            {
                ja0[li+lj] = j / 3;
                aa0[li+lj] = aa[k];
                lj++;
            }else if (j % 3 == 1)
            {
                ja0[li+lj] = nnm + j / 3;
                aa0[li+lj] = aa[k];
                lj++;
            }else{
                ja0[li+lj] = 2*nnm + j / 3;
                aa0[li+lj] = aa[k];
                lj++;
            }
        }
    }

    // generate w-direction collumn address ja0(k),
    // k=ia0(2*nnm+i)(1)ia0(2*nnm+i+1)-1,i=1(1)nnm.
    for ( i = 0; i < nnm; i++)
    {
        li = ia0[2*nnm+i];
        lj = 0;
        ik = 3*i+2;
        istart = ia[ik];
        iend = ia[ik+1];
        for ( k = istart; k < iend; k++)
        {
            j = ja[k];

            if (j % 3 == 0)
            {
                ja0[li+lj] = j / 3;
                aa0[li+lj] = aa[k];
                lj++;
            }else if (j % 3 == 1)
            {
                ja0[li+lj] = nnm + j / 3;
                aa0[li+lj] = aa[k];
                lj++;
            }else{
                ja0[li+lj] = 2*nnm + j / 3;
                aa0[li+lj] = aa[k];
                lj++;
            }
        }
    }

    for ( i = 0; i < n; i++)
    {
        if (i % 3 == 0)
        {
            b0[i/3] = b[i];
        }else if (i % 3 == 1)
        {
            b0[nnm+i/3] = b[i];
        }else
        {
            b0[2*nnm+i/3] = b[i];
        }
    }


    // output
    for ( i = 0; i < n; i++){
        b[i] = b0[i];
        ia[i] = ia0[i];
    }
    ia[n+1] = ia0[n+1];

    for ( i = 0; i < nnz; i++){
        ja[i] = ja0[i];
        aa[i] = aa0[i];
    }
    // clean up
    free(ia0);
    free(ja0);
    free(aa0);
    free(b0);


    // output
    // ia = ia0;
    // ja = ja0;
    // aa = aa0;
    //  b = b0;

    // return
    return 0;
}

// 先几何节点后物理量，转化为先物理量后几何节点
JXF_Int TranMatrixOrder_new( JXF_Int n, JXF_Int nnz, JXF_Int *ia, JXF_Int *ja, JXF_Real *aa , JXF_Real *b)
{
    JXF_Int *ia0, *ja0;
    JXF_Real *aa0, *b0;
    JXF_Int RowA = 0 , RowC = 0;
    JXF_Int ng_p_two , sub_num_rows;
    JXF_Int num_nonzerosC,i, j, mdo, col, row_end;
    JXF_Int Rowx = 0, Rowy;

   ng_p_two = 6;                     //自由度个数
   sub_num_rows = n/ng_p_two;   //子块维数

    if (n%sub_num_rows != 0)
    {
        printf("\033[31mERROR:\033[00m the number of rows in a matrix cannot be divided by %d!!!\n",sub_num_rows);
        exit(-1);
    }

   ia0 = (JXF_Int *)malloc(sizeof(JXF_Int) * (n+1));
   ja0 = (JXF_Int *)malloc(sizeof(JXF_Int) * nnz);
   aa0 = (JXF_Real *)malloc(sizeof(JXF_Real) * nnz);
   b0  = (JXF_Real *)malloc(sizeof(JXF_Real) * n);

   ia0[0] = 0;
   num_nonzerosC = 0;
 for (RowC = 0; RowC < n; RowC ++)
   {
      mdo = RowC / sub_num_rows;
      RowA = ng_p_two * (RowC - mdo * sub_num_rows) + mdo;
      row_end = ia[RowA+1];
      for (j = ia[RowA]; j < row_end; j ++)
      {
         col = ja[j];
         mdo = col % ng_p_two;
         ja0[num_nonzerosC] = (col - mdo) / ng_p_two + mdo * sub_num_rows;
         aa0[num_nonzerosC] = aa[j];
         num_nonzerosC ++;
      }
      ia0[RowC+1] = num_nonzerosC;
   }

   for (Rowy = 0; Rowy < n; Rowy ++)      // 遍历向量的所有元素
   {
      mdo = Rowy / sub_num_rows;                             // 表示在第几块
      Rowx = ng_p_two * (Rowy - mdo * sub_num_rows) + mdo;   // 3*
      b0[Rowy] = b[Rowx];
   }


    // output
    for ( i = 0; i < n; i++){
        b[i] = b0[i];
        ia[i] = ia0[i];
    }
    ia[n+1] = ia0[n+1];

    for ( i = 0; i < nnz; i++){
        ja[i] = ja0[i];
        aa[i] = aa0[i];
    }
    // clean up
    free(ia0);
    free(ja0);
    free(aa0);
    free(b0);
    return 0;
}

#if 0

JXF_Int
jxf_LU(jxf_ParCSRMatrix *par_matrix,
               jxf_ParVector    *par_rhs,
               jxf_ParVector    *par_app)
{
   MPI_Comm         comm         = jxf_ParCSRMatrixComm(par_matrix);
   jxf_CSRMatrix    *A_diag       = jxf_ParCSRMatrixDiag(par_matrix);
   jxf_ParCSRCommPkg    *comm_pkg = jxf_ParCSRMatrixCommPkg(par_matrix);

   JXF_Int             n_global = jxf_ParCSRMatrixGlobalNumRows(par_matrix);
   JXF_Int             n        = jxf_CSRMatrixNumRows(A_diag);
   JXF_Int             first_index   = jxf_ParVectorFirstIndex(par_app);

   jxf_Vector      *u_local = jxf_ParVectorLocalVector(par_app);
   JXF_Real         *u_data  = jxf_VectorData(u_local);
   // JXF_LOW_REAL         *u_data  = jxf_VectorData(u_local);

   jxf_CSRMatrix    *A_CSR;
   JXF_Int             *A_CSR_i;
   JXF_Int             *A_CSR_j;
   JXF_Real          *A_CSR_data;
   // JXF_LOW_REAL          *A_CSR_data;

   jxf_Vector       *f_vector;
   JXF_Real          *f_vector_data;
   // JXF_LOW_REAL          *f_vector_data;

   JXF_Int             i;
   JXF_Int             jj;
   JXF_Int             column;
   JXF_Int             relax_error = 0;
   JXF_Int             num_procs, my_id;

   // JXF_Real         *A_mat;
   // JXF_Real         *b_vec;

   JXF_LOW_REAL         *A_mat;
   JXF_LOW_REAL         *b_vec;

   jxf_MPI_Comm_size(comm, &num_procs);
   jxf_MPI_Comm_rank(comm, &my_id);

  /*-------------------------------------------------------------------------
   * added by peghoty, if the comm_pkg of par_matrix is not created
   * previously, something will be wrong when the "relax" function
   * is called on multi-processors occasions.  2009/07/24
   *----------------------------------------------------------------------- */

   if (!comm_pkg)
   {
      jxf_MatvecCommPkgCreate(par_matrix);
      comm_pkg = jxf_ParCSRMatrixCommPkg(par_matrix);
   }

      //------------------------------------------------------------------------------//
      //                   Direct solve: use gaussian elimination                     //
      //------------------------------------------------------------------------------//

        /*-----------------------------------------------------------------
         *  Generate CSR matrix from ParCSRMatrix par_matrix
         *-----------------------------------------------------------------*/
#ifdef JXF_NO_GLOBAL_PARTITION
         /* all processors are needed for these routines */
         A_CSR = jxf_ParCSRMatrixToCSRMatrixAll(par_matrix);
         f_vector = jxf_ParVectorToVectorAll(par_rhs);
	 if (n)
	 {
#else
	 if (n)
	 {
	     A_CSR = jxf_ParCSRMatrixToCSRMatrixAll(par_matrix);
	     f_vector = jxf_ParVectorToVectorAll(par_rhs);
#endif
         A_CSR_i = jxf_CSRMatrixI(A_CSR);
         A_CSR_j = jxf_CSRMatrixJ(A_CSR);
         A_CSR_data = jxf_CSRMatrixData(A_CSR);
   	   f_vector_data = jxf_VectorData(f_vector);

            // A_mat = jxf_CTAlloc(JXF_Real, n_global*n_global);

            // b_vec = jxf_CTAlloc(JXF_Real, n_global);

            A_mat = jxf_CTAlloc(JXF_LOW_REAL, n_global*n_global);
            b_vec = jxf_CTAlloc(JXF_LOW_REAL, n_global);

         //   /*---------------------------------------------------------------
         //    *  Load CSR matrix into A_mat.
         //    *---------------------------------------------------------------*/

            for (i = 0; i < n_global; i ++)
            {
               for (jj = A_CSR_i[i]; jj < A_CSR_i[i+1]; jj ++)
               {
                  column = A_CSR_j[jj];
                  A_mat[i*n_global+column] = A_CSR_data[jj];
               }
               b_vec[i] = f_vector_data[i];
            }

            relax_error = jxf_gselim_lu(A_mat,b_vec,n_global);

            /* use version with pivoting */
            /* relax_error = jxf_gselim_piv(A_mat,b_vec,n_global);*/

            for (i = 0; i < n; i ++)
            {
               u_data[i] = b_vec[first_index+i];
            }

	         jxf_TFree(A_mat);
            jxf_TFree(b_vec);
            jxf_CSRMatrixDestroy(A_CSR);
            A_CSR = NULL;
            jxf_SeqVectorDestroy(f_vector);
            f_vector = NULL;

         }
#ifdef JXF_NO_GLOBAL_PARTITION
         else
         {
            jxf_CSRMatrixDestroy(A_CSR);
            A_CSR = NULL;
            jxf_SeqVectorDestroy(f_vector);
            f_vector = NULL;
         }
#endif

   // return(relax_error);
   jxf_printf("hjmhjm:%d\n",relax_error);
   return 0;
}

JXF_Int
jxf_gselim_lu( JXF_LOW_REAL *A_matrix, JXF_LOW_REAL *x_vector, JXF_Int size )
{
   JXF_Int    err_flag = 0;
   JXF_Int    j,k,m;
   JXF_LOW_REAL factor;

   if (size == 1) /* A_matrix is 1x1 */
   {
      if (A_matrix[0] != 0.0)
      {
         x_vector[0] = x_vector[0] / A_matrix[0];
         return(err_flag);
      }
      else
      {
         err_flag = 1;
         return(err_flag);
      }
   }
   else /* A_matrix is nxn.  Forward elimination */
   {
      for (k = 0; k < size-1; k ++)
      {
          if (A_matrix[k*size+k] != 0.0)
          {
             for (j = k+1; j < size; j ++)
             {
                 if (A_matrix[j*size+k] != 0.0)
                 {
                    factor = A_matrix[j*size+k] / A_matrix[k*size+k];
                    for (m = k+1; m < size; m ++)
                    {
                        A_matrix[j*size+m] -= factor * A_matrix[k*size+m];
                    }
                    x_vector[j] -= factor * x_vector[k];   /* Elimination step for rhs */
                 }
             }
          }
       }
       /* Back Substitution */
       for (k = size-1; k > 0; -- k)
       {
           x_vector[k] /= A_matrix[k*size+k];
           for (j = 0; j < k; j ++)
           {
               if (A_matrix[j*size+k] != 0.0)
               {
                  x_vector[j] -= x_vector[k] * A_matrix[j*size+k];
               }
           }
       }
       x_vector[0] /= A_matrix[0];
       return(err_flag);
    }
}


JXF_Int
jxf_Pre_Pardiso(Pardiso_data *pdata,
               jxf_ParCSRMatrix *par_matrix,
               jxf_ParVector    *par_rhs,
               jxf_ParVector    *par_app)
{
   MPI_Comm         comm         = jxf_ParCSRMatrixComm(par_matrix);
   jxf_CSRMatrix    *A_diag       = jxf_ParCSRMatrixDiag(par_matrix);
   jxf_ParCSRCommPkg    *comm_pkg = jxf_ParCSRMatrixCommPkg(par_matrix);

   JXF_Int             n_global = jxf_ParCSRMatrixGlobalNumRows(par_matrix);
   JXF_Int             n        = jxf_CSRMatrixNumRows(A_diag);
   JXF_Int             first_index   = jxf_ParVectorFirstIndex(par_app);

   jxf_Vector      *u_local = jxf_ParVectorLocalVector(par_app);
   JXF_Real         *u_data  = jxf_VectorData(u_local);

   jxf_CSRMatrix    *A_CSR;
   JXF_Int             *A_CSR_i;
   JXF_Int             *A_CSR_j;
   JXF_Real          *A_CSR_data;

   jxf_Vector       *f_vector;
   JXF_Real          *f_vector_data;

   JXF_Int             i;
   JXF_Int             k;
   JXF_Int             column;
   JXF_Int             relax_error = 0;
   JXF_Int             num_procs, my_id;


   jxf_MPI_Comm_size(comm, &num_procs);
   jxf_MPI_Comm_rank(comm, &my_id);

  /*-------------------------------------------------------------------------
   * added by peghoty, if the comm_pkg of par_matrix is not created
   * previously, something will be wrong when the "relax" function
   * is called on multi-processors occasions.  2009/07/24
   *----------------------------------------------------------------------- */
   if (!comm_pkg)
   {
      jxf_MatvecCommPkgCreate(par_matrix);
      comm_pkg = jxf_ParCSRMatrixCommPkg(par_matrix);
   }
   // printf("jxf_pardiso, %s, %s, %d\n", __FILE__, __FUNCTION__, __LINE__);

   //------------------------------------------------------------------------------//
   //                   Direct solve: use PARDISO                                  //
   //------------------------------------------------------------------------------//

   /*-----------------------------------------------------------------
   *  Generate CSR matrix from ParCSRMatrix par_matrix
   *-----------------------------------------------------------------*/
#ifdef JXF_NO_GLOBAL_PARTITION
         /* all processors are needed for these routines */
         A_CSR = jxf_ParCSRMatrixToCSRMatrixAll(par_matrix);
         f_vector = jxf_ParVectorToVectorAll(par_rhs);
	if (n)
	{
#else
	if (n)
	{
	   A_CSR = jxf_ParCSRMatrixToCSRMatrixAll(par_matrix);
	   f_vector = jxf_ParVectorToVectorAll(par_rhs);
#endif

      // PARDISO
      // printf("jxf_qsort1, %s, %s, %d\n", __FILE__, __FUNCTION__, __LINE__);
      for ( i = 0; i < A_CSR->num_rows; i++)
      {
         k = A_CSR->i[i];
         jxf_qsort1( &A_CSR->j[k], &A_CSR->data[k], 0, A_CSR->i[i+1] - A_CSR->i[i] - 1 );
      }
      // printf("jxf_pardiso, %s, %s, %d\n", __FILE__, __FUNCTION__, __LINE__);
      jxf_pardiso_float(A_CSR, f_vector, u_local);
      // jxf_pardiso_double(A_CSR, f_vector, u_local);

      // printf("jxf_pardiso, %s, %s, %d\n", __FILE__, __FUNCTION__, __LINE__);

 	   // free
      jxf_CSRMatrixDestroy(A_CSR);
      A_CSR = NULL;
      jxf_SeqVectorDestroy(f_vector);
      f_vector = NULL;

   }
#ifdef JXF_NO_GLOBAL_PARTITION
   else
   {
      jxf_CSRMatrixDestroy(A_CSR);
      A_CSR = NULL;
      jxf_SeqVectorDestroy(f_vector);
      f_vector = NULL;
   }
#endif

   return 0;
}

JXF_Int
jxf_pardiso_float(jxf_CSRMatrix *AA, jxf_Vector *bb, jxf_Vector *xx)
{
    int prtlvl = 0;
    int i;
    // Matrix
    MKL_INT n = AA->num_rows;
    MKL_INT m = AA->num_cols;
    MKL_INT nnz = AA->num_nonzeros;
    MKL_INT *ia = AA->i;
    MKL_INT *ja = AA->j;
    JXF_Real *a1 = AA->data;
    // vector
    JXF_Real *b1 = bb->data;
    JXF_Real *x1 = xx->data;

    float *a;
    // vector
    float *b;
    float *x;

    a = (float*)malloc(sizeof(float) * nnz);
    b = (float*)malloc(sizeof(float) * n);
    x = (float*)malloc(sizeof(float) * n);

    for(i=0;i<nnz;i++) a[i] = a1[i];
    for(i=0;i<n;i++){
        b[i] = b1[i];
    }

    MKL_INT mtype = 11;    /* Real unsymmetric matrix */
    MKL_INT nrhs = 1;      /* Number of right hand sides */
    MKL_INT idum;          /* Integer dummy */
    MKL_INT iparm[64];     /* Pardiso control parameters */
    MKL_INT maxfct, mnum, phase, error, msglvl;    /* Auxiliary variables */

    void *pt[64];          /* Internal solver memory pointer pt */
    float ddum;           /* Double dummy */
    clock_t start_time = clock();

#if DEBUG_MODE
    printf("### DEBUG: %s ...... [Start]\n", __FUNCTION__);
    printf("### DEBUG: nr=%d, nc=%d, nnz=%d\n", m, n, nnz);
#endif

    PARDISOINIT(pt, &mtype, iparm); /* Initialize */
    iparm[34] = 1;        /* Use 0-based indexing */
    iparm[27] = 1;        /* 0: Use double , 1: Use single*/

    // iparm[13] = 1;        /* Output: Number of perturbed pivots */
    // iparm[17] = 1;       /* Output: Number of nonzeros in the factor LU */
    // iparm[18] = 1;       /* Output: Mflops for LU factorization */

    maxfct = 1;           /* Maximum number of numerical factorizations */
    mnum = 1;             /* Which factorization to use */
    msglvl = 0;           /* Do not print statistical information in file */
    error = 0;            /* Initialize error flag */

    phase = 11; /* Reordering and symbolic factorization */
    PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
    if ( error != 0 ) {
        printf ("### ERROR: Symbolic factorization failed %d!\n", error);
        exit (1);
    }

    phase = 22; /* Numerical factorization */
    PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
    if ( error != 0 ) {
        printf ("\n### ERROR: Numerical factorization failed %d!\n", error);
        exit (2);
    }

    phase = 33; /* Back substitution and iterative refinement */
    PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs, iparm, &msglvl, b, x, &error);

    if ( error != 0 ) {
        printf ("\n### ERROR: Solution failed %d!\n", error);
        exit (3);
    }

    if ( prtlvl > PRINT_MIN ) {
        clock_t end_time = clock();
        double solve_time = (double)(end_time - start_time)/(double)(CLOCKS_PER_SEC);
        printf("PARDISO costs %f seconds.\n", solve_time);
    }

    phase = -1; /* Release internal memory */
    PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
             &n, &ddum, ia, ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error);

#if DEBUG_MODE
    printf("### DEBUG: %s ...... [Finish]\n", __FUNCTION__);
#endif
    for(i=0;i<n;i++){
        x1[i] = x[i];
    }

    return 1;
}

JXF_Int
jxf_pardiso_double(jxf_CSRMatrix *AA, jxf_Vector *bb, jxf_Vector *xx)
{
    int prtlvl = 0;
    int i;
    // Matrix
    MKL_INT n = AA->num_rows;
    MKL_INT m = AA->num_cols;
    MKL_INT nnz = AA->num_nonzeros;
    MKL_INT *ia = AA->i;
    MKL_INT *ja = AA->j;
    JXF_Real *a1 = AA->data;
    // vector
    JXF_Real *b1 = bb->data;
    JXF_Real *x1 = xx->data;

    double *a;
    // vector
    double *b;
    double *x;

    a = (double*)malloc(sizeof(double) * nnz);
    b = (double*)malloc(sizeof(double) * n);
    x = (double*)malloc(sizeof(double) * n);

    for(i=0;i<nnz;i++) a[i] = a1[i];
    for(i=0;i<n;i++){
        b[i] = b1[i];
    }

    MKL_INT mtype = 11;    /* Real unsymmetric matrix */
    MKL_INT nrhs = 1;      /* Number of right hand sides */
    MKL_INT idum;          /* Integer dummy */
    MKL_INT iparm[64];     /* Pardiso control parameters */
    MKL_INT maxfct, mnum, phase, error, msglvl;    /* Auxiliary variables */

    void *pt[64];          /* Internal solver memory pointer pt */
    double ddum;           /* Double dummy */
    clock_t start_time = clock();

#if DEBUG_MODE
    printf("### DEBUG: %s ...... [Start]\n", __FUNCTION__);
    printf("### DEBUG: nr=%d, nc=%d, nnz=%d\n", m, n, nnz);
#endif

    PARDISOINIT(pt, &mtype, iparm); /* Initialize */
    iparm[34] = 1;        /* Use 0-based indexing */
    iparm[27] = 0;        /* 0: Use double , 1: Use single*/

    // iparm[13] = 1;        /* Output: Number of perturbed pivots */
    // iparm[17] = 1;       /* Output: Number of nonzeros in the factor LU */
    // iparm[18] = 1;       /* Output: Mflops for LU factorization */

    maxfct = 1;           /* Maximum number of numerical factorizations */
    mnum = 1;             /* Which factorization to use */
    msglvl = 0;           /* Do not print statistical information in file */
    error = 0;            /* Initialize error flag */

    phase = 11; /* Reordering and symbolic factorization */
    PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
    if ( error != 0 ) {
        printf ("### ERROR: Symbolic factorization failed %d!\n", error);
        exit (1);
    }

    phase = 22; /* Numerical factorization */
    PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
    if ( error != 0 ) {
        printf ("\n### ERROR: Numerical factorization failed %d!\n", error);
        exit (2);
    }

    phase = 33; /* Back substitution and iterative refinement */
    PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs, iparm, &msglvl, b, x, &error);

    if ( error != 0 ) {
        printf ("\n### ERROR: Solution failed %d!\n", error);
        exit (3);
    }

    if ( prtlvl > PRINT_MIN ) {
        clock_t end_time = clock();
        double solve_time = (double)(end_time - start_time)/(double)(CLOCKS_PER_SEC);
        printf("PARDISO costs %f seconds.\n", solve_time);
    }

    phase = -1; /* Release internal memory */
    PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
             &n, &ddum, ia, ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error);

#if DEBUG_MODE
    printf("### DEBUG: %s ...... [Finish]\n", __FUNCTION__);
#endif
    for(i=0;i<n;i++){
        x1[i] = x[i];
    }

    return 1;
}

#endif