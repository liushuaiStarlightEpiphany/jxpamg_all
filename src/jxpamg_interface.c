
//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2021        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 * jxpamg_interface.c -- this is an interface function that calls jxpamg's
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
 *  solver_id = 27: BILU-GMRES
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

#include "jx_pamg.h"
#include "jx_ilu.h"
#include "jx_krylov.h"
#include "jx_diagscale.h"
#include "jx_euclid.h"
#include "jx_combined.h"
#include "jx_parbilu.h"
#include "jx_parbsr_mv.h"
#include "jx_cpr.h"
// #define JX_USING_BIG_DOUBLE 1
#define JX_LOW_REAL float

#define PRINT_MIN  0
#define DEBUG_MODE 0

// #include "jx_pamg.h"
// #include "jx_apctl.h"
// #include "jx_euclid.h"
// #include "jx_multils.h"
// #include "jx_smp_forloop.h"
// #include "jx_combined.h"
// #include "jx_ilu.h"
// #include "jx_mv.h"
// #include "jx_util.h"
// #include "jx_diagscale.h"
// #include "jx_krylov.h"

#if 0
JX_Int
jx_LU( jx_ParCSRMatrix *par_matrix,
               jx_ParVector    *par_rhs,
               jx_ParVector    *par_app);
               JX_Int
jx_gselim_lu( JX_LOW_REAL *A_matrix, JX_LOW_REAL *x_vector, JX_Int size );
JX_Int jx_Pre_Pardiso(Pardiso_data *pdata,jx_ParCSRMatrix *par_matrix,
               jx_ParVector    *par_rhs,
               jx_ParVector    *par_app);
JX_Int
jx_pardiso_double(jx_CSRMatrix *AA, jx_Vector *bb, jx_Vector *xx);

JX_Int
jx_pardiso_float(jx_CSRMatrix *AA, jx_Vector *bb, jx_Vector *xx);
#endif
JX_Int jx_CSRMatrixFilter( jx_CSRMatrix *matrix, JX_Real eps, jx_CSRMatrix *matrixOut);
JX_Int JXPAMG_Solver(JX_Int                 argc,
                     char                *argv[],
                     jx_CSRMatrix        *As,
                     MPI_Comm            comm,
                     jx_ParCSRMatrix     *par_matrix,
                     jx_ParVector        *par_rhs,
                     jx_ParVector        *par_sol,
                     JX_Real              tol,
                     JX_Int                 num_f,
                     JX_Int                 solver_id,
                     JX_Int                 *iter);

// 先几何节点后物理量(3个量)，转化为先物理量后几何节点
JX_Int TranMatrixOrder( JX_Int n, JX_Int nnz, JX_Int *ia, JX_Int *ja, JX_Real *aa , JX_Real *b);
JX_Int TranMatrixOrder_new( JX_Int n, JX_Int nnz, JX_Int *ia, JX_Int *ja, JX_Real *aa , JX_Real *b);





JX_Int JXPAMG_Solver_Interface(JX_Int argc, char *argv[],MPI_Comm comm,JX_Int row, JX_Int *ia, JX_Int  *ja, JX_Real *a,
                               JX_Real *ser_x, JX_Real *ser_b, JX_Real tol, JX_Int solver_id, JX_Int num_functions, JX_Int *iter)
{
   JX_Int i,j;
   int myid,num_procs;

   JX_Int n = row;
   JX_Int nnz;
   JX_Real startwtime, endwtime, t1, t2;

   jx_CSRMatrix    *A_Ser;
   jx_Vector       *b_Ser;
   jx_Vector       *x_Ser;

   jx_ParCSRMatrix  *A_Par;
   jx_ParVector     *b_Par;
   jx_ParVector     *x_Par;

   JX_Int       *row_part=NULL;
   JX_Int       *col_part=NULL;
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
   //    t1 = jx_MPI_Wtime();
   //    TranMatrixOrder( n, nnz, ia, ja, a, ser_b);
   //    t2 = jx_MPI_Wtime();
   //    jx_printf("\n >>> 先几何节点后物理量(3个量)转化为先物理量后几何节点的时间:  %.4f(s)\n\n", t2 - t1);
   // }
   if (myid == 0) t1 = jx_MPI_Wtime();
   // 创建串行矩阵，并划分矩阵
   if (myid ==0 ){
      // TranMatrixOrder_new(n,nnz,ia,ja,a,ser_b);
      A_Ser = jx_CSRMatrixCreate(n, n, nnz);
      jx_CSRMatrixI(A_Ser) = ia;
      jx_CSRMatrixJ(A_Ser) = ja;
      jx_CSRMatrixData(A_Ser) = a;
      jx_CSRMatrixInitialize(A_Ser);
      jx_CSRMatrixReorder( (jx_CSRMatrix *)A_Ser);

      b_Ser = jx_SeqVectorCreate(n);
      jx_VectorData(b_Ser)= ser_b;
      jx_SeqVectorInitialize(b_Ser);

      x_Ser = jx_SeqVectorCreate(n);
      jx_VectorData(x_Ser)=ser_x;
      jx_SeqVectorInitialize(x_Ser);

      if (num_functions > 1)
      {
         JX_Int num_nodes,size,rest;
         num_nodes = n/num_functions;

         if (n != num_functions*num_nodes)
         {
            row_part = NULL;
            col_part = NULL;
         }
         else
         {
            row_part = jx_CTAlloc(JX_Int, num_procs+1);
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
   A_Par = jx_CSRMatrixToParCSRMatrix(comm, A_Ser, row_part, col_part);

   int *partitioning;
   jx_ParCSRMatrixGetRowPartitioning(A_Par, &partitioning);
   b_Par = jx_VectorToParVector(comm, b_Ser, partitioning);
   x_Par = jx_ParVectorCreate(comm, jx_ParVectorGlobalSize(b_Par), partitioning);
   jx_ParVectorSetPartitioningOwner(x_Par, 0);
   jx_ParVectorInitialize(x_Par);
   jx_ParVectorSetConstantValues(x_Par, 0.0);
   if (myid == 0){
      t2 = jx_MPI_Wtime();
      printf("\n >>> 串行矩阵转并行矩阵的时间 :  %.4f(s)\n\n", t2 - t1);
   }

   /*--------------------------------------------------------------------------
   * Solving the system using boomer AMG as follows
   *--------------------------------------------------------------------------*/
   if (myid == 0) startwtime = jx_MPI_Wtime();
   // mumps_Interface_float(comm,row, nnz, ia,ja, a, ser_x, ser_b);
   JXPAMG_Solver(argc, argv, A_Ser, comm, A_Par, b_Par, x_Par, tol, solver_id, num_functions, iter);


   if (myid == 0)
   {
      endwtime = jx_MPI_Wtime();
      printf(" >>>Time of JXPAMG_Solver :  %.4f(s)\n", endwtime-startwtime);
   }

   if (myid == 0) t1 = jx_MPI_Wtime();
   //printf("%s  %s : %d  \n",__FILE__,  __FUNCTION__,__LINE__ );
   x_Ser = jx_ParVectorToVectorAll(x_Par);

   /// output
   if (myid == 0){
      for (i = 0; i < n; i++)  ser_x [i] = x_Ser->data[i];
   }
   if (myid == 0){
      t2 = jx_MPI_Wtime();
      printf("\n >>> 并行向量转串行向量的时间 :  %.4f(s)\n", t2 - t1);
   }


   // if (myid == 0){
   jx_ParCSRMatrixDestroy(A_Par);
   A_Par = NULL;
   jx_ParVectorDestroy(b_Par);
   b_Par = NULL;
   jx_ParVectorDestroy(x_Par);
   x_Par = NULL;
   jx_SeqVectorDestroy(x_Ser);
   x_Ser =NULL;
   // }
   ///
   return (0);
}


JX_Int JXPAMG_Solver_Interface_omp(JX_Int argc, char *argv[], JX_Int row, JX_Int *ia, JX_Int  *ja, JX_Real *a,
                               JX_Real *ser_x, JX_Real *ser_b, JX_Real tol, JX_Int solver_id, JX_Int num_functions, JX_Int *iter)
{
   JX_Int i,j;
   int myid,num_procs;

   JX_Int n = row;
   JX_Int nnz;
   JX_Real startwtime, endwtime, t1, t2;

   jx_CSRMatrix    *A_Ser;
   jx_Vector       *b_Ser;
   jx_Vector       *x_Ser;

   jx_ParCSRMatrix  *A_Par;
   jx_ParVector     *b_Par;
   jx_ParVector     *x_Par;

   JX_Int       *row_part=NULL;
   JX_Int       *col_part=NULL;
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
   //    t1 = jx_MPI_Wtime();
   //    TranMatrixOrder( n, nnz, ia, ja, a, ser_b);
   //    t2 = jx_MPI_Wtime();
   //    jx_printf("\n >>> 先几何节点后物理量(3个量)转化为先物理量后几何节点的时间:  %.4f(s)\n\n", t2 - t1);
   // }
   if (myid == 0) t1 = jx_MPI_Wtime();
   // 创建串行矩阵，并划分矩阵
   if (myid ==0 ){
      A_Ser = jx_CSRMatrixCreate(n, n, nnz);
      jx_CSRMatrixI(A_Ser) = ia;
      jx_CSRMatrixJ(A_Ser) = ja;
      jx_CSRMatrixData(A_Ser) = a;
      jx_CSRMatrixInitialize(A_Ser);
      jx_CSRMatrixReorder( (jx_CSRMatrix *)A_Ser);

      b_Ser = jx_SeqVectorCreate(n);
      jx_VectorData(b_Ser)= ser_b;
      jx_SeqVectorInitialize(b_Ser);

      x_Ser = jx_SeqVectorCreate(n);
      jx_VectorData(x_Ser)=ser_x;
      jx_SeqVectorInitialize(x_Ser);

      if (num_functions > 1)
      {
         JX_Int num_nodes,size,rest;
         num_nodes = n/num_functions;

         if (n != num_functions*num_nodes)
         {
            row_part = NULL;
            col_part = NULL;
         }
         else
         {
            row_part = jx_CTAlloc(JX_Int, num_procs+1);
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

   A_Par = jx_CSRMatrixToParCSRMatrix_sp(comm, A_Ser);
   b_Par = jx_VectorToParVector_sp(comm, b_Ser);
   x_Par = jx_VectorToParVector_sp(comm, x_Ser);


   /*--------------------------------------------------------------------------
   * Solving the system using boomer AMG as follows
   *--------------------------------------------------------------------------*/
   if (myid == 0) startwtime = jx_MPI_Wtime();
   JXPAMG_Solver(argc, argv, A_Ser, comm, A_Par, b_Par, x_Par, tol, solver_id, num_functions, iter);

   if (myid == 0)
   {
      endwtime = jx_MPI_Wtime();
      jx_printf(" >>>Time of JXPAMG_Solver :  %.4f(s)\n", endwtime-startwtime);
   }

   ///
   return (0);
}




/* BILU-GMRES preconditioner callbacks */
static JX_Int bilu_gmres_setup(JX_Solver solver, JX_Matrix A, JX_Vector b, JX_Vector x)
{
    (void)A; (void)b; (void)x;
    return 0;
}

static JX_Int bilu_gmres_solve(JX_Solver solver, JX_Matrix A, JX_Vector b, JX_Vector x)
{
    (void)A;
    jx_ParBILUData *ilu_data = (jx_ParBILUData *)solver;
    return jx_BILUSolve((void*)ilu_data, ilu_data->matA,
                        (jx_ParVector*)b, (jx_ParVector*)x);
}


JX_Int JXPAMG_Solver(JX_Int              argc,
                  char                *argv[],
                  jx_CSRMatrix        *As,
                  MPI_Comm            comm,
                  jx_ParCSRMatrix     *par_matrix,
                  jx_ParVector        *par_rhs,
                  jx_ParVector        *par_sol,
                  JX_Real              tol,
                  JX_Int               solver_id,
                  JX_Int               num_functions,
                  JX_Int               *iter)
{
   // MPI_Comm  comm = MPI_COMM_WORLD;
   JX_Int       myid, nprocs;
   JX_Int       arg_index   = 0;
   JX_Int       print_usage = 0;

#if JX_USING_OPENMP || defined (JX_USING_PGCC_SMP)
   JX_Int       nthreads;
#endif

   JX_Real starttime, endtime;
   JX_Real starttimeT, endtimeT;

   char *MatFile = NULL;
   char *RhsFile = NULL;
   char *GusFile = NULL;
   char  AppFile[120];

   // jx_ParCSRMatrix *par_matrix = NULL;
   // jx_ParVector    *par_rhs    = NULL;
   // jx_ParVector    *par_sol    = NULL;
   jx_CSRMatrix    *ser_matrix = NULL, *As_hat = NULL;
   jx_ParCSRMatrix   *Ap_hat = NULL;

   JX_Int *partitioning = NULL;

   JX_Int **grid_relax_points = NULL;
   JX_Int *num_grid_sweeps = NULL;

   /* DiagScale Precond */
   JX_Solver ds_solver;

   /* ILU Precond */
   JX_Solver ilu_solver;
   JX_Real drop_tol;

   /* Euclid solver */
   JX_Solver euclid_solver;
   JX_Int euclid_level;
   JX_Int euclid_bj;

   /* Combined */
   JX_Solver combined_solver;
   JX_Int theta_psi;
   JX_Int theta_rho;
   JX_Int theta_phi;
   JX_Real theta_dis;

   /* JXPAMG solver */
   JX_Solver   amg_solver;
   JX_Int       max_levels;
   JX_Int       cycle_type;
   JX_Int       relax_type;
   JX_Int       measure_type;
   JX_Int       rap2;
   // JX_Int       num_functions;
   JX_Int       ns_down;
   JX_Int       ns_up;
   JX_Int       ns_coarse;
   JX_Int       restri_type;
   JX_Int       keepTranspose;
   JX_Int       coarsen_type;
   JX_Int       coarse_solver;
   JX_Int       interp_type;
   JX_Int       P_max_elmts;
   JX_Int       agg_num_levels;
   JX_Int       ai_measure_type;
   JX_Int       ai_relax_type;
   JX_Real    strong_threshold;
   JX_Real    max_row_sum;

   JX_Real    relax_wt;
   JX_Real    outer_wt;

   JX_Real     S_commpkg_switch;
   JX_Real     AIR_strong_th;

   JX_Int       coarse_threshold;
   JX_Real     coarse_ratio;
   JX_Int       coarsestsolverid;
   JX_Int       conv_criteria;
   JX_Int       amg_print_level;
   JX_Int       CF;

   /* iterative method */
   JX_Solver   solver;
   JX_Real     resdown_0_threshold;
   JX_Real     convfac_threshold_2;
   JX_Int       max_iter;
   JX_Int       k_dim;
   JX_Int       is_check_restarted;  /* peghoty, 2011/11/08 */
   JX_Int       twonorm;
   JX_Int       problem_id;
   JX_Int       file_base;
   JX_Int       print_level;
   JX_Int       keepsol;
   JX_Int       TTest;
   JX_Int       cgs;
   JX_Int       unroll;

   /* other variables */
   JX_Int       lu_length;
   JX_Int       glosize, i;
   JX_Int       last_precond_type;
   JX_Int       initguess = 0;
   JX_Int       num_iterations;
   JX_Real      final_res_norm;
   JX_Real      norm;
   JX_Real      eps = 1e-8;
   JX_Real      trunc_factor;
   //--------------------------
   // 获取进程数和进程编号
   //--------------------------
   jx_MPI_Comm_rank(comm, &myid);
   jx_MPI_Comm_size(comm, &nprocs);

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
   print_level         = 3;      /* 0: 关闭；1：Setup参数；2：Solve参数；3：Setup+Solve参数 */
   keepsol             = 0;      /* 是否保存解向量 */
   TTest               = 1;      /* 是否测试时间 */
   cgs                 = 1;      /* COGMRES: if 2 performs reorthogonalization */
   unroll              = 0;      /* COGMRES: Set number of unrolling in mass funcyions, can be 4 or 8. Default: no unrolling */
   // solver_id           = 22;
   problem_id          = 1;
   file_base           = 1;
#if JX_USING_OPENMP || defined (JX_USING_PGCC_SMP)
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
#if JX_USING_OPENMP || defined (JX_USING_PGCC_SMP)
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
      jx_printf("\n");
      jx_printf("  Usage: %s [<options>]\n", argv[0]);
      jx_printf("\n");
      jx_printf("    -eps <val>   : threshold for filtering out small elements, zhao li\n");
      jx_printf("    -sid <val>   : solver id\n");
      jx_printf("    -pid <val>   : problem id\n");
      jx_printf("    -nf  <val>   : number of functions\n");
      jx_printf("    -nts <val>   : threads number\n");
      jx_printf("    -rap2        : 2nd implementation of RAP\n");
      jx_printf("    -kt          : keep transpose\n");
      jx_printf("    -Pmx <val>   : maximal number of elements per row for interpolation\n");
      jx_printf(" -agg_nl <val>   : num_levels for aggressive coarsening\n");
      jx_printf("   -tnrm <val>   : two_norm in PCG\n");
      jx_printf("   -kdim <val>   : krylov dimension\n");
      jx_printf("   -dtol <val>   : Drop-tolerance for ILU(0) factorization\n");
      jx_printf("-euc_lvl <val>   : level of fill-in for Euclid\n");
      jx_printf(" -euc_bj <val>   : PILU or Block Jacobi ILU\n");
      jx_printf("  -rlx <val>     : relaxation type\n");
      jx_printf("  -ai_rlx <val>  : AI relaxation type\n");
      jx_printf("  -ai_mt  <val>  : AI measure type\n");
      jx_printf("     -ct <val>   : coarsening type\n");
      jx_printf("    -ipt <val>   : interpolation type\n");
      jx_printf("   -mxct <val>   : max. size on coarsest grid\n");
      jx_printf("    -str <val>   : AMG strength threshold\n");
      jx_printf("   -mxrs <val>   : maximum row sum threshold for dependency weakening\n");
      jx_printf("-amg_ptlv <val>  : print_level of AMG when AMG as preconditioner\n");
      jx_printf("-theta_ms <iiiv> : threshold for multi-scale judgement\n");
      jx_printf("   -ptlv <val>   : print_level\n");
      jx_printf("   -help         : using help message\n\n");
      exit(1);
   }


   if (myid == 0)
   {
#if JX_USING_BIG_INT
      jx_printf(" Using BIG_INT,");
#endif
#if JX_USING_BIG_DOUBLE
      jx_printf(" BIG_DOUBLE,");
#endif
#if JX_USING_OPENMP || defined (JX_USING_PGCC_SMP)
      jx_printf(" With OpenMP using %d threads,", nthreads);
#endif
      jx_printf(" JXPAMG MPI using %d processors +++++++++++++++++++++\n\n", nprocs);
   }

   //----------------------------------------------------------------
   // 设定线程数
   //----------------------------------------------------------------
#if JX_USING_OPENMP || defined (JX_USING_PGCC_SMP)
   omp_set_num_threads(nthreads);
#endif

   if (restri_type) /* Set Restriction to be AIR */
   {
      interp_type = 100; /* 1-pt Interp */
      relax_type = 3;
      ns_down = 3;
      ns_up = 3;
      grid_relax_points = jx_CTAlloc(JX_Int *, 4);
      grid_relax_points[0] = NULL;
      grid_relax_points[1] = jx_CTAlloc(JX_Int, ns_down);
      grid_relax_points[2] = jx_CTAlloc(JX_Int, ns_up);
      grid_relax_points[3] = jx_CTAlloc(JX_Int, ns_coarse);
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
      grid_relax_points = jx_CTAlloc(JX_Int *, 4);
      grid_relax_points[0] = jx_CTAlloc(JX_Int, ns_down);   // 最细网格层
      grid_relax_points[1] = jx_CTAlloc(JX_Int, ns_down);   // 前磨光
      grid_relax_points[2] = jx_CTAlloc(JX_Int, ns_up);     // 后磨光
      grid_relax_points[3] = jx_CTAlloc(JX_Int, ns_coarse); // 粗空间

      num_grid_sweeps   = jx_CTAlloc(JX_Int,4);
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

      grid_relax_points = jx_CTAlloc(JX_Int *, 4);
      grid_relax_points[0] = jx_CTAlloc(JX_Int, ns_down);   // 最细网格层
      grid_relax_points[1] = jx_CTAlloc(JX_Int, ns_down);   // 前磨光
      grid_relax_points[2] = jx_CTAlloc(JX_Int, ns_up);     // 后磨光
      grid_relax_points[3] = jx_CTAlloc(JX_Int, ns_coarse); // 粗空间

      num_grid_sweeps   = jx_CTAlloc(JX_Int,4);
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

      grid_relax_points = jx_CTAlloc(JX_Int *, 4);
      grid_relax_points[0] = jx_CTAlloc(JX_Int, ns_down);   // 最细网格层
      grid_relax_points[1] = jx_CTAlloc(JX_Int, ns_down);   // 前磨光
      grid_relax_points[2] = jx_CTAlloc(JX_Int, ns_up);     // 后磨光
      grid_relax_points[3] = jx_CTAlloc(JX_Int, ns_coarse); // 粗空间

      num_grid_sweeps   = jx_CTAlloc(JX_Int,4);
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
   if (TTest) starttime = jx_MPI_Wtime();

   par_matrix = jx_BuildMatParFromOneFile(MatFile, 1, file_base);
   par_rhs    = jx_BuildRhsParFromOneFile(RhsFile, par_matrix);

   if (!GusFile)
   {
      glosize      = jx_ParVectorGlobalSize(par_rhs);
      partitioning = jx_ParVectorPartitioning(par_rhs);
      par_sol      = jx_ParVectorCreate(comm, glosize, partitioning);
      jx_ParVectorSetPartitioningOwner(par_sol, 0);
      jx_ParVectorInitialize(par_sol);
      if (initguess == 0)
      {
         jx_ParVectorSetConstantValues(par_sol, 0.0);
      }
      else
      {
         jx_ParVectorSetRandomValues(par_sol, 22775);
         norm = jx_ParVectorInnerProd(par_sol, par_sol);
         norm = 1./sqrt(norm);
         jx_ParVectorScale(norm, par_sol);
      }
   }
   else
   {
      par_sol = jx_BuildRhsParFromOneFile(GusFile, par_matrix);
   }

   endtime = jx_MPI_Wtime();
   jx_GetWallTime(comm, "BuildParLinearSystem", starttime, endtime, 0, 2);
#endif

   //-----------------------------------------------------------
   //  求解线性代数系统
   //-----------------------------------------------------------
   starttimeT = jx_MPI_Wtime();

   switch (solver_id)
   {
      #if 0
      case 11011:  /* 直接法 */
	   {
         jx_DMUMPS_data *ddata;
         ddata = jx_CTAlloc(jx_DMUMPS_data, 1);


         jx_Mumps_double_setup(ddata,par_matrix);
         jx_Mumps_double_solve(ddata,par_matrix,par_rhs,par_sol);

         ddata->id.job = -2;
         dmumps_c(&ddata->id); /* Terminate instance */
      }
      break;

      case 201:  /* 直接法 */
	   {
         // jx_Mumps_float(NULL,par_matrix, par_rhs, par_sol);
         JX_Real *ADATA = jx_CSRMatrixData(As);
         JX_Int *AIA = jx_CSRMatrixI(As);
         JX_Int *AJA = jx_CSRMatrixJ(As);
         JX_Int num_rows = jx_CSRMatrixNumRows(As);
         JX_Int num_nonzeros = jx_CSRMatrixNumNonzeros(As);
         jx_Vector *b = jx_ParVectorLocalVector(par_rhs);
         JX_Real *bDATA = jx_VectorData(b);
         jx_Vector *x = jx_ParVectorLocalVector(par_sol);
         JX_Real *xDATA = jx_VectorData(x);

         mumps_Interface_float(comm, num_rows,num_nonzeros, AIA, AJA,ADATA,xDATA, bDATA);

         // jx_SeqVectorDestroy(b);
         // jx_SeqVectorDestroy(x);

      }
      break;

      case 101:  /* 直接法 */
	   {
         //  jx_DMUMPS_data *ddata;
         //  ddata = jx_CTAlloc(jx_DMUMPS_data, 1);


         //  jx_Mumps_double_setup(ddata,par_matrix);
         //  jx_Mumps_double_solve(ddata,par_matrix,par_rhs,par_sol);
          jx_SMUMPS_data *ddata;
          ddata = jx_CTAlloc(jx_SMUMPS_data, 1);


          jx_Mumps_float_setup(ddata,par_matrix);
          jx_Mumps_float_solve(ddata,par_matrix,par_rhs,par_sol);
         //  jx_Mumps_float_solve(ddata,par_matrix,par_rhs,par_sol);

         ddata->id.job = -2;
         smumps_c(&ddata->id); /* Terminate instance */


		//   jx_pardiso(par_matrix, par_rhs, par_sol);
		//   jx_PAMGRelax10(par_matrix,par_rhs,NULL,0,0.0,0.0,par_sol,NULL);
      //   Pardiso_data *pdata;
      //   jx_Pre_Pardiso(pdata,par_matrix, par_rhs, par_sol);
      //   jx_Mumps(NULL,par_matrix, par_rhs, par_sol);
      //   jx_Mumps_double(NULL,par_matrix, par_rhs, par_sol);
         //   jx_Mumps_float(NULL,par_matrix, par_rhs, par_sol);
      //  jx_LU(par_matrix,par_rhs,par_sol);
      //   jx_printf("1046\n");
	   }
      break;
      case 1011:  /* MUMPS-F-GMRES */
	   {
         if (myid == 0) jx_printf("\n >>> Solver: MUMPS-GMRES(%d) \n\n", k_dim);

         if (TTest) starttime = jx_MPI_Wtime();

          jx_SMUMPS_data *ddata;
          ddata = jx_CTAlloc(jx_SMUMPS_data, 1);

         JX_ParCSRGMRESCreate(comm, &solver);
         JX_GMRESSetKDim(solver, k_dim);
         JX_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JX_GMRESSetMaxIter(solver, max_iter);
         JX_GMRESSetTol(solver, tol);
         JX_GMRESSetLogging(solver, 1);
         JX_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */

         JX_GMRESSetPrecond(solver, (JX_PtrToSolverFcn)jx_Mumps_float_solve,
                                    (JX_PtrToSolverFcn)jx_Mumps_float_setup, (JX_Solver)ddata);
         jx_Mumps_float_setup(ddata,par_matrix);

         JX_GMRESSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG-GMRES Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jx_MPI_Wtime();

         JX_GMRESSolve(solver, (JX_Matrix)par_matrix, // preOperater
                               (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG-GMRES Solve", starttime, endtime, 0, 2);
         }

         JX_GMRESGetNumIterations(solver, &num_iterations);
         JX_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         ddata->id.job = -2;
         smumps_c(&ddata->id); /* Terminate instance */
         JX_ParCSRGMRESDestroy(solver);
	   }
      break;

      case 102:  /* PARDISO-GMRES(%d) */
	   {

        if (myid == 0) jx_printf("\n >>> Solver: PARDISO-GMRES(%d) \n\n", k_dim);

         if (TTest) starttime = jx_MPI_Wtime();

         JX_ParCSRGMRESCreate(comm, &solver);
         JX_GMRESSetKDim(solver, k_dim);
         JX_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JX_GMRESSetMaxIter(solver, max_iter);
         JX_GMRESSetTol(solver, tol);
         JX_GMRESSetLogging(solver, 1);
         JX_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */

         JX_GMRESSetPrecond(solver, (JX_PtrToSolverFcn)jx_Pre_Pardiso,
                                       NULL, NULL);


         JX_GMRESSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG-GMRES Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jx_MPI_Wtime();

         JX_GMRESSolve(solver, (JX_Matrix)par_matrix, // preOperater
                               (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG-GMRES Solve", starttime, endtime, 0, 2);
         }

         JX_GMRESGetNumIterations(solver, &num_iterations);
         JX_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JX_ParCSRGMRESDestroy(solver);
	   }
      break;

      case 103:  /* PARDISO-FlexGMRES(%d)  */
	   {
       if (myid == 0) jx_printf("\n >>> Solver: PARDISO-FlexGMRES(%d) \n\n", k_dim);

         if (TTest) starttime = jx_MPI_Wtime();

         JX_ParCSRFlexGMRESCreate(comm, &solver);
         JX_FlexGMRESSetKDim(solver, k_dim);
         JX_FlexGMRESSetIsCheckRestarted(solver, is_check_restarted);
         JX_FlexGMRESSetMaxIter(solver, max_iter);
         JX_FlexGMRESSetTol(solver, tol);
         JX_FlexGMRESSetLogging(solver, 1);
         JX_FlexGMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */

         JX_FlexGMRESSetPrecond(solver, (JX_PtrToSolverFcn)jx_Pre_Pardiso,
                                       NULL, NULL);

         /* this is optional - could be a user defined one instead */
         JX_FlexGMRESSetModifyPC(solver, (JX_PtrToModifyPCFcn)jx_FlexGMRESModifyPCDefault);

         JX_FlexGMRESSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG-FlexGMRES Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jx_MPI_Wtime();

         JX_FlexGMRESSolve(solver, (JX_Matrix)par_matrix, // preOperater
                               (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG-FlexGMRES Solve", starttime, endtime, 0, 2);
         }

         JX_FlexGMRESGetNumIterations(solver, &num_iterations);
         JX_FlexGMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JX_PAMGDestroy(amg_solver);
         JX_ParCSRFlexGMRESDestroy(solver);
      }
      break;

      case 3021:  /* Mumps-BiCGSTAB */
      {
         if (myid == 0) jx_printf("\n >>> Solver: Mumps-BiCGSTAB(%d) \n\n");

         if (TTest) starttime = jx_MPI_Wtime();

         jx_SMUMPS_data *ddata;
         ddata = jx_CTAlloc(jx_SMUMPS_data, 1);

         JX_ParCSRBiCGSTABCreate(comm, &solver);
         JX_BiCGSTABSetMaxIter(solver, max_iter);
         JX_BiCGSTABSetTol(solver, tol);
         JX_BiCGSTABSetAbsoluteTol(solver, 0.0);
         JX_BiCGSTABSetConvCriteria(solver, 0);
         JX_BiCGSTABSetLogging(solver, 1);
         JX_BiCGSTABSetPrintLevel(solver, print_level);

         JX_BiCGSTABSetPrecond(solver, (JX_PtrToSolverFcn)jx_Mumps_float_solve,
                                    (JX_PtrToSolverFcn)jx_Mumps_float_setup, (JX_Solver)ddata);
         jx_Mumps_float_setup(ddata,par_matrix);

         JX_BiCGSTABSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG-BiCGSTAB Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jx_MPI_Wtime();

         JX_BiCGSTABSolve(solver, (JX_Matrix)par_matrix, // preOperater
                                  (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG-BiCGSTAB Solve", starttime, endtime, 0, 2);
         }

         JX_BiCGSTABGetNumIterations(solver, &num_iterations);
         JX_BiCGSTABGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         ddata->id.job = -2;
         smumps_c(&ddata->id); /* Terminate instance */
         JX_ParCSRBiCGSTABDestroy(solver);
      }
      break;

      case 3022:  /* PARDISO-FlexGMRES(%d)  */
	   {
       if (myid == 0) jx_printf("\n >>> Solver: PARDISO-FlexGMRES(%d) \n\n", k_dim);

         if (TTest) starttime = jx_MPI_Wtime();

         jx_SMUMPS_data *ddata;
         ddata = jx_CTAlloc(jx_SMUMPS_data, 1);

         JX_ParCSRFlexGMRESCreate(comm, &solver);
         JX_FlexGMRESSetKDim(solver, k_dim);
         JX_FlexGMRESSetIsCheckRestarted(solver, is_check_restarted);
         JX_FlexGMRESSetMaxIter(solver, max_iter);
         JX_FlexGMRESSetTol(solver, tol);
         JX_FlexGMRESSetLogging(solver, 1);
         JX_FlexGMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */

         JX_FlexGMRESSetPrecond(solver, (JX_PtrToSolverFcn)jx_Mumps_float_solve,
                                    (JX_PtrToSolverFcn)jx_Mumps_float_setup, (JX_Solver)ddata);
         jx_Mumps_float_setup(ddata,par_matrix);

         /* this is optional - could be a user defined one instead */
         JX_FlexGMRESSetModifyPC(solver, (JX_PtrToModifyPCFcn)jx_FlexGMRESModifyPCDefault);

         JX_FlexGMRESSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG-FlexGMRES Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jx_MPI_Wtime();

         JX_FlexGMRESSolve(solver, (JX_Matrix)par_matrix, // preOperater
                               (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG-FlexGMRES Solve", starttime, endtime, 0, 2);
         }

         JX_FlexGMRESGetNumIterations(solver, &num_iterations);
         JX_FlexGMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         ddata->id.job = -2;
         smumps_c(&ddata->id); /* Terminate instance */
         JX_ParCSRFlexGMRESDestroy(solver);
      }
      break;


      case 302:  /* PARDISO-BiCGSTAB */
      {
         if (myid == 0) jx_printf("\n >>> Solver: PARDISO-BiCGSTAB(%d) \n\n");

         if (TTest) starttime = jx_MPI_Wtime();

         JX_ParCSRBiCGSTABCreate(comm, &solver);
         JX_BiCGSTABSetMaxIter(solver, max_iter);
         JX_BiCGSTABSetTol(solver, tol);
         JX_BiCGSTABSetAbsoluteTol(solver, 0.0);
         JX_BiCGSTABSetConvCriteria(solver, 0);
         JX_BiCGSTABSetLogging(solver, 1);
         JX_BiCGSTABSetPrintLevel(solver, print_level);

         JX_BiCGSTABSetPrecond(solver, (JX_PtrToSolverFcn)jx_Pre_Pardiso,
                                       NULL, NULL);

         JX_BiCGSTABSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG-BiCGSTAB Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jx_MPI_Wtime();

         JX_BiCGSTABSolve(solver, (JX_Matrix)par_matrix, // preOperater
                                  (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG-BiCGSTAB Solve", starttime, endtime, 0, 2);
         }

         JX_BiCGSTABGetNumIterations(solver, &num_iterations);
         JX_BiCGSTABGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JX_ParCSRBiCGSTABDestroy(solver);
      }
      break;
#endif
      case 0:  /* PAMG */
      {
         if (myid == 0) jx_printf("\n >>> Solver: PAMG \n\n");

         if (TTest) starttime = jx_MPI_Wtime();

         JX_PAMGCreate(&amg_solver);
         if (restri_type)
         {
            jx_assert(restri_type >= 0);
            JX_PAMGSetRestriction(amg_solver, restri_type);
            JX_PAMGSetGridRelaxPoints(amg_solver, grid_relax_points);
         }
         JX_PAMGSetMaxLevels(amg_solver, max_levels);
         JX_PAMGSetMaxIter(amg_solver, max_iter);
         JX_PAMGSetNumFunctions(amg_solver, num_functions);
         JX_PAMGSetCycleType(amg_solver, cycle_type);
         JX_PAMGSetMeasureType(amg_solver, measure_type);
         JX_PAMGSetRAP2(amg_solver, rap2);
         JX_PAMGSetKeepTranspose(amg_solver, keepTranspose);
         JX_PAMGSetTol(amg_solver, tol);
         JX_PAMGSetConvCriteria(amg_solver, conv_criteria);
         JX_PAMGSetCoarsenType(amg_solver, coarsen_type);
         JX_PAMGSetInterpType(amg_solver, interp_type);
         JX_PAMGSetPMaxElmts(amg_solver, P_max_elmts);
         JX_PAMGSetAggNumLevels(amg_solver, agg_num_levels);
         JX_PAMGSetAIMeasureType(amg_solver, ai_measure_type);
         JX_PAMGSetAIRelaxType(amg_solver, ai_relax_type);
         JX_PAMGSetStrongThreshold(amg_solver, strong_threshold);
         JX_PAMGSetMaxRowSum(amg_solver, max_row_sum);
         JX_PAMGSetPrintLevel(amg_solver, print_level);
         JX_PAMGSetCoarsestSolverID(amg_solver, coarsestsolverid);
         JX_PAMGSetCoarseThreshold(amg_solver, coarse_threshold);
         JX_PAMGSetCoarseRatio(amg_solver, coarse_ratio);
         JX_PAMGSetRelaxWt(amg_solver, relax_wt);
         JX_PAMGSetOuterWt(amg_solver, outer_wt);
         if (ns_down > -1) JX_PAMGSetCycleNumSweeps(amg_solver, ns_down, 1);           /* sweep for "down" */
         if (ns_up > -1) JX_PAMGSetCycleNumSweeps(amg_solver, ns_up, 2);           /* sweep for "up" */
         JX_PAMGSetCycleNumSweeps(amg_solver, ns_coarse, 3);           /* sweep for "coarsest" */
         JX_PAMGSetCycleRelaxType(amg_solver, relax_type, 1);  /* relax_type for "down" */
         JX_PAMGSetCycleRelaxType(amg_solver, relax_type, 2);  /* relax_type for "up" */
         JX_PAMGSetCycleRelaxType(amg_solver, coarse_solver, 3);           /* relax_type for "coarsest" */

         //------------------------------------------------------------
         //    JX_PAMG Setup
         //------------------------------------------------------------
         if (max_levels != 1)
         {
            JX_PAMGSetup(amg_solver, (JX_ParCSRMatrix) par_matrix);
         }

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG Setup", starttime, endtime, 0, 2);
         }

         //------------------------------------------------------------
         //    JX_PAMG Solve
         //------------------------------------------------------------
         if (TTest) starttime = jx_MPI_Wtime();

         JX_PAMGSolve(amg_solver, (JX_ParCSRMatrix)par_matrix, (JX_ParVector)par_rhs, (JX_ParVector)par_sol);

         JX_PAMGGetNumIterations(amg_solver, &num_iterations);
         JX_PAMGGetFinalRelativeResidualNorm(amg_solver, &final_res_norm);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG Solve", starttime, endtime, 0, 2);
         }

         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JX_PAMGDestroy(amg_solver);
      }
      break;

      case 11:  /* CG */
      {
         if (myid == 0) jx_printf("\n >>> Solver: CG \n\n");

         JX_ParCSRPCGCreate(comm, &solver);

         JX_PCGSetMaxIter(solver, max_iter);
         JX_PCGSetTol(solver, tol);
         JX_PCGSetTwoNorm(solver, twonorm);  // 0: B 范数； 1：l2 范数
         JX_PCGSetLogging(solver, 1);
         JX_PCGSetPrintLevel(solver, print_level);

         JX_PCGSetup(solver, (JX_Matrix) par_matrix, (JX_Vector) par_rhs, (JX_Vector) par_sol);

         JX_PCGSolve(solver, (JX_Matrix)par_matrix, // preOperater
                             (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         JX_PCGGetNumIterations(solver, &num_iterations);
         JX_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);

         JX_ParCSRPCGDestroy(solver);
      }
      break;

      case 12:  /* PAMG-CG */
      {
         if (myid == 0) jx_printf("\n >>> Solver: PAMG-CG \n\n");

         starttime = jx_MPI_Wtime();

         JX_PAMGCreate(&amg_solver);
         JX_PAMGSetMaxLevels(amg_solver, max_levels);
         JX_PAMGSetMaxIter(amg_solver, 1);
         JX_PAMGSetCycleType(amg_solver, cycle_type);
         JX_PAMGSetMeasureType(amg_solver, measure_type);
         JX_PAMGSetRAP2(amg_solver, rap2);
         JX_PAMGSetKeepTranspose(amg_solver, keepTranspose);
         JX_PAMGSetCoarsenType(amg_solver, coarsen_type);
         JX_PAMGSetCoarsestSolverID(amg_solver, coarsestsolverid);
         JX_PAMGSetInterpType(amg_solver, interp_type);
         JX_PAMGSetPMaxElmts(amg_solver, P_max_elmts);
         JX_PAMGSetTruncFactor(amg_solver, trunc_factor);//zhaoli 2021.06.28
         JX_PAMGSetAggNumLevels(amg_solver, agg_num_levels);
         JX_PAMGSetAIMeasureType(amg_solver, ai_measure_type);
         JX_PAMGSetAIRelaxType(amg_solver, ai_relax_type);
         JX_PAMGSetStrongThreshold(amg_solver, strong_threshold);
         JX_PAMGSetMaxRowSum(amg_solver, max_row_sum);
         JX_PAMGSetPrintLevel(amg_solver, amg_print_level);
         JX_PAMGSetCoarseThreshold(amg_solver, coarse_threshold);
         JX_PAMGSetRelaxWt(amg_solver, relax_wt);
         JX_PAMGSetOuterWt(amg_solver, outer_wt);

         if (CF != -1) {
            JX_PAMGSetNumGridSweeps(amg_solver, num_grid_sweeps);       // zhaoli
            JX_PAMGSetGridRelaxPoints(amg_solver, grid_relax_points);   // zhaoli
		   }
         // JX_Int smooth_num_sweeps = 2;
         // JX_PAMGSetSmoothNumSweeps(amg_solver, smooth_num_sweeps);
         if (ns_down > -1) JX_PAMGSetCycleNumSweeps(amg_solver, ns_down, 1);           /* sweep for "down" */
         if (ns_up > -1) JX_PAMGSetCycleNumSweeps(amg_solver, ns_up, 2);           /* sweep for "up" */
         JX_PAMGSetCycleNumSweeps(amg_solver, ns_coarse, 3);           /* sweep for "coarsest" */
         JX_PAMGSetCycleRelaxType(amg_solver, relax_type, 1);  /* relax_type for "down" */
         JX_PAMGSetCycleRelaxType(amg_solver, relax_type, 2);  /* relax_type for "up" */
         JX_PAMGSetCycleRelaxType(amg_solver, coarse_solver, 3);           /* relax_type for "coarsest" */

         JX_ParCSRPCGCreate(comm, &solver);

         JX_PCGSetMaxIter(solver, max_iter);
         JX_PCGSetTol(solver, tol);
         JX_PCGSetTwoNorm(solver, twonorm);  // 0: B 范数； 1：l2 范数
         JX_PCGSetLogging(solver, 1);
         JX_PCGSetPrintLevel(solver, print_level);

         JX_PCGSetPrecond(solver, (JX_PtrToSolverFcn)JX_PAMGPrecond, (JX_PtrToSolverFcn)JX_PAMGSetup, amg_solver);

         JX_PAMGSetup(amg_solver, (JX_ParCSRMatrix)par_matrix);

         JX_PCGSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         endtime = jx_MPI_Wtime();
         jx_GetWallTime(comm, "PAMG-CG Setup", starttime, endtime, 0, 2);

         starttime = jx_MPI_Wtime();

         JX_PCGSolve(solver, (JX_Matrix)par_matrix, // preOperater
                             (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         endtime = jx_MPI_Wtime();
         jx_GetWallTime(comm, "PAMG-CG Solve", starttime, endtime, 0, 2);


         // jx_ParCSRMatrixMatvec(-1, par_matrix, par_sol, 1, par_rhs);
         // JX_Real yL2 = jx_ParVectorInnerProd( par_rhs, par_rhs );
         // printf(" >>>   AMG-CG :  ||b-Ax||_2 = %e\n", sqrt(yL2));


         JX_PCGGetNumIterations(solver, &num_iterations);
         JX_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);

		 //jx_printf("vvvv %e\n", final_res_norm);

         JX_PAMGDestroy(amg_solver);
         JX_ParCSRPCGDestroy(solver);
      }
      break;

      // 改进 P \approx \hat{A}^{-1}, zhaoli,2021.06.24
      case 112:  /* PAMG-CG-zl */
      {
         if (myid == 0) jx_printf("\n >>> Solver: PAMG-CG-zl \n\n");

         starttime = jx_MPI_Wtime();

         // zhaoli
         if (myid == 0)
         {
            jx_printf("eps = %e, zhaoli 2021.06.24!\n", eps);
            As_hat = jx_CSRMatrixCreate(As->num_rows, As->num_cols, As->num_nonzeros);
            jx_CSRMatrixInitialize(As_hat);
            jx_CSRMatrixNumNonzeros(As_hat) = As->num_nonzeros;
            jx_CSRMatrixNumCols(As_hat) = As->num_cols;
            // jx_printf("1\n");
            jx_CSRMatrixFilter(As, eps, As_hat);
            // jx_printf("2 As_hat->num_cols = %d\n", As_hat->num_cols);
         }
         Ap_hat = jx_CSRMatrixToParCSRMatrix(comm, As_hat, par_matrix->row_starts, par_matrix->col_starts);
         jx_CSRMatrixDestroy(As_hat);

         JX_PAMGCreate(&amg_solver);
         JX_PAMGSetMaxLevels(amg_solver, max_levels);
         JX_PAMGSetMaxIter(amg_solver, 1);
         JX_PAMGSetCycleType(amg_solver, cycle_type);
         JX_PAMGSetMeasureType(amg_solver, measure_type);
         JX_PAMGSetRAP2(amg_solver, rap2);
         JX_PAMGSetKeepTranspose(amg_solver, keepTranspose);
         JX_PAMGSetCoarsenType(amg_solver, coarsen_type);
         JX_PAMGSetInterpType(amg_solver, interp_type);
         JX_PAMGSetPMaxElmts(amg_solver, P_max_elmts);
         JX_PAMGSetAggNumLevels(amg_solver, agg_num_levels);
         JX_PAMGSetAIMeasureType(amg_solver, ai_measure_type);
         JX_PAMGSetAIRelaxType(amg_solver, ai_relax_type);
         JX_PAMGSetStrongThreshold(amg_solver, strong_threshold);
         JX_PAMGSetMaxRowSum(amg_solver, max_row_sum);
         JX_PAMGSetPrintLevel(amg_solver, amg_print_level);
         JX_PAMGSetCoarseThreshold(amg_solver, coarse_threshold);
         JX_PAMGSetRelaxWt(amg_solver, relax_wt);
         JX_PAMGSetOuterWt(amg_solver, outer_wt);
         if (ns_down > -1) JX_PAMGSetCycleNumSweeps(amg_solver, ns_down, 1);           /* sweep for "down" */
         if (ns_up > -1) JX_PAMGSetCycleNumSweeps(amg_solver, ns_up, 2);           /* sweep for "up" */
         JX_PAMGSetCycleNumSweeps(amg_solver, ns_coarse, 3);           /* sweep for "coarsest" */
         JX_PAMGSetCycleRelaxType(amg_solver, relax_type, 1);  /* relax_type for "down" */
         JX_PAMGSetCycleRelaxType(amg_solver, relax_type, 2);  /* relax_type for "up" */
         JX_PAMGSetCycleRelaxType(amg_solver, coarse_solver, 3);           /* relax_type for "coarsest" */

         JX_ParCSRPCGCreate(comm, &solver);

         JX_PCGSetMaxIter(solver, max_iter);
         JX_PCGSetTol(solver, tol);
         JX_PCGSetTwoNorm(solver, twonorm);  // 0: B 范数； 1：l2 范数
         JX_PCGSetLogging(solver, 1);
         JX_PCGSetPrintLevel(solver, print_level);

         JX_PCGSetPrecond(solver, (JX_PtrToSolverFcn)JX_PAMGPrecond, (JX_PtrToSolverFcn)JX_PAMGSetup, amg_solver);

         // JX_PAMGSetup(amg_solver, (JX_ParCSRMatrix)par_matrix);
         JX_PAMGSetup(amg_solver, (JX_ParCSRMatrix)Ap_hat);

         JX_PCGSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         endtime = jx_MPI_Wtime();
         jx_GetWallTime(comm, "PAMG-CG Setup", starttime, endtime, 0, 2);

         starttime = jx_MPI_Wtime();

         // system("free -h");
         // JX_PCGSolve(solver, (JX_Matrix)par_matrix, // preOperater
         JX_PCGSolve(solver, (JX_Matrix)Ap_hat, // preOperater
                             (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         // system("free -h");
         endtime = jx_MPI_Wtime();
         jx_GetWallTime(comm, "PAMG-CG Solve", starttime, endtime, 0, 2);

         JX_PCGGetNumIterations(solver, &num_iterations);
         JX_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);

         JX_PAMGDestroy(amg_solver);
         JX_ParCSRPCGDestroy(solver);
         // jx_ParCSRMatrixDestroy(Ap_hat);
      }
      break;

      case 13:  /* DS-CG */
      {
         if (myid == 0) jx_printf("\n >>> Solver: DS-CG \n\n");

         starttime = jx_MPI_Wtime();

         ds_solver = NULL;

         JX_ParCSRPCGCreate(comm, &solver);
         JX_PCGSetMaxIter(solver, max_iter);
         JX_PCGSetTol(solver, tol);
         JX_PCGSetTwoNorm(solver, twonorm);  // 0: B 范数； 1：l2 范数
         JX_PCGSetLogging(solver, 1);
         JX_PCGSetPrintLevel(solver, print_level);

         JX_PCGSetPrecond(solver, (JX_PtrToSolverFcn)JX_DiagScalePrecond,
                                  (JX_PtrToSolverFcn)JX_DiagScaleSetup, ds_solver);

         JX_DiagScaleSetup(ds_solver, (JX_ParCSRMatrix)par_matrix);

         JX_PCGSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         endtime = jx_MPI_Wtime();
         jx_GetWallTime(comm, "DS-CG Setup", starttime, endtime, 0, 2);

         starttime = jx_MPI_Wtime();

         JX_PCGSolve(solver, (JX_Matrix)par_matrix, // preOperater
                             (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         endtime = jx_MPI_Wtime();
         jx_GetWallTime(comm, "DS-CG Solve", starttime, endtime, 0, 2);

         JX_PCGGetNumIterations(solver, &num_iterations);
         JX_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);

         JX_ParCSRPCGDestroy(solver);
      }
      break;

      case 14:  /* Euclid-CG */
      {
         if (myid == 0) jx_printf("\n >>> Solver: Euclid-CG \n\n");

         starttime = jx_MPI_Wtime();

         JX_EuclidCreate(comm, &euclid_solver);
         //JX_EuclidSetParams(euclid_solver, argc, argv);
         JX_EuclidSetLevel(euclid_solver, euclid_level);
         JX_EuclidSetBJ(euclid_solver, euclid_bj);

         JX_ParCSRPCGCreate(comm, &solver);
         JX_PCGSetMaxIter(solver, max_iter);
         JX_PCGSetTol(solver, tol);
         JX_PCGSetTwoNorm(solver, twonorm);  // 0: B 范数； 1：l2 范数
         JX_PCGSetLogging(solver, 1);
         JX_PCGSetPrintLevel(solver, print_level);

         JX_PCGSetPrecond(solver, (JX_PtrToSolverFcn)JX_EuclidSolve,
                                  (JX_PtrToSolverFcn)JX_EuclidSetup, euclid_solver);

         JX_EuclidSetup(euclid_solver, (JX_ParCSRMatrix)par_matrix);

         JX_PCGSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         endtime = jx_MPI_Wtime();
         jx_GetWallTime(comm, "Euclid-CG Setup", starttime, endtime, 0, 2);

         starttime = jx_MPI_Wtime();

         JX_PCGSolve(solver, (JX_Matrix)par_matrix, // preOperater
                             (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         endtime = jx_MPI_Wtime();
         jx_GetWallTime(comm, "Euclid-CG Solve", starttime, endtime, 0, 2);

         JX_PCGGetNumIterations(solver, &num_iterations);
         JX_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);

         JX_EuclidDestroy(euclid_solver);
         JX_ParCSRPCGDestroy(solver);
      }
      break;

      case 17:  /* ILU-PAMG-ILU-CG */
      {
         if (myid == 0) jx_printf("\n >>> Solver: ILU(0)-PAMG-ILU(0)-CG \n\n");

         if (TTest) starttime = jx_MPI_Wtime();

         JX_CombinedPrecondDataCreate(&combined_solver, comm);
         JX_CombinedPrecondDataSetPreID(combined_solver, 17);
         JX_CombinedPrecondDataSetDropTol(combined_solver, drop_tol);
         JX_CombinedPrecondDataSetInterpType(combined_solver, interp_type);
         JX_CombinedPrecondDataSetCoarsenType(combined_solver, coarsen_type);
         JX_CombinedPrecondDataSetCycleRelaxType(combined_solver, relax_type);
         JX_CombinedPrecondDataSetStrongThreshold(combined_solver, strong_threshold);
         JX_CombinedPrecondDataInitialize(combined_solver, euclid_level); /* Yue Xiaoqiang 2013/12/11 */
         //JX_CombinedPrecondDataInitializeP(combined_solver, argc, argv); /* Yue Xiaoqiang 2014/05/24 */

         JX_ParCSRPCGCreate(comm, &solver);
         JX_PCGSetMaxIter(solver, max_iter);
         JX_PCGSetTol(solver, tol);
         JX_PCGSetTwoNorm(solver, twonorm);  // 0: B 范数； 1：l2 范数
         JX_PCGSetLogging(solver, 1);
         JX_PCGSetPrintLevel(solver, print_level);

         JX_PCGSetPrecond(solver, (JX_PtrToSolverFcn)JX_CombinedPrecondDataSolve,
                                  (JX_PtrToSolverFcn)JX_CombinedPrecondDataSetup, combined_solver);

         JX_CombinedPrecondDataSetup(combined_solver, (JX_ParCSRMatrix)par_matrix);

         JX_PCGSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "ILU-PAMG-ILU(0)-CG Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jx_MPI_Wtime();

         JX_PCGSolve(solver, (JX_Matrix)par_matrix, // preOperater
                             (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "ILU-PAMG-ILU(0)-CG Solve", starttime, endtime, 0, 2);
         }

         JX_CombinedPrecondDataGetLULength(combined_solver, &lu_length);
         if (myid == 0)
         {
            jx_printf(" >>> Memory Complexity of ILU(0) = %d\n", lu_length);
         }

         JX_PCGGetNumIterations(solver, &num_iterations);
         JX_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JX_CombinedPrecondDataDestroy(combined_solver);
         JX_ParCSRPCGDestroy(solver);
      }
      break;

      case 18:  /* PAMG-ILU-PAMG-CG */
      {
         if (myid == 0) jx_printf("\n >>> Solver: PAMG-ILU(0)-PAMG-CG \n\n");

         if (TTest) starttime = jx_MPI_Wtime();

         JX_CombinedPrecondDataCreate(&combined_solver, comm);
         JX_CombinedPrecondDataSetPreID(combined_solver, 19);
         JX_CombinedPrecondDataSetDropTol(combined_solver, drop_tol);
         JX_CombinedPrecondDataSetInterpType(combined_solver, interp_type);
         JX_CombinedPrecondDataSetCoarsenType(combined_solver, coarsen_type);
         JX_CombinedPrecondDataSetCycleRelaxType(combined_solver, relax_type);
         JX_CombinedPrecondDataSetStrongThreshold(combined_solver, strong_threshold);
         JX_CombinedPrecondDataInitialize(combined_solver, euclid_level); /* Yue Xiaoqiang 2013/12/11 */
         //JX_CombinedPrecondDataInitializeP(combined_solver, argc, argv); /* Yue Xiaoqiang 2014/05/24 */

         JX_ParCSRPCGCreate(comm, &solver);
         JX_PCGSetMaxIter(solver, max_iter);
         JX_PCGSetTol(solver, tol);
         JX_PCGSetTwoNorm(solver, twonorm);  // 0: B 范数； 1：l2 范数
         JX_PCGSetLogging(solver, 1);
         JX_PCGSetPrintLevel(solver, print_level);


         JX_PCGSetPrecond(solver, (JX_PtrToSolverFcn)JX_CombinedPrecondDataSolve,
                                  (JX_PtrToSolverFcn)JX_CombinedPrecondDataSetup, combined_solver);

         JX_CombinedPrecondDataSetup(combined_solver, (JX_ParCSRMatrix)par_matrix);

         JX_PCGSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG-ILU(0)-PAMG-CG Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jx_MPI_Wtime();

         JX_PCGSolve(solver, (JX_Matrix)par_matrix, // preOperater
                             (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG-ILU(0)-PAMG-CG Solve", starttime, endtime, 0, 2);
         }

         JX_CombinedPrecondDataGetLULength(combined_solver, &lu_length);
         if (myid == 0)
         {
            jx_printf(" >>> Memory Complexity of ILU(0) = %d\n", lu_length);
         }

         JX_PCGGetNumIterations(solver, &num_iterations);
         JX_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JX_CombinedPrecondDataDestroy(combined_solver);
         JX_ParCSRPCGDestroy(solver);
      }
      break;

      case 21:  /* GMRES */
      {
         if (myid == 0) jx_printf("\n >>> Solver: GMRES(%d) \n\n", k_dim);

         JX_ParCSRGMRESCreate(comm, &solver);
         JX_GMRESSetKDim(solver, k_dim);
         JX_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JX_GMRESSetMaxIter(solver, max_iter);
         JX_GMRESSetTol(solver, tol);
         JX_GMRESSetLogging(solver, 1);
         JX_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */

         JX_GMRESSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         JX_GMRESSolve(solver, (JX_Matrix)par_matrix, // preOperater
                               (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         JX_GMRESGetNumIterations(solver, &num_iterations);
         JX_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         JX_ParCSRGMRESDestroy(solver);
      }
      break;

      case 220:  /* CPR-GMRES */
      {
         if (myid == 0) jx_printf("\n >>> Solver: CPR-GMRES(%d) \n\n", k_dim);
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         // 创建CPR预条件器
         jx_CPRPrecond *cpr_solver = JX_CPRCreate(comm);
         
         // 设置CPR参数
         JX_CPRSetParameter(cpr_solver, "pressure_index", 0);
         JX_CPRSetParameter(cpr_solver, "stage1_maxit", 1);
         JX_CPRSetParameter(cpr_solver, "stage2_maxit", 1);
         JX_CPRSetParameter(cpr_solver, "stage1_solver_type", 1);  // AMG
         JX_CPRSetParameter(cpr_solver, "stage2_solver_type", 3);  // BILU-GMRES (was: 2 for BGS)
         JX_CPRSetRealParameter(cpr_solver, "threshold", 1e-15);
         
         // 创建GMRES求解器
         JX_ParCSRGMRESCreate(comm, &solver);
         JX_GMRESSetKDim(solver, k_dim);
         JX_GMRESSetIsCheckRestarted(solver, is_check_restarted);
         JX_GMRESSetMaxIter(solver, max_iter);
         JX_GMRESSetTol(solver, tol);
         JX_GMRESSetLogging(solver, 1);
         JX_GMRESSetPrintLevel(solver, print_level);
         
         // 设置预条件器为CPR
         JX_GMRESSetPrecond(solver, 
                           (JX_PtrToSolverFcn)JX_CPRPrecond,
                           (JX_PtrToSolverFcn)JX_CPRSetup,
                           cpr_solver);
         
          // 设置CPR预条件器
          JX_CPRSetup(cpr_solver, (jx_ParBSRMatrix*)par_matrix);
         
         // 设置GMRES求解器
         JX_GMRESSetup(solver, (JX_Matrix)par_matrix, 
                        (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "CPR-GMRES Setup", starttime, endtime, 0, 2);
         }
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         // 求解
         JX_GMRESSolve(solver, (JX_Matrix)par_matrix,
                              (JX_Matrix)par_matrix,
                              (JX_Vector)par_rhs,
                              (JX_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "CPR-GMRES Solve", starttime, endtime, 0, 2);
         }
         
         JX_GMRESGetNumIterations(solver, &num_iterations);
         JX_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
         
         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }
         
         // 打印CPR统计信息
         JX_CPRPrint(cpr_solver, 2);
         
         // 销毁
         JX_CPRDestroy(&cpr_solver);
         JX_ParCSRGMRESDestroy(solver);

         break;
      }
      case 22:  /* PAMG-GMRES */
      {
         if (myid == 0) jx_printf("\n >>> Solver: PAMG-GMRES(%d) \n\n", k_dim);

         if (TTest) starttime = jx_MPI_Wtime();

         JX_PAMGCreate(&amg_solver);
         if (restri_type)
         {
            jx_assert(restri_type >= 0);
            JX_PAMGSetRestriction(amg_solver, restri_type);
            JX_PAMGSetGridRelaxPoints(amg_solver, grid_relax_points);
         }
         JX_PAMGSetMaxLevels(amg_solver, max_levels);
         JX_PAMGSetMaxIter(amg_solver, ns_up);
         JX_PAMGSetCycleType(amg_solver, cycle_type);
         JX_PAMGSetMeasureType(amg_solver, measure_type);
         JX_PAMGSetRAP2(amg_solver, rap2);
         JX_PAMGSetKeepTranspose(amg_solver, keepTranspose);
         JX_PAMGSetCoarsenType(amg_solver, coarsen_type);
         JX_PAMGSetInterpType(amg_solver, interp_type);
         JX_PAMGSetPMaxElmts(amg_solver, P_max_elmts);
         JX_PAMGSetAggNumLevels(amg_solver, agg_num_levels);
         JX_PAMGSetAIMeasureType(amg_solver, ai_measure_type);
         JX_PAMGSetAIRelaxType(amg_solver, ai_relax_type);
         JX_PAMGSetStrongThreshold(amg_solver, strong_threshold);
         JX_PAMGSetMaxRowSum(amg_solver, max_row_sum);
         JX_PAMGSetPrintLevel(amg_solver, amg_print_level);
         JX_PAMGSetCoarseThreshold(amg_solver, coarse_threshold);
         JX_PAMGSetRelaxWt(amg_solver, relax_wt);
         JX_PAMGSetOuterWt(amg_solver, outer_wt);
         JX_PAMGSetSCommPkgSwitch(amg_solver, S_commpkg_switch);
         JX_PAMGSetAIRStrongTh(amg_solver, AIR_strong_th);
         if (ns_down > -1) JX_PAMGSetCycleNumSweeps(amg_solver, ns_down, 1);           /* sweep for "down" */
         if (ns_up > -1) JX_PAMGSetCycleNumSweeps(amg_solver, ns_up, 2);           /* sweep for "up" */
         JX_PAMGSetCycleNumSweeps(amg_solver, ns_coarse, 3);           /* sweep for "coarsest" */
         JX_PAMGSetCycleRelaxType(amg_solver, relax_type, 1);  /* relax_type for "down" */
         JX_PAMGSetCycleRelaxType(amg_solver, relax_type, 2);  /* relax_type for "up" */
         JX_PAMGSetCycleRelaxType(amg_solver, coarse_solver, 3);           /* relax_type for "coarsest" */

         // printf("relax_type:%d \n\n",relax_type);

         JX_ParCSRGMRESCreate(comm, &solver);
         JX_GMRESSetKDim(solver, k_dim);
         JX_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JX_GMRESSetMaxIter(solver, max_iter);
         JX_GMRESSetTol(solver, tol);
         JX_GMRESSetLogging(solver, 1);
         JX_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */

         JX_GMRESSetPrecond(solver, (JX_PtrToSolverFcn)JX_PAMGPrecond,
                                    (JX_PtrToSolverFcn)JX_PAMGSetup, amg_solver);

         JX_PAMGSetup(amg_solver, (JX_ParCSRMatrix)par_matrix);

         JX_GMRESSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG-GMRES Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jx_MPI_Wtime();

         JX_GMRESSolve(solver, (JX_Matrix)par_matrix, // preOperater
                               (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG-GMRES Solve", starttime, endtime, 0, 2);
         }

         JX_GMRESGetNumIterations(solver, &num_iterations);

         JX_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         // if (print_level == 0 && myid == 0)
         if (myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }
         JX_PAMGDestroy(amg_solver);
         JX_ParCSRGMRESDestroy(solver);
      }
      break;

      case 23:  /* DS-GMRES */
      {
         if (myid == 0) jx_printf("\n >>> Solver: DS-GMRES(%d) \n\n", k_dim);

         if (TTest) starttime = jx_MPI_Wtime();

         ds_solver = NULL;

         JX_ParCSRGMRESCreate(comm, &solver);
         JX_GMRESSetKDim(solver, k_dim);
         JX_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JX_GMRESSetMaxIter(solver, max_iter);
         JX_GMRESSetTol(solver, tol);
         JX_GMRESSetLogging(solver, 1);
         JX_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */

         JX_GMRESSetPrecond(solver, (JX_PtrToSolverFcn)JX_DiagScalePrecond,
                                    (JX_PtrToSolverFcn)JX_DiagScaleSetup, ds_solver);

         JX_DiagScaleSetup(ds_solver, (JX_ParCSRMatrix)par_matrix);

         JX_GMRESSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "DS-GMRES Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jx_MPI_Wtime();

         JX_GMRESSolve(solver, (JX_Matrix)par_matrix, // preOperater
                               (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "DS-GMRES Solve", starttime, endtime, 0, 2);
         }

         JX_GMRESGetNumIterations(solver, &num_iterations);
         JX_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JX_ParCSRGMRESDestroy(solver);
      }
      break;

      case 24:  /* Euclid-GMRES */
      {
         if (myid == 0) jx_printf("\n >>> Solver: Euclid-GMRES(%d) \n\n", k_dim);

         if (TTest) starttime = jx_MPI_Wtime();

         JX_EuclidCreate(comm, &euclid_solver);
         //JX_EuclidSetParams(euclid_solver, argc, argv);
         JX_EuclidSetLevel(euclid_solver, euclid_level);
         JX_EuclidSetBJ(euclid_solver, euclid_bj);

         JX_ParCSRGMRESCreate(comm, &solver);
         JX_GMRESSetKDim(solver, k_dim);
         JX_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JX_GMRESSetMaxIter(solver, max_iter);
         JX_GMRESSetTol(solver, tol);
         JX_GMRESSetLogging(solver, 1);
         JX_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */

         JX_GMRESSetPrecond(solver, (JX_PtrToSolverFcn)JX_EuclidSolve,
                                    (JX_PtrToSolverFcn)JX_EuclidSetup, euclid_solver);

         JX_EuclidSetup(euclid_solver, (JX_ParCSRMatrix)par_matrix);

         JX_GMRESSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "Euclid-GMRES Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jx_MPI_Wtime();

         JX_GMRESSolve(solver, (JX_Matrix)par_matrix, // preOperater
                               (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "Euclid-GMRES Solve", starttime, endtime, 0, 2);
         }

         JX_GMRESGetNumIterations(solver, &num_iterations);
         JX_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JX_EuclidDestroy(euclid_solver);
         JX_ParCSRGMRESDestroy(solver);
      }
      break;

      case 25:
      {
         if (myid == 0) jx_printf("\n >>> Solver: Block Jacobi ILU-GMRES(%d) \n\n", k_dim);

         if (TTest) starttime = jx_MPI_Wtime();

         JX_ILUCreate(&ilu_solver);
         JX_ILUSetType(ilu_solver, 3);
         JX_ILUSetLevelOfFill(ilu_solver, 0);
         JX_ILUSetsweep(ilu_solver, 3);
         JX_ILUSetLocalReordering(ilu_solver, 1);
         JX_ILUSetTriSolve(ilu_solver, 14);
         JX_ILUSetLowerJacobiIters(ilu_solver, 3);
         JX_ILUSetUpperJacobiIters(ilu_solver, 3);
         JX_ILUSetIRIters(ilu_solver, 1);
         JX_ILUSetPrintLevel(ilu_solver, 0);
         JX_ILUSetMaxIter(ilu_solver, 1);
         JX_ILUSetTol(ilu_solver, 0.0);
         JX_ILUSetMaxNnzPerRow(ilu_solver, 1000);
         JX_ILUSetDropThreshold(ilu_solver, drop_tol);

         JX_ParCSRGMRESCreate(comm, &solver);
         JX_GMRESSetKDim(solver, k_dim);
         JX_GMRESSetIsCheckRestarted(solver, is_check_restarted);
         JX_GMRESSetMaxIter(solver, max_iter);
         JX_GMRESSetTol(solver, tol);
         JX_GMRESSetLogging(solver, 1);
         JX_GMRESSetPrintLevel(solver, print_level);

         JX_GMRESSetPrecond(solver,
                            (JX_PtrToSolverFcn)JX_ILUSolve,
                            (JX_PtrToSolverFcn)JX_ILUSetup,
                            ilu_solver);

         JX_ILUSetup(ilu_solver, (JX_ParCSRMatrix)par_matrix, (JX_ParVector)par_rhs, (JX_ParVector)par_sol);

         JX_GMRESSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "Block Jacobi ILU-GMRES Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jx_MPI_Wtime();

         JX_GMRESSolve(solver, (JX_Matrix)par_matrix,
                               (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "Block Jacobi ILU-GMRES Solve", starttime, endtime, 0, 2);
         }

         JX_GMRESGetNumIterations(solver, &num_iterations);
         JX_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         // JX_ILUDestroy(ilu_solver);
         JX_ParCSRGMRESDestroy(solver);
      }
      break;

      case 27:  /* BILU-GMRES */
      {
         if (myid == 0) jx_printf("\n >>> Solver: BILU-GMRES(%d) \n\n", k_dim);

         if (TTest) starttime = jx_MPI_Wtime();

         JX_BILUCreate(&ilu_solver);
         JX_ILUSetType(ilu_solver, 3);
         JX_ILUSetLevelOfFill(ilu_solver, 0);
         JX_ILUSetMaxIter(ilu_solver, 1);
         JX_ILUSetTol(ilu_solver, tol);
         JX_ILUSetTriSolve(ilu_solver, 14);
         JX_ILUSetLogging(ilu_solver, 1);
          JX_ILUSetPrintLevel(ilu_solver, print_level);
          JX_ILUSetsweep(ilu_solver, 2);

         ((jx_ParBILUData*)ilu_solver)->matA = (jx_ParBSRMatrix*)par_matrix;
         jx_BILUSetup((void*)ilu_solver, (jx_ParBSRMatrix*)par_matrix, par_rhs, par_sol);

         JX_ParCSRGMRESCreate(comm, &solver);
         JX_GMRESSetKDim(solver, k_dim);
         JX_GMRESSetIsCheckRestarted(solver, is_check_restarted);
         JX_GMRESSetMaxIter(solver, max_iter);
         JX_GMRESSetTol(solver, tol);
         JX_GMRESSetLogging(solver, 1);
         JX_GMRESSetPrintLevel(solver, print_level);

         JX_GMRESSetPrecond(solver, (JX_PtrToSolverFcn)bilu_gmres_solve,
                                    (JX_PtrToSolverFcn)bilu_gmres_setup, ilu_solver);

         JX_GMRESSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "BILU-GMRES Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jx_MPI_Wtime();

         JX_GMRESSolve(solver, (JX_Matrix)par_matrix,
                               (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "BILU-GMRES Solve", starttime, endtime, 0, 2);
         }

         JX_GMRESGetNumIterations(solver, &num_iterations);
         JX_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JX_BILUDestroy(ilu_solver);
         JX_ParCSRGMRESDestroy(solver);
      }
      break;

      // case 81:  /* ILU-GMRES */
      // {
      //    if (myid == 0) jx_printf("\n >>> Solver: ILU(0)-GMRES(%d) \n\n", k_dim);

      //    if (TTest) starttime = jx_MPI_Wtime();

      //    JX_ILUZeroFactorDataCreate(&ilu_solver, comm);
      //    JX_ILUZeroFactorDataSetMaxIter(ilu_solver, 1);
      //    JX_ILUZeroFactorDataSetDropTol(ilu_solver, drop_tol);

      //    JX_ParCSRGMRESCreate(comm, &solver);
      //    JX_GMRESSetKDim(solver, k_dim);
      //    JX_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
      //    JX_GMRESSetMaxIter(solver, max_iter);
      //    JX_GMRESSetTol(solver, tol);
      //    JX_GMRESSetLogging(solver, 1);
      //    JX_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */

      //    JX_GMRESSetPrecond(solver, (JX_PtrToSolverFcn)JX_ILUZeroFactorDataPrecond,
      //                               (JX_PtrToSolverFcn)JX_ILUZeroFactorDataSetup, ilu_solver);

      //    JX_ILUZeroFactorDataSetup(ilu_solver, (JX_ParCSRMatrix)par_matrix);

      //    JX_GMRESSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

      //    if (TTest)
      //    {
      //       endtime = jx_MPI_Wtime();
      //       jx_GetWallTime(comm, "ILU(0)-GMRES Setup", starttime, endtime, 0, 2);
      //    }

      //    if (TTest) starttime = jx_MPI_Wtime();

      //    JX_GMRESSolve(solver, (JX_Matrix)par_matrix, // preOperater
      //                          (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

      //    if (TTest)
      //    {
      //       endtime = jx_MPI_Wtime();
      //       jx_GetWallTime(comm, "ILU(0)-GMRES Solve", starttime, endtime, 0, 2);
      //    }

      //    JX_ILUZeroFactorDataGetLULength(ilu_solver, &lu_length);
      //    if (myid == 0)
      //    {
      //       jx_printf(" >>> Memory Complexity of ILU(0) = %d\n", lu_length);
      //    }

      //    JX_GMRESGetNumIterations(solver, &num_iterations);
      //    JX_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

      //    if (print_level == 0 && myid == 0)
      //    {
      //       jx_printf(" >>> num_iterations = %d\n", num_iterations);
      //       jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
      //    }

      //    JX_ILUZeroFactorDataDestroy(ilu_solver);
      //    JX_ParCSRGMRESDestroy(solver);
      // }
      // break;

      case 26:  /* ILU-AdaptiveGMRES */
      {
         if (myid == 0) jx_printf("\n >>> Solver: ILU(0)-AdaptiveGMRES(%d) \n\n", k_dim);

         if (TTest) starttime = jx_MPI_Wtime();

         JX_ILUZeroFactorDataCreate(&ilu_solver, comm);
         JX_ILUZeroFactorDataSetMaxIter(ilu_solver, 1);
         JX_ILUZeroFactorDataSetDropTol(ilu_solver, drop_tol);

         JX_ParCSRGMRESCreate(comm, &solver);
         JX_GMRESSetKDim(solver, k_dim);
         JX_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JX_GMRESSetMaxIter(solver, max_iter);
         JX_GMRESSetTol(solver, tol);
         JX_GMRESSetLogging(solver, 1);
         JX_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */
         JX_GMRESSetResDownZeroThreshold(solver, resdown_0_threshold);
         JX_GMRESSetConvFacThresholdTwo(solver, convfac_threshold_2);

         JX_GMRESSetPrecond(solver, (JX_PtrToSolverFcn)JX_ILUZeroFactorDataPrecond,
                                    (JX_PtrToSolverFcn)JX_ILUZeroFactorDataSetup, ilu_solver);

         JX_ILUZeroFactorDataSetup(ilu_solver, (JX_ParCSRMatrix)par_matrix);

         JX_GMRESSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "ILU(0)-AdaptiveGMRES Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jx_MPI_Wtime();

         JX_GMRESAdaptiveSolve(solver, (JX_Matrix)par_matrix, // preOperater
                                       (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "ILU(0)-AdaptiveGMRES Solve", starttime, endtime, 0, 2);
         }

         JX_ILUZeroFactorDataGetLULength(ilu_solver, &lu_length);
         if (myid == 0)
         {
            jx_printf(" >>> Memory Complexity of ILU(0) = %d\n", lu_length);
         }

         JX_GMRESGetNumIterations(solver, &num_iterations);
         JX_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JX_ILUZeroFactorDataDestroy(ilu_solver);
         JX_ParCSRGMRESDestroy(solver);
      }
      break;

      case 28:  /* Euclid2-AdaptiveGMRES */
      {
         if (myid == 0) jx_printf("\n >>> Solver: Euclid2-AdaptiveGMRES(%d) \n\n", k_dim);

         if (TTest) starttime = jx_MPI_Wtime();

         JX_CombinedPrecondDataCreate(&combined_solver, comm);
         JX_CombinedPrecondDataSetPreID(combined_solver, 1);
         JX_CombinedPrecondDataSetEuclidLevel(combined_solver, euclid_level);
         JX_CombinedPrecondDataSetDropTol(combined_solver, drop_tol);
         JX_CombinedPrecondDataSetAMGParameters(combined_solver, max_levels, relax_type, amg_print_level,
                                                interp_type, P_max_elmts, measure_type, coarsen_type,
                                                agg_num_levels, coarse_threshold, strong_threshold);

         JX_ParCSRGMRESCreate(comm, &solver);
         JX_GMRESSetKDim(solver, k_dim);
         JX_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JX_GMRESSetMaxIter(solver, max_iter);
         JX_GMRESSetTol(solver, tol);
         JX_GMRESSetLogging(solver, 1);
         JX_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */
         JX_GMRESSetResDownZeroThreshold(solver, resdown_0_threshold);
         JX_GMRESSetConvFacThresholdTwo(solver, convfac_threshold_2);

         JX_GMRESSetPrecond(solver, (JX_PtrToSolverFcn)JX_CombinedPrecondDataAdaptiveSolve,
                                    (JX_PtrToSolverFcn)JX_CombinedPrecondDataAdaptiveSetup2, combined_solver);

         JX_CombinedPrecondDataAdaptiveSetup2(combined_solver, (JX_ParCSRMatrix)par_matrix);

         JX_GMRESSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "Euclid2-AdaptiveGMRES Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jx_MPI_Wtime();

         JX_GMRESAdaptiveSolve(solver, (JX_Matrix)par_matrix, // preOperater
                                       (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "Euclid2-AdaptiveGMRES Solve", starttime, endtime, 0, 2);
         }

         JX_GMRESGetNumIterations(solver, &num_iterations);
         JX_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JX_GMRESGetLastPrecondType(solver, &last_precond_type);
         if (myid == 0)
         {
            if (last_precond_type == 1)
            {
               jx_printf("\n >>> Last Precond --- Euclid(%d) with Drop-Tol = %.6lf \n\n", euclid_level, drop_tol);
            }
            else if (last_precond_type == 21)
            {
               jx_printf("\n >>> Last Precond --- Euclid(%d)-PAMG with Drop-Tol = %.6lf \n\n", euclid_level, drop_tol);
            }
         }

         JX_CombinedPrecondDataDestroy(combined_solver);
         JX_ParCSRGMRESDestroy(solver);
      }
      break;

      case 29:  /* Euclid3-AdaptiveGMRES */
      {
         if (myid == 0) jx_printf("\n >>> Solver: Euclid3-AdaptiveGMRES(%d) \n\n", k_dim);

         if (TTest) starttime = jx_MPI_Wtime();

         JX_CombinedPrecondDataCreate(&combined_solver, comm);
         JX_CombinedPrecondDataSetPreID(combined_solver, 1);
         JX_CombinedPrecondDataSetEuclidLevel(combined_solver, euclid_level);
         JX_CombinedPrecondDataSetDropTol(combined_solver, drop_tol);
         JX_CombinedPrecondDataSetThetaPsiRhoPhi(combined_solver, theta_psi, theta_rho, theta_phi, theta_dis);
         JX_CombinedPrecondDataSetAMGParameters(combined_solver, max_levels, relax_type, amg_print_level,
                                                interp_type, P_max_elmts, measure_type, coarsen_type,
                                                agg_num_levels, coarse_threshold, strong_threshold);

         JX_ParCSRGMRESCreate(comm, &solver);
         JX_GMRESSetKDim(solver, k_dim);
         JX_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JX_GMRESSetMaxIter(solver, max_iter);
         JX_GMRESSetTol(solver, tol);
         JX_GMRESSetLogging(solver, 1);
         JX_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */
         JX_GMRESSetResDownZeroThreshold(solver, resdown_0_threshold);
         JX_GMRESSetConvFacThresholdTwo(solver, convfac_threshold_2);

         JX_GMRESSetPrecond(solver, (JX_PtrToSolverFcn)JX_CombinedPrecondDataAdaptiveSolve,
                                    (JX_PtrToSolverFcn)JX_CombinedPrecondDataAdaptiveSetup3, combined_solver);

         JX_CombinedPrecondDataAdaptiveSetup3(combined_solver, (JX_ParCSRMatrix)par_matrix);

         JX_GMRESSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "Euclid3-AdaptiveGMRES Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jx_MPI_Wtime();

         JX_GMRESAdaptiveSolve(solver, (JX_Matrix)par_matrix, // preOperater
                                       (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "Euclid3-AdaptiveGMRES Solve", starttime, endtime, 0, 2);
         }

         JX_GMRESGetNumIterations(solver, &num_iterations);
         JX_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JX_GMRESGetLastPrecondType(solver, &last_precond_type);
         if (myid == 0)
         {
            if (last_precond_type == 1)
            {
               jx_printf("\n >>> Last Precond --- Euclid(%d) with Drop-Tol = %.6lf \n\n", euclid_level, drop_tol);
            }
            else if (last_precond_type == 3)
            {
               jx_printf("\n >>> Last Precond --- PAMG \n\n");
            }
            else if (last_precond_type == 21)
            {
               jx_printf("\n >>> Last Precond --- Euclid(%d)-PAMG with Drop-Tol = %.6lf \n\n", euclid_level, drop_tol);
            }
         }

         JX_CombinedPrecondDataDestroy(combined_solver);
         JX_ParCSRGMRESDestroy(solver);
      }
      break;

      case 51:  /* PAMG-Euclid-PAMG-GMRES */
      {
         if (myid == 0) jx_printf("\n >>> Solver: PAMG-Euclid-PAMG-GMRES(%d) \n\n", k_dim);

         if (TTest) starttime = jx_MPI_Wtime();

         JX_CombinedPrecondDataCreate(&combined_solver, comm);
         JX_CombinedPrecondDataSetPreID(combined_solver, 11);
         JX_CombinedPrecondDataInitialize(combined_solver, euclid_level); /* Yue Xiaoqiang 2013/12/11 */
         //JX_CombinedPrecondDataInitializeP(combined_solver, argc, argv); /* Yue Xiaoqiang 2014/05/24 */

         JX_ParCSRGMRESCreate(comm, &solver);
         JX_GMRESSetKDim(solver, k_dim);
         JX_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JX_GMRESSetMaxIter(solver, max_iter);
         JX_GMRESSetTol(solver, tol);
         JX_GMRESSetLogging(solver, 1);
         JX_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */

         JX_GMRESSetPrecond(solver, (JX_PtrToSolverFcn)JX_CombinedPrecondDataSolve,
                                    (JX_PtrToSolverFcn)JX_CombinedPrecondDataSetup, combined_solver);

         JX_CombinedPrecondDataSetup(combined_solver, (JX_ParCSRMatrix)par_matrix);

         JX_GMRESSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG-Euclid-PAMG-GMRES Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jx_MPI_Wtime();

         JX_GMRESSolve(solver, (JX_Matrix)par_matrix, // preOperater
                               (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG-Euclid-PAMG-GMRES Solve", starttime, endtime, 0, 2);
         }

         JX_GMRESGetNumIterations(solver, &num_iterations);
         JX_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JX_CombinedPrecondDataDestroy(combined_solver);
         JX_ParCSRGMRESDestroy(solver);
      }
      break;

      case 52:  /* Euclid-PAMG-GMRES */
      {
         if (myid == 0) jx_printf("\n >>> Solver: Euclid-PAMG-GMRES(%d) \n\n", k_dim);

         if (TTest) starttime = jx_MPI_Wtime();

         JX_CombinedPrecondDataCreate(&combined_solver, comm);
         JX_CombinedPrecondDataSetPreID(combined_solver, 12);
         JX_CombinedPrecondDataInitialize(combined_solver, euclid_level); /* Yue Xiaoqiang 2013/12/11 */
         //JX_CombinedPrecondDataInitializeP(combined_solver, argc, argv); /* Yue Xiaoqiang 2014/05/24 */

         JX_ParCSRGMRESCreate(comm, &solver);
         JX_GMRESSetKDim(solver, k_dim);
         JX_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JX_GMRESSetMaxIter(solver, max_iter);
         JX_GMRESSetTol(solver, tol);
         JX_GMRESSetLogging(solver, 1);
         JX_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */

         JX_GMRESSetPrecond(solver, (JX_PtrToSolverFcn)JX_CombinedPrecondDataSolve,
                                    (JX_PtrToSolverFcn)JX_CombinedPrecondDataSetup, combined_solver);

         JX_CombinedPrecondDataSetup(combined_solver, (JX_ParCSRMatrix)par_matrix);

         JX_GMRESSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "Euclid-PAMG-GMRES Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jx_MPI_Wtime();

         JX_GMRESSolve(solver, (JX_Matrix)par_matrix, // preOperater
                               (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "Euclid-PAMG-GMRES Solve", starttime, endtime, 0, 2);
         }

         JX_GMRESGetNumIterations(solver, &num_iterations);
         JX_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JX_CombinedPrecondDataDestroy(combined_solver);
         JX_ParCSRGMRESDestroy(solver);
      }
      break;

      case 53:  /* PAMG-Euclid-GMRES */
      {
         if (myid == 0) jx_printf("\n >>> Solver: PAMG-Euclid-GMRES(%d) \n\n", k_dim);

         if (TTest) starttime = jx_MPI_Wtime();

         JX_CombinedPrecondDataCreate(&combined_solver, comm);
         JX_CombinedPrecondDataSetPreID(combined_solver, 13);
         JX_CombinedPrecondDataInitialize(combined_solver, euclid_level); /* Yue Xiaoqiang 2013/12/11 */
         //JX_CombinedPrecondDataInitializeP(combined_solver, argc, argv); /* Yue Xiaoqiang 2014/05/24 */

         JX_ParCSRGMRESCreate(comm, &solver);
         JX_GMRESSetKDim(solver, k_dim);
         JX_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JX_GMRESSetMaxIter(solver, max_iter);
         JX_GMRESSetTol(solver, tol);
         JX_GMRESSetLogging(solver, 1);
         JX_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */

         JX_GMRESSetPrecond(solver, (JX_PtrToSolverFcn)JX_CombinedPrecondDataSolve,
                                    (JX_PtrToSolverFcn)JX_CombinedPrecondDataSetup, combined_solver);

         JX_CombinedPrecondDataSetup(combined_solver, (JX_ParCSRMatrix)par_matrix);

         JX_GMRESSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG-Euclid-GMRES Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jx_MPI_Wtime();

         JX_GMRESSolve(solver, (JX_Matrix)par_matrix, // preOperater
                               (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG-Euclid-GMRES Solve", starttime, endtime, 0, 2);
         }

         JX_GMRESGetNumIterations(solver, &num_iterations);
         JX_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JX_CombinedPrecondDataDestroy(combined_solver);
         JX_ParCSRGMRESDestroy(solver);
      }
      break;

      case 54:  /* Euclid-PAMG-Euclid-GMRES */
      {
         if (myid == 0) jx_printf("\n >>> Solver: Euclid-PAMG-Euclid-GMRES(%d) \n\n", k_dim);

         if (TTest) starttime = jx_MPI_Wtime();

         JX_CombinedPrecondDataCreate(&combined_solver, comm);
         JX_CombinedPrecondDataSetPreID(combined_solver, 14);
         JX_CombinedPrecondDataInitialize(combined_solver, euclid_level); /* Yue Xiaoqiang 2013/12/11 */
         //JX_CombinedPrecondDataInitializeP(combined_solver, argc, argv); /* Yue Xiaoqiang 2014/05/24 */

         JX_ParCSRGMRESCreate(comm, &solver);
         JX_GMRESSetKDim(solver, k_dim);
         JX_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JX_GMRESSetMaxIter(solver, max_iter);
         JX_GMRESSetTol(solver, tol);
         JX_GMRESSetLogging(solver, 1);
         JX_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */

         JX_GMRESSetPrecond(solver, (JX_PtrToSolverFcn)JX_CombinedPrecondDataSolve,
                                    (JX_PtrToSolverFcn)JX_CombinedPrecondDataSetup, combined_solver);

         JX_CombinedPrecondDataSetup(combined_solver, (JX_ParCSRMatrix)par_matrix);

         JX_GMRESSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "Euclid-PAMG-Euclid-GMRES Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jx_MPI_Wtime();

         JX_GMRESSolve(solver, (JX_Matrix)par_matrix, // preOperater
                               (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "Euclid-PAMG-Euclid-GMRES Solve", starttime, endtime, 0, 2);
         }

         JX_GMRESGetNumIterations(solver, &num_iterations);
         JX_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JX_CombinedPrecondDataDestroy(combined_solver);
         JX_ParCSRGMRESDestroy(solver);
      }
      break;

      case 56:  /* ILU-PAMG-GMRES */
      {
         if (myid == 0) jx_printf("\n >>> Solver: ILU(0)-PAMG-GMRES(%d) \n\n", k_dim);

         if ((nprocs > 1) && (myid == 0))
         {
            ser_matrix = jx_CSRMatrixRead(MatFile, file_base);
            jx_CSRMatrixReorder(ser_matrix);
         }

         if (TTest) starttime = jx_MPI_Wtime();

         JX_CombinedPrecondDataCreate(&combined_solver, comm);
         JX_CombinedPrecondDataSetPreID(combined_solver, 16);
         JX_CombinedPrecondDataSetDropTol(combined_solver, drop_tol);
         JX_CombinedPrecondDataInitialize(combined_solver, euclid_level); /* Yue Xiaoqiang 2013/12/11 */
         //JX_CombinedPrecondDataInitializeP(combined_solver, argc, argv); /* Yue Xiaoqiang 2014/05/24 */
         JX_CombinedPrecondDataSetILUMatA(combined_solver, ser_matrix); /* 0号进程上的矩阵 */

         JX_ParCSRGMRESCreate(comm, &solver);
         JX_GMRESSetKDim(solver, k_dim);
         JX_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JX_GMRESSetMaxIter(solver, max_iter);
         JX_GMRESSetTol(solver, tol);
         JX_GMRESSetLogging(solver, 1);
         JX_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */

         JX_GMRESSetPrecond(solver, (JX_PtrToSolverFcn)JX_CombinedPrecondDataSolve,
                                    (JX_PtrToSolverFcn)JX_CombinedPrecondDataSetup, combined_solver);

         JX_CombinedPrecondDataSetup(combined_solver, (JX_ParCSRMatrix)par_matrix);

         JX_GMRESSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "ILU-PAMG-GMRES Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jx_MPI_Wtime();

         JX_GMRESSolve(solver, (JX_Matrix)par_matrix, // preOperater
                               (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "ILU-PAMG-GMRES Solve", starttime, endtime, 0, 2);
         }

         JX_CombinedPrecondDataGetLULength(combined_solver, &lu_length);
         if (myid == 0)
         {
            jx_printf(" >>> Memory Complexity of ILU(0) = %d\n", lu_length);
         }

         JX_GMRESGetNumIterations(solver, &num_iterations);
         JX_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         if ((nprocs > 1) && (myid == 0))
         {
            jx_CSRMatrixDestroy(ser_matrix);
         }
         JX_CombinedPrecondDataDestroy(combined_solver);
         JX_ParCSRGMRESDestroy(solver);
      }
      break;

      case 57:  /* ILU-PAMG-ILU-GMRES */
      {
         if (myid == 0) jx_printf("\n >>> Solver: ILU(0)-PAMG-ILU(0)-GMRES(%d) \n\n", k_dim);

         if (TTest) starttime = jx_MPI_Wtime();

         JX_CombinedPrecondDataCreate(&combined_solver, comm);
         JX_CombinedPrecondDataSetPreID(combined_solver, 17);
         JX_CombinedPrecondDataSetDropTol(combined_solver, drop_tol);
         JX_CombinedPrecondDataInitialize(combined_solver, euclid_level); /* Yue Xiaoqiang 2013/12/11 */
         //JX_CombinedPrecondDataInitializeP(combined_solver, argc, argv); /* Yue Xiaoqiang 2014/05/24 */

         JX_ParCSRGMRESCreate(comm, &solver);
         JX_GMRESSetKDim(solver, k_dim);
         JX_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JX_GMRESSetMaxIter(solver, max_iter);
         JX_GMRESSetTol(solver, tol);
         JX_GMRESSetLogging(solver, 1);
         JX_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */

         JX_GMRESSetPrecond(solver, (JX_PtrToSolverFcn)JX_CombinedPrecondDataSolve,
                                    (JX_PtrToSolverFcn)JX_CombinedPrecondDataSetup, combined_solver);

         JX_CombinedPrecondDataSetup(combined_solver, (JX_ParCSRMatrix)par_matrix);

         JX_GMRESSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "ILU-PAMG-ILU(0)-GMRES Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jx_MPI_Wtime();

         JX_GMRESSolve(solver, (JX_Matrix)par_matrix, // preOperater
                               (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "ILU-PAMG-ILU(0)-GMRES Solve", starttime, endtime, 0, 2);
         }

         JX_CombinedPrecondDataGetLULength(combined_solver, &lu_length);
         if (myid == 0)
         {
            jx_printf(" >>> Memory Complexity of ILU(0) = %d\n", lu_length);
         }

         JX_GMRESGetNumIterations(solver, &num_iterations);
         JX_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JX_CombinedPrecondDataDestroy(combined_solver);
         JX_ParCSRGMRESDestroy(solver);
      }
      break;

      case 31:  /* BiCGSTAB */
      {
         if (myid == 0) jx_printf("\n >>> Solver: BiCGSTAB \n\n");

         if (TTest) starttime = jx_MPI_Wtime();

         JX_ParCSRBiCGSTABCreate(comm, &solver);
         JX_BiCGSTABSetMaxIter(solver, max_iter);
         JX_BiCGSTABSetTol(solver, tol);
         JX_BiCGSTABSetAbsoluteTol(solver, 0.0);
         JX_BiCGSTABSetConvCriteria(solver, 0);
         JX_BiCGSTABSetLogging(solver, 1);
         JX_BiCGSTABSetPrintLevel(solver, print_level);

         JX_BiCGSTABSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "BiCGSTAB Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jx_MPI_Wtime();

         JX_BiCGSTABSolve(solver, (JX_Matrix)par_matrix, // preOperater
                                  (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "BiCGSTAB Solve", starttime, endtime, 0, 2);
         }

         JX_BiCGSTABGetNumIterations(solver, &num_iterations);
         JX_BiCGSTABGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JX_ParCSRBiCGSTABDestroy(solver);
      }
      break;

      case 32:  /* PAMG-BiCGSTAB */
      {
         if (myid == 0) jx_printf("\n >>> Solver: PAMG-BiCGSTAB \n\n");

         if (TTest) starttime = jx_MPI_Wtime();

         JX_PAMGCreate(&amg_solver);
         JX_PAMGSetMaxLevels(amg_solver, max_levels);
         JX_PAMGSetMaxIter(amg_solver, 1);
         JX_PAMGSetCycleType(amg_solver, cycle_type);
         JX_PAMGSetMeasureType(amg_solver, measure_type);
         JX_PAMGSetRAP2(amg_solver, rap2);
         JX_PAMGSetKeepTranspose(amg_solver, keepTranspose);
         JX_PAMGSetCoarsenType(amg_solver, coarsen_type);
         JX_PAMGSetInterpType(amg_solver, interp_type);
         JX_PAMGSetPMaxElmts(amg_solver, P_max_elmts);
         JX_PAMGSetAggNumLevels(amg_solver, agg_num_levels);
         JX_PAMGSetAIMeasureType(amg_solver, ai_measure_type);
         JX_PAMGSetAIRelaxType(amg_solver, ai_relax_type);
         JX_PAMGSetStrongThreshold(amg_solver, strong_threshold);
         JX_PAMGSetMaxRowSum(amg_solver, max_row_sum);
         JX_PAMGSetPrintLevel(amg_solver, amg_print_level);
         JX_PAMGSetCoarseThreshold(amg_solver, coarse_threshold);
         JX_PAMGSetRelaxWt(amg_solver, relax_wt);
         JX_PAMGSetOuterWt(amg_solver, outer_wt);
         if (ns_down > -1) JX_PAMGSetCycleNumSweeps(amg_solver, ns_down, 1);           /* sweep for "down" */
         if (ns_up > -1) JX_PAMGSetCycleNumSweeps(amg_solver, ns_up, 2);           /* sweep for "up" */
         JX_PAMGSetCycleNumSweeps(amg_solver, ns_coarse, 3);           /* sweep for "coarsest" */
         JX_PAMGSetCycleRelaxType(amg_solver, relax_type, 1);  /* relax_type for "down" */
         JX_PAMGSetCycleRelaxType(amg_solver, relax_type, 2);  /* relax_type for "up" */
         JX_PAMGSetCycleRelaxType(amg_solver, coarse_solver, 3);           /* relax_type for "coarsest" */

         JX_ParCSRBiCGSTABCreate(comm, &solver);
         JX_BiCGSTABSetMaxIter(solver, max_iter);
         JX_BiCGSTABSetTol(solver, tol);
         JX_BiCGSTABSetAbsoluteTol(solver, 0.0);
         JX_BiCGSTABSetConvCriteria(solver, 0);
         JX_BiCGSTABSetLogging(solver, 1);
         JX_BiCGSTABSetPrintLevel(solver, print_level);

         JX_BiCGSTABSetPrecond(solver, (JX_PtrToSolverFcn)JX_PAMGPrecond,
                                       (JX_PtrToSolverFcn)JX_PAMGSetup, amg_solver);

         JX_PAMGSetup(amg_solver, (JX_ParCSRMatrix)par_matrix);

         JX_BiCGSTABSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG-BiCGSTAB Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jx_MPI_Wtime();

         JX_BiCGSTABSolve(solver, (JX_Matrix)par_matrix, // preOperater
                                  (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG-BiCGSTAB Solve", starttime, endtime, 0, 2);
         }

         JX_BiCGSTABGetNumIterations(solver, &num_iterations);
         JX_BiCGSTABGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JX_PAMGDestroy(amg_solver);
         JX_ParCSRBiCGSTABDestroy(solver);
      }
      break;

      case 33:  /* DS-BiCGSTAB */
      {
         if (myid == 0) jx_printf("\n >>> Solver: DS-BiCGSTAB \n\n");

         if (TTest) starttime = jx_MPI_Wtime();

         ds_solver = NULL;

         JX_ParCSRBiCGSTABCreate(comm, &solver);
         JX_BiCGSTABSetMaxIter(solver, max_iter);
         JX_BiCGSTABSetTol(solver, tol);
         JX_BiCGSTABSetAbsoluteTol(solver, 0.0);
         JX_BiCGSTABSetConvCriteria(solver, 0);
         JX_BiCGSTABSetLogging(solver, 1);
         JX_BiCGSTABSetPrintLevel(solver, print_level);

         JX_BiCGSTABSetPrecond(solver, (JX_PtrToSolverFcn)JX_DiagScalePrecond,
                                       (JX_PtrToSolverFcn)JX_DiagScaleSetup, ds_solver);

         JX_DiagScaleSetup(ds_solver, (JX_ParCSRMatrix)par_matrix);

         JX_BiCGSTABSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "DS-BiCGSTAB Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jx_MPI_Wtime();

         JX_BiCGSTABSolve(solver, (JX_Matrix)par_matrix, // preOperater
                                  (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "DS-BiCGSTAB Solve", starttime, endtime, 0, 2);
         }

         JX_BiCGSTABGetNumIterations(solver, &num_iterations);
         JX_BiCGSTABGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JX_ParCSRBiCGSTABDestroy(solver);
      }
      break;

      case 34:  /* Euclid-BiCGSTAB */
      {
         if (myid == 0) jx_printf("\n >>> Solver: Euclid-BiCGSTAB \n\n");

         if (TTest) starttime = jx_MPI_Wtime();

         JX_EuclidCreate(comm, &euclid_solver);
         //JX_EuclidSetParams(euclid_solver, argc, argv);
         JX_EuclidSetLevel(euclid_solver, euclid_level);
         JX_EuclidSetBJ(euclid_solver, euclid_bj);

         JX_ParCSRBiCGSTABCreate(comm, &solver);
         JX_BiCGSTABSetMaxIter(solver, max_iter);
         JX_BiCGSTABSetTol(solver, tol);
         JX_BiCGSTABSetAbsoluteTol(solver, 0.0);
         JX_BiCGSTABSetConvCriteria(solver, 0);
         JX_BiCGSTABSetLogging(solver, 1);
         JX_BiCGSTABSetPrintLevel(solver, print_level);

         JX_BiCGSTABSetPrecond(solver, (JX_PtrToSolverFcn)JX_EuclidSolve,
                                       (JX_PtrToSolverFcn)JX_EuclidSetup, euclid_solver);

         JX_EuclidSetup(euclid_solver, (JX_ParCSRMatrix)par_matrix);

         JX_BiCGSTABSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "Euclid-BiCGSTAB Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jx_MPI_Wtime();

         JX_BiCGSTABSolve(solver, (JX_Matrix)par_matrix, // preOperater
                                  (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "Euclid-BiCGSTAB Solve", starttime, endtime, 0, 2);
         }

         JX_BiCGSTABGetNumIterations(solver, &num_iterations);
         JX_BiCGSTABGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JX_EuclidDestroy(euclid_solver);
         JX_ParCSRBiCGSTABDestroy(solver);
      }
      break;

      case 42:  /* PAMG-FlexGMRES */
      {
         if (myid == 0) jx_printf("\n >>> Solver: PAMG-FlexGMRES(%d) \n\n", k_dim);

         if (TTest) starttime = jx_MPI_Wtime();

         JX_PAMGCreate(&amg_solver);
         if (restri_type)
         {
            jx_assert(restri_type >= 0);
            JX_PAMGSetRestriction(amg_solver, restri_type);
            JX_PAMGSetGridRelaxPoints(amg_solver, grid_relax_points);
         }
         JX_PAMGSetMaxLevels(amg_solver, max_levels);
         JX_PAMGSetMaxIter(amg_solver, 1);
         JX_PAMGSetCycleType(amg_solver, cycle_type);
         JX_PAMGSetMeasureType(amg_solver, measure_type);
         JX_PAMGSetRAP2(amg_solver, rap2);
         JX_PAMGSetKeepTranspose(amg_solver, keepTranspose);
         JX_PAMGSetCoarsenType(amg_solver, coarsen_type);
         JX_PAMGSetInterpType(amg_solver, interp_type);
         JX_PAMGSetPMaxElmts(amg_solver, P_max_elmts);
         JX_PAMGSetAggNumLevels(amg_solver, agg_num_levels);
         JX_PAMGSetAIMeasureType(amg_solver, ai_measure_type);
         JX_PAMGSetAIRelaxType(amg_solver, ai_relax_type);
         JX_PAMGSetStrongThreshold(amg_solver, strong_threshold);
         JX_PAMGSetMaxRowSum(amg_solver, max_row_sum);
         JX_PAMGSetPrintLevel(amg_solver, amg_print_level);
         JX_PAMGSetCoarseThreshold(amg_solver, coarse_threshold);
         JX_PAMGSetRelaxWt(amg_solver, relax_wt);
         JX_PAMGSetOuterWt(amg_solver, outer_wt);
         JX_PAMGSetSCommPkgSwitch(amg_solver, S_commpkg_switch);
         JX_PAMGSetAIRStrongTh(amg_solver, AIR_strong_th);
         if (ns_down > -1) JX_PAMGSetCycleNumSweeps(amg_solver, ns_down, 1);           /* sweep for "down" */
         if (ns_up > -1) JX_PAMGSetCycleNumSweeps(amg_solver, ns_up, 2);           /* sweep for "up" */
         JX_PAMGSetCycleNumSweeps(amg_solver, ns_coarse, 3);           /* sweep for "coarsest" */
         JX_PAMGSetCycleRelaxType(amg_solver, relax_type, 1);  /* relax_type for "down" */
         JX_PAMGSetCycleRelaxType(amg_solver, relax_type, 2);  /* relax_type for "up" */
         JX_PAMGSetCycleRelaxType(amg_solver, coarse_solver, 3);           /* relax_type for "coarsest" */

         JX_ParCSRFlexGMRESCreate(comm, &solver);
         JX_FlexGMRESSetKDim(solver, k_dim);
         JX_FlexGMRESSetIsCheckRestarted(solver, is_check_restarted);
         JX_FlexGMRESSetMaxIter(solver, max_iter);
         JX_FlexGMRESSetTol(solver, tol);
         JX_FlexGMRESSetLogging(solver, 1);
         JX_FlexGMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */

         JX_FlexGMRESSetPrecond(solver, (JX_PtrToSolverFcn)JX_PAMGPrecond,
                                    (JX_PtrToSolverFcn)JX_PAMGSetup, amg_solver);

         JX_PAMGSetup(amg_solver, (JX_ParCSRMatrix)par_matrix);

         /* this is optional - could be a user defined one instead */
         JX_FlexGMRESSetModifyPC(solver, (JX_PtrToModifyPCFcn)jx_FlexGMRESModifyPCDefault);

         JX_FlexGMRESSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG-FlexGMRES Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jx_MPI_Wtime();

         JX_FlexGMRESSolve(solver, (JX_Matrix)par_matrix, // preOperater
                               (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG-FlexGMRES Solve", starttime, endtime, 0, 2);
         }

         JX_FlexGMRESGetNumIterations(solver, &num_iterations);
         JX_FlexGMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JX_PAMGDestroy(amg_solver);
         JX_ParCSRFlexGMRESDestroy(solver);
      }
      break;

      case 62:  /* PAMG-COGMRES */
      {
         if (myid == 0) jx_printf("\n >>> Solver: PAMG-COGMRES(%d) \n\n", k_dim);

         if (TTest) starttime = jx_MPI_Wtime();

         JX_PAMGCreate(&amg_solver);
         if (restri_type)
         {
            jx_assert(restri_type >= 0);
            JX_PAMGSetRestriction(amg_solver, restri_type);
            JX_PAMGSetGridRelaxPoints(amg_solver, grid_relax_points);
         }
         JX_PAMGSetMaxLevels(amg_solver, max_levels);
         JX_PAMGSetMaxIter(amg_solver, 1);
         JX_PAMGSetCycleType(amg_solver, cycle_type);
         JX_PAMGSetMeasureType(amg_solver, measure_type);
         JX_PAMGSetRAP2(amg_solver, rap2);
         JX_PAMGSetKeepTranspose(amg_solver, keepTranspose);
         JX_PAMGSetCoarsenType(amg_solver, coarsen_type);
         JX_PAMGSetInterpType(amg_solver, interp_type);
         JX_PAMGSetPMaxElmts(amg_solver, P_max_elmts);
         JX_PAMGSetAggNumLevels(amg_solver, agg_num_levels);
         JX_PAMGSetAIMeasureType(amg_solver, ai_measure_type);
         JX_PAMGSetAIRelaxType(amg_solver, ai_relax_type);
         JX_PAMGSetStrongThreshold(amg_solver, strong_threshold);
         JX_PAMGSetMaxRowSum(amg_solver, max_row_sum);
         JX_PAMGSetPrintLevel(amg_solver, amg_print_level);
         JX_PAMGSetCoarseThreshold(amg_solver, coarse_threshold);
         JX_PAMGSetRelaxWt(amg_solver, relax_wt);
         JX_PAMGSetOuterWt(amg_solver, outer_wt);
         JX_PAMGSetSCommPkgSwitch(amg_solver, S_commpkg_switch);
         JX_PAMGSetAIRStrongTh(amg_solver, AIR_strong_th);
         if (ns_down > -1) JX_PAMGSetCycleNumSweeps(amg_solver, ns_down, 1);           /* sweep for "down" */
         if (ns_up > -1) JX_PAMGSetCycleNumSweeps(amg_solver, ns_up, 2);           /* sweep for "up" */
         JX_PAMGSetCycleNumSweeps(amg_solver, ns_coarse, 3);           /* sweep for "coarsest" */
         JX_PAMGSetCycleRelaxType(amg_solver, relax_type, 1);  /* relax_type for "down" */
         JX_PAMGSetCycleRelaxType(amg_solver, relax_type, 2);  /* relax_type for "up" */
         JX_PAMGSetCycleRelaxType(amg_solver, coarse_solver, 3);           /* relax_type for "coarsest" */

         JX_ParCSRCOGMRESCreate(comm, &solver);
         JX_COGMRESSetKDim(solver, k_dim);
         JX_COGMRESSetUnroll(solver, unroll);
         JX_COGMRESSetCGS(solver, cgs);
         JX_COGMRESSetIsCheckRestarted(solver, is_check_restarted);
         JX_COGMRESSetMaxIter(solver, max_iter);
         JX_COGMRESSetTol(solver, tol);
         JX_COGMRESSetLogging(solver, 1);
         JX_COGMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */

         JX_COGMRESSetPrecond(solver, (JX_PtrToSolverFcn)JX_PAMGPrecond,
                                    (JX_PtrToSolverFcn)JX_PAMGSetup, amg_solver);

         JX_PAMGSetup(amg_solver, (JX_ParCSRMatrix)par_matrix);

         JX_COGMRESSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG-COGMRES Setup", starttime, endtime, 0, 2);
         }

         if (TTest) starttime = jx_MPI_Wtime();

         JX_COGMRESSolve(solver, (JX_Matrix)par_matrix, // preOperater
                               (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG-COGMRES Solve", starttime, endtime, 0, 2);
         }

         JX_COGMRESGetNumIterations(solver, &num_iterations);
         JX_COGMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }

         JX_PAMGDestroy(amg_solver);
         JX_ParCSRCOGMRESDestroy(solver);
      }
      break;
   }

   if (TTest)
   {
      endtimeT = jx_MPI_Wtime();
      jx_GetWallTime(comm, "Total Sove Time", starttimeT, endtimeT, 0, 2);
   }

   *iter = num_iterations;
   return 0;
}



#define Debug_zl 0
JX_Int
jx_CSRMatrixFilter( jx_CSRMatrix *matrix, JX_Real eps, jx_CSRMatrix *matrixOut)
{
   JX_Real    *matrix_data, val;
   JX_Int     *matrix_i;
   JX_Int     *matrix_j;
   JX_Int      num_rows;
   JX_Int      i, j, istart, iend;

   matrix_data = jx_CSRMatrixData(matrix);
   matrix_i    = jx_CSRMatrixI(matrix);
   matrix_j    = jx_CSRMatrixJ(matrix);
   num_rows    = jx_CSRMatrixNumRows(matrix);


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
   JX_Real max, th;
   JX_Int count = 0;
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
JX_Int TranMatrixOrder( JX_Int n, JX_Int nnz, JX_Int *ia, JX_Int *ja, JX_Real *aa , JX_Real *b)
{
    JX_Int i, k, j, ik;
    JX_Int istart, iend;
    JX_Real val;
    JX_Int *ia0, *ja0;
    JX_Real *aa0, *b0;
    JX_Int nnm = n/3;

    if (n%3 != 0)
    {
        printf("\033[31mERROR:\033[00m the number of rows in a matrix cannot be divided by 3!!!\n");
        exit(-1);
    }

    ia0 = (JX_Int *)malloc(sizeof(JX_Int) * (n+1));
    ja0 = (JX_Int *)malloc(sizeof(JX_Int) * nnz);
    aa0 = (JX_Real *)malloc(sizeof(JX_Real) * nnz);
    b0  = (JX_Real *)malloc(sizeof(JX_Real) * n);


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


    JX_Int li, lj;
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
JX_Int TranMatrixOrder_new( JX_Int n, JX_Int nnz, JX_Int *ia, JX_Int *ja, JX_Real *aa , JX_Real *b)
{
    JX_Int *ia0, *ja0;
    JX_Real *aa0, *b0;
    JX_Int RowA = 0 , RowC = 0;
    JX_Int ng_p_two , sub_num_rows;
    JX_Int num_nonzerosC,i, j, mdo, col, row_end;
    JX_Int Rowx = 0, Rowy;

   ng_p_two = 6;                     //自由度个数
   sub_num_rows = n/ng_p_two;   //子块维数

    if (n%sub_num_rows != 0)
    {
        printf("\033[31mERROR:\033[00m the number of rows in a matrix cannot be divided by %d!!!\n",sub_num_rows);
        exit(-1);
    }

   ia0 = (JX_Int *)malloc(sizeof(JX_Int) * (n+1));
   ja0 = (JX_Int *)malloc(sizeof(JX_Int) * nnz);
   aa0 = (JX_Real *)malloc(sizeof(JX_Real) * nnz);
   b0  = (JX_Real *)malloc(sizeof(JX_Real) * n);

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

JX_Int
jx_LU(jx_ParCSRMatrix *par_matrix,
               jx_ParVector    *par_rhs,
               jx_ParVector    *par_app)
{
   MPI_Comm         comm         = jx_ParCSRMatrixComm(par_matrix);
   jx_CSRMatrix    *A_diag       = jx_ParCSRMatrixDiag(par_matrix);
   jx_ParCSRCommPkg    *comm_pkg = jx_ParCSRMatrixCommPkg(par_matrix);

   JX_Int             n_global = jx_ParCSRMatrixGlobalNumRows(par_matrix);
   JX_Int             n        = jx_CSRMatrixNumRows(A_diag);
   JX_Int             first_index   = jx_ParVectorFirstIndex(par_app);

   jx_Vector      *u_local = jx_ParVectorLocalVector(par_app);
   JX_Real         *u_data  = jx_VectorData(u_local);
   // JX_LOW_REAL         *u_data  = jx_VectorData(u_local);

   jx_CSRMatrix    *A_CSR;
   JX_Int             *A_CSR_i;
   JX_Int             *A_CSR_j;
   JX_Real          *A_CSR_data;
   // JX_LOW_REAL          *A_CSR_data;

   jx_Vector       *f_vector;
   JX_Real          *f_vector_data;
   // JX_LOW_REAL          *f_vector_data;

   JX_Int             i;
   JX_Int             jj;
   JX_Int             column;
   JX_Int             relax_error = 0;
   JX_Int             num_procs, my_id;

   // JX_Real         *A_mat;
   // JX_Real         *b_vec;

   JX_LOW_REAL         *A_mat;
   JX_LOW_REAL         *b_vec;

   jx_MPI_Comm_size(comm, &num_procs);
   jx_MPI_Comm_rank(comm, &my_id);

  /*-------------------------------------------------------------------------
   * added by peghoty, if the comm_pkg of par_matrix is not created
   * previously, something will be wrong when the "relax" function
   * is called on multi-processors occasions.  2009/07/24
   *----------------------------------------------------------------------- */

   if (!comm_pkg)
   {
      jx_MatvecCommPkgCreate(par_matrix);
      comm_pkg = jx_ParCSRMatrixCommPkg(par_matrix);
   }

      //------------------------------------------------------------------------------//
      //                   Direct solve: use gaussian elimination                     //
      //------------------------------------------------------------------------------//

        /*-----------------------------------------------------------------
         *  Generate CSR matrix from ParCSRMatrix par_matrix
         *-----------------------------------------------------------------*/
#ifdef JX_NO_GLOBAL_PARTITION
         /* all processors are needed for these routines */
         A_CSR = jx_ParCSRMatrixToCSRMatrixAll(par_matrix);
         f_vector = jx_ParVectorToVectorAll(par_rhs);
	 if (n)
	 {
#else
	 if (n)
	 {
	     A_CSR = jx_ParCSRMatrixToCSRMatrixAll(par_matrix);
	     f_vector = jx_ParVectorToVectorAll(par_rhs);
#endif
         A_CSR_i = jx_CSRMatrixI(A_CSR);
         A_CSR_j = jx_CSRMatrixJ(A_CSR);
         A_CSR_data = jx_CSRMatrixData(A_CSR);
   	   f_vector_data = jx_VectorData(f_vector);

            // A_mat = jx_CTAlloc(JX_Real, n_global*n_global);

            // b_vec = jx_CTAlloc(JX_Real, n_global);

            A_mat = jx_CTAlloc(JX_LOW_REAL, n_global*n_global);
            b_vec = jx_CTAlloc(JX_LOW_REAL, n_global);

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

            relax_error = jx_gselim_lu(A_mat,b_vec,n_global);

            /* use version with pivoting */
            /* relax_error = jx_gselim_piv(A_mat,b_vec,n_global);*/

            for (i = 0; i < n; i ++)
            {
               u_data[i] = b_vec[first_index+i];
            }

	         jx_TFree(A_mat);
            jx_TFree(b_vec);
            jx_CSRMatrixDestroy(A_CSR);
            A_CSR = NULL;
            jx_SeqVectorDestroy(f_vector);
            f_vector = NULL;

         }
#ifdef JX_NO_GLOBAL_PARTITION
         else
         {
            jx_CSRMatrixDestroy(A_CSR);
            A_CSR = NULL;
            jx_SeqVectorDestroy(f_vector);
            f_vector = NULL;
         }
#endif

   // return(relax_error);
   jx_printf("hjmhjm:%d\n",relax_error);
   return 0;
}

JX_Int
jx_gselim_lu( JX_LOW_REAL *A_matrix, JX_LOW_REAL *x_vector, JX_Int size )
{
   JX_Int    err_flag = 0;
   JX_Int    j,k,m;
   JX_LOW_REAL factor;

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


JX_Int
jx_Pre_Pardiso(Pardiso_data *pdata,
               jx_ParCSRMatrix *par_matrix,
               jx_ParVector    *par_rhs,
               jx_ParVector    *par_app)
{
   MPI_Comm         comm         = jx_ParCSRMatrixComm(par_matrix);
   jx_CSRMatrix    *A_diag       = jx_ParCSRMatrixDiag(par_matrix);
   jx_ParCSRCommPkg    *comm_pkg = jx_ParCSRMatrixCommPkg(par_matrix);

   JX_Int             n_global = jx_ParCSRMatrixGlobalNumRows(par_matrix);
   JX_Int             n        = jx_CSRMatrixNumRows(A_diag);
   JX_Int             first_index   = jx_ParVectorFirstIndex(par_app);

   jx_Vector      *u_local = jx_ParVectorLocalVector(par_app);
   JX_Real         *u_data  = jx_VectorData(u_local);

   jx_CSRMatrix    *A_CSR;
   JX_Int             *A_CSR_i;
   JX_Int             *A_CSR_j;
   JX_Real          *A_CSR_data;

   jx_Vector       *f_vector;
   JX_Real          *f_vector_data;

   JX_Int             i;
   JX_Int             k;
   JX_Int             column;
   JX_Int             relax_error = 0;
   JX_Int             num_procs, my_id;


   jx_MPI_Comm_size(comm, &num_procs);
   jx_MPI_Comm_rank(comm, &my_id);

  /*-------------------------------------------------------------------------
   * added by peghoty, if the comm_pkg of par_matrix is not created
   * previously, something will be wrong when the "relax" function
   * is called on multi-processors occasions.  2009/07/24
   *----------------------------------------------------------------------- */
   if (!comm_pkg)
   {
      jx_MatvecCommPkgCreate(par_matrix);
      comm_pkg = jx_ParCSRMatrixCommPkg(par_matrix);
   }
   // printf("jx_pardiso, %s, %s, %d\n", __FILE__, __FUNCTION__, __LINE__);

   //------------------------------------------------------------------------------//
   //                   Direct solve: use PARDISO                                  //
   //------------------------------------------------------------------------------//

   /*-----------------------------------------------------------------
   *  Generate CSR matrix from ParCSRMatrix par_matrix
   *-----------------------------------------------------------------*/
#ifdef JX_NO_GLOBAL_PARTITION
         /* all processors are needed for these routines */
         A_CSR = jx_ParCSRMatrixToCSRMatrixAll(par_matrix);
         f_vector = jx_ParVectorToVectorAll(par_rhs);
	if (n)
	{
#else
	if (n)
	{
	   A_CSR = jx_ParCSRMatrixToCSRMatrixAll(par_matrix);
	   f_vector = jx_ParVectorToVectorAll(par_rhs);
#endif

      // PARDISO
      // printf("jx_qsort1, %s, %s, %d\n", __FILE__, __FUNCTION__, __LINE__);
      for ( i = 0; i < A_CSR->num_rows; i++)
      {
         k = A_CSR->i[i];
         jx_qsort1( &A_CSR->j[k], &A_CSR->data[k], 0, A_CSR->i[i+1] - A_CSR->i[i] - 1 );
      }
      // printf("jx_pardiso, %s, %s, %d\n", __FILE__, __FUNCTION__, __LINE__);
      jx_pardiso_float(A_CSR, f_vector, u_local);
      // jx_pardiso_double(A_CSR, f_vector, u_local);

      // printf("jx_pardiso, %s, %s, %d\n", __FILE__, __FUNCTION__, __LINE__);

 	   // free
      jx_CSRMatrixDestroy(A_CSR);
      A_CSR = NULL;
      jx_SeqVectorDestroy(f_vector);
      f_vector = NULL;

   }
#ifdef JX_NO_GLOBAL_PARTITION
   else
   {
      jx_CSRMatrixDestroy(A_CSR);
      A_CSR = NULL;
      jx_SeqVectorDestroy(f_vector);
      f_vector = NULL;
   }
#endif

   return 0;
}

JX_Int
jx_pardiso_float(jx_CSRMatrix *AA, jx_Vector *bb, jx_Vector *xx)
{
    int prtlvl = 0;
    int i;
    // Matrix
    MKL_INT n = AA->num_rows;
    MKL_INT m = AA->num_cols;
    MKL_INT nnz = AA->num_nonzeros;
    MKL_INT *ia = AA->i;
    MKL_INT *ja = AA->j;
    JX_Real *a1 = AA->data;
    // vector
    JX_Real *b1 = bb->data;
    JX_Real *x1 = xx->data;

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

    for(i=0;i<n;i++){
        x1[i] = x[i];
    }

    return 1;
}

JX_Int
jx_pardiso_double(jx_CSRMatrix *AA, jx_Vector *bb, jx_Vector *xx)
{
    int prtlvl = 0;
    int i;
    // Matrix
    MKL_INT n = AA->num_rows;
    MKL_INT m = AA->num_cols;
    MKL_INT nnz = AA->num_nonzeros;
    MKL_INT *ia = AA->i;
    MKL_INT *ja = AA->j;
    JX_Real *a1 = AA->data;
    // vector
    JX_Real *b1 = bb->data;
    JX_Real *x1 = xx->data;

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

    for(i=0;i<n;i++){
        x1[i] = x[i];
    }

    return 1;
}

#endif