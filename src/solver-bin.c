/*
This is the reference code of SolverChallenge to optimize by the participants
@Code version: 1.0
@Update date: 2021/5/17
@Author: Dechuang Yang, Haocheng Lian

Added some parallel packages: jxfpamg
@Update date: 2021/5/28
@Author: Li Zhao
*/

// typedef long double Real;
typedef double Real;
typedef int Int;
// typedef long long int Int;

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include "mmio_highlevel.h"
#include "my_solver.h"
#include "mpi.h"
#include "new_Matrix.h"


// Real check_correctness(Int n, Int *row_ptr, Int *col_idx, Real *val, Real *x, Real *b, Real *Res);
// void check_correctness(int n, int *row_ptr, int *col_idx, long double *val, long double *x, long double *b);
void check_correctness(int n, int *row_ptr, int *col_idx, Real *val, Real *x, Real *b);
void store_x(Int n, Real *x, char *filename);
void load_b(Int n, Real *b, char *filename);

Int main(Int argc, char **argv)
{
    Int     *row_ptr; // the csr row pointer array of matrix A
    Int     *col_idx; // the csr column index array of matrix A
    Real  *val;  // the csr value array of matrix A
    Real  *x, *b, *x1, *x2; 
    int *row_ptr_a11_n,*row_ptr_a12_n,*row_ptr_a21_n,*row_ptr_a22_n,*col_idx_a11_n,*col_idx_a12_n,*col_idx_a21_n,*col_idx_a22_n;
    double *val_a11_n, *val_a12_n, *val_a21_n, *val_a22_n, *b1_n, *b2_n;

    char    *filename_matrix = argv[1];  // the filename of matrix A
    char    *nrows = argv[2];       // type 0: CSR matrix and vector; type 1: binary matrix and vector
    char    *number_of_zeros = argv[3];       // type 0: CSR matrix and vector; type 1: binary matrix and vector
    char    *filename_x = argv[4];       // the filename of solution vector x
    char    *matrix_tolerance = argv[5]; // the tolerance of input matrix
    char    *pbb = argv[6];              // the number of blocks

    int     n           = atoi(nrows);
    int     nnzA        = atoi(number_of_zeros);
    int     pb          = atoi(pbb);
    // char    *sid = argv[5];              // solver id
    Int     iter = 0;                    // the number of iterations
    Real  residule = 0.0, RRN;
    Real  tolerance = atof(matrix_tolerance);
    Real  coff = 1;
    int     rank;
    int     size;
    Int     i;
    // Int     solver_id   = atoi(sid);
    Int     solver_id   = 12;
    Int     precond_id  = 3;
    Int     num_functions = 1;
    int n1 = 0;
    int n2 = 0;
    MPI_Comm comm = MPI_COMM_WORLD;
   
    // system("free -h");
    MPI_Init(NULL,NULL); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size); 
    
    //-----------------------
    //  命令行修改参数
    //-----------------------
    Int arg_index = 1;
    while (arg_index < argc)
    {
        if ( strcmp(argv[arg_index], "-sid") == 0 )
        {
            arg_index ++;
            solver_id = atoi(argv[arg_index++]);
        }
        else if ( strcmp(argv[arg_index], "-nf") == 0 )
        {
            arg_index ++;
            num_functions = atoi(argv[arg_index++]);
        }else{
            arg_index ++;
        }
    }

    if (rank == 0)
    {
        printf("===================================================================\n");
        printf("A  : %s\n", filename_matrix);
        printf("x  : %s\n", filename_x);
        printf("tol: %s\n", matrix_tolerance);
        printf("sid: %d\n", solver_id);
        printf("nf : %d\n", num_functions);
        printf("===================================================================\n");

        printf("\n\n+++++++++++++++++++++ MPI using %d processors +++++++++++++++++++++\n\n", size);

        //load matrix
        printf("load matrix:\n");

        row_ptr = (int *)malloc(sizeof(int) * n+1);
        col_idx = (int *)malloc(sizeof(int) * nnzA);
        val = (double *)malloc(sizeof(double) * nnzA);
        b = (double *)malloc(sizeof(double) * n);

    //load matrix
    // 打开文件
    char target[200];
    int size;
    FILE *fp;

    sprintf(target, "%s/Ai.bin\0", filename_matrix);
    fp = fopen(target, "rb");
    if (fp == NULL) {
        printf("cannot open %s \n", target);
        return -1;
    }
    size = fread(row_ptr, sizeof(int), n+1, fp);
    fclose(fp);

    sprintf(target, "%s/Aj.bin\0", filename_matrix);
    fp = fopen(target, "rb");
    if (fp == NULL) {
        printf("cannot open %s \n", target);
        return -1;
    }
    size = fread(col_idx, sizeof(int), nnzA, fp);
    fclose(fp);

    sprintf(target, "%s/Av.bin\0", filename_matrix);
    fp = fopen(target, "rb");
    if (fp == NULL) {
        printf("cannot open %s \n", target);
        return -1;
    }
    size = fread(val, sizeof(double), nnzA, fp);
    fclose(fp);

    printf("pb=%d",pb);
    // return 0;
    if(pb==7){

        sprintf(target, "%s/b_t0.bin\0", filename_matrix);
        fp = fopen(target, "rb");
        if (fp == NULL) {
            printf("cannot open %s \n", target);
            return -1;
        }
        size = fread(b, sizeof(double), n, fp);
        fclose(fp);
    }
    else{
        sprintf(target, "%s/b.bin\0", filename_matrix);
        fp = fopen(target, "rb");
        if (fp == NULL) {
            printf("cannot open %s \n", target);
            return -1;
        }
        size = fread(b, sizeof(double), n, fp);
        fclose(fp);
    }

        printf("matrix info: m: %lld, n: %lld, nnz: %lld\n", n, n, nnzA);


        int *mark = (int *)malloc(sizeof(int) * n);
        sprintf(target, "%s/mark_pb05_one.txt\0", filename_matrix);
        fp = fopen(target, "rb");
        for (i = 0; i < n; i++)
            fscanf(fp, "%d\n", &mark[i]);
        fclose(fp);


        for (i = 0; i < n; i++){
            if (mark[i] == 1){
                n1++;
            }else if (mark[i] == 2)
            {
                n2++;
            }
        }

        printf(" ===> matrix split by mark....\n"); 



        matrix(n, nnzA, mark, b, row_ptr, col_idx, val, 
            &row_ptr_a11_n, &col_idx_a11_n, &val_a11_n, 
            &row_ptr_a12_n, &col_idx_a12_n, &val_a12_n, 
            &row_ptr_a21_n, &col_idx_a21_n, &val_a21_n, 
            &row_ptr_a22_n, &col_idx_a22_n, &val_a22_n, 
            &b1_n, &b2_n);

        int nnz11 = row_ptr_a11_n[n1];
        int nnz12 = row_ptr_a12_n[n1];
        int nnz21 = row_ptr_a21_n[n2];
        int nnz22 = row_ptr_a22_n[n2];
        printf("a11 row : %d, a22 row : %d, a row : %d \n", n1, n2, n1+n2);
        printf("a11 nnz : %d, a12 nnz : %d, a21 nnz : %d, a22 nnz : %d, a nnz : %d \n", nnz11, nnz12, nnz21, nnz22, nnz11+nnz12+nnz21+nnz22);

        
        x = (Real *)malloc(sizeof(Real) * n);
        x1 = (Real *)malloc(sizeof(Real) * n1);
        x2 = (Real *)malloc(sizeof(Real) * n2);
        for (i = 0; i < n; i++) x[i] = 0; // 初始化为 0 
        for (i = 0; i < n1; i++) x1[i] = 0; // 初始化为 0 
        for (i = 0; i < n2; i++) x2[i] = 0; // 初始化为 0 


        // 精确求解 a11 x1 = b1
        for (i = 0; i < n1; i++){
            x1[i] = b1_n[i]/val_a11_n[i] ;
        }

        // 计算 a21x1 
        Real *b_correct = (Real *)malloc(sizeof(Real) * n2);
        spmv(n2, row_ptr_a21_n, col_idx_a21_n, val_a21_n, x1, b_correct);
        // 更新 a22x2 = b2 - a21x1
        for (i = 0; i < n2; i++){
            b2_n[i] -= b_correct[i] ;
        }
    }

    // solve x and record wall-time
    struct timeval t_start, t_stop;
    gettimeofday(&t_start, NULL);

    //-------------------------------------------------------------------------
    //  boomer_amg
    //-------------------------------------------------------------------------  
    if (rank == 0) printf("\nJXFPAMG_Solver_Interface(int, long double -- version):\n");    
    
    JXFPAMG_Solver_Interface(argc, argv,comm, n2, row_ptr_a22_n, col_idx_a22_n, val_a22_n, x2, b2_n, tolerance, solver_id, num_functions, &iter);
    // JXFPAMG_Solver_Interface(argc, argv,comm, n, row_ptr, col_idx, val, x, b, tolerance, solver_id, num_functions, &iter);
    
    gettimeofday(&t_stop, NULL);
    double total_time = (t_stop.tv_sec - t_start.tv_sec) * 1000.0 + (t_stop.tv_usec - t_start.tv_usec) / 1000.0;
    
    if (rank == 0)
    {
        //store x to a file
        // store_x(n, x, filename_x);

        //check the correctness
        //print the #iteration, residual and total_time
        //RRN = check_correctness(n, row_ptr, col_idx, val, x, b, &residule);
        check_correctness(n, row_ptr, col_idx, val, x, b);
        // printf("\n\n===============================================================================\n");
        // printf("iteration = %d, residual = %Le, RRN = %Le, total_time = %.5Lf sec\n", iter, residule, RRN, total_time / 1000.0);
        // printf("===============================================================================\n");
        printf("total_time = %.5lf sec\n", total_time / 1000.0);
    }
    
    ///
    MPI_Finalize();
    return 0;
}

//validate the x by b-A*x
// Real check_correctness(Int n, Int *row_ptr, Int *col_idx, Real *val, Real *x, Real *b, Real *Res)
// {
//     Real *b_new = (Real *)malloc(sizeof(Real) * n);
//     Real *check_b = (Real *)malloc(sizeof(Real) * n);
//     spmv(n, row_ptr, col_idx, val, x, b_new);
//     Int i;
//     for (i = 0; i < n; i++)
//         check_b[i] = b_new[i] - b[i];

//     *Res =  vec2norm(check_b, n);
//     Real b_L2 = vec2norm(b, n);
//     // printf("\nZL: ||b-Ax||_2: %Le, ||b||_2: %Le\n", vec2norm(check_b, n), b_L2);
//     return *Res / b_L2;
// }
void check_correctness(int n, int *row_ptr, int *col_idx, Real *val, Real *x, Real *b)
{
    Real *b_new = (Real *)malloc(sizeof(Real) * n);
    Real *check_b = (Real *)malloc(sizeof(Real) * n);
    spmv(n, row_ptr, col_idx, val, x, b_new);
    int i ;
    for (i = 0; i < n; i++)
        check_b[i] = b_new[i] - b[i];

    double answer1 = vec2norm(check_b, n);
    //double answer2 = max_check(check_b, n);
    double answer3 = answer1 / vec2norm(b, n);
    fprintf(stdout, "Check || AX - B || 2             =  %12.6e\n", answer1);
    //fprintf(stdout, "Check || AX - B || MAX           =  %12.6e\n", answer2);
    fprintf(stdout, "Check || AX - B || 2 / || B || 2 =  %12.6e\n", answer3);

    free(b_new);
    free(check_b);
}
// void check_correctness(int n, int *row_ptr, int *col_idx, long double *val, long double *x, long double *b)
// {
//     long double *b_new = (long double *)malloc(sizeof(long double) * n);
//     long double *check_b = (long double *)malloc(sizeof(long double) * n);
//     spmv(n, row_ptr, col_idx, val, x, b_new);
//     int i ;
//     for (i = 0; i < n; i++)
//         check_b[i] = b_new[i] - b[i];

//     double answer1 = vec2norm(check_b, n);
//     //double answer2 = max_check(check_b, n);
//     double answer3 = answer1 / vec2norm(b, n);
//     fprintf(stdout, "Check || AX - B || 2             =  %12.6e\n", answer1);
//     //fprintf(stdout, "Check || AX - B || MAX           =  %12.6e\n", answer2);
//     fprintf(stdout, "Check || AX - B || 2 / || B || 2 =  %12.6e\n", answer3);

//     free(b_new);
//     free(check_b);
// }


//store x to a file
void store_x(Int n, Real *x, char *filename)
{
    FILE *p = fopen(filename, "w");
    fprintf(p, "%lld\n", n);
    Int i;
    for (i = 0; i < n; i++)
        fprintf(p, "%Lf\n", x[i]);
    fclose(p);
}

//load right-hand side vector b
void load_b(Int n, Real *b, char *filename)
{
    FILE *p = fopen(filename, "r");
    Int n_right;
    Int r = fscanf(p, "%lld", &n_right);
    // printf("n_right: %lld, n: %lld.\n", n_right, n);
    if (n_right != n)
    {
        fclose(p);
        printf("Invalid size of b.\n");
        return;
    }
    Int i;
    for (i = 0; i < n_right; i++)
        r = fscanf(p, "%Lf", &b[i]);
    fclose(p);
}
