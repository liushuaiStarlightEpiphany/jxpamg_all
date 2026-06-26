/*
This is the reference code of SolverChallenge to optimize by the participants
@Code version: 1.0
@Update date: 2021/5/17
@Author: Dechuang Yang, Haocheng Lian

Added some parallel packages: jxpamg
@Update date: 2021/5/28
@Author: Li Zhao
*/

#define HAVE_OMP 1
#define DEBUG_MODE 0
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include "mmio_highlevel.h"
#include "my_solver.h"
#include "multi_scale_atom.h"

#ifdef HAVE_OMP
#include "omp.h"
#endif

//double check_correctness(int n, int *row_ptr, int *col_idx, double *val, double *x, double *x0, double *b, double *res);
double check_correctness(int n, int *row_ptr, int *col_idx, double *val, double *x, double *b, double *res);
void store_x(int n, double *x, char *filename);
void load_x(int n, double *x, char *filename);
void load_b(int n, double *b, char *filename);
void Pb8MatrixCheck(int n, int nnz, int *ia, int *ja, double *aa, double *bb);
void PerRowNNZCheck(int probID, int n, int nnz, int *ia, int *ja, double *aa);
void CheckMatrixPN( int n, int nnz, int *ia, int *ja, double *aa);
int FASP_Matrix_Check(int probID, int row, int nnz, int *ia, int  *ja, double *aa, double *bb, int isPlot, int isWriteAb);
int FASP_Solver_Interface(int row, int nnz, int *ia, int  *ja, double *aa, double *ser_x, double *ser_b, int solver_type,  int precond_type, double tol, int *iter);
int FASP_BSRSOL_Interface(int row, int nnz, int nb, int *ia, int  *ja, double *aa, double *ser_x, double *ser_b, int itmethod,  double tol, int *iter);

int main(int argc, char **argv)
{
    printf("1\n");

    
    int     m, isSymmetricA;
    int     *row_ptr; // the csr row pointer array of matrix A
    int     *col_idx; // the csr column index array of matrix A
    double  *val;  // the csr value array of matrix A
    double  *x, *b,*x0; 
    char    *filename_matrix = argv[1];  // the filename of matrix A
    char    *nrows = argv[2];       // type 0: CSR matrix and vector; type 1: binary matrix and vector
    char    *number_of_zeros = argv[3];       // type 0: CSR matrix and vector; type 1: binary matrix and vector
    char    *filename_x = argv[4];       // the filename of solution vector x
    char    *matrix_tolerance = argv[5]; // the tolerance of input matrix
    char    *type_str = argv[6];         // work type 
    char    *sid = argv[7];              // solver id
    char    *pid = argv[8];              // precond id
    char    *nbb = argv[9];              // the number of blocks
    char    *pbb = argv[10];              // the number of blocks

    int     iter = 0;                    // the number of iterations
    double  residule = 0.0, rrn = 0.0;
    double  tolerance = atof(matrix_tolerance);
    printf("3\n");
    
    int     myid;
    int     nthreads;
    int     i;
    int     n           = atoi(nrows);
    int     nnzA        = atoi(number_of_zeros);
    int     type        = atoi(type_str);
    int     solver_id   = atoi(sid);
    int     precond_id  = atoi(pid);
    int     nb          = atoi(nbb);
    int     pb          = atoi(pbb);
    int     probem_id   = precond_id;
    int     isPlot      = solver_id;
    int     isWriteAb   = nb;
    printf("2\n");
#ifdef HAVE_OMP
#pragma omp parallel
{
    nthreads = omp_get_num_threads();
}
#endif

    printf("===================================================================\n");
    printf("1 A   : %s\n", filename_matrix);
    printf("2.1 n   : %d\n", n);
    printf("2.2 nnz   : %d\n", nnzA);
    printf("3 x   : %s\n", filename_x);
    printf("4 tol : %s\n", matrix_tolerance);
    printf("5 type: %d (1: FASP_CSRSOL, 2: FASP_BSRSOL, 3: Matrix_Check)\n", type);
    printf("6 sid : %d (solver id, or Is Plot Matrix picture: isPlot)\n", solver_id);
    printf("7 pid : %d (preconde id, or Matrix Check: probem_id)\n", precond_id);
    printf("8 nb  : %d (the number of blocks, or Is write matrix and rhs)\n", nb);
    printf("===================================================================\n");
    
#ifdef HAVE_OMP    
    printf("\n\n+++++++++++++++++++++ Number of threads %d +++++++++++++++++++++\n\n", nthreads);
#endif

    //load matrix
    //mmio_allinone(&m, &n, &nnzA, &isSymmetricA, &row_ptr, &col_idx, &val, filename_matrix);
    // mmio_allinone(&m, &n, &nnzA, &isSymmetricA, &row_ptr, &col_idx, &val, filename_matrix);

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
        // b = (double *)malloc(sizeof(double) * n);
        // b = (double *)malloc(sizeof(double) * n);
        // b = (double *)malloc(sizeof(double) * n);

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

    m=n;
    printf("matrix info: m: %d, n: %d, nnz: %d, isSymmertirc: %d\n", m, n, nnzA, isSymmetricA);
    if (m != n)
    {
        printf("Invalid matrix size.\n");
        return 0;
    }

    x = (double *)malloc(sizeof(double) * n);

    // // load right-hand side vector b
    // printf("load right-hand side vector:\n");
    // load_b(n, b, filename_b);

    // // load x
    // printf("load initial guess vector x:\n");
    // load_x(n, x, filename_x);

    // // load x
    // printf("load initial guess vector x:\n");
    // load_x(n, x0, filename_x);

    /* // load x
    printf("load initial vector x:\n");
    load_x(n, x, filename_x); */

    // 初值为0
    for (i = 0; i < n; i++) x[i] = 0; 
    // for (i = 0; i < n; i++) x[i] = 0; 

    // check the norm of the right-hand side vector b and r0
// printf("\n\n==============DEBUG START===============\n");
#if DEBUG_MODE > 0
    double norm_b = 0;
    norm_b = vec2norm(b, m);
    printf("### DEBUG: the norm of b is %e\n", norm_b);
    double max_b = fabs(b[0]), min_b = fabs(b[0]);
    for (i=1; i<m; i++)
        if(fabs(b[i]) > max_b) max_b = fabs(b[i]);
        else if (fabs(b[i]) < min_b) min_b = fabs(b[i]); 
    printf("The max element of b is %e, and the min element of b is %e\n", max_b, min_b); 

    double norm_x0;
    norm_x0 = vec2norm(x, m);
    printf("### DEBUG: the norm of x0 is %e\n", norm_x0);
    double max_x0 = fabs(x[0]), min_x0 = fabs(x[0]);
    for (i=1; i<m; i++)
        if(fabs(x[i]) > max_x0) max_x0 = fabs(x[i]);
        else if (fabs(x[i]) < min_x0) min_x0 = fabs(x[i]); 
    printf("The max element of x0 is %e, and the min element of x0 is %e\n", max_x0, min_x0); 

    double norm_r0;
    double *b0 = (double *)malloc(sizeof(double) * m);
    double *r0 = (double *)malloc(sizeof(double) * m);
    spmv(n, row_ptr, col_idx, val, x, b0);
    for (i = 1; i < n; i++)
        r0[i] = b[i] - b0[i];
    norm_r0 = vec2norm(r0, n);
    printf("### DEBUG: the norm of r0 is %e\n", norm_r0);
    double max_r0 = fabs(r0[0]), min_r0 = fabs(r0[0]);
    for (i=0; i<m; i++)
        if(fabs(r0[i]) > max_r0) max_r0 = fabs(r0[i]);
        else if(fabs(r0[i]) < min_r0) min_r0 = fabs(r0[i]); 
    printf("The max element of  r0 is %e, and the min element of r0 is %e\n", max_r0, min_r0); 
printf("===============DEBUG END================\n\n");
#endif

    // solve x and record wall-time
    struct timeval t_start, t_stop;
    gettimeofday(&t_start, NULL);

    // my_solver(n, row_ptr, col_idx, val, x, b, &iter, tolerance);
 
    if (type == 1)
    {
        //-------------------------------------------------------------------------
        //  faspsolver
        //------------------------------------------------------------------------- 
        printf("FASP_CSRSOL\n");    
        FASP_Solver_Interface(n, nnzA, row_ptr, col_idx, val, x, b, solver_id, precond_id, tolerance, &iter);
    }else if (type == 2)
    {
        //-------------------------------------------------------------------------
        //  fasp4blkoil
        //-------------------------------------------------------------------------    
        printf("FASP_BSRSOL, nb = %d\n", nb);
        FASP_BSRSOL_Interface(n, nnzA, nb, row_ptr, col_idx, val, x, b, solver_id, tolerance, &iter);

    }else{
        printf("FASP_Matrix_Check\n"); 

        // 将 CSR 对角元优先
        CSRMatrixDiagonalElementFirst( n, nnzA, row_ptr, col_idx, val );

        // 检测矩阵对角元和非对角元的正负情况、个数
        CheckMatrixPN(n, nnzA, row_ptr, col_idx, val);

        //FASP矩阵性态检测
        FASP_Matrix_Check(probem_id, n, nnzA, row_ptr, col_idx, val, b, isPlot, isWriteAb);
        // Pb8MatrixCheck(n, nnzA, row_ptr, col_idx, val, b);
        
        //每行非零元检测
        PerRowNNZCheck(probem_id, n, nnzA, row_ptr, col_idx, val);
        
        // Check the Multi-Scale Property   
        DoubleCSRMatrixMultiScaleProperty( n, row_ptr, col_idx, val, 1 );
        DoubleCSRMatrixMultiScaleProperty( n, row_ptr, col_idx, val, 2 );
        DoubleCSRMatrixMultiScaleProperty( n, row_ptr, col_idx, val, 3 );
        DoubleCSRMatrixMultiScaleProperty( n, row_ptr, col_idx, val, 4 );
        DoubleCSRMatrixMultiScaleProperty( n, row_ptr, col_idx, val, 5 );
        DoubleCSRMatrixMultiScaleProperty( n, row_ptr, col_idx, val, 6 );
        DoubleCSRMatrixMultiScaleProperty( n, row_ptr, col_idx, val, 7 );
        DoubleCSRMatrixMultiScaleProperty( n, row_ptr, col_idx, val, 8 );

        //检索给定元素范围所在文件行（绝对值意义下）
        SearchRows(n, row_ptr, col_idx, val);
    }
    

    gettimeofday(&t_stop, NULL);
    double total_time = (t_stop.tv_sec - t_start.tv_sec) * 1000.0 + (t_stop.tv_usec - t_start.tv_usec) / 1000.0;
    
    //store x to a file
    store_x(n, x, "x.txt");

    //check the correctness
    //print the #iteration, residual and total_time
    rrn = check_correctness(n, row_ptr, col_idx, val, x, b, &residule);
    // rrn = check_correctness(n, row_ptr, col_idx, val, x, x0, b, &residule);
    printf("\n\n===============================================================================\n");
    printf("iteration = %d, residule = %e, RRN = %e, total_time = %.5lf sec\n", iter, residule, rrn, total_time / 1000.0);
    printf("===============================================================================\n");
    
    return 0;
}

/* //validate the x by b-A*x
double check_correctness(int n, int *row_ptr, int *col_idx, double *val, double *x, double *x0, double *b, double *res)
{
    double b_L2 = vec2norm(b, n);
    double *b_new = (double *)malloc(sizeof(double) * n);
    double *b_init = (double *)malloc(sizeof(double) * n);
    double *check_b = (double *)malloc(sizeof(double) * n);
    double *check_b0 = (double *)malloc(sizeof(double) * n);
    spmv(n, row_ptr, col_idx, val, x, b_new);
    spmv(n, row_ptr, col_idx, val, x0, b_init);
    int i;
    for (i = 0; i < n; i++)
    {
        check_b[i] = b_new[i] - b[i];
        check_b0[i] = b_init[i] - b[i];
    }
    *res = vec2norm(check_b, n);  
    return *res / vec2norm(check_b0, n);
} */


//validate the x by b-A*x
double check_correctness(int n, int *row_ptr, int *col_idx, double *val, double *x, double *b, double *res)
{
    double b_L2 = vec2norm(b, n);
    double *b_new = (double *)malloc(sizeof(double) * n);
    double *check_b = (double *)malloc(sizeof(double) * n);
    spmv(n, row_ptr, col_idx, val, x, b_new);
    int i;
    for (i = 0; i < n; i++)
        check_b[i] = b_new[i] - b[i];
    *res = vec2norm(check_b, n);    
    return *res / b_L2;
}


//store x to a file
void store_x(int n, double *x, char *filename)
{
    FILE *p = fopen(filename, "w");
    fprintf(p, "%d\n", n);
    int i;
    for (i = 0; i < n; i++)
        fprintf(p, "%lf\n", x[i]);
    fclose(p);
}

//load right-hand side vector b
void load_b(int n, double *b, char *filename)
{
    FILE *p = fopen(filename, "r");
    int n_right;
    int r = fscanf(p, "%d", &n_right);
    if (n_right != n)
    {
        fclose(p);
        printf("Invalid size of b.\n");
        return;
    }
    int i;
    for (i = 0; i < n_right; i++)
        r = fscanf(p, "%lf", &b[i]);
    fclose(p);
}

// load x 开发原子大赛
void load_x(int n, double *x, char *filename)
{
    FILE *p = fopen(filename, "r");
    int n_right;
    int r = fscanf(p, "%d", &n_right);
    if (n_right != n)
    {
        fclose(p);
        printf("Invalid size of x.\n");
        return;
    }

    int i;
    int number;
    double ix_tmp, iy_tmp, iz_tmp;
    for (i = 0; i < n_right; i++)
        r = fscanf(p, "%d %lf %lf %lf %lf", &number, &x[i], &ix_tmp, &iy_tmp, &iz_tmp);
    fclose(p);

    /* int i, number;
    for (i = 0; i < n_right; i++)
        r = fscanf(p, "%d %lf\n", &number, &x[i]);
    fclose(p);  */

}

void PerRowNNZCheck(int probID, int n, int nnz, int *ia, int *ja, double *aa)
{
    int i,k,j;
    int istart, iend, rowMax = 0, rowMin = 1e+5;
    double val;

    int *per_row_num = (int *)malloc(sizeof(int) * n);

    for ( i = 0; i < n; i++)
    {
        per_row_num[i] = 0;

        istart = ia[i]; 
        iend   = ia[i+1];
        for ( k = istart; k < iend; k++)
        {
            j = ja[k];
            val = aa[k];
            if (val != 0)
            {
                per_row_num[i]++;
            }
        }
        if (per_row_num[i] != 0)
        {
            // 统计每行非零元的最大数和最小数
            if (per_row_num[i] > rowMax)
            {
                rowMax = per_row_num[i];
            }else if(per_row_num[i] < rowMin){
                rowMin = per_row_num[i];
            }
        }
        
    }
    
    printf("\n\nper_row_nonzeros_min: %d, per_row_nonzeros_max: %d\n", rowMin, rowMax);
#if 0
    char file[128]; 
    sprintf(file, "out/per_row_num_%d.out", probID);  
    printf("Output the matrix file: matfile = %s\n", file);  
    FILE *fid = fopen(file, "w");
    
    for ( i = 0; i < n; i++)
    {
        fprintf(fid, "%d    %d\n", i, per_row_num[i]);
    }
    fclose(fid);
    printf("write out/per_row_num.txt, finish!!!\n");
#endif
    free(per_row_num);
}


// 检测矩阵对角元和非对角元的正负情况、个数
void CheckMatrixPN( int n, int nnz, int *ia, int *ja, double *aa)
{
    int i,k,j;
    int istart, iend, diag0 = 0, diagP = 0, diagN = 0, non_diagP = 0, non_diagN = 0;
    double val;

    for ( i = 0; i < n; i++)
    {
        istart = ia[i]; 
        iend   = ia[i+1];
        for ( k = istart; k < iend; k++)
        {
            j   = ja[k];
            val = aa[k];

            if (j == i) //对角元
            {
                if (val > 0) diagP++;
                else if(val < 0){
                    diagN++;
                }else{
                    diag0++;
                }
            }else{ // 非对角元
                if (val > 0) non_diagP++;
                else if(val < 0){
                    non_diagN++;
                }
            }
        } 
    }
    printf("\n检测矩阵对角元和非对角元的正负情况、个数（矩阵阶数：%d，非零元：%d）\n", n, nnz);
    printf("正对角元个数：%d, 负对角元个数：%d, 零对角元个数：%d\n", diagP, diagN, diag0);
    printf("正非对角元个数：%d, 负非对角元个数：%d\n", non_diagP, non_diagN);
    printf("对角元+非对角元总个数（非零元）：%d\n", diagP+diagN+non_diagP+non_diagN);
}


void Pb8MatrixCheck(int n, int nnz, int *ia, int *ja, double *aa, double *bb)
{
    int bound = 3114327;//3103639;
    int i,k,j;
    int istart, iend;
    double val;

    // PerRowNNZCheck(00, n, nnz, ia, ja, aa);

    // 1-3103639
    // A   B
    // C   D
    
    // A
    int nzA = 0, *iaA, *jaA;
    int rowA = bound;
    double *aaA, *bA;

    bA  = (double *)malloc(sizeof(double) * rowA);
    iaA = (int *)malloc(sizeof(int) * (bound+1));
    iaA[0] = 0;
    for ( i = 0; i < bound; i++)
    {
        bA[i] = bb[i];
        
        istart = ia[i]; 
        iend   = ia[i+1];
        for ( k = istart; k < iend; k++)
        {
            j = ja[k];
            if (j < bound)
            {
                val = aa[k];
                if (val != 0) nzA++;
            }
        }
        iaA[i+1] = nzA;
    }
    jaA = (int *)malloc(sizeof(int) * nzA);
    aaA = (double *)malloc(sizeof(double) * nzA);
    
    nzA = 0;
    for ( i = 0; i < bound; i++)
    {
        istart = ia[i]; 
        iend   = ia[i+1];
        for ( k = istart; k < iend; k++)
        {
            j = ja[k];
            val = aa[k];
            if (j < bound)
            {
                if (val != 0){
                    jaA[nzA] = j;
                    aaA[nzA] = val;
                    nzA++;
                }
            }
        }
    }

    printf("FASP_Matrix_Check(A)\n");     
    FASP_Matrix_Check(11, rowA, nzA, iaA, jaA, aaA, bA, 0, 0);
    // PerRowNNZCheck(11, rowA, nzA, iaA, jaA, aaA);


    // D
    int nzD = 0, *iaD, *jaD;
    int rowD = n - bound;
    double *aaD, *bD;

    bD  = (double *)malloc(sizeof(double) * rowD);
    iaD = (int *)malloc(sizeof(int) * (rowD+1));
    iaD[0] = 0;
    for ( i = bound; i < n; i++)
    {
        bD[i-bound] = bb[i-bound];
        istart = ia[i]; 
        iend   = ia[i+1];
        for ( k = istart; k < iend; k++)
        {
            j = ja[k];
            val = aa[k];
            if (j >= bound)
            {
                if (val != 0) nzD++;
            }
        }
        iaD[i-bound+1] = nzD;
    }
    // printf("iaD[%d] = %d, nzD = %d\n", rowD, iaD[rowD], nzD);

    jaD = (int *)malloc(sizeof(int) * nzD);
    aaD = (double *)malloc(sizeof(double) * nzD);
    
    nzD = 0;
    for ( i = bound; i < n; i++)
    {
        istart = ia[i]; 
        iend   = ia[i+1];
        for ( k = istart; k < iend; k++)
        {
            j = ja[k];
            val = aa[k];
            if (j >= bound)
            {
                if (val != 0){
                    jaD[nzD] = j - bound;
                    aaD[nzD] = aa[k];
                    nzD++;
                }
            }
        }
    }
    // printf("nzD = %d\n", nzD);
    printf("\n\nFASP_Matrix_Check(D)\n");     
    FASP_Matrix_Check(22, rowD, nzD, iaD, jaD, aaD, bD, 0, 0);
    PerRowNNZCheck(22, rowD, nzD, iaD, jaD, aaD);

    // free 
    free(iaA);
    free(jaA);
    free(aaA);
    free(bA);
    free(iaD);
    free(jaD);
    free(aaD);
    free(bD);
}


#if 0
// 先几何节点后物理量(3个量)，转化为先物理量后几何节点
int TranMatrixOrder( int n, int nnz, int *ia, int *ja, double *aa , double *b)
{
    int i, k, j, ik;
    int istart, iend;
    double val;
    int *ia0, *ja0;
    double *aa0, *b0;
    int nnm = n/3;

    if (n%3 != 0)
    {
        printf("\033[31mERROR:\033[00m the number of rows in a matrix cannot be divided by 3!!!\n");
        exit(-1);
    }

    ia0 = (int *)malloc(sizeof(int) * (n+1));
    ja0 = (int *)malloc(sizeof(int) * nnz);
    aa0 = (double *)malloc(sizeof(double) * nnz);
    b0  = (double *)malloc(sizeof(double) * n);
	

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
    
    
    int li, lj;
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

    // clean up
    free(ia);
    free(ja);
    free(aa);
    free(b);

    // output
    ia = ia0;
    ja = ja0;
    aa = aa0;
     b = b0;

    // return
    return 0;
}

#endif
