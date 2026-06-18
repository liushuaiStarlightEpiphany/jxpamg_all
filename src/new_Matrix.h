#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
// #include "utlise.h"


// void spmv(int n, int *row_ptr, int *col_idx, double *val, double *x, double *y);

int matrix(int n, int nnzA, int* mark, double* b, int* row_ptr, int* col_idx, double* val, 
           int** row_ptr_a11_n, int** col_idx_a11_n, double** val_a11_n, 
           int** row_ptr_a12_n, int** col_idx_a12_n, double** val_a12_n, 
           int** row_ptr_a21_n, int** col_idx_a21_n, double** val_a21_n, 
           int** row_ptr_a22_n, int** col_idx_a22_n, double** val_a22_n, 
           double** b1_n, double** b2_n) {
    int i, j;

    double *b_temp = NULL; // right-hand side vector b, (b: real number, b_im: imaginary number)
    double *b1_temp = NULL; // right-hand side vector b, (b: real number, b_im: imaginary number)
    double *b0_temp = NULL; // right-hand side vector b, (b: real number, b_im: imaginary number)

    int countb12 = 0;  
    int* countrowb12 = NULL;     
    int* countrowb12_00 = NULL;     

    int countb12_1 = 0;  
    int* countrowb12_0 = NULL;     
    int* countrowb12_1 = NULL;     

    int* col_idx_a11_temp = NULL;         // the csr column index array of matrix A
    double* val_a11_temp = NULL;         // the csr value array of matrix A (real number)
    int nnzAa11 = 0;

    int* col_idx_a12_temp = NULL;         // the csr column index array of matrix A
    double* val_a12_temp = NULL;         // the csr value array of matrix A (real number)
    int nnzAa12 = 0;

    int* col_idx_a21_temp = NULL;         // the csr column index array of matrix A
    double* val_a21_temp = NULL;         // the csr value array of matrix A (real number)
    int nnzAa21 = 0;

    int* col_idx_a22_temp = NULL;         // the csr column index array of matrix A
    double* val_a22_temp = NULL;         // the csr value array of matrix A (real number)
    int nnzAa22 = 0;

    double *x = NULL;
    double *x1 = NULL; 
    double *x2 = NULL;

    countrowb12 = (int*)malloc(n * sizeof(int));
    countrowb12_00 = (int*)malloc(n * sizeof(int));
    countrowb12_0 = (int*)malloc(n * sizeof(int));
    countrowb12_1 = (int*)malloc(n * sizeof(int));

    for(i = 0; i < n; i++) {
        countrowb12[i] = -1;
        countrowb12_1[i] = -1;
        countrowb12_00[i] = -1;
        countrowb12_0[i] = -1;
    }

    b_temp = (double*)malloc(sizeof(double) * n);
    b1_temp = (double*)malloc(sizeof(double) * n);
    for(i = 0; i < n; i++) {
        b_temp[i] = 0.0;
        b1_temp[i] = 0.0;
    }

    countb12 = 0;
    countb12_1 = 0;

    for (i = 0; i < n; i++) {
        if(mark[i] == 1) {
            countrowb12_00[countb12] = i;
            countrowb12[i] = countb12;
            b_temp[countb12] = b[i];
            countb12 += 1;
        } else if(mark[i] == 2) {
            countrowb12_0[countb12_1] = i;
            countrowb12_1[i] = countb12_1;
            b1_temp[countb12_1] = b[i];
            countb12_1 += 1;
        } else {
            fprintf(stdout, "error mark.\n");
        }
    }
   fprintf(stdout, "test a11: n,n.   %d    %d\n",countb12,countb12_1);   
    
    double *b1 = (double*)malloc(sizeof(double) * countb12);
    double *b2 = (double*)malloc(sizeof(double) * countb12_1);

    memcpy(b1, b_temp, sizeof(double) * countb12);
    memcpy(b2, b1_temp, sizeof(double) * countb12_1);

    free(b_temp);
    free(b1_temp);

    // �ֿ����A11��A12
    int* row_ptr_a11 = (int*)malloc((countb12 + 1) * sizeof(int));
    int* row_ptr_a12 = (int*)malloc((countb12 + 1) * sizeof(int));
    col_idx_a11_temp = (int*)malloc(nnzA * sizeof(int));
    col_idx_a12_temp = (int*)malloc(nnzA * sizeof(int));
    val_a11_temp = (double*)malloc(nnzA * sizeof(double));
    val_a12_temp = (double*)malloc(nnzA * sizeof(double));

    nnzAa11 = 0;
    nnzAa12 = 0;

    // ��ȡ�ֿ����A11��A12
    for (i = 0; i < countb12; i++) {
        row_ptr_a11[i] = nnzAa11;
        row_ptr_a12[i] = nnzAa12;        
        int row_temp = countrowb12_00[i];
        for (j = row_ptr[row_temp]; j < row_ptr[row_temp + 1]; j++) {
            int clo_temp = col_idx[j];
            if(mark[clo_temp] == 1) {
                col_idx_a11_temp[nnzAa11] = countrowb12[clo_temp];
                val_a11_temp[nnzAa11] = val[j];
                nnzAa11 += 1;
            } else if(mark[clo_temp] == 2) {
                col_idx_a12_temp[nnzAa12] = countrowb12_1[clo_temp];
                val_a12_temp[nnzAa12] = val[j];
                nnzAa12 += 1;    
            } else {
                fprintf(stdout, "error mark.\n");
            }
        }
    }
    row_ptr_a11[countb12] = nnzAa11;
    row_ptr_a12[countb12] = nnzAa12;        
    int* col_idx_a11 = (int*)malloc(nnzAa11 * sizeof(int));
    double* val_a11 = (double*)malloc(nnzAa11 * sizeof(double));
    int* col_idx_a12 = (int*)malloc(nnzAa12 * sizeof(int));
    double* val_a12 = (double*)malloc(nnzAa12 * sizeof(double));

//    for (i=0;i<countb12;i++){
//            for (j=row_ptr_a11[i];j<row_ptr_a11[i+1];j++){
//                    fprintf(stdout, "a11.  %d   %d    %d   %lf  %d \n",j,i,col_idx_a11_temp[j],val_a11_temp[j],countrowb12_00[i]);
//            }
//            if(i>10)return 0;
//    }

    memcpy(col_idx_a11, col_idx_a11_temp, sizeof(int) * nnzAa11);
    memcpy(val_a11, val_a11_temp, sizeof(double) * nnzAa11);
    memcpy(col_idx_a12, col_idx_a12_temp, sizeof(int) * nnzAa12);
    memcpy(val_a12, val_a12_temp, sizeof(double) * nnzAa12);

    free(col_idx_a11_temp);
    free(val_a11_temp);
    free(col_idx_a12_temp);
    free(val_a12_temp);

    // �ֿ����A21��A22
    int* row_ptr_a21 = (int*)malloc((countb12_1 + 1) * sizeof(int));
    int* row_ptr_a22 = (int*)malloc((countb12_1 + 1) * sizeof(int));
    col_idx_a21_temp = (int*)malloc(nnzA * sizeof(int));
    col_idx_a22_temp = (int*)malloc(nnzA * sizeof(int));
    val_a21_temp = (double*)malloc(nnzA * sizeof(double));
    val_a22_temp = (double*)malloc(nnzA * sizeof(double));
    nnzAa21 = 0;
    nnzAa22 = 0;

    // ��ȡ�ֿ����A21��A22
    for (i = 0; i < countb12_1; i++) {
        row_ptr_a21[i] = nnzAa21;
        row_ptr_a22[i] = nnzAa22;        
        int row_temp = countrowb12_0[i];
        for (j = row_ptr[row_temp]; j < row_ptr[row_temp + 1]; j++) {
            int clo_temp = col_idx[j];
            if(mark[clo_temp] == 1) {
                col_idx_a21_temp[nnzAa21] = countrowb12[clo_temp];
                val_a21_temp[nnzAa21] = val[j];
                nnzAa21 += 1;
            } else if(mark[clo_temp] == 2) {
                col_idx_a22_temp[nnzAa22] = countrowb12_1[clo_temp];
                val_a22_temp[nnzAa22] = val[j];
                nnzAa22 += 1;    
            } else {
                fprintf(stdout, "error mark.\n");
            }
        }
    }
    row_ptr_a21[countb12_1] = nnzAa21;
    row_ptr_a22[countb12_1] = nnzAa22;        
    int* col_idx_a21 = (int*)malloc(nnzAa21 * sizeof(int));
    double* val_a21 = (double*)malloc(nnzAa21 * sizeof(double));
    int* col_idx_a22 = (int*)malloc(nnzAa22 * sizeof(int));
    double* val_a22 = (double*)malloc(nnzAa22 * sizeof(double));

    memcpy(col_idx_a21, col_idx_a21_temp, sizeof(int) * nnzAa21);
    memcpy(val_a21, val_a21_temp, sizeof(double) * nnzAa21);
    memcpy(col_idx_a22, col_idx_a22_temp, sizeof(int) * nnzAa22);
    memcpy(val_a22, val_a22_temp, sizeof(double) * nnzAa22);

    free(col_idx_a21_temp);
    free(val_a21_temp);
    free(col_idx_a22_temp);
    free(val_a22_temp);
    
    *col_idx_a11_n = col_idx_a11; 
    *col_idx_a12_n = col_idx_a12; 
    *col_idx_a21_n = col_idx_a21; 
    *col_idx_a22_n = col_idx_a22; 
      
    *val_a11_n = val_a11;
    *val_a12_n = val_a12;
    *val_a21_n = val_a21;
    *val_a22_n = val_a22;
    
    *row_ptr_a11_n = row_ptr_a11;
    *row_ptr_a12_n = row_ptr_a12;
    *row_ptr_a21_n = row_ptr_a21;
    *row_ptr_a22_n = row_ptr_a22;

    *b1_n = b1;
    *b2_n = b2;
    
    // ��֤�ֿ�������ȷ��
    double tol_test = 1e-3;

    x = (double*)malloc(sizeof(double) * n);
    x1 = (double*)malloc(sizeof(double) * countb12);
    x2 = (double*)malloc(sizeof(double) * countb12_1);

    b_temp = (double*)malloc(sizeof(double) * countb12);
    b1_temp = (double*)malloc(sizeof(double) * countb12);
    b0_temp = (double*)malloc(sizeof(double) * n);

    for (i = 0; i < n; i++) x[i] = i;
    for (i = 0; i < countb12; i++) x1[i] = x[countrowb12_00[i]];
    for (i = 0; i < countb12_1; i++) x2[i] = x[countrowb12_0[i]];

    for (i = 0; i < countb12; i++) b_temp[i] = 0.0;
    for (i = 0; i < countb12; i++) b1_temp[i] = 0.0;
    for (i = 0; i < n; i++) b0_temp[i] = 0.0;

    spmv(countb12, row_ptr_a11, col_idx_a11, val_a11, x1, b_temp);
    spmv(countb12, row_ptr_a12, col_idx_a12, val_a12, x2, b1_temp);

    spmv(n, row_ptr, col_idx, val, x, b0_temp);

    fprintf(stdout, "tol_test=   %lf\n", tol_test);
    for (i = 0; i < countb12; i++) {
        double atemp1 = b_temp[i] + b1_temp[i];
        double atemp2 = b0_temp[countrowb12_00[i]];
        double atemp3 = fabs(atemp2 - atemp1);
        if (atemp3 > tol_test) {
            fprintf(stdout, "test a11,a12 .%d , %lf ,%lf , %lf\n", i, b_temp[i], b1_temp[i], b0_temp[countrowb12_00[i]]);
            for (j = row_ptr_a11[i]; j < row_ptr_a11[i + 1]; j++) {
                fprintf(stdout, "error val_a11.   %d    %d   %d   %d  %d  %lf\n", i, j, col_idx_a11[j], countrowb12_00[i], countrowb12_00[col_idx_a11[j]], val_a11[j]);
            }
            for (j = row_ptr_a12[i]; j < row_ptr_a12[i + 1]; j++) {
                fprintf(stdout, "error val_a12.   %d  %d   %d  %d   %d    %lf\n", i, j, col_idx_a12[j], countrowb12_00[i], countrowb12_0[col_idx_a12[j]], val_a12[j]);
            }
            if (i > 5) return 0;
        }
    }
    fprintf(stdout, "test A11&A12 ok\n");
    free(b_temp);
    free(b1_temp);

    // ��֤�ֿ�������ȷ��
    b_temp = (double*)malloc(sizeof(double) * countb12_1);
    b1_temp = (double*)malloc(sizeof(double) * countb12_1);

    for (i = 0; i < countb12_1; i++) b_temp[i] = 0.0;
    for (i = 0; i < countb12_1; i++) b1_temp[i] = 0.0;

    spmv(countb12_1, row_ptr_a21, col_idx_a21, val_a21, x1, b_temp);
    spmv(countb12_1, row_ptr_a22, col_idx_a22, val_a22, x2, b1_temp);

    for (i = 0; i < countb12_1; i++) {
        double atemp1 = b_temp[i] + b1_temp[i];
        double atemp2 = b0_temp[countrowb12_0[i]];
        double atemp3 = fabs(atemp2 - atemp1);
        if (atemp3 > tol_test) {
            fprintf(stdout, "test a11,a12 .%d , %lf ,%lf , %lf\n", i, b_temp[i], b1_temp[i], b0_temp[countrowb12_0[i]]);
            for (j = row_ptr_a21[i]; j < row_ptr_a21[i + 1]; j++) {
                fprintf(stdout, "error val_a11.   %d    %d   %d   %d  %d  %lf\n", i, j, col_idx_a21[j], countrowb12_0[i], countrowb12_00[col_idx_a21[j]], val_a21[j]);
            }
            for (j = row_ptr_a12[i]; j < row_ptr_a12[i + 1]; j++) {
                fprintf(stdout, "error val_a12.   %d  %d   %d  %d   %d    %lf\n", i, j, col_idx_a12[j], countrowb12_0[i], countrowb12_0[col_idx_a12[j]], val_a12[j]);
            }
            if (i > 5) return 0;
        }
    }
    fprintf(stdout, "test A21&A22 ok\n");
    free(b_temp);
    free(b1_temp);
    free(b0_temp);
    free(x);
    free(x1);
    free(x2);    

    free(countrowb12);
    free(countrowb12_0);
    free(countrowb12_1);
    free(countrowb12_00);
    return 0;
}
