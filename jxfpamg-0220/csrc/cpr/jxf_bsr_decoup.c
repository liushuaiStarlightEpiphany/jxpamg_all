#include "jxf_bsr_decoup.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// 内部辅助宏和函数
#define JXF_MIN(a, b) ((a) < (b) ? (a) : (b))
#define JXF_MAX(a, b) ((a) > (b) ? (a) : (b))
#define JXF_ABS(x) ((x) >= 0 ? (x) : -(x))

// 交换两个数
#define JXF_SWAP(a, b) \
    do { \
        JXF_Real temp = (a); \
        (a) = (b); \
        (b) = temp; \
    } while (0)

// 内部辅助函数声明
static void jxf_smat_inv_2x2(JXF_Real* a);
static void jxf_smat_inv_3x3(JXF_Real* a);
static void jxf_smat_inv_4x4(JXF_Real* a);
static void jxf_smat_inv_nc(JXF_Real* a, JXF_Int n);
static void jxf_smat_identity(JXF_Real* A, JXF_Int n);
static void jxf_smat_mul(JXF_Real* A, JXF_Real* B, JXF_Real* C, JXF_Int n);
static void jxf_smat_vec_mul(JXF_Real* A, JXF_Real* b, JXF_Real* c, JXF_Int n);
static void jxf_smat_mul_2x2(JXF_Real* A, JXF_Real* B, JXF_Real* C);
static void jxf_smat_mul_4x4(JXF_Real* A, JXF_Real* B, JXF_Real* C);
static void jxf_smat_vec_mul_2(JXF_Real* A, JXF_Real* b, JXF_Real* c);
static void jxf_smat_vec_mul_4(JXF_Real* A, JXF_Real* b, JXF_Real* c);

static void jxf_decouple_abf(JXF_Real* val, JXF_Real* rhs, JXF_Int* rpt, 
                           JXF_Int* cpt, JXF_Int nb, JXF_Int nrow);
static void jxf_decouple_abf_2x2(JXF_Real* val, JXF_Real* rhs, JXF_Int* rpt, 
                               JXF_Int* cpt, JXF_Int nrow);
static void jxf_decouple_abf_4x4(JXF_Real* val, JXF_Real* rhs, JXF_Int* rpt, 
                               JXF_Int* cpt, JXF_Int nrow);
static void jxf_decouple_abf_nb(JXF_Real* val, JXF_Real* rhs, JXF_Int* rpt, 
                              JXF_Int* cpt, JXF_Int nb, JXF_Int nrow);

static void jxf_decouple_anl(JXF_Real* val, JXF_Real* rhs, JXF_Int* rpt, 
                           JXF_Int* cpt, JXF_Int nb, JXF_Int nrow, JXF_Int is_thermal);
static void jxf_decouple_sem(JXF_Real* val, JXF_Real* rhs, JXF_Int* rpt, 
                           JXF_Int* cpt, JXF_Int nb, JXF_Int nrow, JXF_Int is_thermal);
static void jxf_decouple_qi(JXF_Real* val, JXF_Real* rhs, JXF_Int* rpt, 
                          JXF_Int* cpt, JXF_Int nb, JXF_Int nrow, JXF_Int is_thermal);

/*========================================================================*
 *                       基础矩阵运算函数（保持不变）                      *
 *========================================================================*/
// 2x2矩阵求逆
static void jxf_smat_inv_2x2(JXF_Real* a) {
    const JXF_Real a11 = a[0], a12 = a[1];
    const JXF_Real a21 = a[2], a22 = a[3];
    
    const JXF_Real det = a11 * a22 - a12 * a21;
    
    if (JXF_ABS(det) < 1e-15) {
        // 接近奇异，设为单位阵
        a[0] = 1.0; a[1] = 0.0;
        a[2] = 0.0; a[3] = 1.0;
    } else {
        JXF_Real det_inv = 1.0 / det;
        a[0] =  a22 * det_inv;
        a[1] = -a12 * det_inv;
        a[2] = -a21 * det_inv;
        a[3] =  a11 * det_inv;
    }
}

// 3x3矩阵求逆
static void jxf_smat_inv_3x3(JXF_Real* a) {
    const JXF_Real a11 = a[0], a12 = a[1], a13 = a[2];
    const JXF_Real a21 = a[3], a22 = a[4], a23 = a[5];
    const JXF_Real a31 = a[6], a32 = a[7], a33 = a[8];
    
    // 计算余子式
    const JXF_Real M11 = a22 * a33 - a23 * a32;
    const JXF_Real M12 = a21 * a33 - a23 * a31;
    const JXF_Real M13 = a21 * a32 - a22 * a31;
    const JXF_Real M21 = a12 * a33 - a13 * a32;
    const JXF_Real M22 = a11 * a33 - a13 * a31;
    const JXF_Real M23 = a11 * a32 - a12 * a31;
    const JXF_Real M31 = a12 * a23 - a13 * a22;
    const JXF_Real M32 = a11 * a23 - a13 * a21;
    const JXF_Real M33 = a11 * a22 - a12 * a21;
    
    // 计算行列式
    const JXF_Real det = a11 * M11 - a12 * M12 + a13 * M13;
    
    if (JXF_ABS(det) < 1e-15) {
        // 接近奇异，设为单位阵
        for (int i = 0; i < 9; i++) a[i] = 0.0;
        a[0] = a[4] = a[8] = 1.0;
    } else {
        JXF_Real det_inv = 1.0 / det;
        // 伴随矩阵并转置
        a[0] =  M11 * det_inv;
        a[1] = -M21 * det_inv;
        a[2] =  M31 * det_inv;
        a[3] = -M12 * det_inv;
        a[4] =  M22 * det_inv;
        a[5] = -M32 * det_inv;
        a[6] =  M13 * det_inv;
        a[7] = -M23 * det_inv;
        a[8] =  M33 * det_inv;
    }
}

// 4x4矩阵求逆（油藏模拟常用）
static void jxf_smat_inv_4x4(JXF_Real* a) {
    const JXF_Real a11 = a[0], a12 = a[1], a13 = a[2], a14 = a[3];
    const JXF_Real a21 = a[4], a22 = a[5], a23 = a[6], a24 = a[7];
    const JXF_Real a31 = a[8], a32 = a[9], a33 = a[10], a34 = a[11];
    const JXF_Real a41 = a[12], a42 = a[13], a43 = a[14], a44 = a[15];
    
    // 计算余子式
    const JXF_Real M11 = a22*a33*a44 + a23*a34*a42 + a24*a32*a43 
                       - a22*a34*a43 - a23*a32*a44 - a24*a33*a42;
    const JXF_Real M12 = a12*a34*a43 + a13*a32*a44 + a14*a33*a42 
                       - a12*a33*a44 - a13*a34*a42 - a14*a32*a43;
    const JXF_Real M13 = a12*a23*a44 + a13*a24*a42 + a14*a22*a43 
                       - a12*a24*a43 - a13*a22*a44 - a14*a23*a42;
    const JXF_Real M14 = a12*a24*a33 + a13*a22*a34 + a14*a23*a32 
                       - a12*a23*a34 - a13*a24*a32 - a14*a22*a33;
    
    const JXF_Real det = a11*M11 - a12*M12 + a13*M13 - a14*M14;
    
    if (JXF_ABS(det) < 1e-15) {
        // 接近奇异，设为单位阵
        for (int i = 0; i < 16; i++) a[i] = 0.0;
        a[0] = a[5] = a[10] = a[15] = 1.0;
    } else {
        JXF_Real det_inv = 1.0 / det;
        
        // 继续计算其他余子式
        const JXF_Real M21 = a21*a34*a43 + a23*a31*a44 + a24*a33*a41 
                           - a21*a33*a44 - a23*a34*a41 - a24*a31*a43;
        const JXF_Real M22 = a11*a33*a44 + a13*a34*a41 + a14*a31*a43 
                           - a11*a34*a43 - a13*a31*a44 - a14*a33*a41;
        const JXF_Real M23 = a11*a24*a43 + a13*a21*a44 + a14*a23*a41 
                           - a11*a23*a44 - a13*a24*a41 - a14*a21*a43;
        const JXF_Real M24 = a11*a23*a34 + a13*a24*a31 + a14*a21*a33 
                           - a11*a24*a33 - a13*a21*a34 - a14*a23*a31;
        
        const JXF_Real M31 = a21*a32*a44 + a22*a34*a41 + a24*a31*a42 
                           - a21*a34*a42 - a22*a31*a44 - a24*a32*a41;
        const JXF_Real M32 = a11*a34*a42 + a12*a31*a44 + a14*a32*a41 
                           - a11*a32*a44 - a12*a34*a41 - a14*a31*a42;
        const JXF_Real M33 = a11*a22*a44 + a12*a24*a41 + a14*a21*a42 
                           - a11*a24*a42 - a12*a21*a44 - a14*a22*a41;
        const JXF_Real M34 = a11*a24*a32 + a12*a21*a34 + a14*a22*a31 
                           - a11*a22*a34 - a12*a24*a31 - a14*a21*a32;
        
        const JXF_Real M41 = a21*a33*a42 + a22*a31*a43 + a23*a32*a41 
                           - a21*a32*a43 - a22*a33*a41 - a23*a31*a42;
        const JXF_Real M42 = a11*a32*a43 + a12*a33*a41 + a13*a31*a42 
                           - a11*a33*a42 - a12*a31*a43 - a13*a32*a41;
        const JXF_Real M43 = a11*a23*a42 + a12*a21*a43 + a13*a22*a41 
                           - a11*a22*a43 - a12*a23*a41 - a13*a21*a42;
        const JXF_Real M44 = a11*a22*a33 + a12*a23*a31 + a13*a21*a32 
                           - a11*a23*a32 - a12*a21*a33 - a13*a22*a31;
        
        // 伴随矩阵并转置
        a[0]  =  M11 * det_inv;
        a[1]  = -M21 * det_inv;
        a[2]  =  M31 * det_inv;
        a[3]  = -M41 * det_inv;
        a[4]  = -M12 * det_inv;
        a[5]  =  M22 * det_inv;
        a[6]  = -M32 * det_inv;
        a[7]  =  M42 * det_inv;
        a[8]  =  M13 * det_inv;
        a[9]  = -M23 * det_inv;
        a[10] =  M33 * det_inv;
        a[11] = -M43 * det_inv;
        a[12] = -M14 * det_inv;
        a[13] =  M24 * det_inv;
        a[14] = -M34 * det_inv;
        a[15] =  M44 * det_inv;
    }
}

// 通用n×n矩阵求逆（高斯消元法）
static void jxf_smat_inv_nc(JXF_Real* a, JXF_Int n) {
    if (n == 1) {
        a[0] = (JXF_ABS(a[0]) > 1e-15) ? 1.0 / a[0] : 1.0;
        return;
    } else if (n == 2) {
        jxf_smat_inv_2x2(a);
        return;
    } else if (n == 3) {
        jxf_smat_inv_3x3(a);
        return;
    } else if (n == 4) {
        jxf_smat_inv_4x4(a);
        return;
    }
    
    // 对于更大的块，使用高斯消元法
    JXF_Int i, j, k, max_idx;
    JXF_Real max_val, tmp;
    JXF_Int* pivot = (JXF_Int*)malloc(n * sizeof(JXF_Int));
    
    // 初始化pivot数组
    for (i = 0; i < n; i++) pivot[i] = i;
    
    // 高斯消元（部分选主元）
    for (i = 0; i < n; i++) {
        // 查找主元
        max_val = JXF_ABS(a[i*n + i]);
        max_idx = i;
        for (j = i+1; j < n; j++) {
            if (JXF_ABS(a[j*n + i]) > max_val) {
                max_val = JXF_ABS(a[j*n + i]);
                max_idx = j;
            }
        }
        
        // 如果主元太小，矩阵可能奇异
        if (max_val < 1e-15) {
            free(pivot);
            // 设为单位阵
            for (j = 0; j < n*n; j++) a[j] = 0.0;
            for (j = 0; j < n; j++) a[j*n + j] = 1.0;
            return;
        }
        
        // 交换行
        if (max_idx != i) {
            for (j = 0; j < n; j++) {
                JXF_SWAP(a[i*n + j], a[max_idx*n + j]);
            }
            JXF_SWAP(pivot[i], pivot[max_idx]);
        }
        
        // 归一化当前行
        tmp = a[i*n + i];
        for (j = 0; j < n; j++) {
            a[i*n + j] /= tmp;
        }
        
        // 消去其他行
        for (j = 0; j < n; j++) {
            if (j != i) {
                tmp = a[j*n + i];
                for (k = 0; k < n; k++) {
                    a[j*n + k] -= tmp * a[i*n + k];
                }
            }
        }
    }
    
    // 根据pivot重新排列列
    JXF_Real* temp = (JXF_Real*)malloc(n * sizeof(JXF_Real));
    for (i = 0; i < n; i++) {
        if (pivot[i] != i) {
            // 保存第i列
            for (j = 0; j < n; j++) temp[j] = a[j*n + i];
            // 复制pivot[i]列到第i列
            for (j = 0; j < n; j++) a[j*n + i] = a[j*n + pivot[i]];
            // 将temp复制到pivot[i]列
            for (j = 0; j < n; j++) a[j*n + pivot[i]] = temp[j];
            
            // 更新pivot
            for (j = i+1; j < n; j++) {
                if (pivot[j] == i) {
                    pivot[j] = pivot[i];
                    break;
                }
            }
        }
    }
    
    free(temp);
    free(pivot);
}

// 创建单位矩阵
static void jxf_smat_identity(JXF_Real* A, JXF_Int n) {
    JXF_Int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            A[i*n + j] = (i == j) ? 1.0 : 0.0;
        }
    }
}

// 矩阵乘法：C = A * B
static void jxf_smat_mul(JXF_Real* A, JXF_Real* B, JXF_Real* C, JXF_Int n) {
    JXF_Int i, j, k;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            C[i*n + j] = 0.0;
            for (k = 0; k < n; k++) {
                C[i*n + j] += A[i*n + k] * B[k*n + j];
            }
        }
    }
}

// 2x2矩阵乘法优化
static void jxf_smat_mul_2x2(JXF_Real* A, JXF_Real* B, JXF_Real* C) {
    C[0] = A[0]*B[0] + A[1]*B[2];
    C[1] = A[0]*B[1] + A[1]*B[3];
    C[2] = A[2]*B[0] + A[3]*B[2];
    C[3] = A[2]*B[1] + A[3]*B[3];
}

// 4x4矩阵乘法优化
static void jxf_smat_mul_4x4(JXF_Real* A, JXF_Real* B, JXF_Real* C) {
    C[0] = A[0]*B[0] + A[1]*B[4] + A[2]*B[8] + A[3]*B[12];
    C[1] = A[0]*B[1] + A[1]*B[5] + A[2]*B[9] + A[3]*B[13];
    C[2] = A[0]*B[2] + A[1]*B[6] + A[2]*B[10] + A[3]*B[14];
    C[3] = A[0]*B[3] + A[1]*B[7] + A[2]*B[11] + A[3]*B[15];
    
    C[4] = A[4]*B[0] + A[5]*B[4] + A[6]*B[8] + A[7]*B[12];
    C[5] = A[4]*B[1] + A[5]*B[5] + A[6]*B[9] + A[7]*B[13];
    C[6] = A[4]*B[2] + A[5]*B[6] + A[6]*B[10] + A[7]*B[14];
    C[7] = A[4]*B[3] + A[5]*B[7] + A[6]*B[11] + A[7]*B[15];
    
    C[8] = A[8]*B[0] + A[9]*B[4] + A[10]*B[8] + A[11]*B[12];
    C[9] = A[8]*B[1] + A[9]*B[5] + A[10]*B[9] + A[11]*B[13];
    C[10] = A[8]*B[2] + A[9]*B[6] + A[10]*B[10] + A[11]*B[14];
    C[11] = A[8]*B[3] + A[9]*B[7] + A[10]*B[11] + A[11]*B[15];
    
    C[12] = A[12]*B[0] + A[13]*B[4] + A[14]*B[8] + A[15]*B[12];
    C[13] = A[12]*B[1] + A[13]*B[5] + A[14]*B[9] + A[15]*B[13];
    C[14] = A[12]*B[2] + A[13]*B[6] + A[14]*B[10] + A[15]*B[14];
    C[15] = A[12]*B[3] + A[13]*B[7] + A[14]*B[11] + A[15]*B[15];
}

// 矩阵向量乘法：c = A * b
static void jxf_smat_vec_mul(JXF_Real* A, JXF_Real* b, JXF_Real* c, JXF_Int n) {
    JXF_Int i, j;
    for (i = 0; i < n; i++) {
        c[i] = 0.0;
        for (j = 0; j < n; j++) {
            c[i] += A[i*n + j] * b[j];
        }
    }
}

// 2x2矩阵向量乘法优化
static void jxf_smat_vec_mul_2(JXF_Real* A, JXF_Real* b, JXF_Real* c) {
    c[0] = A[0]*b[0] + A[1]*b[1];
    c[1] = A[2]*b[0] + A[3]*b[1];
}

// 4x4矩阵向量乘法优化
static void jxf_smat_vec_mul_4(JXF_Real* A, JXF_Real* b, JXF_Real* c) {
    c[0] = A[0]*b[0] + A[1]*b[1] + A[2]*b[2] + A[3]*b[3];
    c[1] = A[4]*b[0] + A[5]*b[1] + A[6]*b[2] + A[7]*b[3];
    c[2] = A[8]*b[0] + A[9]*b[1] + A[10]*b[2] + A[11]*b[3];
    c[3] = A[12]*b[0] + A[13]*b[1] + A[14]*b[2] + A[15]*b[3];
}

/*========================================================================*
 *                       解耦算法实现（同时处理矩阵和右端项）              *
 *========================================================================*/

// ABF解耦（2x2块专用优化版本）
static void jxf_decouple_abf_2x2(JXF_Real* val, JXF_Real* rhs, JXF_Int* rpt, 
                               JXF_Int* cpt, JXF_Int nrow) {
    const JXF_Int nb = 2;
    const JXF_Int nb2 = nb * nb;
    
    JXF_Real binv[4];  // 逆矩阵
    JXF_Real smat[4];  // 临时矩阵
    JXF_Real svec[2];  // 临时向量
    
    JXF_Int i, j, ibegin, iend;
    JXF_Int block_idx = 0;
    
    for (i = 0; i < nrow; ++i) {
        // 获取对角块并求逆
        binv[0] = val[block_idx * nb2 + 0];
        binv[1] = val[block_idx * nb2 + 1];
        binv[2] = val[block_idx * nb2 + 2];
        binv[3] = val[block_idx * nb2 + 3];
        
        jxf_smat_inv_2x2(binv);
        
        ibegin = rpt[i];
        iend = rpt[i + 1];
        
        // 处理这一行的所有矩阵块
        for (j = ibegin; j < iend; ++j) {
            JXF_Real* block = val + (block_idx + (j - ibegin)) * nb2;
            
            if (j == ibegin) {
                // 对角块设为单位阵
                block[0] = 1.0; block[1] = 0.0;
                block[2] = 0.0; block[3] = 1.0;
            } else {
                // 非对角块：smat = binv * block
                jxf_smat_mul_2x2(binv, block, smat);
                block[0] = smat[0]; block[1] = smat[1];
                block[2] = smat[2]; block[3] = smat[3];
            }
        }
        
        // 处理右端项：svec = binv * rhs
        svec[0] = rhs[i * nb];
        svec[1] = rhs[i * nb + 1];
        
        jxf_smat_vec_mul_2(binv, svec, svec);
        
        rhs[i * nb] = svec[0];
        rhs[i * nb + 1] = svec[1];
        
        block_idx += (iend - ibegin);
    }
}

// ABF解耦（4x4块专用优化版本）
static void jxf_decouple_abf_4x4(JXF_Real* val, JXF_Real* rhs, JXF_Int* rpt, 
                               JXF_Int* cpt, JXF_Int nrow) {
    const JXF_Int nb = 4;
    const JXF_Int nb2 = nb * nb;
    
    JXF_Real* binv = (JXF_Real*)malloc(nb2 * sizeof(JXF_Real));
    JXF_Real* smat = (JXF_Real*)malloc(nb2 * sizeof(JXF_Real));
    JXF_Real* svec = (JXF_Real*)malloc(nb * sizeof(JXF_Real));
    
    JXF_Int i, j, ibegin, iend;
    JXF_Int block_idx = 0;
    
    for (i = 0; i < nrow; ++i) {
        // 获取对角块并求逆
        memcpy(binv, val + block_idx * nb2, nb2 * sizeof(JXF_Real));
        jxf_smat_inv_4x4(binv);
        
        ibegin = rpt[i];
        iend = rpt[i + 1];
        
        // 处理这一行的所有矩阵块
        for (j = ibegin; j < iend; ++j) {
            JXF_Real* block = val + (block_idx + (j - ibegin)) * nb2;
            
            if (j == ibegin) {
                // 对角块设为单位阵
                jxf_smat_identity(block, nb);
            } else {
                // 非对角块：smat = binv * block
                jxf_smat_mul_4x4(binv, block, smat);
                memcpy(block, smat, nb2 * sizeof(JXF_Real));
            }
        }
        
        // 处理右端项：svec = binv * rhs
        memcpy(svec, rhs + i * nb, nb * sizeof(JXF_Real));
        jxf_smat_vec_mul_4(binv, svec, svec);
        memcpy(rhs + i * nb, svec, nb * sizeof(JXF_Real));
        
        block_idx += (iend - ibegin);
    }
    
    free(binv);
    free(smat);
    free(svec);
}

// ABF解耦（通用版本）
static void jxf_decouple_abf_nb(JXF_Real* val, JXF_Real* rhs, JXF_Int* rpt, 
                              JXF_Int* cpt, JXF_Int nb, JXF_Int nrow) {
    const JXF_Int nb2 = nb * nb;
    
    JXF_Real* binv = (JXF_Real*)malloc(nb2 * sizeof(JXF_Real));
    JXF_Real* smat = (JXF_Real*)malloc(nb2 * sizeof(JXF_Real));
    JXF_Real* svec = (JXF_Real*)malloc(nb * sizeof(JXF_Real));
    
    JXF_Int i, j, ibegin, iend;
    JXF_Int block_idx = 0;
    
    for (i = 0; i < nrow; ++i) {
        // 获取对角块并求逆
        memcpy(binv, val + block_idx * nb2, nb2 * sizeof(JXF_Real));
        jxf_smat_inv_nc(binv, nb);
        
        ibegin = rpt[i];
        iend = rpt[i + 1];
        
        // 处理这一行的所有矩阵块
        for (j = ibegin; j < iend; ++j) {
            JXF_Real* block = val + (block_idx + (j - ibegin)) * nb2;
            
            if (j == ibegin) {
                // 对角块设为单位阵
                jxf_smat_identity(block, nb);
            } else {
                // 非对角块：smat = binv * block
                jxf_smat_mul(binv, block, smat, nb);
                memcpy(block, smat, nb2 * sizeof(JXF_Real));
            }
        }
        
        // 处理右端项：svec = binv * rhs
        memcpy(svec, rhs + i * nb, nb * sizeof(JXF_Real));
        jxf_smat_vec_mul(binv, svec, svec, nb);
        memcpy(rhs + i * nb, svec, nb * sizeof(JXF_Real));
        
        block_idx += (iend - ibegin);
    }
    
    free(binv);
    free(smat);
    free(svec);
}

// ABF解耦（主函数）
static void jxf_decouple_abf(JXF_Real* val, JXF_Real* rhs, JXF_Int* rpt, 
                           JXF_Int* cpt, JXF_Int nb, JXF_Int nrow) {
    switch (nb) {
        case 2:
            jxf_decouple_abf_2x2(val, rhs, rpt, cpt, nrow);
            break;
        case 4:
            jxf_decouple_abf_4x4(val, rhs, rpt, cpt, nrow);
            break;
        default:
            jxf_decouple_abf_nb(val, rhs, rpt, cpt, nb, nrow);
            break;
    }
}

// 解析解耦（ANL）
static void jxf_decouple_anl(JXF_Real* val, JXF_Real* rhs, JXF_Int* rpt, 
                           JXF_Int* cpt, JXF_Int nb, JXF_Int nrow, JXF_Int is_thermal) {
    const JXF_Int nb2 = nb * nb;
    
    JXF_Real* binv = (JXF_Real*)malloc(nb2 * sizeof(JXF_Real));
    JXF_Real* smat = (JXF_Real*)malloc(nb2 * sizeof(JXF_Real));
    JXF_Real* svec = (JXF_Real*)malloc(nb * sizeof(JXF_Real));
    
    JXF_Int i, j, l, ibegin, iend;
    JXF_Int block_idx = 0;
    
    for (i = 0; i < nrow; ++i) {
        // 创建解析解耦矩阵（单位阵，第一行修改）
        jxf_smat_identity(binv, nb);
        
        JXF_Real* diag_block = val + block_idx * nb2;
        
        if (is_thermal) {
            /**
             * A =
             * [PP  PN1  PN2  ... PT ]
             * [N1P N1N1 N1N2 ... N1T]
             * [N2P N2N1 N2N2 ... N2T]
             *         ...
             * [TP  TN1  TN2  ... TT]
             * */
            for (l = 0; l < nb - 2; l++) {
                binv[1 + l] = -diag_block[1 + l];
                binv[(nb - 1) * nb + 1 + l] = -diag_block[(nb - 1) * nb + 1 + l];
            }
        } else {
            for (l = 0; l < nb - 1; l++) {
                binv[1 + l] = -diag_block[1 + l];
            }
        }
        
        ibegin = rpt[i];
        iend = rpt[i + 1];
        
        // 处理这一行的所有矩阵块
        for (j = ibegin; j < iend; ++j) {
            JXF_Real* block = val + (block_idx + (j - ibegin)) * nb2;
            
            // smat = binv * block
            jxf_smat_mul(binv, block, smat, nb);
            memcpy(block, smat, nb2 * sizeof(JXF_Real));
        }
        
        // 处理右端项：svec = binv * rhs
        memcpy(svec, rhs + i * nb, nb * sizeof(JXF_Real));
        jxf_smat_vec_mul(binv, svec, svec, nb);
        memcpy(rhs + i * nb, svec, nb * sizeof(JXF_Real));
        
        block_idx += (iend - ibegin);
    }
    
    free(binv);
    free(smat);
    free(svec);
}

// 半解析解耦（SEM）
static void jxf_decouple_sem(JXF_Real* val, JXF_Real* rhs, JXF_Int* rpt, 
                           JXF_Int* cpt, JXF_Int nb, JXF_Int nrow, JXF_Int is_thermal) {
    const JXF_Int nb2 = nb * nb;
    
    JXF_Real* binv = (JXF_Real*)malloc(nb2 * sizeof(JXF_Real));
    JXF_Real* decoup_mat = (JXF_Real*)malloc(nb2 * sizeof(JXF_Real));
    JXF_Real* smat = (JXF_Real*)malloc(nb2 * sizeof(JXF_Real));
    JXF_Real* svec = (JXF_Real*)malloc(nb * sizeof(JXF_Real));
    
    JXF_Int i, j, l, ibegin, iend;
    JXF_Int block_idx = 0;
    
    for (i = 0; i < nrow; ++i) {
        // 步骤1：计算对角块的逆（ABF部分）
        memcpy(binv, val + block_idx * nb2, nb2 * sizeof(JXF_Real));
        jxf_smat_inv_nc(binv, nb);
        
        // 步骤2：用解析解耦替换第一行（和最后一行，如果是热力模型）
        memcpy(decoup_mat, binv, nb2 * sizeof(JXF_Real));
        
        JXF_Real* diag_block = val + block_idx * nb2;
        
        if (is_thermal) {
            // 热力模型：替换第一行和最后一行
            decoup_mat[0] = 1.0;
            decoup_mat[nb - 1] = 0.0;
            decoup_mat[(nb - 1) * nb] = 0.0;
            decoup_mat[nb2 - 1] = 1.0;
            
            for (l = 0; l < nb - 2; l++) {
                decoup_mat[1 + l] = -diag_block[1 + l];
                decoup_mat[(nb - 1) * nb + 1 + l] = -diag_block[(nb - 1) * nb + 1 + l];
            }
        } else {
            // 非热力模型：只替换第一行
            decoup_mat[0] = 1.0;
            for (l = 0; l < nb - 1; l++) {
                decoup_mat[1 + l] = -diag_block[1 + l];
            }
        }
        
        ibegin = rpt[i];
        iend = rpt[i + 1];
        
        // 处理这一行的所有矩阵块
        for (j = ibegin; j < iend; ++j) {
            JXF_Real* block = val + (block_idx + (j - ibegin)) * nb2;
            
            // smat = decoup_mat * block
            jxf_smat_mul(decoup_mat, block, smat, nb);
            memcpy(block, smat, nb2 * sizeof(JXF_Real));
        }
        
        // 处理右端项：svec = decoup_mat * rhs
        memcpy(svec, rhs + i * nb, nb * sizeof(JXF_Real));
        jxf_smat_vec_mul(decoup_mat, svec, svec, nb);
        memcpy(rhs + i * nb, svec, nb * sizeof(JXF_Real));
        
        block_idx += (iend - ibegin);
    }
    
    free(binv);
    free(decoup_mat);
    free(smat);
    free(svec);
}

// Quasi-IMPES解耦（QI）
static void jxf_decouple_qi(JXF_Real* val, JXF_Real* rhs, JXF_Int* rpt, 
                          JXF_Int* cpt, JXF_Int nb, JXF_Int nrow, JXF_Int is_thermal) {
    const JXF_Int nb2 = nb * nb;
    
    JXF_Real* decoup_mat = (JXF_Real*)malloc(nb2 * sizeof(JXF_Real));
    JXF_Real* smat = (JXF_Real*)malloc(nb2 * sizeof(JXF_Real));
    JXF_Real* svec = (JXF_Real*)malloc(nb * sizeof(JXF_Real));
    
    JXF_Int i, j, l, ibegin, iend;
    JXF_Int block_idx = 0;
    
    for (i = 0; i < nrow; ++i) {
        // 构建QI解耦矩阵
        jxf_smat_identity(decoup_mat, nb);
        
        JXF_Real* diag_block = val + block_idx * nb2;
        
        if (is_thermal) {
            // 热力模型
            for (l = 0; l < nb - 2; l++) {
                // 使用对角元归一化
                decoup_mat[1 + l] = -diag_block[1 + l] / diag_block[(l + 1) * nb + l + 1];
                decoup_mat[(nb - 1) * nb + 1 + l] = -diag_block[(nb - 1) * nb + 1 + l] / 
                                                   diag_block[(l + 1) * nb + l + 1];
            }
        } else {
            // 非热力模型
            for (l = 0; l < nb - 1; l++) {
                decoup_mat[1 + l] = -diag_block[1 + l] / diag_block[(l + 1) * nb + l + 1];
            }
        }
        
        ibegin = rpt[i];
        iend = rpt[i + 1];
        
        // 处理这一行的所有矩阵块
        for (j = ibegin; j < iend; ++j) {
            JXF_Real* block = val + (block_idx + (j - ibegin)) * nb2;
            
            // smat = decoup_mat * block
            jxf_smat_mul(decoup_mat, block, smat, nb);
            memcpy(block, smat, nb2 * sizeof(JXF_Real));
        }
        
        // 处理右端项：svec = decoup_mat * rhs
        memcpy(svec, rhs + i * nb, nb * sizeof(JXF_Real));
        jxf_smat_vec_mul(decoup_mat, svec, svec, nb);
        memcpy(rhs + i * nb, svec, nb * sizeof(JXF_Real));
        
        block_idx += (iend - ibegin);
    }
    
    free(decoup_mat);
    free(smat);
    free(svec);
}

/*========================================================================*
 *                       True-IMPES 解耦实现                              *
 *========================================================================*/

// 更简化的True-IMPES解耦（2x2块）
static void jxf_decouple_timpes_2x2(JXF_Real* val, JXF_Real* rhs, JXF_Int* rpt, 
                                         JXF_Int* cpt, JXF_Int nrow) {
    const JXF_Int nb = 2;
    const JXF_Int nb2 = nb * nb;
    
    // 步骤1: 计算列累计矩阵 Ac = 列求和(A)
    JXF_Real* Ac = (JXF_Real*)calloc(nrow * nb2, sizeof(JXF_Real));
    
    JXF_Int block_idx = 0;
    for (JXF_Int i = 0; i < nrow; ++i) {
        JXF_Int ibegin = rpt[i];
        JXF_Int iend = rpt[i + 1];
        
        for (JXF_Int j = ibegin; j < iend; ++j) {
            JXF_Int col = cpt[j];
            JXF_Real* block = val + block_idx * nb2;
            JXF_Real* ac_block = Ac + col * nb2;
            
            ac_block[0] += block[0];
            ac_block[1] += block[1];
            ac_block[2] += block[2];
            ac_block[3] += block[3];
            
            block_idx++;
        }
    }
    
    // 步骤2: 计算解耦系数（模仿FASP的smat_solution_nc2）
    JXF_Real* coeff = (JXF_Real*)malloc(nrow * sizeof(JXF_Real));
    
    for (JXF_Int i = 0; i < nrow; ++i) {
        JXF_Real* ac_block = Ac + i * nb2;
        
        // 求解: Ac_ii * x = [1, 0]^T
        // 对于2x2: [a11 a12; a21 a22] * [x1; x2] = [1; 0]
        const JXF_Real a11 = ac_block[0];
        const JXF_Real a12 = ac_block[1];
        const JXF_Real a21 = ac_block[2];
        const JXF_Real a22 = ac_block[3];
        
        const JXF_Real det = a11 * a22 - a12 * a21;
        
        if (JXF_ABS(det) < 1e-15) {
            coeff[i] = 0.0;  // 奇异情况
        } else {
            // 求解线性系统: x = Ac_ii^{-1} * [1, 0]^T
            // 只需要第一个分量: x1 = a22 / det
            // 但True-IMPES实际上需要的是系数来消除饱和度对压力的影响
            // 更常见的是: coeff = -a12 / a22 (如果a22不为0)
            if (JXF_ABS(a22) > 1e-15) {
                coeff[i] = -a12 / a22;
            } else {
                coeff[i] = 0.0;
            }
        }
    }
    
    // 步骤3: 应用解耦到矩阵和右端项
    block_idx = 0;
    for (JXF_Int i = 0; i < nrow; ++i) {
        JXF_Int ibegin = rpt[i];
        JXF_Int iend = rpt[i + 1];
        JXF_Real c = coeff[i];
        
        for (JXF_Int j = ibegin; j < iend; ++j) {
            JXF_Real* block = val + block_idx * nb2;
            
            // 应用解耦: 压力方程 += c * 饱和度方程
            block[0] = block[0] + c * block[2];
            block[1] = block[1] + c * block[3];
            // 饱和度方程保持不变
            
            block_idx++;
        }
        
        // 应用解耦到右端项
        rhs[i * nb] = rhs[i * nb] + c * rhs[i * nb + 1];
    }
    
    free(Ac);
    free(coeff);
}
// True-IMPES 解耦（2x2块专用优化版本）
// static void jxf_decouple_timpes_2x2(JXF_Real* val, JXF_Real* rhs, JXF_Int* rpt, 
//                                   JXF_Int* cpt, JXF_Int nrow) {
//     const JXF_Int nb = 2;
//     const JXF_Int nb2 = nb * nb;
    
//     // 步骤1: 计算列累计矩阵 Ac = 列求和(A)
//     JXF_Real* Ac = (JXF_Real*)calloc(nrow * nb2, sizeof(JXF_Real));
    
//     // 遍历所有块，累加到对应列
//     JXF_Int block_idx = 0;
//     for (JXF_Int i = 0; i < nrow; ++i) {
//         JXF_Int ibegin = rpt[i];
//         JXF_Int iend = rpt[i + 1];
        
//         for (JXF_Int j = ibegin; j < iend; ++j) {
//             JXF_Int col = cpt[j];
//             JXF_Real* block = val + block_idx * nb2;
            
//             // 累加到对应列块
//             JXF_Real* ac_block = Ac + col * nb2;
//             ac_block[0] += block[0];
//             ac_block[1] += block[1];
//             ac_block[2] += block[2];
//             ac_block[3] += block[3];
            
//             block_idx++;
//         }
//     }
    
//     // 步骤2: 计算解耦系数（基于列累计矩阵）
//     JXF_Real* coeff = (JXF_Real*)malloc(nrow * sizeof(JXF_Real));
    
//     for (JXF_Int i = 0; i < nrow; ++i) {
//         JXF_Real* ac_block = Ac + i * nb2;
        
//         // 从列累计块计算解耦系数
//         // 计算 Ac_ii^{-1} 的第一行
//         const JXF_Real a11 = ac_block[0];
//         const JXF_Real a12 = ac_block[1];
//         const JXF_Real a21 = ac_block[2];
//         const JXF_Real a22 = ac_block[3];
        
//         const JXF_Real det = a11 * a22 - a12 * a21;
        
//         if (JXF_ABS(det) < 1e-15) {
//             coeff[i] = 0.0;
//         } else {
//             JXF_Real det_inv = 1.0 / det;
//             // 计算逆矩阵的第一行: [a22, -a12] * det_inv
//             // 但True-IMPES只需要第一个元素
//             coeff[i] = a22 * det_inv;  // 这是 (1,0,0) * Ac_ii^{-1} 的第一个元素
//         }
//     }
    
//     // 步骤3: 应用解耦到矩阵
//     block_idx = 0;
//     for (JXF_Int i = 0; i < nrow; ++i) {
//         JXF_Int ibegin = rpt[i];
//         JXF_Int iend = rpt[i + 1];
        
//         for (JXF_Int j = ibegin; j < iend; ++j) {
//             JXF_Real* block = val + block_idx * nb2;
            
//             // 应用True-IMPES解耦: M = (1,0,0) * Ac^{-1} * A
//             // 对于2x2: 新矩阵 = [coeff[0], coeff[1]] * A
//             JXF_Real* row_block = block; // 2x2块的每一行
            
//             // 计算解耦后的行: [coeff[i]*row1[0] + coeff[i]*row1[1], ...]
//             // 实际上True-IMPES只修改第一行
//             JXF_Real new_row0_0 = coeff[i] * (row_block[0] * a22 - row_block[1] * a21);
//             JXF_Real new_row0_1 = coeff[i] * (row_block[1] * a11 - row_block[0] * a12);
            
//             // 保持第二行不变（饱和度方程）
//             block[0] = new_row0_0;
//             block[1] = new_row0_1;
//             // block[2] 和 block[3] 保持原状
            
//             block_idx++;
//         }
//     }
    
//     // 步骤4: 应用解耦到右端项
//     for (JXF_Int i = 0; i < nrow; ++i) {
//         JXF_Real* ac_block = Ac + i * nb2;
//         const JXF_Real a11 = ac_block[0];
//         const JXF_Real a12 = ac_block[1];
//         const JXF_Real a21 = ac_block[2];
//         const JXF_Real a22 = ac_block[3];
        
//         const JXF_Real det = a11 * a22 - a12 * a21;
        
//         if (JXF_ABS(det) > 1e-15) {
//             JXF_Real det_inv = 1.0 / det;
//             JXF_Real coeff0 = a22 * det_inv;
//             JXF_Real coeff1 = -a12 * det_inv;
            
//             // 应用解耦: z = (1,0,0) * Ac^{-1} * r
//             rhs[i * nb] = coeff0 * rhs[i * nb] + coeff1 * rhs[i * nb + 1];
//             // 饱和度方程的右端项保持不变
//         }
//     }
    
//     free(Ac);
//     free(coeff);
// }

// True-IMPES 解耦（3x3块专用优化版本）
static void jxf_decouple_timpes_3x3(JXF_Real* val, JXF_Real* rhs, JXF_Int* rpt, 
                                  JXF_Int* cpt, JXF_Int nrow) {
    const JXF_Int nb = 3;
    const JXF_Int nb2 = nb * nb;
    
    // 步骤1: 计算列累计矩阵
    JXF_Real* Ac = (JXF_Real*)calloc(nrow * nb2, sizeof(JXF_Real));
    
    JXF_Int block_idx = 0;
    for (JXF_Int i = 0; i < nrow; ++i) {
        JXF_Int ibegin = rpt[i];
        JXF_Int iend = rpt[i + 1];
        
        for (JXF_Int j = ibegin; j < iend; ++j) {
            JXF_Int col = cpt[j];
            JXF_Real* block = val + block_idx * nb2;
            JXF_Real* ac_block = Ac + col * nb2;
            
            // 累加到对应列块
            for (JXF_Int k = 0; k < nb2; ++k) {
                ac_block[k] += block[k];
            }
            
            block_idx++;
        }
    }
    
    // 步骤2: 计算解耦系数
    JXF_Real* coeff = (JXF_Real*)malloc(nrow * 2 * sizeof(JXF_Real)); // 每个块需要2个系数
    
    for (JXF_Int i = 0; i < nrow; ++i) {
        JXF_Real* ac_block = Ac + i * nb2;
        
        // 计算 Ac_ii^{-1}
        JXF_Real binv[9];
        memcpy(binv, ac_block, 9 * sizeof(JXF_Real));
        jxf_smat_inv_3x3(binv);
        
        // 保存 (1,0,0) * Ac_ii^{-1} 的前两个元素（忽略第三个）
        coeff[i*2] = binv[0];     // 对应压力方程系数
        coeff[i*2+1] = binv[1];   // 对应第一个组分方程系数
    }
    
    // 步骤3: 应用解耦到矩阵
    block_idx = 0;
    for (JXF_Int i = 0; i < nrow; ++i) {
        JXF_Int ibegin = rpt[i];
        JXF_Int iend = rpt[i + 1];
        
        JXF_Real c0 = coeff[i*2];
        JXF_Real c1 = coeff[i*2+1];
        
        for (JXF_Int j = ibegin; j < iend; ++j) {
            JXF_Real* block = val + block_idx * nb2;
            
            // 应用解耦：新第一行 = [c0, c1, 0] * block
            JXF_Real new_row0_0 = c0 * block[0] + c1 * block[3];
            JXF_Real new_row0_1 = c0 * block[1] + c1 * block[4];
            JXF_Real new_row0_2 = c0 * block[2] + c1 * block[5];
            
            // 保持其他行不变
            block[0] = new_row0_0;
            block[1] = new_row0_1;
            block[2] = new_row0_2;
            // block[3]-block[8] 保持不变
            
            block_idx++;
        }
    }
    
    // 步骤4: 应用解耦到右端项
    for (JXF_Int i = 0; i < nrow; ++i) {
        JXF_Real* ac_block = Ac + i * nb2;
        JXF_Real binv[9];
        memcpy(binv, ac_block, 9 * sizeof(JXF_Real));
        jxf_smat_inv_3x3(binv);
        
        // 应用解耦: z = (1,0,0) * Ac^{-1} * r
        JXF_Real* r = rhs + i * nb;
        rhs[i * nb] = binv[0] * r[0] + binv[1] * r[1] + binv[2] * r[2];
        // 其他方程的右端项保持不变
    }
    
    free(Ac);
    free(coeff);
}

// True-IMPES 解耦（通用版本）
static void jxf_decouple_timpes_nb(JXF_Real* val, JXF_Real* rhs, JXF_Int* rpt, 
                                 JXF_Int* cpt, JXF_Int nb, JXF_Int nrow) {
    const JXF_Int nb2 = nb * nb;
    
    // 步骤1: 计算列累计矩阵
    JXF_Real* Ac = (JXF_Real*)calloc(nrow * nb2, sizeof(JXF_Real));
    
    JXF_Int block_idx = 0;
    for (JXF_Int i = 0; i < nrow; ++i) {
        JXF_Int ibegin = rpt[i];
        JXF_Int iend = rpt[i + 1];
        
        for (JXF_Int j = ibegin; j < iend; ++j) {
            JXF_Int col = cpt[j];
            JXF_Real* block = val + block_idx * nb2;
            JXF_Real* ac_block = Ac + col * nb2;
            
            for (JXF_Int k = 0; k < nb2; ++k) {
                ac_block[k] += block[k];
            }
            
            block_idx++;
        }
    }
    
    // 步骤2: 计算解耦系数
    JXF_Real* decoup_row = (JXF_Real*)malloc(nb * sizeof(JXF_Real));
    
    for (JXF_Int i = 0; i < nrow; ++i) {
        JXF_Real* ac_block = Ac + i * nb2;
        
        // 计算 Ac_ii^{-1}
        JXF_Real* binv = (JXF_Real*)malloc(nb2 * sizeof(JXF_Real));
        memcpy(binv, ac_block, nb2 * sizeof(JXF_Real));
        jxf_smat_inv_nc(binv, nb);
        
        // 提取 (1,0,0,...) * Ac_ii^{-1} 作为解耦行
        for (JXF_Int k = 0; k < nb; ++k) {
            decoup_row[k] = binv[k];
        }
        
        // 步骤3: 应用解耦到矩阵的对应行
        JXF_Int row_start = rpt[i];
        JXF_Int row_end = rpt[i + 1];
        
        for (JXF_Int j = row_start; j < row_end; ++j) {
            JXF_Int global_idx = 0;
            // 找到这个块的全局索引（简化处理）
            for (JXF_Int k = 0; k < i; ++k) {
                global_idx += (rpt[k+1] - rpt[k]);
            }
            global_idx += (j - row_start);
            
            JXF_Real* block = val + global_idx * nb2;
            
            // 计算解耦后的第一行
            JXF_Real new_row0[nb];
            for (JXF_Int col = 0; col < nb; ++col) {
                new_row0[col] = 0.0;
                for (JXF_Int k = 0; k < nb; ++k) {
                    new_row0[col] += decoup_row[k] * block[k*nb + col];
                }
            }
            
            // 替换第一行
            for (JXF_Int col = 0; col < nb; ++col) {
                block[col] = new_row0[col];
            }
            // 其他行保持不变
        }
        
        // 步骤4: 应用解耦到右端项
        JXF_Real* r = rhs + i * nb;
        JXF_Real new_rhs0 = 0.0;
        for (JXF_Int k = 0; k < nb; ++k) {
            new_rhs0 += decoup_row[k] * r[k];
        }
        rhs[i * nb] = new_rhs0;
        
        free(binv);
    }
    
    free(decoup_row);
    free(Ac);
}

// True-IMPES 解耦（主函数）
static void jxf_decouple_timpes(JXF_Real* val, JXF_Real* rhs, JXF_Int* rpt, 
                              JXF_Int* cpt, JXF_Int nb, JXF_Int nrow) {
    switch (nb) {
        case 2:
            jxf_decouple_timpes_2x2(val, rhs, rpt, cpt, nrow);
            break;
        case 3:
            jxf_decouple_timpes_3x3(val, rhs, rpt, cpt, nrow);
            break;
        default:
            jxf_decouple_timpes_nb(val, rhs, rpt, cpt, nb, nrow);
            break;
    }
}

// True-IMPES 解耦版本2（模仿FASP的第二种实现）
static void jxf_decouple_timpes2(JXF_Real* val, JXF_Real* rhs, JXF_Int* rpt, 
                               JXF_Int* cpt, JXF_Int nb, JXF_Int nrow) {
    const JXF_Int nb2 = nb * nb;
    
    // 步骤1: 计算列累计矩阵
    JXF_Real* Ac = (JXF_Real*)calloc(nrow * nb2, sizeof(JXF_Real));
    
    JXF_Int block_idx = 0;
    for (JXF_Int i = 0; i < nrow; ++i) {
        JXF_Int ibegin = rpt[i];
        JXF_Int iend = rpt[i + 1];
        
        for (JXF_Int j = ibegin; j < iend; ++j) {
            JXF_Int col = cpt[j];
            JXF_Real* block = val + block_idx * nb2;
            JXF_Real* ac_block = Ac + col * nb2;
            
            for (JXF_Int k = 0; k < nb2; ++k) {
                ac_block[k] += block[k];
            }
            
            block_idx++;
        }
    }
    
    // 步骤2: 为每个块计算解耦系数并直接应用到矩阵和右端项
    if (nb == 2) {
        JXF_Real* coeff = (JXF_Real*)malloc(nrow * sizeof(JXF_Real));
        
        // 计算解耦系数
        for (JXF_Int i = 0; i < nrow; ++i) {
            JXF_Real* ac_block = Ac + i * nb2;
            
            const JXF_Real a00 = ac_block[0];
            const JXF_Real a01 = ac_block[1];
            const JXF_Real a10 = ac_block[2];
            const JXF_Real a11 = ac_block[3];
            
            const JXF_Real det = a00 * a11 - a01 * a10;
            
            if (JXF_ABS(det) < 1e-15) {
                coeff[i] = 0.0;
            } else {
                // 计算逆矩阵的第一行
                coeff[i] = -a01 / a11;  // 简化版本：消去压力方程中的饱和度系数
            }
        }
        
        // 应用解耦：修改矩阵
        block_idx = 0;
        for (JXF_Int i = 0; i < nrow; ++i) {
            JXF_Int ibegin = rpt[i];
            JXF_Int iend = rpt[i + 1];
            JXF_Real c = coeff[i];
            
            for (JXF_Int j = ibegin; j < iend; ++j) {
                JXF_Real* block = val + block_idx * nb2;
                
                // 应用解耦：压力方程 += c * 饱和度方程
                block[0] = block[0] + c * block[2];
                block[1] = block[1] + c * block[3];
                // 饱和度方程保持不变
                
                block_idx++;
            }
        }
        
        // 应用解耦：修改右端项
        for (JXF_Int i = 0; i < nrow; ++i) {
            JXF_Real c = coeff[i];
            rhs[i * nb] = rhs[i * nb] + c * rhs[i * nb + 1];
        }
        
        free(coeff);
    } else if (nb == 3) {
        JXF_Real* coeff = (JXF_Real*)malloc(nrow * 2 * sizeof(JXF_Real));
        
        // 计算解耦系数
        for (JXF_Int i = 0; i < nrow; ++i) {
            JXF_Real* ac_block = Ac + i * nb2;
            JXF_Real binv[9];
            memcpy(binv, ac_block, 9 * sizeof(JXF_Real));
            jxf_smat_inv_3x3(binv);
            
            // 保存解耦系数（从逆矩阵的第一行）
            coeff[i*2] = binv[0];
            coeff[i*2+1] = binv[1];
        }
        
        // 应用解耦：修改矩阵
        block_idx = 0;
        for (JXF_Int i = 0; i < nrow; ++i) {
            JXF_Int ibegin = rpt[i];
            JXF_Int iend = rpt[i + 1];
            JXF_Real c0 = coeff[i*2];
            JXF_Real c1 = coeff[i*2+1];
            
            for (JXF_Int j = ibegin; j < iend; ++j) {
                JXF_Real* block = val + block_idx * nb2;
                
                // 应用解耦：压力方程 += c0 * 组分1方程 + c1 * 组分2方程
                block[0] = block[0] + c0 * block[3] + c1 * block[6];
                block[1] = block[1] + c0 * block[4] + c1 * block[7];
                block[2] = block[2] + c0 * block[5] + c1 * block[8];
                
                block_idx++;
            }
        }
        
        // 应用解耦：修改右端项
        for (JXF_Int i = 0; i < nrow; ++i) {
            JXF_Real c0 = coeff[i*2];
            JXF_Real c1 = coeff[i*2+1];
            rhs[i * nb] = rhs[i * nb] + c0 * rhs[i * nb + 1] + c1 * rhs[i * nb + 2];
        }
        
        free(coeff);
    } else {
        // 通用情况
        JXF_Real** coeff = (JXF_Real**)malloc(nrow * sizeof(JXF_Real*));
        
        for (JXF_Int i = 0; i < nrow; ++i) {
            JXF_Real* ac_block = Ac + i * nb2;
            JXF_Real* binv = (JXF_Real*)malloc(nb2 * sizeof(JXF_Real));
            memcpy(binv, ac_block, nb2 * sizeof(JXF_Real));
            jxf_smat_inv_nc(binv, nb);
            
            coeff[i] = (JXF_Real*)malloc((nb-1) * sizeof(JXF_Real));
            for (JXF_Int k = 0; k < nb-1; ++k) {
                coeff[i][k] = binv[k+1]; // 使用逆矩阵的第一行（跳过第一个元素）
            }
            
            free(binv);
        }
        
        // 应用解耦
        block_idx = 0;
        for (JXF_Int i = 0; i < nrow; ++i) {
            JXF_Int ibegin = rpt[i];
            JXF_Int iend = rpt[i + 1];
            
            for (JXF_Int j = ibegin; j < iend; ++j) {
                JXF_Real* block = val + block_idx * nb2;
                
                // 计算新的压力方程
                for (JXF_Int col = 0; col < nb; ++col) {
                    JXF_Real sum = block[col]; // 压力方程原值
                    for (JXF_Int k = 0; k < nb-1; ++k) {
                        sum += coeff[i][k] * block[(k+1)*nb + col];
                    }
                    block[col] = sum;
                }
                
                block_idx++;
            }
        }
        
        // 释放内存
        for (JXF_Int i = 0; i < nrow; ++i) {
            free(coeff[i]);
        }
        free(coeff);
    }
    
    free(Ac);
}

/*========================================================================*
 *                       公共接口函数                                     *
 *========================================================================*/

JXF_Int jxf_BSRMatrixDecouple(jxf_BSRMatrix* A_bsr, 
                           jxf_Vector* rhs,
                           JXF_DecoupType decoup_type,
                           JXF_Int is_thermal) {
    if (A_bsr == NULL || rhs == NULL) {
        ERROR_PRINTF("jxf_BSRMatrixDecouple: NULL matrix or rhs\n");
        return jxf_error_flag;
    }

    printf("\njxf_BSRMatrixDecouple...\n");

    fflush(stdout); 
        
    
    JXF_Int nb = jxf_BSRMatrixBlockSize(A_bsr);
    JXF_Int nrows = jxf_BSRMatrixNumRows(A_bsr);
    JXF_Int* rpt = jxf_BSRMatrixI(A_bsr);
    JXF_Int* cpt = jxf_BSRMatrixJ(A_bsr);
    JXF_Real* val = jxf_BSRMatrixData(A_bsr);
    JXF_Real* rhs_data = jxf_VectorData(rhs);
    
    if (val == NULL || rhs_data == NULL || rpt == NULL || cpt == NULL) {
        ERROR_PRINTF("jxf_BSRMatrixDecouple: Invalid matrix or rhs data\n");
        return jxf_error_flag;
    }
    
    // 检查向量尺寸是否匹配
    if (jxf_VectorSize(rhs) != nrows * nb) {
        ERROR_PRINTF("jxf_BSRMatrixDecouple: RHS size mismatch. Expected %d, got %d\n",
                    nrows * nb, jxf_VectorSize(rhs));
        return jxf_error_flag;
    }
    
    // 检查块大小是否支持
    if (nb < 1 || nb > 10) {
        ERROR_PRINTF("jxf_BSRMatrixDecouple: Unsupported block size %d\n", nb);
        return jxf_error_flag;
    }
    
    // 根据解耦类型调用相应的函数
    switch (decoup_type) {
        case JXF_DECOUP_NONE:
            // 不解耦，直接返回
            break;
            
        case JXF_DECOUP_ABF:
            jxf_decouple_abf(val, rhs_data, rpt, cpt, nb, nrows);
            break;
            
        case JXF_DECOUP_ANL:
            jxf_decouple_anl(val, rhs_data, rpt, cpt, nb, nrows, is_thermal);
            break;
            
        case JXF_DECOUP_SEM:
            jxf_decouple_sem(val, rhs_data, rpt, cpt, nb, nrows, is_thermal);
            break;
            
        case JXF_DECOUP_QI:
            jxf_decouple_qi(val, rhs_data, rpt, cpt, nb, nrows, is_thermal);
            break;
            
        case JXF_DECOUP_TIMPES:
            jxf_decouple_timpes(val, rhs_data, rpt, cpt, nb, nrows);
            break;
            
        case JXF_DECOUP_TIMPES2:
            jxf_decouple_timpes2(val, rhs_data, rpt, cpt, nb, nrows);
            break;
            
        default:
            ERROR_PRINTF("jxf_BSRMatrixDecouple: Unknown decoupling type %d\n", decoup_type);
            return jxf_error_flag;
    }
    
    return JXF_SUCCESS;
}

JXF_Int jxf_ParBSRMatrixDecouple(jxf_ParBSRMatrix* A_parbsr,
                              jxf_ParVector* rhs,
                              JXF_DecoupType decoup_type,
                              JXF_Int is_thermal) {
    if (A_parbsr == NULL || rhs == NULL) {
        ERROR_PRINTF("jxf_ParBSRMatrixDecouple: NULL matrix or rhs\n");
        return jxf_error_flag;
    }
    
    int myid;
    MPI_Comm comm = jxf_ParBSRMatrixComm(A_parbsr);
    MPI_Comm_rank(comm, &myid);
    
    // 获取对角部分和非对角部分
    jxf_BSRMatrix* diag = jxf_ParBSRMatrixDiag(A_parbsr);
    jxf_BSRMatrix* offd = jxf_ParBSRMatrixOffd(A_parbsr);
    
    if (diag == NULL) {
        ERROR_PRINTF("jxf_ParBSRMatrixDecouple: NULL diagonal matrix\n");
        return jxf_error_flag;
    }
    
    // 获取本地向量部分
    jxf_Vector* local_rhs = jxf_ParVectorLocalVector(rhs);
    if (local_rhs == NULL) {
        ERROR_PRINTF("jxf_ParBSRMatrixDecouple: NULL local rhs\n");
        return jxf_error_flag;
    }
    
    // 先解耦对角部分和本地右端项
    JXF_Int result = jxf_BSRMatrixDecouple(diag, local_rhs, decoup_type, is_thermal);
    if (result != JXF_SUCCESS) {
        ERROR_PRINTF("jxf_ParBSRMatrixDecouple: Failed to decouple diagonal part\n");
        return result;
    }
    
    // 如果存在非对角部分，也需要解耦
    if (offd != NULL) {
        jxf_Vector* temp_rhs = jxf_SeqVectorCreate(jxf_BSRMatrixNumRows(offd) * 
                                               jxf_BSRMatrixBlockSize(offd));
        jxf_SeqVectorInitialize(temp_rhs);
        
        result = jxf_BSRMatrixDecouple(offd, temp_rhs, decoup_type, is_thermal);
        
        jxf_SeqVectorDestroy(temp_rhs);
        
        if (result != JXF_SUCCESS) {
            ERROR_PRINTF("jxf_ParBSRMatrixDecouple: Failed to decouple off-diagonal part\n");
            return result;
        }
    }
    
    return JXF_SUCCESS;
}

// 更新枚举类型字符串转换函数
const char* jxf_DecoupTypeToString(JXF_DecoupType decoup_type) {
    switch (decoup_type) {
        case JXF_DECOUP_NONE:    return "NONE";
        case JXF_DECOUP_ABF:     return "ABF";
        case JXF_DECOUP_ANL:     return "ANL";
        case JXF_DECOUP_SEM:     return "SEM";
        case JXF_DECOUP_QI:      return "QI";
        case JXF_DECOUP_TIMPES:  return "True-IMPES";
        case JXF_DECOUP_TIMPES2: return "True-IMPES2";
        default:                return "UNKNOWN";
    }
}