# include "naviixmsh.h"
# include "jx_pamg.h"


JX_Int jx_CSMatrixAssemble( MshTopology *matrix, Matrix mat, jx_CSRMatrix *matrixOut);


JX_Int jx_CSMatrixAssemble( MshTopology *matrix, Matrix mat)
{
    for (int elem = 0; elem < mshPtr->elementsNum; elem++) {
        int ncols = 0;
        int nvals = 0;
        int row = mshPtr->elements[elem].gId;
        int col[100];  // 假设最大列数为100，实际应用中应根据需要调整大小
        double val[100];  // 假设最大值为100，实际应用中应根据需要调整大小

        // 设置对角线上的值
        col[ncols] = mshPtr->elements[elem].gId;
        val[nvals] = mat->diag[elem];
        ncols++;
        nvals++;

        // 设置非对角线上的值（邻接元素）
        for (int j = 0; j < mshPtr->elements[elem].adjElemsNum; j++) {
            col[ncols] = mshPtr->elements[elem].adjElems[j].gId;  // 假设adjElems是结构体数组，包含gId
            val[nvals] = mat->off_diag[elem][j];
            ncols++;
            nvals++;
        }

        int num = mshPtr->elements[elem].adjElemsNum + 1;  // 邻接元素数量加1（包括对角线元素）

        // 设置矩阵的值
        JX_IJMatrixSetValues(Hypre_A, 1, &num, &row, col, val);

        // 设置向量的值
        JX_IJMatrixSetValues(Hypre_b, 1, &row, &mat->rhs[elem]);
    }
}
