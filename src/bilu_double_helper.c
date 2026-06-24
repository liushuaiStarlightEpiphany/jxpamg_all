#include "jx_parbilu.h"
#include "jx_parbsr_mv.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef struct {
    JX_Solver solver;
    jx_ParBSRMatrix* mat;
    jx_ParVector* save_x;
    jx_ParVector* save_r;
    jx_ParVector* work;
} BiluCtx;

void* bilu_double_create(void* A, int myid) {
    jx_ParBSRMatrix* mat = (jx_ParBSRMatrix*)A;
    JX_Solver solver = NULL;
    JX_BILUCreate(&solver);
    JX_ILUSetType(solver, 3);
    JX_ILUSetLevelOfFill(solver, 0);
    JX_ILUSetMaxIter(solver, 1);
    JX_ILUSetTol(solver, 0.0);
    JX_ILUSetTriSolve(solver, 14);
    JX_ILUSetLogging(solver, 0);
    JX_ILUSetsweep(solver, 3);
    JX_BigInt nrows = jx_ParBSRMatrixGlobalNumRows(mat);
    JX_Int blk = jx_ParBSRMatrixBlockSize(mat);
    JX_Int* part;
    jx_ParBSRMatrixGetRowPartitioning(mat, &part);
    jx_ParVector* tmp = jx_ParVectorCreate(jx_ParBSRMatrixComm(mat), nrows * blk, part);
    jx_ParVectorInitialize(tmp);
    ((jx_ParBILUData*)solver)->matA = mat;
    jx_BILUSetup((void*)solver, mat, tmp, tmp);
    BiluCtx* ctx = (BiluCtx*)malloc(sizeof(BiluCtx));
    ctx->solver = solver;
    ctx->mat = mat;
    ctx->save_x = jx_ParVectorCreate(jx_ParBSRMatrixComm(mat), nrows * blk, part);
    jx_ParVectorInitialize(ctx->save_x);
    ctx->save_r = jx_ParVectorCreate(jx_ParBSRMatrixComm(mat), nrows * blk, part);
    jx_ParVectorInitialize(ctx->save_r);
    ctx->work = jx_ParVectorCreate(jx_ParBSRMatrixComm(mat), nrows * blk, part);
    jx_ParVectorInitialize(ctx->work);
    if (myid == 0) printf("Double BILU(BSR) setup done.\n");
    return (void*)ctx;
}

void bilu_double_apply(void* vctx, float* x_data, float* b_data, JX_Int nloc) {
    BiluCtx* ctx = (BiluCtx*)vctx;
    double* xd = jx_VectorData(jx_ParVectorLocalVector(ctx->save_x));
    double* rd = jx_VectorData(jx_ParVectorLocalVector(ctx->save_r));
    double* wd = jx_VectorData(jx_ParVectorLocalVector(ctx->work));
    for (JX_Int i = 0; i < nloc; i++) xd[i] = (double)x_data[i];
    for (JX_Int i = 0; i < nloc; i++) rd[i] = (double)b_data[i];
    jx_ParVectorCopy(ctx->save_r, ctx->save_r);
    jx_ParBSRMatrixMatvec(-1.0, ctx->mat, ctx->save_x, 1.0, ctx->save_r);
    jx_ParVectorSetConstantValues(ctx->work, 0.0);
    jx_BILUSolve((void*)ctx->solver, ctx->mat, ctx->save_r, ctx->work);
    for (JX_Int i = 0; i < nloc; i++) { xd[i] += wd[i]; x_data[i] = (float)xd[i]; }
}

void bilu_print_residual(void* A, void* b, void* x, MPI_Comm comm, const char* label) {
    (void)comm;
    jx_ParBSRMatrix* mat = (jx_ParBSRMatrix*)A;
    jx_ParVector* pb = (jx_ParVector*)b;
    jx_ParVector* px = (jx_ParVector*)x;
    JX_BigInt nr = jx_ParBSRMatrixGlobalNumRows(mat);
    JX_Int bk = jx_ParBSRMatrixBlockSize(mat);
    JX_Int* pt; jx_ParBSRMatrixGetRowPartitioning(mat, &pt);
    jx_ParVector* tr = jx_ParVectorCreate(jx_ParBSRMatrixComm(mat), nr * bk, pt);
    jx_ParVectorInitialize(tr);
    jx_ParVectorCopy(pb, tr); jx_ParBSRMatrixMatvec(-1.0, mat, px, 1.0, tr);
    double rn = sqrt(jx_ParVectorInnerProd(tr, tr));
    jx_ParVectorCopy(pb, tr); double bn = sqrt(jx_ParVectorInnerProd(tr, tr));
    printf("%s %.6e (rel: %.6e)\n", label, rn, rn / (bn + 1e-16));
    fflush(stdout);
    jx_ParVectorDestroy(tr);
}

void bilu_double_destroy(void* vctx) {
    if (!vctx) return;
    BiluCtx* ctx = (BiluCtx*)vctx;
    if (ctx->solver) { ((jx_ParBILUData*)ctx->solver)->matA = NULL; JX_BILUDestroy(ctx->solver); }
    jx_ParVectorDestroy(ctx->save_x);
    jx_ParVectorDestroy(ctx->save_r);
    jx_ParVectorDestroy(ctx->work);
    free(ctx);
}
