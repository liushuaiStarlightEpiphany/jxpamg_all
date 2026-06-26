#include "naviixLogical.h"
#include <time.h>
#include "naviixMsh.h"
#include "naviixParallel.h"
#include "naviixMatrix.h"
#include "naviixHypreAPI.h"
#include "_hypre_utilities.h"
#include "krylov.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_krylov.h"

void HypreAMG(Matrix* mat, int MaxIter, Real mtol, int* iternum, Real* residual, Real* costTime, Real* x)
{
    HYPRE_IJMatrix Hypre_A;
    HYPRE_IJVector Hypre_x;
    HYPRE_IJVector Hypre_b;

    int ilower, iupper, local_size;
    int nbelements;

    nbelements = mshPtr->elementsNum;

    clock_t start, end;

    start = clock();

    ilower = mshPtr->elements[0].gId;
    iupper = mshPtr->elements[nbelements - 1].gId;
    local_size = iupper - ilower + 1;


    //矩阵初始化
    HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &Hypre_A);
    HYPRE_IJMatrixSetObjectType(Hypre_A, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(Hypre_A);

    HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &Hypre_b);
    HYPRE_IJVectorSetObjectType(Hypre_b, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(Hypre_b);

    HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &Hypre_x);
    HYPRE_IJVectorSetObjectType(Hypre_x, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(Hypre_x);

    //矩阵装配
    int num;
    int row;
    int ncols;
    int col[MAX_FACES + 1];
    int nvals;
    double val[MAX_FACES + 1];

    for (int elem = 0; elem < mshPtr->elementsNum; elem++)
    {
        ncols = 0;
        nvals = 0;

        row = mshPtr->elements[elem].gId;

        col[ncols] = mshPtr->elements[elem].gId;
        ncols++;

        val[nvals] = mat->diag[elem];
        nvals++;

        for (int j = 0; j < mshPtr->elements[elem].adjElemsNum; j++)
        {
            col[ncols] = mshPtr->elements[elem].adjElems[j];
            ncols++;

            val[nvals] = mat->off_diag[elem][j];
            nvals++;
        }

        num = mshPtr->elements[elem].adjElemsNum + 1;

        HYPRE_IJMatrixSetValues(Hypre_A, 1, &num, &mshPtr->elements[elem].gId, col, val);
        HYPRE_IJVectorSetValues(Hypre_b, 1, &mshPtr->elements[elem].gId, &mat->rhs[elem]);
    }

    HYPRE_IJMatrixAssemble(Hypre_A);
    HYPRE_IJVectorAssemble(Hypre_b);
    HYPRE_IJVectorAssemble(Hypre_x);

    //矩阵求解

    Real mid_value;

    HYPRE_ParCSRMatrix parcsr_AMGp;
    HYPRE_Solver par_solver;
    HYPRE_Solver par_precond;

    HYPRE_ParVector par_x1;
    HYPRE_ParVector par_b1;

    HYPRE_IJMatrixGetObject(Hypre_A, (void**)&parcsr_AMGp);
    HYPRE_IJVectorGetObject(Hypre_x, (void**)&par_x1);
    HYPRE_IJVectorGetObject(Hypre_b, (void**)&par_b1);

    HYPRE_ParCSRBiCGSTABCreate(MPI_COMM_WORLD, &par_solver);
    HYPRE_BiCGSTABSetMaxIter(par_solver, MaxIter);
    HYPRE_BiCGSTABSetTol(par_solver, mtol);

    HYPRE_BoomerAMGCreate(&par_precond);
    HYPRE_BoomerAMGSetCoarsenType(par_precond, 10);
    HYPRE_BoomerAMGSetInterpType(par_precond, 6);

    HYPRE_BoomerAMGSetStrongThreshold(par_precond, 0.3);
    HYPRE_BoomerAMGSetTol(par_precond, 0.0);
    HYPRE_BoomerAMGSetMaxIter(par_precond, 1);
    HYPRE_ParCSRBiCGSTABSetPrecond(par_solver,
        HYPRE_BoomerAMGSolve,
        HYPRE_BoomerAMGSetup,
        par_precond);

    HYPRE_ParCSRBiCGSTABSetup(par_solver, parcsr_AMGp, par_b1, par_x1);

    HYPRE_ParCSRBiCGSTABSolve(par_solver, parcsr_AMGp, par_b1, par_x1);

    HYPRE_BiCGSTABGetNumIterations(par_solver, iternum);
    HYPRE_BiCGSTABGetFinalRelativeResidualNorm(par_solver, residual);

    HYPRE_ParCSRBiCGSTABDestroy(par_solver);

    HYPRE_BoomerAMGDestroy(par_precond);

    for (int i = ilower; i <= iupper; i++)
    {
        HYPRE_IJVectorGetValues(Hypre_x, 1, &i, &mid_value);
        x[i - ilower] = mid_value;
    }

    //内存释放
    HYPRE_IJMatrixDestroy(Hypre_A);
    HYPRE_IJVectorDestroy(Hypre_x);
    HYPRE_IJVectorDestroy(Hypre_b);

    end = clock();

    (*costTime) = ((Real)end - (Real)start) / CLOCKS_PER_SEC;
}