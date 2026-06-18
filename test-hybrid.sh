#!/bin/bash
# ============================================================
# Hybrid CPR-GMRES 测试脚本
# 双精度 GMRES + 单精度 CPR 预条件
# ============================================================
set -e

# ===================== 可调参数 =============================
# --- 矩阵参数 ---
MATRIX_FILE="/home/spring/l_s/jxpamg_all/JXFPAMG-bsr-fmt/data/spe10_bsr/A_bsr_spe10.bin"
RHS_FILE="/home/spring/l_s/jxpamg_all/JXFPAMG-bsr-fmt/data/spe10_bsr/b_spe10.dat"
MATRIX_TYPE=1           # 0=CSR, 1=BSR
BINARY=1                # 0=文本, 1=二进制

# --- MPI 进程数 ---
NP_LIST="1 2 4 8 16 32"

# --- GMRES 参数（见 cpr.tex 表~\ref{tab:params_hybrid_gmres}）---
KDIM=30                 # Krylov 子空间维度
MAX_ITER=30             # 最大 GMRES 迭代次数
TOL=1e-4                # 收敛容差

# --- CPR 参数（见 cpr.tex 表~\ref{tab:params_cpr}）---
DECOUP_TYPE="TIMPES"    # ABF / TIMPES / TIMPES2 / ANL / SEM / QI / NONE
STAGE1_MAXIT=1          # Stage 1 (AMG) 最大迭代
STAGE2_MAXIT=2          # Stage 2 迭代次数
STAGE2_TYPE=2           # Stage 2 类型: 2=HGS, 3=HSGS
PRESSURE_INDEX=0        # 压力变量索引
PRINT_LEVEL=1           # 打印级别 0=最小, 1=正常, 2=调试
# ============================================================

# ----- 路径与编译环境 -----
SRC_DIR="/home/spring/l_s/jxpamg_all/JXFPAMG-bsr-fmt"
SRC_FILE="$SRC_DIR/src/test_bsr_cprgmres_hybrid.c"
BIN="$SRC_DIR/test_bsr_cprgmres_hybrid"
MPIDIR="/opt/intel/oneapi/mpi/2021.6.0"
METHOD="hybrid"

# ----- 参数映射 -----
case "$DECOUP_TYPE" in
    NONE)    DECOUP_VAL=0 ;;
    ABF)     DECOUP_VAL=1 ;;
    ANL)     DECOUP_VAL=2 ;;
    SEM)     DECOUP_VAL=3 ;;
    QI)      DECOUP_VAL=4 ;;
    TIMPES)  DECOUP_VAL=5 ;;
    TIMPES2) DECOUP_VAL=6 ;;
    *) echo "Error: unknown DECOUP_TYPE=$DECOUP_TYPE"; exit 1 ;;
esac

# ----- 修改源码参数 -----
echo "Configuring parameters..."
sed -i "s/int k_dim = [0-9]*;/int k_dim = $KDIM;/" "$SRC_FILE"
sed -i "s/^int max_iterations = [0-9]*;/int max_iterations = $MAX_ITER;/" "$SRC_FILE"
sed -i "s/^JX_Real tolerance = [0-9.e\-]*;/JX_Real tolerance = $TOL;/" "$SRC_FILE"
sed -i "s/JX_DecoupType decoup_type = JX_DECOUP_[A-Z0-9]*;/JX_DecoupType decoup_type = JX_DECOUP_$DECOUP_TYPE;/" "$SRC_FILE"

# sed for CPR params (inside solve_with_mixed_precision_cpr_gmres)
sed -i "s/JXF_CPRSetParameter(cpr, \"stage1_maxit\", [0-9]*);/JXF_CPRSetParameter(cpr, \"stage1_maxit\", $STAGE1_MAXIT);/" "$SRC_FILE"
sed -i "s/JXF_CPRSetParameter(cpr, \"stage2_maxit\", [0-9]*);/JXF_CPRSetParameter(cpr, \"stage2_maxit\", $STAGE2_MAXIT);/" "$SRC_FILE"
sed -i "s/JXF_CPRSetParameter(cpr, \"print_level\", [0-9]*);/JXF_CPRSetParameter(cpr, \"print_level\", $PRINT_LEVEL);/" "$SRC_FILE"

echo "  KDIM=$KDIM MAX_ITER=$MAX_ITER TOL=$TOL DECOUP=$DECOUP_TYPE"
echo "  STAGE1_MAXIT=$STAGE1_MAXIT STAGE2_MAXIT=$STAGE2_MAXIT STAGE2_TYPE=$STAGE2_TYPE"

# ----- 编译 -----
echo "Compiling..."
make -C "$SRC_DIR" test_hybrid MPIDIR="$MPIDIR" 2>&1 | tail -2

# ----- 日志目录 -----
mydate=$(date +%Y-%m-%d)
log_dir="$SRC_DIR/log/$mydate"
mkdir -p "$log_dir"

# ----- 循环运行 -----
for np in $NP_LIST; do
    echo ""
    echo "===== NP=$np ====="
    mat_base=$(basename "$MATRIX_FILE")
    mat_name="${mat_base%.*}"
    log_file="$log_dir/${METHOD}-${mat_name}-np${np}.log"
    echo "Log: $log_file"

    export LD_LIBRARY_PATH="$MPIDIR/lib:$LD_LIBRARY_PATH"
    cmd="mpirun -np $np $BIN $MATRIX_FILE $MATRIX_TYPE $BINARY $RHS_FILE"
    echo "Running: $cmd"
    eval "timeout 600 $cmd" 2>&1 | tee "$log_file"
done

echo ""
echo "All done. Logs: $log_dir/"
