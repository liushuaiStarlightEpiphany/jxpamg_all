#!/bin/bash
# ============================================================
# Mixed (IR) CPR-GMRES 测试脚本
# 双精度外校正 + 单精度内层 CPR-GMRES
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

# --- IR 参数（见 cpr.tex 表~\ref{tab:params_ir}）---
MAX_IR_ITER=100         # 最大 IR 外迭代次数
IR_TOL=1e-4             # IR 外收敛容差
INNER_MAX_ITER=10       # 内层 GMRES 最大迭代（固定次数，不检测内收敛）
INNER_TOL=1e-3          # 内层 GMRES 收敛容差（实际由固定迭代控制）

# --- 内层 CPR 参数（见 cpr.tex 表~\ref{tab:params_cpr}）---
DECOUP_TYPE="TIMPES"    # ABF / TIMPES / TIMPES2 / ANL / SEM / QI / NONE
STAGE1_MAXIT=1          # Stage 1 (AMG) 最大迭代
STAGE2_MAXIT=2          # Stage 2 迭代次数
STAGE2_TYPE=2           # Stage 2 类型: 2=HGS, 3=HSGS
KDIM=30                 # 内层 GMRES Krylov 子空间维度
PRINT_LEVEL=1           # 打印级别
# ============================================================

# ----- 路径与编译环境 -----
SRC_DIR="/home/spring/l_s/jxpamg_all/JXFPAMG-bsr-fmt"
SRC_FILE="$SRC_DIR/src/test_bsr_cprgmres_mixed.c"
BIN="$SRC_DIR/test_bsr_cprgmres_mixed"
MPIDIR="/opt/intel/oneapi/mpi/2021.6.0"
METHOD="mixed"

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
sed -i "s/int max_ir_iterations = [0-9]*;/int max_ir_iterations = $MAX_IR_ITER;/" "$SRC_FILE"
sed -i "s/Real_double ir_tolerance = [0-9.e\-]*;/Real_double ir_tolerance = $IR_TOL;/" "$SRC_FILE"
sed -i "s/int inner_max_iterations = [0-9]*;/int inner_max_iterations = $INNER_MAX_ITER;/" "$SRC_FILE"
sed -i "s/Real_float inner_tolerance = [0-9.e\-]*;/Real_float inner_tolerance = $INNER_TOL;/" "$SRC_FILE"
sed -i "s/int k_dim = [0-9]*;/int k_dim = $KDIM;/" "$SRC_FILE"
sed -i "s/JX_DecoupType decoup_type = JX_DECOUP_[A-Z0-9]*;/JX_DecoupType decoup_type = JX_DECOUP_$DECOUP_TYPE;/" "$SRC_FILE"
sed -i "s/JXF_CPRSetParameter(cpr, \"stage1_maxit\", [0-9]*);/JXF_CPRSetParameter(cpr, \"stage1_maxit\", $STAGE1_MAXIT);/" "$SRC_FILE"
sed -i "s/JXF_CPRSetParameter(cpr, \"stage2_maxit\", [0-9]*);/JXF_CPRSetParameter(cpr, \"stage2_maxit\", $STAGE2_MAXIT);/" "$SRC_FILE"
sed -i "s/JXF_CPRSetParameter(cpr, \"print_level\", [0-9]*);/JXF_CPRSetParameter(cpr, \"print_level\", $PRINT_LEVEL);/" "$SRC_FILE"

echo "  MAX_IR_ITER=$MAX_IR_ITER IR_TOL=$IR_TOL INNER_MAX_ITER=$INNER_MAX_ITER INNER_TOL=$INNER_TOL"
echo "  KDIM=$KDIM DECOUP=$DECOUP_TYPE STAGE1_MAXIT=$STAGE1_MAXIT STAGE2_MAXIT=$STAGE2_MAXIT"

# ----- 编译 -----
echo "Compiling..."
make -C "$SRC_DIR" test_mixed MPIDIR="$MPIDIR" 2>&1 | tail -2

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
