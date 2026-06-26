#!/bin/bash
#SBATCH --job-name=test_bsr        # 作业名
#SBATCH -N 1                        # 节点数
#SBATCH -n 16                       # MPI任务数
#SBATCH --partition=mt_test         # 计算分区
#SBATCH --exclusive                 # 不共享节点

# ============================================================
#   参数配置区域
# ============================================================

# ----------------
# 1. 并行配置
# ----------------
NP_LIST="1"               # MPI 进程数列表，如 "1 2 4 8"

# ----------------
# 2. 矩阵配置
# ----------------
# 小测试矩阵
# MATRIX_FILE="./data/A_bsr_302X302X3.dat"
# MATRIX_TYPE="1"            # 0=CSR, 1=BSR
# BINARY="0"                 # 0=文本, 1=二进制
# RHS_FILE="./data/b_302x1.dat"

# SPE10 大矩阵
MATRIX_FILE="./data/spe10_bsr/A_bsr_spe10.bin"
MATRIX_TYPE="1"            # 0=CSR, 1=BSR
BINARY="1"                 # 0=文本, 1=二进制
RHS_FILE="./data/spe10_bsr/b_spe10.dat"

# ----------------
# 3. 方法标识 (用于日志文件名)
# ----------------
# stage2_solver_type: 2=BGS, 3=BILU-GMRES
METHOD="BILU-GMRES"

# ----------------
# 4. 解耦类型
# ----------------
# 0=NONE 1=ABF 2=ANL 3=SEM 4=QI 5=TIMPES 6=TIMPES2
# 设为 "all" 遍历所有类型
DECOUP_TYPE="5"

# ============================================================
#   环境配置
# ============================================================

export LD_LIBRARY_PATH=/opt/intel/oneapi/mpi/2021.6.0/lib:LD_LIBRARY_PATH
# export UCX_TLS=^cma

# ============================================================
#   日志配置
# ============================================================

DATE=$(date +%Y%m%d)
TIME=$(date +%H%M%S)
LOG_BASE="./log/${DATE}"
MATRIX_NAME=$(basename "$MATRIX_FILE" | sed 's/\.[^.]*$//')

mkdir -p "$LOG_BASE"

# ============================================================
#   运行函数
# ============================================================

decoup_label() {
    case "$1" in
        0) echo "NONE" ;;
        1) echo "ABF" ;;
        2) echo "ANL" ;;
        3) echo "SEM" ;;
        4) echo "QI" ;;
        5) echo "TIMPES" ;;
        6) echo "TIMPES2" ;;
        *) echo "UNKNOWN" ;;
    esac
}

run_test() {
    local np=$1
    local dt=$2
    local dt_label=$(decoup_label "$dt")
    local logfile="$LOG_BASE/${MATRIX_NAME}-${dt_label}-np${np}-${TIME}.log"
    
    echo "============================================================"
    echo "  进程数: $np"
    echo "  解耦:   $dt_label ($dt)"
    echo "  日志:   $logfile"
    echo "============================================================"
    
    {
        echo "============================================================"
        echo "  Test Configuration"
        echo "============================================================"
        echo "  Matrix:       $MATRIX_FILE"
        echo "  Matrix type:  $([ "$MATRIX_TYPE" = "1" ] && echo "BSR" || echo "CSR")"
        echo "  Binary:       $([ "$BINARY" = "1" ] && echo "Yes" || echo "No")"
        echo "  RHS:          ${RHS_FILE:-Default (all ones)}"
        echo "  Method:       $METHOD"
        echo "  Decoupling:   $dt_label ($dt)"
        echo "  NP:           $np"
        echo "  Start time:   $(date '+%Y-%m-%d %H:%M:%S')"
        echo "============================================================"
        echo ""
        
        if [ -n "$RHS_FILE" ] && [ -f "$RHS_FILE" ]; then
            mpirun -n "$np" ./test_bsr_cprgmres "$MATRIX_FILE" "$MATRIX_TYPE" "$BINARY" "$RHS_FILE" "$dt"
        else
            mpirun -n "$np" ./test_bsr_cprgmres "$MATRIX_FILE" "$MATRIX_TYPE" "$BINARY" "" "$dt"
        fi
        
        echo ""
        echo "============================================================"
        echo "  End time:     $(date '+%Y-%m-%d %H:%M:%S')"
        echo "============================================================"
    } 2>&1 | tee "$logfile"
    
    echo ""
}

# ============================================================
#   主程序
# ============================================================

echo ""
echo "============================================================"
echo "  BSR CPR-GMRES Test Suite"
echo "============================================================"
echo "  Date:         $DATE"
echo "  Time:         $TIME"
echo "  Matrix:       $MATRIX_FILE"
echo "  Method:       $METHOD"
echo "  NP list:      $NP_LIST"
echo "  Log dir:      $LOG_BASE"
echo "============================================================"
echo ""

# 确定解耦类型列表
if [ "$DECOUP_TYPE" = "all" ]; then
    DT_LIST="0 1 2 3 4 5 6"
else
    DT_LIST="$DECOUP_TYPE"
fi

for np in $NP_LIST; do
    for dt in $DT_LIST; do
        run_test "$np" "$dt"
    done
done

echo ""
echo "============================================================"
echo "  All tests completed!"
echo "  Log directory: $LOG_BASE"
echo "============================================================"
echo ""
