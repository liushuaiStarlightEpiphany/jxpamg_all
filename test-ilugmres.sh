#!/bin/bash
set -e

export OMP_NUM_THREADS=32

NP=1
SID=25
TOL=1e-6

MATRIX_FILE="/home/spring/l_s/jxpamg_all/JXPAMG-bsr/matrix/mat_csr_128X128X128.coo"
RHS_FILE="/home/spring/l_s/jxpamg_all/JXPAMG-bsr/matrix/rhs_128X128X128.dat"
OUTPUT_FILE="test/x.txt"

PARAM_TAG="sid${SID}_np${NP}_tol${TOL}"

mydate=$(date +%Y-%m-%d)
log_dir="./log/$mydate"
mkdir -p "$log_dir"

log_file="$log_dir/$PARAM_TAG.log"
echo "Log: $log_file"

cmd="mpirun -np $NP ./solver"
cmd="$cmd $MATRIX_FILE $RHS_FILE $OUTPUT_FILE $TOL"
cmd="$cmd -sid $SID"

echo "Running: $cmd"
echo "" | tee -a "$log_file"
eval "$cmd" 2>&1 | tee -a "$log_file"

exit ${PIPESTATUS[0]}
