#!/bin/sh
#SBATCH -p q_amd_share
#SBATCH -N 1 
#SBATCH -n 4
#SBATCH --exclusive

module load amd/intel_compiler/2022u2


# create log
if [ ! -d "./log" ];then
mkdir log
fi

# create sol
if [ ! -d "./sol" ];then
mkdir sol
fi

mydate=$(date +%Y-%m-%d)

cd ./log
# create mydate
if [ ! -d "./$mydate" ];then
mkdir $mydate
fi
cd ..

cd ./sol
# create mydate
if [ ! -d "./$mydate" ];then
mkdir $mydate
fi
cd ..

dataroot=/home/export/base/sc100286/sc100286/online1/zlj/data/matrix
pb1="2D-poisson"
pb2="3D-poisson"
pb3="kcs"
nts=1

#### pb1
# C-AMG
for np in 1 2 4 8 16 ; do
        mpirun -np $np ./solver \
        $dataroot/$pb1/csrmat_1023X1023.dat 9104256 272717760 x_${pb}.txt 1e-6 $pb1 $dataroot/$pb1/rhs_1023X1023.dat 0\
        -sid 12 -maxiter 100 -nts $nts -ct 6 -ipt 6  -rlx 3  \
                2>&1 |tee -a ./log/$mydate/$pb-sid$sid-np$np.log
done
# HMIS-AMG
for np in 1 2 4 8 16 ; do
        mpirun -np $np ./solver \
        $dataroot/$pb1/csrmat_1023X1023.dat 9104256 272717760 x_${pb}.txt 1e-6 $pb1 $dataroot/$pb1/rhs_1023X1023.dat 0\
        -sid 22 -maxiter 100 -nts $nts -ct 10 -ipt 6  -rlx 3  \
                2>&1 |tee -a ./log/$mydate/$pb-sid$sid-np$np.log
done
UA-AMG
for np in 1 2 4 8 16 ; do
        mpirun -np $np ./solver \
        $dataroot/$pb1/csrmat_1023X1023.dat 9104256 272717760 x_${pb}.txt 1e-6 $pb1 $dataroot/$pb1/rhs_1023X1023.dat 0\
        -sid 32 -maxiter 100 -nts $nts -ct 66 -ipt 66  -rlx 6 -cyt 2 -str 0.06 -mxct 25 \
                2>&1 |tee -a ./log/$mydate/$pb-sid$sid-np$np.log
done

#pb2
for np in 1 2 4 8 16 ; do
        mpirun -np $np ./solver \
        $dataroot/$pb2/mat_csr_127X127X127.dat 9104256 272717760 x_${pb}.txt 1e-6 $pb2 $dataroot/$pb2/rhs_127X127X127.dat 0\
        -sid 12 -maxiter 100 -nts $nts -ct 6 -ipt 6  -rlx 3  \
                2>&1 |tee -a ./log/$mydate/$pb-sid$sid-np$np.log
done
for np in 1 2 4 8 16 ; do
        mpirun -np $np ./solver \
        $dataroot/$pb2/mat_csr_127X127X127.dat 9104256 272717760 x_${pb}.txt 1e-6 $pb2 $dataroot/$pb2/rhs_127X127X127.dat 0\
        -sid 22 -maxiter 100 -nts $nts -ct 10 -ipt 6  -rlx 3  \
                2>&1 |tee -a ./log/$mydate/$pb-sid$sid-np$np.log
done
for np in 1 2 4 8 16 ; do
        mpirun -np $np ./solver \
        $dataroot/$pb2/mat_csr_127X127X127.dat 9104256 272717760 x_${pb}.txt 1e-6 $pb2 $dataroot/$pb2/rhs_127X127X127.dat 0\
        -sid 32 -maxiter 100 -nts $nts -ct 66 -ipt 66  -rlx 6 -cyt 2 -str 0.06 -mxct 25 \
                2>&1 |tee -a ./log/$mydate/$pb-sid$sid-np$np.log
done
#pb3
for np in 1 2 4 8 16 ; do
        mpirun -np $np ./solver \
        $dataroot/$pb3/matrix_p_101_zlj.dat 9104256 272717760 x_${pb}.txt 1e-4 $pb3 $dataroot/$pb3/vector_rhs_p_101_zlj.dat 2\
        -sid 12 -maxiter 100 -nts $nts -ct 6 -ipt 6  -rlx 3  \
                2>&1 |tee -a ./log/$mydate/$pb-sid$sid-np$np.log
done
for np in 1 2 4 8 16 ; do
        mpirun -np $np ./solver \
        $dataroot/$pb3/matrix_p_101_zlj.dat 9104256 272717760 x_${pb}.txt 1e-4 $pb3 $dataroot/$pb3/vector_rhs_p_101_zlj.dat 2\
        -sid 22 -maxiter 100 -nts $nts -ct 10 -ipt 6  -rlx 3  \
                2>&1 |tee -a ./log/$mydate/$pb-sid$sid-np$np.log
done
for np in 1 2 4 8 16 ; do
        mpirun -np $np ./solver \
        $dataroot/$pb3/matrix_p_101_zlj.dat 9104256 272717760 x_${pb}.txt 1e-4 $pb3 $dataroot/$pb3/vector_rhs_p_101_zlj.dat 2\
        -sid 32 -maxiter 100 -nts $nts -ct 66 -ipt 66  -rlx 6 -cyt 2 -str 0.06 -mxct 25 \
                2>&1 |tee -a ./log/$mydate/$pb-sid$sid-np$np.log
done

