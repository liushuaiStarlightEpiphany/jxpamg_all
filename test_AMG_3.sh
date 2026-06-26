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
pb5="3D-poisson-27pt"
nts=1



for ns in 1 ; do
        for it in  1 ; do
                for np in 1 2 4 8 10 ; do
                        mpirun -np $np ./solver \
                        $dataroot/$pb5/mat_csr_127X127X127_27pt.dat 9104256 272717760 x_${pb}.txt 1e-18 $pb1 $dataroot/$pb5/rhs_127X127X127_27pt.dat 0\
                        -sid 12 -maxiter $it -nts $nts -ct 8 -ipt 5  -rlx 3 -ns_down $ns -ns_up 0\
                                2>&1 |tee -a ./log/$mydate/$pb-sid$sid-np$np.log
                done
        done
done



for ns in 1 ; do
        for rlx in  3 ; do
                for np in 1 2 4 8 10 ; do
                        mpirun -np $np ./solver \
                        $dataroot/$pb5/mat_csr_255X255X255_27pt.dat 9104256 272717760 x_${pb}.txt 1e-40 $pb1 $dataroot/$pb5/rhs_255X255X255_27pt.dat 0\
                        -sid 12 -maxiter 1 -nts $nts -ct 8 -ipt 5  -rlx $rlx -ns_down $ns -ns_up 0  \
                                2>&1 |tee -a ./log/$mydate/$pb-sid$sid-np$np.log
                done
        done
done

