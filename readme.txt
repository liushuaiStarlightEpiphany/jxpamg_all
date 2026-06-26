环境(编译器、版本): oneapi
ONEAPI_ROOT = /online1/public/support/amd/intel_compiler/2022u2/oneapi

文件夹结构说明:
- jxpamg-0220/:jxpamg源代码，默认double类型
- src/:接口函数源代码，其中solver.c里为main函数，读取矩阵右端文件，构造线性系统，并调用JXPAMG_SOLVER接口函数进行求解；jxpamg_interface.c为jxpamg接口函数文件，其余为一些依赖文件
- make.sh:编译脚本
- test_AMG_1.sh:运行脚本示例,对应测试项一
- test_AMG_2.sh:运行脚本示例,对应测试项二
- test_AMG_3.sh:运行脚本示例,对应测试项三
运行脚本中涉及的参数: ./solver $dataroot/pb$pb 9104256 272717760 x_${pb}.txt 1e-12 $pb ...
前五个参数分别为：矩阵右端文件路径（这里是CSR格式二进制文件存储）、矩阵行数、矩阵非零元数、解向量传出路径、求解tol、问题编号，
其余参数均在jxpamg_interface.c中使用到，如：sid为求解方法, maxiter为最大迭代步数, ct为coarsen_type粗化类型, ipt为interp_type插值类型, cyt为cycle_type, rlx为relax_type磨光类型, cs为粗空间求解器


编译(makefile怎么修改、cmake、make):  
1.JXPAMG需要重新编译   
    ①进入~/jxpamg-0220/makefile.pub修改
    mpi的路径: MPIDIR=$INTEL_ROOT/intel_compiler/2020u4/compilers_and_libraries_2020.4.304/linux/mpi/intel64
    mkl的路径: MKL_DIR=$INTEL_ROOT/intel_compiler/2020u4/compilers_and_libraries_2020.4.304/linux/mkl
    ②修改完路径后进行编译
    make clean
    make -j 16 
    cd ..

2.Makefile中的路径修改
    vi Makefile
    mpi的路径: MPIDIR=$INTEL_ROOT/intel_compiler/2020u4/compilers_and_libraries_2020.4.304/linux/mpi/intel64
    mkl的路径: MKL_DIR=$INTEL_ROOT/intel_compiler/2020u4/compilers_and_libraries_2020.4.304/linux/mkl
    make clean
    make -j 16
    cd ..

3. 编译
    sh make.sh
