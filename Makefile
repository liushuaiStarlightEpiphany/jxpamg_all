###################################
#   Edit by Zhao Li 2021.05.28    #
#   Modified by zlj for BSR testing      #
###################################
# .SUFFIXES:.c .f90 .f .for
# mpi 根目录
# MPIDIR = /vol8/appsoftware/mpi-x
# jxfpamg 根目录
JXFPAMGDIR = ./jxfpamg-0220
# jxpamg 根目录
JXPAMGDIR = ../JXPAMG-bsr/jxpamg-0220
# MKL 根目录
MKL_DIR = /opt/intel/oneapi/mkl/2022.1.0
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Use the OpenMP parallelism [YES] or not [#YES]
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
JXFPAMG_USE_OMP = YES
JXF_USE_MUMPS = #YES
JXF_USE_BIG_INT = #YES
JXF_USE_BIG_DOUBLE = #YES
ifdef JXF_USE_MUMPS
topdir = /home/export/base/ycsc_zhaol/xinx/online1/real_mumps_openmp/MUMPS_5.4.0
endif
ifdef JXF_USE_BIG_INT
JXF_BIG_INT_OPTION = -DJXF_USING_BIG_INT=1
else
JXF_BIG_INT_OPTION = -DJXF_USING_BIG_INT=0
endif
ifdef JXF_USE_BIG_DOUBLE
JXF_BIG_DOUBLE_OPTION = -DJXF_USING_BIG_DOUBLE=1
else
JXF_BIG_DOUBLE_OPTION = -DJXF_USING_BIG_DOUBLE=0
endif
ifdef JXF_USE_SINGLE
JXF_SINGLE_OPTION = -DJXF_SINGLE=1
else
JXF_SINGLE_OPTION = -DJXF_SINGLE=0
endif
ifdef JXFPAMG_USE_OMP
CC   = mpicc -fopenmp
F90C = mpifort -fopenmp
else
CC   = mpicc
F90C = mpifort
endif
ifdef JXF_USE_MUMPS
JXF_MUMPS_OPTION = -DJXF_USING_MUMPS=1
libdir = $(topdir)/lib
include $(topdir)/Makefile.inc
MUMPS_INCLUDE_O = $(INCS)
MUMPS_INCLUDE_T = -I$(topdir)/include
MUMPS_INCLUDES = $(MUMPS_INCLUDE_O) $(MUMPS_INCLUDE_T)
MUMPS_LIB_O = -L$(libdir) -ldmumps -lsmumps
MUMPS_LIB_T = -L$(libdir) -lmumps_common
MUMPS_LIBS = $(MUMPS_LIB_O) $(MUMPS_LIB_T) $(LORDERINGS) $(LIBS) $(LIBOTHERS)
else
JXF_MUMPS_OPTION = -DJXF_USING_MUMPS=0
endif
INC_MKL = -I $(MKL_DIR)/include  
LIB_MKL = -L $(MKL_DIR)/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -ldl
INCLUDES = -I $(JXFPAMGDIR)/include -I $(JXPAMGDIR)/include -I ${MPIDIR}/include ${INC_MKL} ${MUMPS_INCLUDES} 
LIB      = -L $(JXPAMGDIR)/lib -L $(JXFPAMGDIR)/lib -lJXFPAMG  -lJXPAMG -L ${MPIDIR}/lib   ${MUMPS_LIBS}  -lmpi -lm #${LIB_MKL}
FLAGS = ${JXF_BIG_INT_OPTION} ${JXF_BIG_DOUBLE_OPTION} ${JXF_SINGLE_OPTION}
ifdef JXFPAMG_USE_OMP
JXF_OMP_CP_OPTION = -DJXF_USING_OPENMP=1
else
JXF_OMP_CP_OPTION = -DJXF_USING_OPENMP=0
endif
C_COMPILE_FLAGS = ${FLAGS} ${JXF_OMP_CP_OPTION} -Wl,--allow-multiple-definition
F_COMPILE_FLAGS = ${FLAGS} ${JXF_OMP_CP_OPTION}
CFLAGS =  ${C_COMPILE_FLAGS} -I. ${INCLUDES} 
FFLAGS =  ${F_COMPILE_FLAGS} -I. ${INCLUDES} 
.cpp.o:
	${CC} -o $@ -c ${CFLAGS} $<
.c.o:
	${CC} -o $@ -c ${CFLAGS} $<
.f.o:
	${F90C} -o $@ -c ${FFLAGS} $<
.for.o:
	${F90C} -o $@ -c ${FFLAGS} $<
.f90.o:
	${F90C} -o $@ -c ${FFLAGS} $<
# 原始求解器
SOLVER_OBJS = 	./src/solver.o \
		./src/jxfpamg_interface.o
# BSR测试程序
BSR_TEST_OBJS = ./src/test_bsr_mv.o
# CPR-GMRES测试程序
CPR_TEST_OBJS = ./src/test_bsr_cprgmres.o
# 添加混合精度目标
MIXED_TEST_OBJS = ./src/test_bsr_cprgmres_mixed.o
# 添加混合精度目标
Hybrid_TEST_OBJS = ./src/test_bsr_cprgmres_hybrid.o
# 编译原始求解器
solver: ${SOLVER_OBJS}
	${CC} -O3 -o solver ${FLAGS} ${SOLVER_OBJS} ${LIB} ${CFLAGS}
# 编译BSR测试程序
test_bsr_mv: ${BSR_TEST_OBJS}
	${CC} -O3 -o test_bsr_mv ${FLAGS} ${BSR_TEST_OBJS} ${LIB} ${CFLAGS}
# 编译CPR-GMRES测试程序
test_cpr: ${CPR_TEST_OBJS}
	${CC} -O3 -o test_bsr_cprgmres ${FLAGS} ${CPR_TEST_OBJS} ${LIB} ${CFLAGS}
test_mixed: ${MIXED_TEST_OBJS}
	${CC} -O3 -o test_bsr_cprgmres_mixed ${FLAGS} ${MIXED_TEST_OBJS} ${LIB} ${CFLAGS}
test_hybrid: ${Hybrid_TEST_OBJS}
	${CC} -O3 -o test_bsr_cprgmres_hybrid ${FLAGS} ${Hybrid_TEST_OBJS} ${LIB} ${CFLAGS}
clean:
	rm -rf ./src/*.o
clean_all:
	rm -rf ./src/*.o
	cd $(JXFPAMGDIR) && make clean