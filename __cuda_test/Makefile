a.out: matmula.cu
	nvcc -std=c++20 -lineinfo -Xcompiler -fopenmp -lcuda -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86 -g matmula.cu  && ./a.out 
ncu: a.out
	ncu --metric l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st,sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm.max,sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_st.max,gpu__time_active.max ./a.out

mem: a.out
	ncu --section MemoryWorkloadAnalysis --metric l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st,sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm.max,sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_st.max,gpu__time_active.max ./a.out
