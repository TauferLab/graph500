cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0074 NEW)
find_package(Kokkos REQUIRED)

project(graph500)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(GRAPH_GENERATOR_MPI "Use MPI for the graph generator" ON)
option(REUSE_CSR_FOR_VALIDATION "Reuse the CSR for the validation step" ON)
option(PROCS_PER_NODE_NOT_POWER_OF_TWO "Use non powers of 2 processes per node" OFF)
option(SIZE_MUST_BE_A_POWER_OF_TWO "Use non powers of 2 for nodes" ON)
option(DEBUGSTATS "Enable debugging stats" OFF)
option(SSSP "Enable SSSP" ON)

if(GRAPH_GENERATOR_MPI)
	add_definitions(-DGRAPH_GENERATOR_MPI)
endif(GRAPH_GENERATOR_MPI)

if(REUSE_CSR_FOR_VALIDATION)
	add_definitions(-DREUSE_CSR_FOR_VALIDATION)
endif(REUSE_CSR_FOR_VALIDATION)

if(PROCS_PER_NODE_NOT_POWER_OF_TWO)
	add_definitions(-DPROCS_PER_NODE_NOT_POWER_OF_TWO)
endif(PROCS_PER_NODE_NOT_POWER_OF_TWO)

if(SIZE_MUST_BE_A_POWER_OF_TWO)
	add_definitions(-DSIZE_MUST_BE_A_POWER_OF_TWO)
endif(SIZE_MUST_BE_A_POWER_OF_TWO)

if(DEBUGSTATS)
	add_definitions(-DDEBUGSTATS)
endif(DEBUGSTATS)

if(SSSP)
	add_definitions(-DSSSP)
endif(SSSP)

add_executable(graph500_reference_bfs_sssp
				src/main.cpp
				src/utils.cpp
				src/validate.cpp
				src/csr_reference.cpp
				src/bfs_reference.cpp
				src/sssp_reference.cpp
				aml/aml.c
				generator/graph_generator.c
				generator/make_graph.c
				generator/splittable_mrg.c
				generator/utils.c
				)

add_executable(graph500_custom_bfs_sssp
				src/main.cpp
				src/utils.cpp
				src/validate.cpp
				src/csr_custom.cpp
				src/bfs_custom.cpp
				src/sssp_custom.cpp
				aml/aml.c
				generator/graph_generator.c
				generator/make_graph.c
				generator/splittable_mrg.c
				generator/utils.c
				)

add_executable(graph500_kokkos_bfs_sssp
				src/main_kokkos.cpp
				src/utils.cpp
				src/validate.cpp
				src/csr_kokkos.cpp
				src/bfs_kokkos.cpp
				src/sssp_kokkos.cpp
				aml/aml.c
				generator/graph_generator.c
				generator/make_graph.c
				generator/splittable_mrg.c
				generator/utils.c
				)

target_link_libraries(graph500_reference_bfs_sssp Kokkos::kokkos)
target_link_libraries(graph500_custom_bfs_sssp Kokkos::kokkos)
target_link_libraries(graph500_kokkos_bfs_sssp Kokkos::kokkos)
