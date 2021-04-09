//Stub for custom BFS implementations

#include "common.h"
#include "common_custom.h"
#include "aml.h"
#include "csr_reference.h"
#include "bitmap_reference.h"
#include <stdint.h>
#include <inttypes.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <limits.h>
#include <assert.h>
#include <stdint.h>

//VISITED bitmap parameters
unsigned long *visited;
int64_t visited_size;
int *frontier;

int64_t *pred_glob,*column;
unsigned int *rowstarts;
oned_csr_graph g;
float* weights;
void* heap;

//user should provide this function which would be called once to do kernel 1: graph convert
void make_graph_data_structure(const tuple_graph* const tg) {
	//graph conversion, can be changed by user by replacing oned_csr.{c,h} with new graph format 
	convert_graph_to_oned_csr(tg, &g);

	column=g.column;
	rowstarts=g.rowstarts;
	//user code to allocate other buffers for bfs
	frontier = xmalloc(g.nglobalverts*sizeof(int));
#ifdef SSSP
	visited_size = (g.nlocalverts + ulong_bits - 1) / ulong_bits;
	visited = xmalloc(visited_size*sizeof(unsigned long));
    weights=g.weights;
    heap = heap_alloc();
#endif
}

//user should provide this function which would be called several times to do kernel 2: breadth first search
//pred[] should be root for root, -1 for unrechable vertices
//prior to calling run_bfs pred is set to -1 by calling clean_pred
void run_bfs(int64_t root, int64_t* pred) {
    pred_glob=pred;

    CLEAN_VISITED();

    int queue_start = 0;
    int queue_end = 0;
    queue_insert(frontier, &queue_start, &queue_end, root);
    pred[root] = root;

    while (queue_start < queue_end) {
        int v = queue_remove(frontier, &queue_start, &queue_end);

        for(long j=rowstarts[v];j<rowstarts[v+1];j++) {
            int u = COLUMN(j); 
            if (pred[u] == -1) {
                pred[u] = v;
                queue_insert(frontier, &queue_start, &queue_end, u);
            }
        }
    }
}

//we need edge count to calculate teps. Validation will check if this count is correct
//user should change this function if another format (not standart CRS) used
void get_edge_count_for_teps(int64_t* edge_visit_count) {
	long i,j;
	long edge_count=0;
	for(i=0;i<g.nlocalverts;i++)
		if(pred_glob[i]!=-1) {
			for(j=g.rowstarts[i];j<g.rowstarts[i+1];j++)
				if(COLUMN(j)<=VERTEX_TO_GLOBAL(my_pe(),i))
					edge_count++;
		}
	aml_long_allsum(&edge_count);
	*edge_visit_count=edge_count;
}

//user provided function to initialize predecessor array to whatevere value user needs
void clean_pred(int64_t* pred) {
	int i;
	for(i=0;i<g.nlocalverts;i++) pred[i]=-1;
}

//user provided function to be called once graph is no longer needed
void free_graph_data_structure(void) {
	free_oned_csr_graph(&g);
    free(frontier);
#ifdef SSSP
	free(visited);
    heap_free(heap);
#endif
}

//user should change is function if distribution(and counts) of vertices is changed
size_t get_nlocalverts_for_pred(void) {
	return g.nlocalverts;
}
