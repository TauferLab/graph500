// Stub for custom SSSP implementation

#include "aml.h"
#include "common.h"
#include "common_custom.h"
#include "csr_reference.h"
#include "bitmap_reference.h"

extern oned_csr_graph g;
extern int64_t* column,*pred_glob,visited_size;
extern unsigned long * visited;
extern void* heap;
// Matrix data
extern int* rowstarts;
extern float *weights;

#ifdef SSSP


//user provided function to be called several times to implement kernel 3: single source shortest path
//function has filled with -1 and -1.0 pred and dist arrays
//at exit dist should have shortest distance to root for each vertice otherwise -1
//pred array should point to next vertie in shortest path or -1 if vertex unreachable
//pred[VERTEX_LOCAL(root)] should be root and dist should be 0.0

void run_sssp(int64_t root,int64_t* pred,float *dist) {
    pred_glob=pred;

    CLEAN_VISITED();

    heap_insert(heap, root, 0);
    pred[root] = root;
    dist[root] = 0.0;

    while (!heap_empty(heap)) {
        int v = heap_remove(heap);
        if (TEST_VISITEDLOC(v)) {
            continue;
        }
        SET_VISITEDLOC(v);
        float dist_v = dist[v];

        for(long j=rowstarts[v];j<rowstarts[v+1];j++) {
            int u = COLUMN(j); 
            float w = weights[j];
            if (!TEST_VISITEDLOC(u)) {
                if (dist[u] == -1 || dist_v + w < dist[u]) {
                    dist[u] = dist_v + w;
                    pred[u] = v;
                    heap_insert(heap, u, dist[u]);
                }
            }
        }
    }
}

//user provided function to prefill dist array with whatever value
void clean_shortest(float* dist) {
	int i;
	for(i=0;i<g.nlocalverts;i++) dist[i]=-1.0;
}
#endif
