/* Copyright (c) 2011-2017 Graph500 Steering Committee
   All rights reserved.
   Developed by:                Anton Korzh anton@korzh.us
                                Graph500 Steering Committee
                                http://www.graph500.org
   New code under University of Illinois/NCSA Open Source License
   see license.txt or https://opensource.org/licenses/NCSA
*/

// Graph500: Kernel 3 SSSP
// Simple parallel delta-stepping with relaxations as Active Messages

#include "aml.h"
#include "common.h"
#include "csr_reference.h"
#include "bitmap_reference.h"
#include <string.h>

#ifdef DEBUGSTATS
extern int64_t nbytes_sent,nbytes_rcvd;
#endif
// variables shared from bfs_reference
extern oned_csr_graph g;
extern int qc,q2c;
extern int* q1,*q2;
extern int* rowstarts;
extern int64_t* column,*pred_glob,visited_size;
extern unsigned long * visited;
#ifdef SSSP
//global variables as those accesed by active message handler
float *glob_dist;
float glob_maxdelta, glob_mindelta; //range for current bucket
float *weights;
volatile int lightphase;

//Relaxation data type 
typedef struct  __attribute__((__packed__)) relaxmsg {
	float w; //weight of an edge
	int dest_vloc; //local index of destination vertex
	int src_vloc; //local index of source vertex
} relaxmsg;

// Active message handler for relaxation
void relaxhndl(int from, void* dat, int sz) {
	relaxmsg* m = (relaxmsg*) dat;
	int vloc = m->dest_vloc; // Destination
	float w = m->w; // Weight
	float *dest_dist = &glob_dist[vloc]; // Current distance for destination
	//check if relaxation is needed: either new path is shorter or vertex not reached earlier
	if (*dest_dist < 0 || *dest_dist > w) {
		*dest_dist = w; //update distance
		pred_glob[vloc]=VERTEX_TO_GLOBAL(from,m->src_vloc); //update path by setting new parent

		if(lightphase && !TEST_VISITEDLOC(vloc)) //Bitmap used to track if was already relaxed with light edge
		{
			if(w < glob_maxdelta) { //if falls into current bucket needs further reprocessing
				q2[q2c++] = vloc;
				SET_VISITEDLOC(vloc);
			}
		}
	}
}

//Sending relaxation active message
void send_relax(int64_t glob, float weight,int fromloc) {
	relaxmsg m = {weight,VERTEX_LOCAL(glob),fromloc};
	aml_send(&m,1,sizeof(relaxmsg),VERTEX_OWNER(glob));
}

void run_sssp(int64_t root,int64_t* pred,float *dist) {

	unsigned int i,j;
	long sum=0;

  // Delta determines bucket size
	float delta = 0.025;
  // Start of bucket
	glob_mindelta=0.0;
  // End of bucket
	glob_maxdelta=delta;
  // Distance array
	glob_dist=dist;
	weights=g.weights;
  // Vertex array
	pred_glob=pred;
  // Index of first free element
  // light edges: < qc
	qc=0;q2c=0;

	aml_register_handler(relaxhndl,1);

	if (VERTEX_OWNER(root) == my_pe()) {
    // Mark root as visited
		q1[0]=VERTEX_LOCAL(root);
    // Set index for next visited vertex
		qc=1;
    // Set root distance to 0
		dist[VERTEX_LOCAL(root)]=0.0;
    // Parent of root is root
		pred[VERTEX_LOCAL(root)]=root;
	}

	aml_barrier();
	sum=1; // at least 1 bucket needs to be processed

  // Epochs
	int64_t lastvisited=1;
	while(sum!=0) {
#ifdef DEBUGSTATS
		double t0 = aml_time();
		nbytes_sent=0;
#endif
		//1. iterate over light edges
		while(sum!=0) {
			CLEAN_VISITED(); // Clear visited list
			lightphase=1; // Notify relax handler that it's dealing with a light edge
			aml_barrier(); // Wait till everyone has cleared their lists
			for(i=0;i<qc;i++)
				for(j=rowstarts[q1[i]];j<rowstarts[q1[i]+1];j++) // Go through edges connected to i
					if(weights[j]<delta) // Check if edge weight is less than delta
						send_relax(COLUMN(j),dist[q1[i]]+weights[j],q1[i]); // Relax if necessary
			aml_barrier(); // Ensure all relaxations/communication is done

      // Switch buffers to edges that need to be reprocessed
			qc=q2c;q2c=0;
      // Swap vertex lists, not sure if necessary
      int *tmp=q1;q1=q2;q2=tmp;
      // Set sum to # of light edges
			sum=qc;
      // Global reduction
			aml_long_allsum(&sum);
		}
		lightphase=0;
		aml_barrier();

		//2. iterate over S and heavy edges
		for(i=0;i<g.nlocalverts;i++) // Go through local vertices
      // If current distance of i is in current bucket
			if(dist[i]>=glob_mindelta && dist[i] < glob_maxdelta) { 
        // Iterate through edges
				for(j=rowstarts[i];j<rowstarts[i+1];j++)
					if(weights[j]>=delta) // Check if edge is heavy
						send_relax(COLUMN(j),dist[i]+weights[j],i); // Send and relax if necessary
			}
		aml_barrier(); // Finish processing heavy edges

    // Move to next bucket
		glob_mindelta=glob_maxdelta;
		glob_maxdelta+=delta;
		qc=0;sum=0;

		//3. Bucket processing and checking termination condition
		int64_t lvlvisited=0; // Debug only
		for(i=0;i<g.nlocalverts;i++) // Iterate through local vertices
			if(dist[i]>=glob_mindelta) { // If currect distance is in current bucket
				sum++; //how many are still to be processed
				if (dist[i] < glob_maxdelta) // Check if in current bucket
					q1[qc++]=i; //this is lowest bucket, (add to bucket)
			} else if(dist[i]!=-1.0) lvlvisited++; // Debug only
		aml_long_allsum(&sum); // Sum vertices in current bucket
#ifdef DEBUGSTATS
		t0-=aml_time();
		aml_long_allsum(&lvlvisited);
		aml_long_allsum(&nbytes_sent);
		if(!my_pe()) printf("--lvl[%1.2f..%1.2f] visited %lld (total %llu) in %5.2fs, network aggr %5.2fGb/s\n",glob_mindelta,glob_maxdelta,lvlvisited-lastvisited,lvlvisited,-t0,-(double)nbytes_sent*8.0/(1.e9*t0));
		lastvisited = lvlvisited;
#endif
	}

}

// Initialize distance
void clean_shortest(float* dist) {
	int i;
	for(i=0;i<g.nlocalverts;i++) dist[i]=-1.0;
}
#endif
