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
//
// Long edge phase
//
// Push Model:
// Scan for all active vertices (u) in bucket B_k
// For each long edge e=<u,v> incident to u, compute tentative distance d'(v) = d(u)+w(e)
// Send info to process owning v
// Note:
//    Let i be bucket holding v
//      i==k => self edge
//      i<k  => backward edge
//      i>k  => forward edge
//    Backward edges already relaxed
//
// Pull Model:
// Scan for all active vertices (v) in later buckets 
// For each long edge e=<u,v> incident to v send request to owner of u if w(e) < d(v) - k*delta
// Owner of u responds with new distance d'(v) if u in current bucket B_k

#include "aml.h"
#include "common.h"
#include "csr_reference.h"
#include "bitmap_reference.h"
#include <string.h>
#include <stdio.h>

#ifdef DEBUGSTATS
extern int64_t nbytes_sent,nbytes_rcvd;
#endif
// variables shared from bfs_reference
extern oned_csr_graph g;
extern int qc,q2c;
extern int* q1,*q2;
extern unsigned int* rowstarts;
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

typedef struct  __attribute__((__packed__)) requestmsg {
	float w; //weight of an edge
  float dv;
	int dest_vloc; //local index of destination vertex
	int src_vloc; //local index of source vertex
} requestmsg;

typedef struct __attribute__((__packed__)) responsemsg {
	relaxmsg msg;
	int dst_rank;
} responsemsg;

responsemsg* responsemsgs;
unsigned int msg_counter = 0;
unsigned int counter = 0;

#ifdef DEBUG
long unsigned int ph1 = 0;
long unsigned int ph2_push = 0;
long unsigned int ph2_pull = 0;
long unsigned int ph1_relax = 0;
long unsigned int ph1_actual_relax = 0;
long unsigned int ph2_relax = 0;
long unsigned int ph2_actual_relax = 0;
long unsigned int ph2_request = 0;
long unsigned int ph2_actual_request = 0;
long unsigned int ph2_response = 0;
long unsigned int ph2_actual_response = 0;
#endif

// Active message handler for relaxation
void relaxhndl(int from, void* dat, int sz) {
	relaxmsg* m = (relaxmsg*) dat;
	int vloc = m->dest_vloc; // Destination
	float w = m->w; // Weight
	float *dest_dist = &glob_dist[vloc]; // Current distance for destination
	//check if relaxation is needed: either new path is shorter or vertex not reached earlier
	if (*dest_dist < 0 || *dest_dist > w) {
#ifdef DEBUG
if(lightphase) {
  ph1_actual_relax++;
} else {
  ph2_actual_relax++;
}
#endif
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

// Pull Model:
// Scan for all active vertices (v) in later buckets 
// For each long edge e=<u,v> incident to v send request to owner of u if w(e) < d(v) - k*delta
// Owner of u responds with new distance d'(v) if u in current bucket B_k

void requesthndl(int from, void* dat, int sz) {
	requestmsg* m = (requestmsg*) dat;
	int uloc = m->src_vloc;
	int vloc = m->dest_vloc;
  float weight = m->w;
	float distance = m->dv;
	float udist = glob_dist[uloc];
	if(udist < glob_maxdelta && udist>=glob_mindelta && udist+weight >= glob_maxdelta ) {
    if(udist+weight < distance || distance == -1.0) {
#ifdef DEBUG
ph2_actual_request++;
#endif
		  responsemsgs[msg_counter].msg.w = udist+weight;
		  responsemsgs[msg_counter].msg.src_vloc = uloc;
		  responsemsgs[msg_counter].msg.dest_vloc = vloc;
		  responsemsgs[msg_counter].dst_rank = from;
		  msg_counter++; 
    }
	}
}

void responsehndl(int from, void* dat, int sz) {
	relaxmsg* m = (relaxmsg*) dat;
	int vloc = m->dest_vloc; // Destination
	float w = m->w; // Weight
	float dest_dist = glob_dist[vloc]; // Current distance for destination
	//check if relaxation is needed: either new path is shorter or vertex not reached earlier
	if (dest_dist < 0 || dest_dist > w) {
#ifdef DEBUG
ph2_actual_response++;
#endif
#ifdef DEBUG
		if(!(lightphase && !TEST_VISITEDLOC(vloc))) {
			if(from == 0) {
				printf("(%d, %d): (%d,%d) = %f\n", from, my_pe(), m->src_vloc, m->dest_vloc, *dest_dist);
			}
			counter++;
		}
#endif
		glob_dist[vloc] = w; //update distance
		pred_glob[vloc]=VERTEX_TO_GLOBAL(from,m->src_vloc); //update path by setting new parent
	}
}

//Sending relaxation active message
void send_relax(int64_t glob, float weight,int fromloc) {
	relaxmsg m = {weight,VERTEX_LOCAL(glob),fromloc};
	aml_send(&m,1,sizeof(relaxmsg),VERTEX_OWNER(glob));
}

void send_request(float weight, float dv, int64_t src, int dst) {
	requestmsg m = {weight, dv, dst, VERTEX_LOCAL(src)};
	aml_send(&m, 2, sizeof(requestmsg), VERTEX_OWNER(src));
}

void send_response(float weight, int src, int64_t dst) {
	relaxmsg m = {weight, VERTEX_LOCAL(dst), src};
	aml_send(&m, 3, sizeof(relaxmsg), VERTEX_OWNER(dst));
}

void run_sssp(int64_t root,int64_t* pred,float *dist) {

	unsigned int i,j;
	unsigned int old_settled = 1; 
	long sum=0;
	unsigned int vdegree = 0;

	// Delta determines bucket size
	float delta = 0.04625;
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

	responsemsgs = xmalloc(4*g.nlocaledges*sizeof(responsemsg));

	aml_register_handler(relaxhndl,1);
	aml_register_handler(requesthndl,2);
	aml_register_handler(responsehndl,3);

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

	unsigned int epoch_counter = 0;
	// Epochs
	int64_t lastvisited=1;
	while(sum!=0) {
#ifdef DEBUGSTATS
		double t0 = aml_time();
		nbytes_sent=0;
#endif
		//1. iterate over light edges
		while(sum!=0) {
#ifdef DEBUG
ph1++;
#endif
			CLEAN_VISITED(); // Clear visited list
			lightphase=1; // Notify relax handler that it's dealing with a light edge
			aml_barrier(); // Wait till everyone has cleared their lists
			for(i=0;i<qc;i++) {
				for(j=rowstarts[q1[i]];j<rowstarts[q1[i]+1];j++) { // Go through edges connected to i
					// Check if edge weight is less than delta
					//					if(weights[j]<delta) // Check if edge weight is less than delta
          float distance = dist[q1[i]]+weights[j];
					if(glob_mindelta <= distance && distance < glob_maxdelta) { 
#ifdef DEBUG
ph1_relax++;
#endif
						send_relax(COLUMN(j),distance,q1[i]); // Relax if necessary
					}
				}
			}
			aml_barrier(); // Ensure all relaxations/communication is done

			// Switch buffers to edges that need to be reprocessed
			qc=q2c;q2c=0;
			// Swap vertex lists
			int *tmp=q1;q1=q2;q2=tmp;
			// Set sum to # of light edges
			sum=qc;

			// Global reduction
			aml_long_allsum(&sum);
		}
		lightphase=0;
		aml_barrier();

		//2. iterate over S and heavy edges
		if(vdegree > 2) {

#ifdef DEBUG
ph2_pull++;
#endif
			for(i=0; i<g.nlocalverts; i++) {
				if(dist[i] == -1.0 || dist[i] >= glob_maxdelta) {
					for(j=rowstarts[i]; j<rowstarts[i+1]; j++) {
						//            if(weights[j] >= delta) {
						if(dist[i] > glob_mindelta+weights[j] || dist[i] == -1.0) {
#ifdef DEBUG
ph2_request++;
#endif
							send_request(weights[j], dist[i], COLUMN(j), i);
						}
					}
				}
			}
			aml_barrier();
			for(i=0; i<msg_counter; i++) {
#ifdef DEBUG
ph2_response++;
#endif
				aml_send(&(responsemsgs[i].msg), 3, sizeof(responsemsg), responsemsgs[i].dst_rank);
			}
			aml_barrier();
			msg_counter = 0;
		} else {
#ifdef DEBUG
ph2_push++;
#endif
			for(i=0;i<g.nlocalverts;i++) { // Go through local vertices
				// If current distance of i is in current bucket
				if(dist[i]>=glob_mindelta && dist[i] < glob_maxdelta) { 
					// Iterate through edges
					for(j=rowstarts[i];j<rowstarts[i+1];j++) {
						// Check if edge is heavy
						//	  				if(weights[j]>=delta) {// Check if edge is heavy
						if(dist[i]+weights[j]>=glob_maxdelta || weights[j] >= delta) { 
#ifdef DEBUG
ph2_relax++;
#endif
							send_relax(COLUMN(j),dist[i]+weights[j],i); // Send and relax if necessary
						}
					}
				}
			}
			aml_barrier(); // Finish processing heavy edges
		}


		// Move to next bucket
		glob_mindelta=glob_maxdelta;
		glob_maxdelta+=delta;
		qc=0;sum=0;

		//3. Bucket processing and checking termination condition
		int64_t lvlvisited=0; // Debug only
		vdegree = 0;
		for(i=0;i<g.nlocalverts;i++) { // Iterate through local vertices
			if(dist[i]>=glob_mindelta) { // If currect distance is in current bucket
				sum++; //how many are still to be processed
				if (dist[i] < glob_maxdelta) { // Check if in current bucket
					q1[qc++]=i; //this is lowest bucket, (add to bucket)
					vdegree += rowstarts[i+1]-rowstarts[i];
				}
			} else if(dist[i]!=-1.0) {
				lvlvisited++; // Debug only
			}
		}
		aml_long_allsum(&sum); // Sum vertices in current bucket
						
		unsigned int  num_settled = g.nglobalverts - sum;
		if (num_settled < old_settled) {
#if DEBUG
      if(delta != 1.0 && my_pe() == 0) {
        printf("Switched to Bellman-Ford\n");
      }
#endif
      delta = 1.0;			
    }
		old_settled = num_settled; 

		aml_long_allsum(&vdegree);
		int qc_temp = qc;
		aml_long_allsum(&qc_temp);
		vdegree /= qc_temp;

#ifdef DEBUGSTATS
				t0-=aml_time();
				aml_long_allsum(&lvlvisited);
				aml_long_allsum(&nbytes_sent);
				if(!my_pe()) printf("--lvl[%1.2f..%1.2f] visited %lld (total %llu) in %5.2fs, network aggr %5.2fGb/s\n",glob_mindelta,glob_maxdelta,lvlvisited-lastvisited,lvlvisited,-t0,-(double)nbytes_sent*8.0/(1.e9*t0));
				lastvisited = lvlvisited;
#endif
	  epoch_counter++;
#if DEBUG
    aml_long_allsum(&ph1_relax);
    aml_long_allsum(&ph1_actual_relax);
    aml_long_allsum(&ph2_relax);
    aml_long_allsum(&ph2_actual_relax);
    aml_long_allsum(&ph2_request);
    aml_long_allsum(&ph2_actual_request);
    aml_long_allsum(&ph2_response);
    aml_long_allsum(&ph2_actual_response);
    aml_barrier();
    if(my_pe() == 0) {
      printf("Epoch: %d, Phase 1: %lu, Phase 2 Push: %lu, Phase 3: Pull: %lu\n", epoch_counter, ph1, ph2_push, ph2_pull);
      printf("Epoch: %d, Phase1 relax msgs: %lu, Phase1 actual relaxations: %lu\n",  epoch_counter, ph1_relax, ph1_actual_relax);
      printf("Epoch: %d, Phase2 request msgs: %lu, Phase2 actual requestss: %lu\n",  epoch_counter, ph2_request, ph2_actual_request);
      printf("Epoch: %d, Phase2 response msgs: %lu, Phase2 actual responses: %lu\n", epoch_counter, ph2_response, ph2_actual_response);
      printf("Epoch: %d, Phase2 relax msgs: %lu, Phase2 actual relaxations: %lu\n",  epoch_counter, ph2_relax, ph2_actual_relax);
    }
ph1_relax = 0;
ph1_actual_relax = 0;
ph2_relax = 0;
ph2_actual_relax = 0;
ph2_request = 0;
ph2_actual_request = 0;
ph2_response = 0;
ph2_actual_response = 0;
#endif
  }
  free(responsemsgs);
#if DEBUG
  ph1 = 0;
  ph2_push = 0;
  ph2_pull = 0;
#endif
}


		// Initialize distance
void clean_shortest(float* dist) {
	int i;
	for(i=0;i<g.nlocalverts;i++) dist[i]=-1.0;
}
#endif
