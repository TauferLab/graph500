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
#include "csr_custom.h"
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
extern float *weights;
volatile int lightphase;
//Kokkos views
Kokkos::View<int*> q1_kokkos, q2_kokkos;
Kokkos::View<int[1]> q1c_kokkos, q2c_kokkos, lightphase_kokkos;
Kokkos::View<int[1]>::HostMirror q1c_kokkos_host, q2c_kokkos_host;
Kokkos::View<unsigned int*> rowstarts_kokkos;
Kokkos::View<int64_t*> column_kokkos, pred_glob_kokkos;
Kokkos::View<int64_t[1]> visited_size_kokkos;
Kokkos::View<unsigned long*> visited_kokkos;
Kokkos::View<float*> weights_kokkos, *glob_dist_kokkos;
// Relax/request views
Kokkos::View<float*> relaxmsgs_d_w;
Kokkos::View<int*> 	 relaxmsgs_d_dst_vloc;
Kokkos::View<int*> 	 relaxmsgs_d_src_vloc;
Kokkos::View<float*> requestmsgs_d_w;
Kokkos::View<float*> requestmsgs_d_dv;
Kokkos::View<int*>   requestmsgs_d_dst_vloc;
Kokkos::View<int*>   requestmsgs_d_src_vloc;
// Host mirrors
Kokkos::View<float*>::HostMirror relaxmsgs_h_w;
Kokkos::View<int*>::HostMirror   relaxmsgs_h_dst_vloc;
Kokkos::View<int*>::HostMirror   relaxmsgs_h_src_vloc;
Kokkos::View<float*>::HostMirror requestmsgs_h_w;
Kokkos::View<float*>::HostMirror requestmsgs_h_dv;
Kokkos::View<int*>::HostMirror   requestmsgs_h_dst_vloc;
Kokkos::View<int*>::HostMirror   requestmsgs_h_src_vloc;
// Counters
Kokkos::View<int[1]> relaxcounter;
Kokkos::View<int[1]> requestcounter;
Kokkos::View<int[1]>::HostMirror relaxcounter_host;
Kokkos::View<int[1]>::HostMirror requestcounter_host;

void setup_sssp_kokkos_globals() {
	// Declare Views
	int lp = lightphase;
	int64_t colalloc = BYTES_PER_VERTEX*g.nlocaledges;
	colalloc += (4095);
	colalloc /= 4096;
	colalloc *= 4096;
	rowstarts_kokkos = Kokkos::View<unsigned int*>("Rowstarts view", g.nlocalverts+1);
	column_kokkos 	= Kokkos::View<int64_t*>("Column view", colalloc/sizeof(int64_t));
	weights_kokkos 	= Kokkos::View<float*>("Weights view", g.nlocaledges);
	q1_kokkos 		= Kokkos::View<int*>("q1 view", g.nlocalverts);
	q2_kokkos 		= Kokkos::View<int*>("q2 view", g.nlocalverts);
	visited_kokkos 	= Kokkos::View<unsigned long*>("Visited view", visited_size);
	// Setup scalars
	q1c_kokkos = Kokkos::View<int[1]>("q1 counter");
	q2c_kokkos = Kokkos::View<int[1]>("q2 counter");
	Kokkos::deep_copy(q1c_kokkos, qc);
	Kokkos::deep_copy(q2c_kokkos, q2c);
	q1c_kokkos_host = Kokkos::create_mirror_view(q1c_kokkos);
	q2c_kokkos_host = Kokkos::create_mirror_view(q2c_kokkos);
	Kokkos::deep_copy(visited_size_kokkos, visited_size);
	Kokkos::deep_copy(lightphase_kokkos, lp);
	// Copy data
	Kokkos::View<unsigned int*>::HostMirror rowstarts_host = Kokkos::create_mirror_view(rowstarts_kokkos);
	Kokkos::parallel_for("Copy rowstarts", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,g.nlocalverts+1),
	KOKKOS_LAMBDA(const size_t i) {
		rowstarts_host(i) = rowstarts[i];
	});
	Kokkos::deep_copy(rowstarts_kokkos, rowstarts_host);
	Kokkos::View<int64_t*>::HostMirror column_host = Kokkos::create_mirror_view(column_kokkos);
	Kokkos::parallel_for("Copy columns", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,colalloc/sizeof(int64_t)),
	KOKKOS_LAMBDA(const size_t i) {
		column_host(i) = column[i];
	});
	Kokkos::deep_copy(column_kokkos, column_host);
	Kokkos::View<float*>::HostMirror weights_host = Kokkos::create_mirror_view(weights_kokkos);
	Kokkos::parallel_for("Copy weights", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,g.nlocaledges),
	KOKKOS_LAMBDA(const size_t i) {
		weights_host(i) = weights[i];
	});
	Kokkos::deep_copy(weights_kokkos, weights_host);
	Kokkos::View<int*>::HostMirror q1_host = Kokkos::create_mirror_view(q1_kokkos);
	Kokkos::parallel_for("Copy q1", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,g.nlocalverts),
	KOKKOS_LAMBDA(const size_t i) {
		q1_host(i) = q1[i];
	});
	Kokkos::deep_copy(q1_kokkos, q1_host);
	Kokkos::View<int*>::HostMirror q2_host = Kokkos::create_mirror_view(q2_kokkos);
	Kokkos::parallel_for("Copy q2", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,g.nlocalverts),
	KOKKOS_LAMBDA(const size_t i) {
		q2_host(i) = q2[i];
	});
	Kokkos::deep_copy(q2_kokkos, q2_host);
	Kokkos::View<unsigned long*>::HostMirror visited_host = Kokkos::create_mirror_view(visited_kokkos);
	Kokkos::parallel_for("Copy visited", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,visited_size),
	KOKKOS_LAMBDA(const size_t i) {
		visited_host(i) = visited[i];
	});
	Kokkos::deep_copy(visited_kokkos, visited_host);
	// Relax/request views
	relaxmsgs_d_w = Kokkos::View<float*>("Relax message weight", g.nlocaledges);
	relaxmsgs_d_dst_vloc = Kokkos::View<int*>("Relax message dst_vloc", g.nlocaledges);
	relaxmsgs_d_src_vloc = Kokkos::View<int*>("Relax message dst_vloc", g.nlocaledges);
	requestmsgs_d_w = Kokkos::View<float*>("request message weight", g.nlocaledges);
	requestmsgs_d_dv = Kokkos::View<float*>("request message dv", g.nlocaledges);
	requestmsgs_d_dst_vloc = Kokkos::View<int*>("request message dst_vloc", g.nlocaledges);
	requestmsgs_d_src_vloc = Kokkos::View<int*>("request message dst_vloc", g.nlocaledges);
	// Host mirrors
	relaxmsgs_h_w = Kokkos::create_mirror_view(relaxmsgs_d_w);
	relaxmsgs_h_dst_vloc = Kokkos::create_mirror_view(relaxmsgs_d_dst_vloc);
	relaxmsgs_h_src_vloc = Kokkos::create_mirror_view(relaxmsgs_d_src_vloc);
	requestmsgs_h_w = Kokkos::create_mirror_view(requestmsgs_d_w);
	requestmsgs_h_dv = Kokkos::create_mirror_view(requestmsgs_d_dv);
	requestmsgs_h_dst_vloc = Kokkos::create_mirror_view(requestmsgs_d_dst_vloc);
	requestmsgs_h_src_vloc = Kokkos::create_mirror_view(requestmsgs_d_src_vloc);
	// Counters
	relaxcounter = Kokkos::View<int[1]>("Relaxation counter");
	requestcounter = Kokkos::View<int[1]>("Request counter");
	relaxcounter_host = Kokkos::create_mirror_view(relaxcounter);
	requestcounter_host = Kokkos::create_mirror_view(requestcounter);
}

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

void relaxhndl_kokkos(int from, void* dat, int sz) {
	relaxmsg* m = (relaxmsg*) dat;
	int vloc = m->dest_vloc; // Destination
	float w = m->w; // Weight
	int vsrc = m->src_vloc;
	Kokkos::parallel_for("relax", Kokkos::RangePolicy<Kokkos::Serial>(0,1), KOKKOS_LAMBDA(const int idx) {
		float dest_dist = glob_dist[vloc];
		if(dest_dist < 0 || dest_dist > w) {
			glob_dist[vloc] = w;
			pred_glob[vloc] = VERTEX_TO_GLOBAL(from, vsrc);
			bool visited_vloc = ((visited_kokkos(vloc >> 6) & (1ULL << (vloc & 63))) != 0);
			if(lightphase && !visited_vloc) {
				if(w < glob_maxdelta) {
					int idx = q2c_kokkos(0)++;
					q2[idx] = vloc;
					visited_kokkos(vloc >> 6) |= (1ULL << (vloc & 63));
				}
			}
		}
	});
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
			responsemsgs[msg_counter].msg.w = udist+weight;
			responsemsgs[msg_counter].msg.src_vloc = uloc;
			responsemsgs[msg_counter].msg.dest_vloc = vloc;
			responsemsgs[msg_counter].dst_rank = from;
			msg_counter++; 
		}
	}
}

void requesthndl_kokkos(int from, void* dat, int sz) {
	requestmsg* m = (requestmsg*) dat;
	int uloc = m->src_vloc;
	int vloc = m->dest_vloc;
	float weight = m->w;
	float distance = m->dv;
	float udist;
	auto subview = Kokkos::subview(*glob_dist_kokkos, uloc);
	Kokkos::deep_copy(udist, subview);
	if(udist < glob_maxdelta && udist>=glob_mindelta && udist+weight >= glob_maxdelta ) {
		if(udist+weight < distance || distance == -1.0) {
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
	int vsrc = m->src_vloc;
	float w = m->w; // Weight
	float dest_dist = glob_dist[vloc]; // Current distance for destination
	//check if relaxation is needed: either new path is shorter or vertex not reached earlier
	if (dest_dist < 0 || dest_dist > w) {
		glob_dist[vloc] = w; //update distance
		pred_glob[vloc]=VERTEX_TO_GLOBAL(from,m->src_vloc); //update path by setting new parent
	}
}

void responsehndl_kokkos(int from, void* dat, int sz) {
	relaxmsg* m = (relaxmsg*) dat;
	int vloc = m->dest_vloc; // Destination
	int vsrc = m->src_vloc;
	float w = m->w; // Weight
	//check if relaxation is needed: either new path is shorter or vertex not reached earlier
	Kokkos::parallel_for("response", Kokkos::RangePolicy<Kokkos::Serial>(0,1), KOKKOS_LAMBDA(const int i) {
		float dest_dist = glob_dist[vloc]; // Current distance for destination
		if(dest_dist < 0 || dest_dist > w) {
			glob_dist[vloc] = w;
			pred_glob[vloc] = VERTEX_TO_GLOBAL(from, vsrc);
		}
	});
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

void run_sssp_kokkos(int64_t root,Kokkos::View<int64_t*>& pred, Kokkos::View<float*>& dist) {
//printf("Rank %d started sssp_kokkos with root %ld\n", rank, root);
//	unsigned int i,j;
	unsigned int i;
	unsigned int old_settled = 1; 
//	long sum=0;
//	unsigned int vdegree = 0;
	Kokkos::View<long[1]> sum_d("Sum view");
	Kokkos::deep_copy(sum_d, 0);
	Kokkos::View<long[1]>::HostMirror sum_h = Kokkos::create_mirror_view(sum_d);
	sum_h(0) = 0;
	Kokkos::View<unsigned int[1]> vdegree_kokkos("Vertex degree view");
	Kokkos::View<unsigned int[1]>::HostMirror vdegree_host = Kokkos::create_mirror_view(vdegree_kokkos);
	Kokkos::deep_copy(vdegree_kokkos, 0);
	vdegree_host(0) = 0;

	glob_dist_kokkos = &dist;

//printf("Rank %d declareded scalars\n", rank);
//if(rank==0)
//printf("Root at %ld\n", root);

	// Delta determines bucket size
	float delta = 0.04625;
	// Start of bucket
	glob_mindelta=0.0;
	// End of bucket
	glob_maxdelta=delta;
	// Distance array
	glob_dist=dist.data();
	weights=g.weights;
	// Vertex array
	pred_glob=pred.data();
	// Index of first free element
	// light edges: < qc
	Kokkos::deep_copy(q1c_kokkos, 0);
	Kokkos::deep_copy(q2c_kokkos, 0);
	q1c_kokkos_host(0) = 0;
	q2c_kokkos_host(0) = 0;
	int* old_q1 = q1;
	int* old_q2 = q2;
	q1 = q1_kokkos.data();
	q2 = q2_kokkos.data();
	column = column_kokkos.data();

//printf("Rank %d initialized scalars\n", rank);

	responsemsgs = (responsemsg*) xmalloc(4*g.nlocaledges*sizeof(responsemsg));
//	// Relax/request views
//	Kokkos::View<float*> relaxmsgs_d_w("Relax message weight", g.nlocaledges);
//	Kokkos::View<int*> 	 relaxmsgs_d_dst_vloc("Relax message dst_vloc", g.nlocaledges);
//	Kokkos::View<int*> 	 relaxmsgs_d_src_vloc("Relax message dst_vloc", g.nlocaledges);
//	Kokkos::View<float*> requestmsgs_d_w("request message weight", g.nlocaledges);
//	Kokkos::View<float*> requestmsgs_d_dv("request message dv", g.nlocaledges);
//	Kokkos::View<int*>   requestmsgs_d_dst_vloc("request message dst_vloc", g.nlocaledges);
//	Kokkos::View<int*>   requestmsgs_d_src_vloc("request message dst_vloc", g.nlocaledges);
//	// Host mirrors
//	Kokkos::View<float*>::HostMirror relaxmsgs_h_w = Kokkos::create_mirror_view(relaxmsgs_d_w);
//	Kokkos::View<int*>::HostMirror   relaxmsgs_h_dst_vloc = Kokkos::create_mirror_view(relaxmsgs_d_dst_vloc);
//	Kokkos::View<int*>::HostMirror   relaxmsgs_h_src_vloc = Kokkos::create_mirror_view(relaxmsgs_d_src_vloc);
//	Kokkos::View<float*>::HostMirror requestmsgs_h_w = Kokkos::create_mirror_view(requestmsgs_d_w);
//	Kokkos::View<float*>::HostMirror requestmsgs_h_dv = Kokkos::create_mirror_view(requestmsgs_d_dv);
//	Kokkos::View<int*>::HostMirror   requestmsgs_h_dst_vloc = Kokkos::create_mirror_view(requestmsgs_d_dst_vloc);
//	Kokkos::View<int*>::HostMirror   requestmsgs_h_src_vloc = Kokkos::create_mirror_view(requestmsgs_d_src_vloc);
//	// Counters
//	Kokkos::View<int[1]> relaxcounter = Kokkos::View<int[1]>("Relaxation counter");
//	Kokkos::View<int[1]> requestcounter = Kokkos::View<int[1]>("Request counter");
//	Kokkos::View<int[1]>::HostMirror relaxcounter_host = Kokkos::create_mirror_view(relaxcounter);
//	Kokkos::View<int[1]>::HostMirror requestcounter_host = Kokkos::create_mirror_view(requestcounter);
	Kokkos::deep_copy(relaxcounter, 0);
	Kokkos::deep_copy(requestcounter, 0);

//printf("Rank %d done with initializing kokkos views\n", rank);

	aml_register_handler(relaxhndl_kokkos,1);
	aml_register_handler(requesthndl_kokkos,2);
	aml_register_handler(responsehndl_kokkos,3);

//printf("Rank %d done registering handlers\n", rank);

	if (VERTEX_OWNER(root) == my_pe()) {
		Kokkos::parallel_for("Update root", Kokkos::RangePolicy<>(0,1), KOKKOS_LAMBDA(const size_t idx) {
			// Mark root as visited
			q1[0] = VERTEX_LOCAL(root);
			// Set index for next visited vertex
			q1c_kokkos(0) = 1;
			// Parent of root is root
			dist(VERTEX_LOCAL(root)) = 0.0;
			pred(VERTEX_LOCAL(root)) = root;
		});
	}
//printf("Rank %d done with initial setup\n", rank);

	aml_barrier();
	sum_h(0)=1; // at least 1 bucket needs to be processed

	// Epochs
	int64_t lastvisited=1;
	while(sum_h(0)!=0) {
#ifdef DEBUGSTATS
		double t0 = aml_time();
		nbytes_sent=0;
#endif
		//1. iterate over light edges
		while(sum_h(0)!=0) {
			Kokkos::deep_copy(visited_kokkos, 0);
			lightphase=1; // Notify relax handler that it's dealing with a light edge
			aml_barrier(); // Wait till everyone has cleared their lists
			Kokkos::parallel_for("lightphase", Kokkos::RangePolicy<>(0,q1c_kokkos_host(0)), KOKKOS_LAMBDA(const unsigned int i) {
				for(unsigned int j=rowstarts_kokkos(q1[i]);j<rowstarts_kokkos(q1[i]+1);j++) { // Go through edges connected to i
					// Check if edge weight is less than delta
					// if(weights[j]<delta) // Check if edge weight is less than delta
					float distance = dist(q1[i])+weights_kokkos(j);
					if(glob_mindelta <= distance && distance < glob_maxdelta) { 
						if(rank == VERTEX_OWNER(COLUMN(j))) {
							int vloc = VERTEX_LOCAL(COLUMN(j));
							float dest_dist = dist(vloc);
							if(dest_dist < 0 || dest_dist > distance) {
								Kokkos::atomic_assign(&(dist(vloc)), distance);
								Kokkos::atomic_assign(&(pred(vloc)), VERTEX_TO_GLOBAL(rank, q1[i]));
								bool visited_vloc = ((visited_kokkos(vloc >> 6) & (1ULL << (vloc & 63))) != 0);
								if(lightphase && !visited_vloc) {
									if(distance < glob_maxdelta) {
										int idx = Kokkos::atomic_fetch_add(&(q2c_kokkos(0)), 1);
										q2[idx] = vloc;
										Kokkos::atomic_or(&(visited_kokkos(vloc >> 6)), (1UL << (vloc & 63)));
									}
								}
							}
						} else {
							int idx = Kokkos::atomic_fetch_add(&(relaxcounter(0)), 1);
							relaxmsgs_d_w(idx) = distance;
							relaxmsgs_d_dst_vloc(idx) = j;
							relaxmsgs_d_src_vloc(idx) = q1[i];
						}
					}
				}
			});
			aml_barrier(); // Ensure all relaxations/communication is done

			Kokkos::deep_copy(relaxcounter_host, relaxcounter);
Kokkos::fence();
			auto relaxmsgs_sub_h_w = Kokkos::subview(relaxmsgs_h_w, Kokkos::make_pair(0,relaxcounter_host(0)));
			auto relaxmsgs_sub_d_w = Kokkos::subview(relaxmsgs_d_w, Kokkos::make_pair(0,relaxcounter_host(0)));
			auto relaxmsgs_sub_h_src_vloc = Kokkos::subview(relaxmsgs_h_src_vloc, Kokkos::make_pair(0,relaxcounter_host(0)));
			auto relaxmsgs_sub_d_src_vloc = Kokkos::subview(relaxmsgs_d_src_vloc, Kokkos::make_pair(0,relaxcounter_host(0)));
			auto relaxmsgs_sub_h_dst_vloc = Kokkos::subview(relaxmsgs_h_dst_vloc, Kokkos::make_pair(0,relaxcounter_host(0)));
			auto relaxmsgs_sub_d_dst_vloc = Kokkos::subview(relaxmsgs_d_dst_vloc, Kokkos::make_pair(0,relaxcounter_host(0)));
			Kokkos::deep_copy(relaxmsgs_sub_h_w, relaxmsgs_sub_d_w);
			Kokkos::deep_copy(relaxmsgs_sub_h_dst_vloc, relaxmsgs_sub_d_dst_vloc);
			Kokkos::deep_copy(relaxmsgs_sub_h_src_vloc, relaxmsgs_sub_d_src_vloc);
Kokkos::fence();
			for(i=0; i<relaxcounter_host(0); i++) {
				int64_t global = COLUMN(relaxmsgs_sub_h_dst_vloc(i));
				send_relax(global, relaxmsgs_sub_h_w(i), relaxmsgs_sub_h_src_vloc(i));
			}
Kokkos::fence();
			aml_barrier();
			Kokkos::deep_copy(relaxcounter, 0);

			// Switch buffers to edges that need to be reprocessed
			Kokkos::deep_copy(q1c_kokkos, q2c_kokkos);
			Kokkos::deep_copy(q1c_kokkos_host, q1c_kokkos);
			Kokkos::deep_copy(q2c_kokkos, 0);
			Kokkos::deep_copy(q2c_kokkos_host, 0);
Kokkos::fence();
			// Swap vertex lists
			int *tmp=q1;q1=q2;q2=tmp;
			// Set sum to # of light edges
			sum_h(0) = q1c_kokkos_host(0);
Kokkos::fence();
			// Global reduction
			aml_long_allsum(&sum_h(0));
		}
		lightphase=0;
		aml_barrier();
Kokkos::fence();

		//2. iterate over S and heavy edges
		if(vdegree_host(0) > 2) {
			Kokkos::parallel_for("S and heavy edges", Kokkos::RangePolicy<>(0,g.nlocalverts), 
			KOKKOS_LAMBDA(const unsigned int i) {
				if(dist(i) == -1.0 || dist(i) >= glob_maxdelta) {
					for(unsigned int j=rowstarts_kokkos(i); j<rowstarts_kokkos(i+1); j++) {
						if(dist(i) > glob_mindelta+weights_kokkos(j) || dist(i) == -1.0) {
							int idx = Kokkos::atomic_fetch_add(&(requestcounter(0)), 1);
							requestmsgs_d_w(idx) = weights_kokkos(j);
							requestmsgs_d_dv(idx) = dist(i);
							requestmsgs_d_dst_vloc(idx) = i;
							requestmsgs_d_src_vloc(idx) = j;
						}
					}
				}
			});
			aml_barrier();
			Kokkos::deep_copy(requestcounter_host, requestcounter);
Kokkos::fence();
			auto requestmsgs_sub_d_w = Kokkos::subview(requestmsgs_d_w, Kokkos::make_pair(0, requestcounter_host(0)));
			auto requestmsgs_sub_h_w = Kokkos::subview(requestmsgs_h_w, Kokkos::make_pair(0, requestcounter_host(0)));
			auto requestmsgs_sub_d_dv = Kokkos::subview(requestmsgs_d_dv, Kokkos::make_pair(0, requestcounter_host(0)));
			auto requestmsgs_sub_h_dv = Kokkos::subview(requestmsgs_h_dv, Kokkos::make_pair(0, requestcounter_host(0)));
			auto requestmsgs_sub_d_dst_vloc = Kokkos::subview(requestmsgs_d_dst_vloc, Kokkos::make_pair(0, requestcounter_host(0)));
			auto requestmsgs_sub_h_dst_vloc = Kokkos::subview(requestmsgs_h_dst_vloc, Kokkos::make_pair(0, requestcounter_host(0)));
			auto requestmsgs_sub_d_src_vloc = Kokkos::subview(requestmsgs_d_src_vloc, Kokkos::make_pair(0, requestcounter_host(0)));
			auto requestmsgs_sub_h_src_vloc = Kokkos::subview(requestmsgs_h_src_vloc, Kokkos::make_pair(0, requestcounter_host(0)));
			Kokkos::deep_copy(requestmsgs_sub_h_w, requestmsgs_sub_d_w);
			Kokkos::deep_copy(requestmsgs_sub_h_dv, requestmsgs_sub_d_dv);
			Kokkos::deep_copy(requestmsgs_sub_h_dst_vloc, requestmsgs_sub_d_dst_vloc);
			Kokkos::deep_copy(requestmsgs_sub_h_src_vloc, requestmsgs_sub_d_src_vloc);
Kokkos::fence();
			for(i=0; i<requestcounter_host(0); i++) {
				int64_t src = COLUMN(requestmsgs_sub_h_src_vloc(i));
				send_request(requestmsgs_sub_h_w(i), requestmsgs_sub_h_dv(i), src, requestmsgs_sub_h_dst_vloc(i));
			}
			aml_barrier();
			Kokkos::deep_copy(requestcounter, 0);
			for(i=0; i<msg_counter; i++) {
				aml_send(&(responsemsgs[i].msg), 3, sizeof(responsemsg), responsemsgs[i].dst_rank);
			}
			aml_barrier();
			msg_counter = 0;
Kokkos::fence();
		} else {
			Kokkos::parallel_for("serial loop", Kokkos::RangePolicy<Kokkos::Serial>(0,g.nlocalverts), KOKKOS_LAMBDA(const unsigned int i) {
				if(dist(i) >= glob_mindelta && dist(i) < glob_maxdelta) {
					for(unsigned int j=rowstarts_kokkos(i); j<rowstarts_kokkos(i+1); j++) {
						if(dist(i)+weights_kokkos(j) >= glob_maxdelta || weights_kokkos(j) >= delta) {
//							send_relax(COLUMN(j), dist(i)+weights_kokkos(j), i);
							int idx = Kokkos::atomic_fetch_add(&(relaxcounter(0)), 1);
							relaxmsgs_d_w(idx) = dist(i)+weights_kokkos(j);
							relaxmsgs_d_dst_vloc(idx) = j;
							relaxmsgs_d_src_vloc(idx) = i;
						}
					}
				}
			});
			Kokkos::deep_copy(relaxcounter_host, relaxcounter);
Kokkos::fence();
			auto relaxmsgs_sub_h_w = Kokkos::subview(relaxmsgs_h_w, Kokkos::make_pair(0,relaxcounter_host(0)));
			auto relaxmsgs_sub_d_w = Kokkos::subview(relaxmsgs_d_w, Kokkos::make_pair(0,relaxcounter_host(0)));
			auto relaxmsgs_sub_h_src_vloc = Kokkos::subview(relaxmsgs_h_src_vloc, Kokkos::make_pair(0,relaxcounter_host(0)));
			auto relaxmsgs_sub_d_src_vloc = Kokkos::subview(relaxmsgs_d_src_vloc, Kokkos::make_pair(0,relaxcounter_host(0)));
			auto relaxmsgs_sub_h_dst_vloc = Kokkos::subview(relaxmsgs_h_dst_vloc, Kokkos::make_pair(0,relaxcounter_host(0)));
			auto relaxmsgs_sub_d_dst_vloc = Kokkos::subview(relaxmsgs_d_dst_vloc, Kokkos::make_pair(0,relaxcounter_host(0)));
			Kokkos::deep_copy(relaxmsgs_sub_h_w, relaxmsgs_sub_d_w);
			Kokkos::deep_copy(relaxmsgs_sub_h_dst_vloc, relaxmsgs_sub_d_dst_vloc);
			Kokkos::deep_copy(relaxmsgs_sub_h_src_vloc, relaxmsgs_sub_d_src_vloc);
Kokkos::fence();
			for(i=0; i<relaxcounter_host(0); i++) {
				int64_t global = COLUMN(relaxmsgs_sub_h_dst_vloc(i));
				send_relax(global, relaxmsgs_sub_h_w(i), relaxmsgs_sub_h_src_vloc(i));
			}
			aml_barrier(); // Finish processing heavy edges
			Kokkos::deep_copy(relaxcounter, 0);
		}


		// Move to next bucket
		glob_mindelta=glob_maxdelta;
		glob_maxdelta+=delta;
		q1c_kokkos_host(0) = 0;
		sum_h(0)=0;
		Kokkos::deep_copy(q1c_kokkos, 0);
		Kokkos::deep_copy(sum_d, 0);

		//3. Bucket processing and checking termination condition
		int64_t lvlvisited=0; // Debug only
		Kokkos::deep_copy(vdegree_kokkos, 0);
Kokkos::fence();
		Kokkos::parallel_for("Bucket processing", Kokkos::RangePolicy<>(0, g.nlocalverts), KOKKOS_LAMBDA(const unsigned int i) {
			if(dist(i) >= glob_mindelta) {
				Kokkos::atomic_increment(&(sum_d(0)));
				if(dist(i) < glob_maxdelta) {
					int idx = Kokkos::atomic_fetch_add(&(q1c_kokkos(0)), 1);
					q1[idx] = i;
					Kokkos::atomic_add(&(vdegree_kokkos(0)), rowstarts_kokkos(i+1) - rowstarts_kokkos(i));
				}
			}
		});
		Kokkos::deep_copy(q1c_kokkos_host, q1c_kokkos);
		Kokkos::deep_copy(vdegree_host, vdegree_kokkos);
		Kokkos::deep_copy(sum_h, sum_d);
Kokkos::fence();
		aml_long_allsum(&sum_h(0)); // Sum vertices in current bucket
						
		unsigned int  num_settled = g.nglobalverts - sum_h(0);
		if (num_settled < old_settled) {
			delta = 1.0;			
		}
		old_settled = num_settled; 

		aml_long_allsum(&vdegree_host(0));
		int qc_temp = q1c_kokkos_host(0);
		aml_long_allsum(&qc_temp);
		vdegree_host(0) /= qc_temp;
		Kokkos::deep_copy(vdegree_kokkos, vdegree_host);
Kokkos::fence();

#ifdef DEBUGSTATS
		t0-=aml_time();
		aml_long_allsum(&lvlvisited);
		aml_long_allsum(&nbytes_sent);
		if(!my_pe()) printf("--lvl[%1.2f..%1.2f] visited %lld (total %llu) in %5.2fs, network aggr %5.2fGb/s\n",glob_mindelta,glob_maxdelta,lvlvisited-lastvisited,lvlvisited,-t0,-(double)nbytes_sent*8.0/(1.e9*t0));
		lastvisited = lvlvisited;
#endif
	}
	q1 = old_q1;
	q2 = old_q2;
	free(responsemsgs);
}

void run_sssp_openmp(int64_t root,int64_t* pred,float *dist) {
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

	responsemsgs = (responsemsg*) xmalloc(4*g.nlocaledges*sizeof(responsemsg));
	relaxmsg *relaxmsgs = (relaxmsg*) xmalloc(g.nlocaledges*sizeof(relaxmsg));
	requestmsg *requestmsgs = (requestmsg*) xmalloc(g.nlocaledges*sizeof(requestmsg));
	int relaxcounter = 0;
	int requestcounter = 0;

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
			CLEAN_VISITED(); // Clear visited list
			lightphase=1; // Notify relax handler that it's dealing with a light edge
			aml_barrier(); // Wait till everyone has cleared their lists
//#pragma omp parallel for private(j)
			for(i=0;i<qc;i++) {
				for(j=rowstarts[q1[i]];j<rowstarts[q1[i]+1];j++) { // Go through edges connected to i
					// Check if edge weight is less than delta
					// if(weights[j]<delta) // Check if edge weight is less than delta
					float distance = dist[q1[i]]+weights[j];
					if(glob_mindelta <= distance && distance < glob_maxdelta) { 
						if(rank == VERTEX_OWNER(COLUMN(j))) {
//							relaxmsg m = {distance, VERTEX_LOCAL(COLUMN(j)), q1[i]};
//							relaxhndl(rank, &m, sizeof(relaxmsg));							
							int vloc = VERTEX_LOCAL(COLUMN(j));
							float dest_dist = glob_dist[vloc];
							if(dest_dist < 0 || dest_dist > distance) {
#pragma omp atomic write
								glob_dist[vloc] = distance;
#pragma omp atomic write
								pred_glob[vloc] = VERTEX_TO_GLOBAL(rank, q1[i]);
								if(lightphase && !TEST_VISITEDLOC(vloc)) {
									if(distance < glob_maxdelta) {
										int idx = 0;
#pragma omp atomic capture
										idx = q2c++;
										q2[idx] = vloc;
//										SET_VISITEDLOC(vloc);
#pragma omp atomic update
										visited[vloc >> 6] |= (1ULL << (vloc % 63));
									}
								}
							}
						} else {
//							send_relax(COLUMN(j),distance,q1[i]); // Relax if necessary
							int idx = 0;
#pragma omp atomic capture
							idx = relaxcounter++;
							relaxmsgs[idx].w = distance;
							relaxmsgs[idx].dest_vloc = j;
							relaxmsgs[idx].src_vloc = q1[i];
						}
					}
				}
			}
			aml_barrier(); // Ensure all relaxations/communication is done

			for(i=0; i<relaxcounter; i++) {
				int64_t global = COLUMN(relaxmsgs[i].dest_vloc);
				relaxmsgs[i].dest_vloc = VERTEX_LOCAL(global);
				aml_send(&(relaxmsgs[i]), 1, sizeof(relaxmsg), VERTEX_OWNER(global));
			}
			aml_barrier();
			relaxcounter = 0;

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
#pragma omp parallel for private(j)
			for(i=0; i<g.nlocalverts; i++) {
				if(dist[i] == -1.0 || dist[i] >= glob_maxdelta) {
					for(j=rowstarts[i]; j<rowstarts[i+1]; j++) {
						//if(weights[j] >= delta) {
						if(dist[i] > glob_mindelta+weights[j] || dist[i] == -1.0) {
							int idx = 0;
#pragma omp atomic capture
							idx = requestcounter++;
							requestmsgs[idx].w = weights[j];
							requestmsgs[idx].dv = dist[i];
							requestmsgs[idx].dest_vloc = i;
							requestmsgs[idx].src_vloc = j;
//							send_request(weights[j], dist[i], COLUMN(j), i);
						}
					}
				}
			}
			aml_barrier();
			for(i=0; i<requestcounter; i++) {
				int64_t src = COLUMN(requestmsgs[i].src_vloc);
				requestmsgs[i].src_vloc = VERTEX_LOCAL(src);
				aml_send(&(requestmsgs[i]), 2, sizeof(requestmsg), VERTEX_OWNER(src));
			}
			aml_barrier();
			requestcounter = 0;
			for(i=0; i<msg_counter; i++) {
				aml_send(&(responsemsgs[i].msg), 3, sizeof(responsemsg), responsemsgs[i].dst_rank);
			}
			aml_barrier();
			msg_counter = 0;
		} else {
			for(i=0;i<g.nlocalverts;i++) { // Go through local vertices
				// If current distance of i is in current bucket
				if(dist[i]>=glob_mindelta && dist[i] < glob_maxdelta) { 
					// Iterate through edges
					for(j=rowstarts[i];j<rowstarts[i+1];j++) {
						// Check if edge is heavy
						//if(weights[j]>=delta) {// Check if edge is heavy
						if(dist[i]+weights[j]>=glob_maxdelta || weights[j] >= delta) { 
							int idx = 0;
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
#pragma omp parallel for
		for(i=0;i<g.nlocalverts;i++) { // Iterate through local vertices
			if(dist[i]>=glob_mindelta) { // If currect distance is in current bucket
#pragma omp atomic update
				sum++; //how many are still to be processed
				if (dist[i] < glob_maxdelta) { // Check if in current bucket
					int idx = 0;
#pragma omp atomic capture
					idx = qc++;
					q1[idx] = i;
#pragma omp atomic update
					vdegree += rowstarts[i+1]-rowstarts[i];
//					q1[qc++]=i; //this is lowest bucket, (add to bucket)
//					vdegree += rowstarts[i+1]-rowstarts[i];
				}
			} else if(dist[i]!=-1.0) {
				lvlvisited++; // Debug only
			}
		}
		aml_long_allsum(&sum); // Sum vertices in current bucket
						
		unsigned int  num_settled = g.nglobalverts - sum;
		if (num_settled < old_settled) {
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
	}
	free(responsemsgs);
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

	responsemsgs = (responsemsg*) xmalloc(4*g.nlocaledges*sizeof(responsemsg));

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
			CLEAN_VISITED(); // Clear visited list
			lightphase=1; // Notify relax handler that it's dealing with a light edge
			aml_barrier(); // Wait till everyone has cleared their lists
			for(i=0;i<qc;i++) {
				for(j=rowstarts[q1[i]];j<rowstarts[q1[i]+1];j++) { // Go through edges connected to i
					// Check if edge weight is less than delta
					// if(weights[j]<delta) // Check if edge weight is less than delta
					float distance = dist[q1[i]]+weights[j];
					if(glob_mindelta <= distance && distance < glob_maxdelta) { 
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
			for(i=0; i<g.nlocalverts; i++) {
				if(dist[i] == -1.0 || dist[i] >= glob_maxdelta) {
					for(j=rowstarts[i]; j<rowstarts[i+1]; j++) {
						//if(weights[j] >= delta) {
						if(dist[i] > glob_mindelta+weights[j] || dist[i] == -1.0) {
							send_request(weights[j], dist[i], COLUMN(j), i);
						}
					}
				}
			}
			aml_barrier();
			for(i=0; i<msg_counter; i++) {
				aml_send(&(responsemsgs[i].msg), 3, sizeof(responsemsg), responsemsgs[i].dst_rank);
			}
			aml_barrier();
			msg_counter = 0;
		} else {
			for(i=0;i<g.nlocalverts;i++) { // Go through local vertices
				// If current distance of i is in current bucket
				if(dist[i]>=glob_mindelta && dist[i] < glob_maxdelta) { 
					// Iterate through edges
					for(j=rowstarts[i];j<rowstarts[i+1];j++) {
						// Check if edge is heavy
						//if(weights[j]>=delta) {// Check if edge is heavy
						if(dist[i]+weights[j]>=glob_maxdelta || weights[j] >= delta) { 
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
	}
	free(responsemsgs);
}


		// Initialize distance
void clean_shortest(float* dist) {
	int i;
	for(i=0;i<g.nlocalverts;i++) dist[i]=-1.0;
}

#endif
