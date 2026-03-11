//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
// Adapted for Low Level Parallel Programming 2017
//
// TAgent represents an agent in the scenario. Each
// agent has a position (x,y) and a number of destinations
// it wants to visit (waypoints). The desired next position
// represents the position it would like to visit next as it
// will bring it closer to its destination.
// Note: the agent will not move by itself, but the movement
// is handled in ped_model.cpp.
//

#ifndef _ped_compute_desired_cuda_h_
#define _ped_compute_desired_cuda_h_ 1

#include <cuda_runtime.h>
#include <string.h>
#include <sys/types.h>

#include "ped_agent.h"
#include "ped_waypoint.h"

struct dscu_agents_s {
	size_t size; // number of agents this struct holds
	int *x, *y;	 // The agents' current position
	double **wps_x, **wps_y, **wps_r;
	size_t *wps_sz;	  // size of each waypoints array
	ssize_t *dst_idx; // The index of each agent's current destination
					  // (may require several steps to reach)
	int *des_x, *des_y;
};

#define THREADS_PER_BLOCK 512

void dscu_init(std::vector<Ped::Tagent *> agents,
			   struct dscu_agents_s *agents_h,
			   struct dscu_agents_s *agents_d);
void dscu_dinit(struct dscu_agents_s *agents_h);
void dscu_compute_next_desired_position(const struct dscu_agents_s *agents);

#endif
