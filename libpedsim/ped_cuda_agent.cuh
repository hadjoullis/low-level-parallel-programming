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

#ifndef _ped_cuda_agent_h_
#define _ped_cuda_agent_h_ 1

#include <cuda_runtime.h>
#include <string.h>

#include "ped_agent.h"
#include "ped_simd_agents.h"
#include "ped_waypoint.h"

#define THREADS_PER_BLOCK 512

void cuda_init(std::vector<Ped::Tagent *> agents, struct agents *agents_s, struct agents *agents_d);
void cuda_dinit(struct agents *agents_s);
void kernel_launch(dim3 blocks, dim3 threads_per_block, const struct agents *agents);

#endif
