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

#include "ped_model.h"
#include <cuda_runtime.h>
#include <math.h>
#include <string.h>

#define THREADS_PER_BLOCK 512

// Update the position according to get closer to the current destination
__global__ void cuda_computeNextDesiredPosition(const struct agents *agents);
void kernel_launch(dim3 blocks, dim3 threads_per_block, const struct agents *agents);

#endif
