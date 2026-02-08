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

#ifndef _ped_simd_agents_h_
#define _ped_simd_agents_h_ 1

#include <float.h>
#include <immintrin.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <sys/types.h>

// max sizeof double vs size_t vs ssize_t vs int: 8
#define STEPS (sizeof(__m512) / 8)
#define ALIGN (512 / STEPS)
#define ROUND (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)

// (x,y): position of the waypoint
// r: radius defines the area of this waypoint, i.e. a circular area with the
// middle point in (x,y). Any point within this area is considered to be
// belonging to this destination.
struct waypoints {
	double **x, **y, **r;
	size_t *sz; // size of each waypoints array
};

struct agents {
	size_t size; // number of agents this struct holds
	int *x, *y;	 // The agents' current position
	int *desiredPositionX,
		*desiredPositionY;		// The agents' desired next position
	struct waypoints waypoints; // The ring-like array of all destinations that
								// each agent still has to visit
	ssize_t *destination_idx;	// The index of each agent's current destination
								// (may require several steps to reach)
};

// Update the position according to get closer to the current destination
void simd_computeNextDesiredPosition(const struct agents *agents,
									 const size_t agent_idx);
void single_computeNextDesiredPosition(const struct agents *agents,
									   const size_t agent_idx);

#endif
