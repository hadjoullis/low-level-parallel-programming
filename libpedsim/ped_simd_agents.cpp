//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_simd_agents.h"
#include <math.h>

#include <stdlib.h>

void struct_agents_computeNextDesiredPosition(struct agents *agents, const size_t agent_idx) {
	ssize_t nextDestination_idx = -1;
	bool agentReachedDestination = false;

	if (agents->destination_idx[agent_idx] != -1) {
		// compute if agent reached its current destination
		const ssize_t dst_idx = agents->destination_idx[agent_idx];
		double diffX = agents->waypoints.x[agent_idx][dst_idx] - agents->x[agent_idx];
		double diffY = agents->waypoints.y[agent_idx][dst_idx] - agents->y[agent_idx];
		double length = sqrt(diffX * diffX + diffY * diffY);
		agentReachedDestination = length < agents->waypoints.r[agent_idx][dst_idx];
	}

	if ((agentReachedDestination || agents->destination_idx[agent_idx] == -1)
		&& agents->waypoints.sz[agent_idx] != 0) {
		// Case 1: agent has reached destination (or has no current destination);
		// get next destination if available
		ssize_t wps_idx = (agents->destination_idx[agent_idx] + 1);
		if (wps_idx == agents->waypoints.sz[agent_idx]) {
			wps_idx = -1;
		}
		nextDestination_idx = wps_idx;
		agents->destination_idx[agent_idx] = wps_idx;
	}
	else {
		// Case 2: agent has not yet reached destination, continue to move towards
		// current destination
		nextDestination_idx = agents->destination_idx[agent_idx];
	}

	agents->destination_idx[agent_idx] = nextDestination_idx;
	if (agents->destination_idx[agent_idx] == -1) {
		return; // no destination, no need to compute where to move to
	}

	const ssize_t dst_idx = agents->destination_idx[agent_idx];
	double diffX = agents->waypoints.x[agent_idx][dst_idx] - agents->x[agent_idx];
	double diffY = agents->waypoints.y[agent_idx][dst_idx] - agents->y[agent_idx];
	double len = sqrt(diffX * diffX + diffY * diffY);
	agents->desiredPositionX[agent_idx] = (int)round(agents->x[agent_idx] + diffX / len);
	agents->desiredPositionY[agent_idx] = (int)round(agents->y[agent_idx] + diffY / len);
}
