//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_cuda_agent.cuh"

__device__ static ssize_t get_nextDestination_idx(double **wps_x, double **wps_y, double **wps_r, size_t *wps_sz, size_t size, int *x, int *y, ssize_t *destination_idx, const size_t agent_idx) {
	ssize_t nextDestination_idx = -1;
	bool agentReachedDestination = false;
	const int agent_x = x[agent_idx];
	const int agent_y = y[agent_idx];

	if (destination_idx[agent_idx] != -1) {
		// compute if agent reached its current destination
		const ssize_t dst_idx = destination_idx[agent_idx];
		const double diffX = wps_x[agent_idx][dst_idx] - agent_x;
		const double diffY = wps_y[agent_idx][dst_idx] - agent_y;
		const double len = sqrt(diffX * diffX + diffY * diffY);
		agentReachedDestination = len < wps_r[agent_idx][dst_idx];
	}

	if ((agentReachedDestination || destination_idx[agent_idx] == -1) &&
		wps_sz[agent_idx] != 0) {
		// Case 1: agent has reached destination (or has no current
		// destination); get next destination if available
		ssize_t wps_idx = destination_idx[agent_idx] + 1;
		if (wps_idx == wps_sz[agent_idx]) {
			wps_idx = -1;
		}
		nextDestination_idx = wps_idx;
	} else {
		// Case 2: agent has not yet reached destination, continue to move
		// towards current destination
		nextDestination_idx = destination_idx[agent_idx];
	}
	return nextDestination_idx;
}

__global__ void cuda_computeNextDesiredPosition(double **wps_x, double **wps_y, double **wps_r, size_t *wps_sz, size_t size, int *x, int *y, int *desiredPositionX, int *desiredPositionY, ssize_t *destination_idx) {
	const size_t agent_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (agent_idx >= size) {
		return;
	}
	ssize_t nextDestination_idx = get_nextDestination_idx(wps_x, wps_y, wps_r, wps_sz, size, x, y, destination_idx, agent_idx);
	destination_idx[agent_idx] = nextDestination_idx;
	if (destination_idx[agent_idx] == -1) {
		return; // no destination, no need to compute where to move to
	}
	const int agent_x = x[agent_idx];
	const int agent_y = y[agent_idx];

	const ssize_t dst_idx = destination_idx[agent_idx];
	const double diffX = wps_x[agent_idx][dst_idx] - agent_x;
	const double diffY = wps_y[agent_idx][dst_idx] - agent_y;
	const double len = sqrt(diffX * diffX + diffY * diffY);
	desiredPositionX[agent_idx] = (int)round(agent_x + diffX / len);
	desiredPositionY[agent_idx] = (int)round(agent_y + diffY / len);
}

void kernel_launch(dim3 blocks, dim3 threads_per_block, const struct agents *agents) {
	cuda_computeNextDesiredPosition<<<blocks, threads_per_block>>>(agents->waypoints.x, agents->waypoints.y, agents->waypoints.r, agents->waypoints.sz, agents->size, agents->x, agents->y, agents->desiredPositionX, agents->desiredPositionY, agents->destination_idx);
}
