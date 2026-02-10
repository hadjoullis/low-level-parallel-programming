//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_cuda_agent.cuh"

void cuda_init(std::vector<Ped::Tagent *> agents, struct agents *agents_s, struct agents *agents_d) {
	// -- host --
	const size_t agents_size = agents.size();
	const size_t int_bytes = sizeof(int) * agents_size;
	const size_t size_t_bytes = sizeof(size_t) * agents_size;
	const size_t ssize_t_bytes = sizeof(ssize_t) * agents_size;
	const size_t ptr_bytes = sizeof(double *) * agents_size;
	cudaMallocHost(&agents_s->x, int_bytes);
	cudaMallocHost(&agents_s->y, int_bytes);
	cudaMallocHost(&agents_s->desiredPositionX, int_bytes);
	cudaMallocHost(&agents_s->desiredPositionY, int_bytes);
	cudaMallocHost(&agents_s->destination_idx, ssize_t_bytes);
	cudaMallocHost(&agents_s->waypoints.x, ptr_bytes);
	cudaMallocHost(&agents_s->waypoints.y, ptr_bytes);
	cudaMallocHost(&agents_s->waypoints.r, ptr_bytes);
	for (size_t i = 0; i < agents_size; i++) {
		const size_t bytes = sizeof(double) * agents[i]->getWaypointsSize();
		cudaMallocHost(&agents_s->waypoints.x[i], bytes);
		cudaMallocHost(&agents_s->waypoints.y[i], bytes);
		cudaMallocHost(&agents_s->waypoints.r[i], bytes);
	}
	cudaMallocHost(&agents_s->waypoints.sz, size_t_bytes);

	agents_s->size = agents_size;
	for (size_t i = 0; i < agents_size; i++) {
		agents_s->x[i] = agents[i]->getX();
		agents_s->y[i] = agents[i]->getY();
		// agents_s->desiredPositionX[i]: not set
		// agents_s->desiredPositionY[i]: not set
		agents_s->destination_idx[i] = -1;
		// agents_s->waypoints: already set
		agents_s->waypoints.sz[i] = agents[i]->getWaypointsSize();
		for (size_t j = 0; j < agents_s->waypoints.sz[i]; j++) {
			auto wp = agents[i]->getWaypoint(j);
			agents_s->waypoints.x[i][j] = wp->getx();
			agents_s->waypoints.y[i][j] = wp->gety();
			agents_s->waypoints.r[i][j] = wp->getr();
		}
	}

	// -- device --
	agents_d->size = agents_size;
	double **wps_x, **wps_y, **wps_r;
	wps_x = (double **)malloc(ptr_bytes);
	wps_y = (double **)malloc(ptr_bytes);
	wps_r = (double **)malloc(ptr_bytes);

	cudaMalloc(&agents_d->x, int_bytes);
	cudaMalloc(&agents_d->y, int_bytes);
	cudaMalloc(&agents_d->desiredPositionX, int_bytes);
	cudaMalloc(&agents_d->desiredPositionY, int_bytes);
	cudaMalloc(&agents_d->destination_idx, ssize_t_bytes);
	cudaMalloc(&agents_d->waypoints.x, ptr_bytes);
	cudaMalloc(&agents_d->waypoints.y, ptr_bytes);
	cudaMalloc(&agents_d->waypoints.r, ptr_bytes);
	for (size_t i = 0; i < agents_size; i++) {
		const size_t bytes = sizeof(double) * agents[i]->getWaypointsSize();
		cudaMalloc(&wps_x[i], bytes);
		cudaMalloc(&wps_y[i], bytes);
		cudaMalloc(&wps_r[i], bytes);
	}
	cudaMalloc(&agents_d->waypoints.sz, size_t_bytes);

	cudaMemcpy(agents_d->waypoints.x, wps_x, ptr_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(agents_d->waypoints.y, wps_y, ptr_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(agents_d->waypoints.r, wps_r, ptr_bytes, cudaMemcpyHostToDevice);

	cudaMemcpy(agents_d->x, agents_s->x, int_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(agents_d->y, agents_s->y, int_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(agents_d->destination_idx, agents_s->destination_idx, ssize_t_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(agents_d->waypoints.sz, agents_s->waypoints.sz, size_t_bytes, cudaMemcpyHostToDevice);
	for (size_t i = 0; i < agents_size; i++) {
		const size_t bytes = sizeof(double) * agents[i]->getWaypointsSize();
		cudaMemcpy(wps_x[i], agents_s->waypoints.x[i], bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(wps_y[i], agents_s->waypoints.y[i], bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(wps_r[i], agents_s->waypoints.r[i], bytes, cudaMemcpyHostToDevice);
	}

	free(wps_x);
	free(wps_y);
	free(wps_r);
}

void cuda_dinit(struct agents *agents_s) {
	// -- host --
	cudaFreeHost(agents_s->x);
	cudaFreeHost(agents_s->y);
	cudaFreeHost(agents_s->desiredPositionX);
	cudaFreeHost(agents_s->desiredPositionY);
	cudaFreeHost(agents_s->destination_idx);
	for (size_t i = 0; i < agents_s->size; i++) {
		cudaFreeHost(agents_s->waypoints.x[i]);
		cudaFreeHost(agents_s->waypoints.y[i]);
		cudaFreeHost(agents_s->waypoints.r[i]);
	}
	cudaFreeHost(agents_s->waypoints.x);
	cudaFreeHost(agents_s->waypoints.y);
	cudaFreeHost(agents_s->waypoints.r);
	cudaFreeHost(agents_s->waypoints.sz);

	// -- device --
	cudaDeviceReset();
}

static __device__ ssize_t get_nextDestination_idx(double **wps_x,
												  double **wps_y,
												  double **wps_r,
												  size_t *wps_sz,
												  const int agent_x,
												  const int agent_y,
												  const ssize_t dst_idx,
												  const size_t agent_idx) {
	ssize_t nextDestination_idx = -1;
	bool agentReachedDestination = false;

	if (dst_idx != -1) {
		// compute if agent reached its current destination
		const double diffX = wps_x[agent_idx][dst_idx] - agent_x;
		const double diffY = wps_y[agent_idx][dst_idx] - agent_y;
		const double len = sqrt(diffX * diffX + diffY * diffY);
		agentReachedDestination = len < wps_r[agent_idx][dst_idx];
	}

	if ((agentReachedDestination || dst_idx == -1) && wps_sz[agent_idx] != 0) {
		// Case 1: agent has reached destination (or has no current
		// destination); get next destination if available
		ssize_t wps_idx = dst_idx + 1;
		if (wps_idx == wps_sz[agent_idx]) {
			wps_idx = -1;
		}
		nextDestination_idx = wps_idx;
	} else {
		// Case 2: agent has not yet reached destination, continue to move
		// towards current destination
		nextDestination_idx = dst_idx;
	}
	return nextDestination_idx;
}

static __global__ void cuda_computeNextDesiredPosition(double **wps_x,
													   double **wps_y,
													   double **wps_r,
													   size_t *wps_sz,
													   size_t size,
													   int *x,
													   int *y,
													   int *desiredPositionX,
													   int *desiredPositionY,
													   ssize_t *destination_idx) {
	const size_t agent_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (agent_idx >= size) {
		return;
	}
	const int agent_x = x[agent_idx];
	const int agent_y = y[agent_idx];
	ssize_t dst_idx = destination_idx[agent_idx];

	ssize_t nextDestination_idx = get_nextDestination_idx(
		wps_x, wps_y, wps_r, wps_sz, agent_x, agent_y, dst_idx, agent_idx);
	destination_idx[agent_idx] = nextDestination_idx;
	dst_idx = nextDestination_idx;
	if (dst_idx == -1) {
		return; // no destination, no need to compute where to move to
	}

	const double diffX = wps_x[agent_idx][dst_idx] - agent_x;
	const double diffY = wps_y[agent_idx][dst_idx] - agent_y;
	const double len = sqrt(diffX * diffX + diffY * diffY);
	desiredPositionX[agent_idx] = (int)round(agent_x + diffX / len);
	desiredPositionY[agent_idx] = (int)round(agent_y + diffY / len);

	x[agent_idx] = desiredPositionX[agent_idx];
	y[agent_idx] = desiredPositionY[agent_idx];
}

void kernel_launch(dim3 blocks, dim3 threads_per_block, const struct agents *agents) {
	cuda_computeNextDesiredPosition<<<blocks, threads_per_block>>>(agents->waypoints.x,
																   agents->waypoints.y,
																   agents->waypoints.r,
																   agents->waypoints.sz,
																   agents->size,
																   agents->x,
																   agents->y,
																   agents->desiredPositionX,
																   agents->desiredPositionY,
																   agents->destination_idx);
}
