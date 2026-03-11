//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_compute_desired_cuda.cuh"

void dscu_init(std::vector<Ped::Tagent *> agents,
			   struct dscu_agents_s *agents_h,
			   struct dscu_agents_s *agents_d) {
	// -- host --
	const size_t agents_size = agents.size();
	const size_t int_bytes = sizeof(int) * agents_size;
	const size_t size_t_bytes = sizeof(size_t) * agents_size;
	const size_t ssize_t_bytes = sizeof(ssize_t) * agents_size;
	const size_t ptr_bytes = sizeof(double *) * agents_size;
	cudaMallocHost(&agents_h->x, int_bytes);
	cudaMallocHost(&agents_h->y, int_bytes);
	cudaMallocHost(&agents_h->dst_idx, ssize_t_bytes);
	cudaMallocHost(&agents_h->des_x, int_bytes);
	cudaMallocHost(&agents_h->des_y, int_bytes);
	cudaMallocHost(&agents_h->wps_x, ptr_bytes);
	cudaMallocHost(&agents_h->wps_y, ptr_bytes);
	cudaMallocHost(&agents_h->wps_r, ptr_bytes);
	for (size_t i = 0; i < agents_size; i++) {
		const size_t bytes = sizeof(double) * agents[i]->getWaypointsSize();
		cudaMallocHost(&agents_h->wps_x[i], bytes);
		cudaMallocHost(&agents_h->wps_y[i], bytes);
		cudaMallocHost(&agents_h->wps_r[i], bytes);
	}
	cudaMallocHost(&agents_h->wps_sz, size_t_bytes);

	agents_h->size = agents_size;
	for (size_t i = 0; i < agents_size; i++) {
		agents_h->x[i] = agents[i]->getX();
		agents_h->y[i] = agents[i]->getY();
		agents_h->dst_idx[i] = -1;
		agents_h->wps_sz[i] = agents[i]->getWaypointsSize();
		for (size_t j = 0; j < agents_h->wps_sz[i]; j++) {
			auto wp = agents[i]->getWaypoint(j);
			agents_h->wps_x[i][j] = wp->getx();
			agents_h->wps_y[i][j] = wp->gety();
			agents_h->wps_r[i][j] = wp->getr();
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
	cudaMalloc(&agents_d->dst_idx, ssize_t_bytes);
	cudaMalloc(&agents_d->des_x, int_bytes);
	cudaMalloc(&agents_d->des_y, int_bytes);
	cudaMalloc(&agents_d->wps_x, ptr_bytes);
	cudaMalloc(&agents_d->wps_y, ptr_bytes);
	cudaMalloc(&agents_d->wps_r, ptr_bytes);
	for (size_t i = 0; i < agents_size; i++) {
		const size_t bytes = sizeof(double) * agents[i]->getWaypointsSize();
		cudaMalloc(&wps_x[i], bytes);
		cudaMalloc(&wps_y[i], bytes);
		cudaMalloc(&wps_r[i], bytes);
	}
	cudaMalloc(&agents_d->wps_sz, size_t_bytes);

	cudaMemcpy(agents_d->wps_x, wps_x, ptr_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(agents_d->wps_y, wps_y, ptr_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(agents_d->wps_r, wps_r, ptr_bytes, cudaMemcpyHostToDevice);

	cudaMemcpy(agents_d->x, agents_h->x, int_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(agents_d->y, agents_h->y, int_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(agents_d->dst_idx, agents_h->dst_idx, ssize_t_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(agents_d->wps_sz, agents_h->wps_sz, size_t_bytes, cudaMemcpyHostToDevice);
	for (size_t i = 0; i < agents_size; i++) {
		const size_t bytes = sizeof(double) * agents[i]->getWaypointsSize();
		cudaMemcpy(wps_x[i], agents_h->wps_x[i], bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(wps_y[i], agents_h->wps_y[i], bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(wps_r[i], agents_h->wps_r[i], bytes, cudaMemcpyHostToDevice);
	}

	free(wps_x);
	free(wps_y);
	free(wps_r);
}

void dscu_dinit(struct dscu_agents_s *agents_h) {
	// -- host --
	cudaFreeHost(agents_h->x);
	cudaFreeHost(agents_h->y);
	cudaFreeHost(agents_h->dst_idx);
	cudaFreeHost(agents_h->des_x);
	cudaFreeHost(agents_h->des_y);
	for (size_t i = 0; i < agents_h->size; i++) {
		cudaFreeHost(agents_h->wps_x[i]);
		cudaFreeHost(agents_h->wps_y[i]);
		cudaFreeHost(agents_h->wps_r[i]);
	}
	cudaFreeHost(agents_h->wps_x);
	cudaFreeHost(agents_h->wps_y);
	cudaFreeHost(agents_h->wps_r);
	cudaFreeHost(agents_h->wps_sz);

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
													   ssize_t *dst_idxs,
													   int *des_x,
													   int *des_y) {
	const size_t agent_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (agent_idx >= size) {
		return;
	}
	const int agent_x = x[agent_idx];
	const int agent_y = y[agent_idx];
	ssize_t dst_idx = dst_idxs[agent_idx];

	ssize_t nextDestination_idx = get_nextDestination_idx(
		wps_x, wps_y, wps_r, wps_sz, agent_x, agent_y, dst_idx, agent_idx);
	dst_idxs[agent_idx] = nextDestination_idx;
	dst_idx = nextDestination_idx;
	if (dst_idx == -1) {
		return; // no destination, no need to compute where to move to
	}

	const double diffX = wps_x[agent_idx][dst_idx] - agent_x;
	const double diffY = wps_y[agent_idx][dst_idx] - agent_y;
	const double len = sqrt(diffX * diffX + diffY * diffY);

	if (len != 0.0) {
		des_x[agent_idx] = (int)round(agent_x + diffX / len);
		des_y[agent_idx] = (int)round(agent_y + diffY / len);
	} else {
		des_x[agent_idx] = agent_x;
		des_y[agent_idx] = agent_y;
	}
}

void dscu_compute_next_desired_position(const struct dscu_agents_s *agents) {
	static dim3 threads_per_block(THREADS_PER_BLOCK, 1, 1);
	static dim3 blocks(((agents->size + threads_per_block.x - 1) / threads_per_block.x), 1, 1);
	cuda_computeNextDesiredPosition<<<blocks, threads_per_block>>>(agents->wps_x,
																   agents->wps_y,
																   agents->wps_r,
																   agents->wps_sz,
																   agents->size,
																   agents->x,
																   agents->y,
																   agents->dst_idx,
																   agents->des_x,
																   agents->des_y);
}
