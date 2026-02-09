//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_model.h"
#include "ped_waypoint.h"
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <stack>
#include <thread>

#ifndef NOCDUA
#include "cuda_testkernel.h"
#endif

#include <stdlib.h>

void Ped::Model::setup(std::vector<Ped::Tagent *> agentsInScenario,
					   std::vector<Twaypoint *> destinationsInScenario,
					   IMPLEMENTATION implementation) {
#ifndef NOCUDA
	// Convenience test: does CUDA work on this machine?
	cuda_test();
#else
	std::cout << "Not compiled for CUDA" << std::endl;
#endif

	// Set
	agents = std::vector<Ped::Tagent *>(agentsInScenario.begin(), agentsInScenario.end());

	// Set up destinations
	destinations = std::vector<Ped::Twaypoint *>(destinationsInScenario.begin(),
												 destinationsInScenario.end());

	// Sets the chosen implemenation. Standard in the given code is SEQ
	this->implementation = implementation;

	// Set up heatmap (relevant for Assignment 4)
	setupHeatmapSeq();

	switch (implementation) {
	case Ped::VECTOR:
		printf("Setting up data structures for SIMD...\n");
		simd_init();
		printf("Data structures set up for SIMD complete.\n");
		break;
	case Ped::CUDA:
		printf("Setting up data structures for cuda...\n");
		cuda_init();
		printf("Data structures set up for cuda complete.\n");
		break;
	default:
		printf("No extra setup needed for given implementation\n");
	}
}

void Ped::Model::simd_init(void) {
	agents_s = {0};
	const size_t agents_size = agents.size();
	const size_t int_bytes = sizeof(int) * agents_size;
	const size_t size_t_bytes = sizeof(size_t) * agents_size;
	const size_t ssize_t_bytes = sizeof(ssize_t) * agents_size;
	const size_t ptr_bytes = sizeof(double *) * agents_size;
	posix_memalign((void **)&agents_s.x, ALIGN, int_bytes);
	posix_memalign((void **)&agents_s.y, ALIGN, int_bytes);
	posix_memalign((void **)&agents_s.desiredPositionX, ALIGN, int_bytes);
	posix_memalign((void **)&agents_s.desiredPositionY, ALIGN, int_bytes);
	posix_memalign((void **)&agents_s.destination_idx, ALIGN, ssize_t_bytes);
	posix_memalign((void **)&agents_s.waypoints.x, ALIGN, ptr_bytes);
	posix_memalign((void **)&agents_s.waypoints.y, ALIGN, ptr_bytes);
	posix_memalign((void **)&agents_s.waypoints.r, ALIGN, ptr_bytes);
	for (size_t i = 0; i < agents_size; i++) {
		// allocate one extra element so when we index with offset '-1' we do
		// not go out of bounds ('simd_computeNextDesiredPosition')
		const size_t bytes = sizeof(double) * (agents[i]->getWaypointsSize() + 1);
		posix_memalign((void **)&agents_s.waypoints.x[i], ALIGN, bytes);
		posix_memalign((void **)&agents_s.waypoints.y[i], ALIGN, bytes);
		posix_memalign((void **)&agents_s.waypoints.r[i], ALIGN, bytes);

		agents_s.waypoints.x[i]++;
		agents_s.waypoints.y[i]++;
		agents_s.waypoints.r[i]++;
	}
	posix_memalign((void **)&agents_s.waypoints.sz, ALIGN, size_t_bytes);

	agents_s.size = agents_size;
	for (size_t i = 0; i < agents_size; i++) {
		agents_s.x[i] = agents[i]->getX();
		agents_s.y[i] = agents[i]->getY();
		// agents_s.desiredPositionX[i]: not set
		// agents_s.desiredPositionY[i]: not set
		agents_s.destination_idx[i] = -1;
		// we need to ensure that if we index with '-1' the condition 'len <
		// dst_r' evaluates to true so we satisfy both clauses of the second
		// if statement
		agents_s.waypoints.x[i][-1] = 0;
		agents_s.waypoints.y[i][-1] = 0;
		agents_s.waypoints.r[i][-1] = DBL_MAX;
		// agents_s.waypoints: already set
		agents_s.waypoints.sz[i] = agents[i]->getWaypointsSize();
		for (size_t j = 0; j < agents_s.waypoints.sz[i]; j++) {
			auto wp = agents[i]->getWaypoint(j);
			agents_s.waypoints.x[i][j] = wp->getx();
			agents_s.waypoints.y[i][j] = wp->gety();
			agents_s.waypoints.r[i][j] = wp->getr();
		}
	}
}

void Ped::Model::cuda_init(void) {
	// -- host --
	agents_s = {0};
	const size_t agents_size = agents.size();
	const size_t int_bytes = sizeof(int) * agents_size;
	const size_t size_t_bytes = sizeof(size_t) * agents_size;
	const size_t ssize_t_bytes = sizeof(ssize_t) * agents_size;
	const size_t ptr_bytes = sizeof(double *) * agents_size;
	cudaMallocHost(&agents_s.x, int_bytes);
	cudaMallocHost(&agents_s.y, int_bytes);
	cudaMallocHost(&agents_s.desiredPositionX, int_bytes);
	cudaMallocHost(&agents_s.desiredPositionY, int_bytes);
	cudaMallocHost(&agents_s.destination_idx, ssize_t_bytes);
	cudaMallocHost(&agents_s.waypoints.x, ptr_bytes);
	cudaMallocHost(&agents_s.waypoints.y, ptr_bytes);
	cudaMallocHost(&agents_s.waypoints.r, ptr_bytes);
	for (size_t i = 0; i < agents_size; i++) {
		const size_t bytes = sizeof(double) * agents[i]->getWaypointsSize();
		cudaMallocHost(&agents_s.waypoints.x[i], bytes);
		cudaMallocHost(&agents_s.waypoints.y[i], bytes);
		cudaMallocHost(&agents_s.waypoints.r[i], bytes);
	}
	cudaMallocHost(&agents_s.waypoints.sz, size_t_bytes);

	agents_s.size = agents_size;
	for (size_t i = 0; i < agents_size; i++) {
		agents_s.x[i] = agents[i]->getX();
		agents_s.y[i] = agents[i]->getY();
		// agents_s.desiredPositionX[i]: not set
		// agents_s.desiredPositionY[i]: not set
		agents_s.destination_idx[i] = -1;
		// agents_s.waypoints: already set
		agents_s.waypoints.sz[i] = agents[i]->getWaypointsSize();
		for (size_t j = 0; j < agents_s.waypoints.sz[i]; j++) {
			auto wp = agents[i]->getWaypoint(j);
			agents_s.waypoints.x[i][j] = wp->getx();
			agents_s.waypoints.y[i][j] = wp->gety();
			agents_s.waypoints.r[i][j] = wp->getr();
		}
	}

	// -- device --
	num_blocks = agents_size / THREADS_PER_BLOCK;

	agents_d = {0};
	double **wps_x, **wps_y, **wps_r;
	wps_x = (double **)malloc(ptr_bytes);
	wps_y = (double **)malloc(ptr_bytes);
	wps_r = (double **)malloc(ptr_bytes);

	cudaMalloc(&agents_d.x, int_bytes);
	cudaMalloc(&agents_d.y, int_bytes);
	cudaMalloc(&agents_d.desiredPositionX, int_bytes);
	cudaMalloc(&agents_d.desiredPositionY, int_bytes);
	cudaMalloc(&agents_d.destination_idx, ssize_t_bytes);
	cudaMalloc(&agents_d.waypoints.x, ptr_bytes);
	cudaMalloc(&agents_d.waypoints.y, ptr_bytes);
	cudaMalloc(&agents_d.waypoints.r, ptr_bytes);
	for (size_t i = 0; i < agents_size; i++) {
		const size_t bytes = sizeof(double) * agents[i]->getWaypointsSize();
		cudaMalloc(&wps_x[i], bytes);
		cudaMalloc(&wps_y[i], bytes);
		cudaMalloc(&wps_r[i], bytes);
	}
	cudaMalloc(&agents_d.waypoints.sz, size_t_bytes);

	cudaMemcpy(agents_d.waypoints.x, wps_x, ptr_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(agents_d.waypoints.y, wps_y, ptr_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(agents_d.waypoints.r, wps_r, ptr_bytes, cudaMemcpyHostToDevice);

	cudaMemcpy(agents_d.x, agents_s.x, int_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(agents_d.y, agents_s.y, int_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(agents_d.destination_idx, agents_s.destination_idx, ssize_t_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(agents_d.waypoints.sz, agents_s.waypoints.sz, size_t_bytes, cudaMemcpyHostToDevice);
	for (size_t i = 0; i < agents_size; i++) {
		const size_t bytes = sizeof(double) * agents[i]->getWaypointsSize();
		cudaMemcpy(wps_x[i], agents_s.waypoints.x[i], bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(wps_y[i], agents_s.waypoints.y[i], bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(wps_r[i], agents_s.waypoints.r[i], bytes, cudaMemcpyHostToDevice);
	}
}

void Ped::Model::pthread_tick(const int k, int id) {
	auto &agents = this->agents;
	const int n = agents.size();

	const int chunk_sz = n / k;
	const int start = id * chunk_sz;
	const int end = ((id != k - 1) ? ((id + 1) * chunk_sz) : n);

	for (int i = start; i < end; i++) {
		auto *agent = agents[i];
		agent->computeNextDesiredPosition();
		const int x = agent->getDesiredX();
		const int y = agent->getDesiredY();
		agent->setX(x);
		agent->setY(y);
	}
}

void Ped::Model::tick() {
	// EDIT HERE FOR ASSIGNMENT 1
	switch (this->implementation) {
	case Ped::SEQ: {
		for (auto *const agent : this->agents) {
			agent->computeNextDesiredPosition();
			const int x = agent->getDesiredX();
			const int y = agent->getDesiredY();
			agent->setX(x);
			agent->setY(y);
		}
		break;
	}
	case Ped::OMP: {
		auto &agents = this->agents;
		const int n = agents.size();

#pragma omp parallel for default(none) shared(n, agents)
		for (int i = 0; i < n; i++) {
			auto *agent = agents[i];
			agent->computeNextDesiredPosition();
			const int x = agent->getDesiredX();
			const int y = agent->getDesiredY();
			agent->setX(x);
			agent->setY(y);
		}
		break;
	}
	case Ped::PTHREAD: {
		static bool once = true;
		static int PTHREAD_NUM_THREADS = 8;
		if (once) {
			char *retval = getenv("PTHREAD_NUM_THREADS");
			if (retval) {
				PTHREAD_NUM_THREADS = atoi(retval);
			}
			once = false;
		}
		static std::vector<std::thread> tid(PTHREAD_NUM_THREADS);
		for (int i = 0; i < PTHREAD_NUM_THREADS; i++) {
			tid[i] = std::thread(&Ped::Model::pthread_tick, this, PTHREAD_NUM_THREADS, i);
		}
		for (auto &t : tid) {
			t.join();
		}
		break;
	}
	case Ped::VECTOR: {
		size_t i;
		for (i = 0; i + STEPS <= agents_s.size; i += STEPS) {
			simd_computeNextDesiredPosition(&agents_s, i);
			const __m256i x = _mm256_load_epi32(&agents_s.desiredPositionX[i]);
			const __m256i y = _mm256_load_epi32(&agents_s.desiredPositionY[i]);
			_mm256_store_epi32(&agents_s.x[i], x);
			_mm256_store_epi32(&agents_s.y[i], y);
		}
		for (; i < agents_s.size; i++) {
			single_computeNextDesiredPosition(&agents_s, i);
			agents_s.x[i] = agents_s.desiredPositionX[i];
			agents_s.y[i] = agents_s.desiredPositionY[i];
		}
		break;
	}
	case Ped::CUDA: {
		static dim3 threads_per_block(THREADS_PER_BLOCK, 1, 1);
		static dim3 blocks(ceil(agents_s.size / threads_per_block.x), 1, 1);
		static const size_t bytes = sizeof(int) * agents_s.size;

		kernel_launch(blocks, threads_per_block, &agents_d);

		cudaMemcpy(agents_s.x, agents_d.x, bytes, cudaMemcpyDeviceToHost);
		cudaMemcpy(agents_s.y, agents_d.y, bytes, cudaMemcpyDeviceToHost);
		break;
	}
	default:
		fprintf(stderr, "ERROR: NOT IMPLEMENTED\n");
		exit(1);
	}
}

////////////
/// Everything below here relevant for Assignment 3.
/// Don't use this for Assignment 1!
///////////////////////////////////////////////

// Moves the agent to the next desired position. If already taken, it will
// be moved to a location close to it.
void Ped::Model::move(Ped::Tagent *agent) {
	// Search for neighboring agents
	set<const Ped::Tagent *> neighbors = getNeighbors(agent->getX(), agent->getY(), 2);

	// Retrieve their positions
	std::vector<std::pair<int, int>> takenPositions;
	for (std::set<const Ped::Tagent *>::iterator neighborIt = neighbors.begin();
		 neighborIt != neighbors.end();
		 ++neighborIt) {
		std::pair<int, int> position((*neighborIt)->getX(), (*neighborIt)->getY());
		takenPositions.push_back(position);
	}

	// Compute the three alternative positions that would bring the agent
	// closer to his desiredPosition, starting with the desiredPosition itself
	std::vector<std::pair<int, int>> prioritizedAlternatives;
	std::pair<int, int> pDesired(agent->getDesiredX(), agent->getDesiredY());
	prioritizedAlternatives.push_back(pDesired);

	int diffX = pDesired.first - agent->getX();
	int diffY = pDesired.second - agent->getY();
	std::pair<int, int> p1, p2;
	if (diffX == 0 || diffY == 0) {
		// Agent wants to walk straight to North, South, West or East
		p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
		p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
	} else {
		// Agent wants to walk diagonally
		p1 = std::make_pair(pDesired.first, agent->getY());
		p2 = std::make_pair(agent->getX(), pDesired.second);
	}
	prioritizedAlternatives.push_back(p1);
	prioritizedAlternatives.push_back(p2);

	// Find the first empty alternative position
	for (std::vector<pair<int, int>>::iterator it = prioritizedAlternatives.begin();
		 it != prioritizedAlternatives.end();
		 ++it) {

		// If the current position is not yet taken by any neighbor
		if (std::find(takenPositions.begin(), takenPositions.end(), *it) == takenPositions.end()) {

			// Set the agent's position
			agent->setX((*it).first);
			agent->setY((*it).second);

			break;
		}
	}
}

/// Returns the list of neighbors within dist of the point x/y. This
/// can be the position of an agent, but it is not limited to this.
/// \date    2012-01-29
/// \return  The list of neighbors
/// \param   x the x coordinate
/// \param   y the y coordinate
/// \param   dist the distance around x/y that will be searched for agents
/// (search field is a square in the current implementation)
set<const Ped::Tagent *> Ped::Model::getNeighbors(int x, int y, int dist) const {

	// create the output list
	// ( It would be better to include only the agents close by, but this
	// programmer is lazy.)
	return set<const Ped::Tagent *>(agents.begin(), agents.end());
}

void Ped::Model::cleanup() {
	// Nothing to do here right now.
}

void Ped::Model::simd_dinit() {
	free(agents_s.x);
	free(agents_s.y);
	free(agents_s.desiredPositionX);
	free(agents_s.desiredPositionY);
	free(agents_s.destination_idx);
	for (size_t i = 0; i < agents_s.size; i++) {
		agents_s.waypoints.x[i]--;
		agents_s.waypoints.y[i]--;
		agents_s.waypoints.r[i]--;
		free(agents_s.waypoints.x[i]);
		free(agents_s.waypoints.y[i]);
		free(agents_s.waypoints.r[i]);
	}
	free(agents_s.waypoints.x);
	free(agents_s.waypoints.y);
	free(agents_s.waypoints.r);
	free(agents_s.waypoints.sz);
}

void Ped::Model::cuda_dinit() {
	// -- device --
	// cudaFree(agents_d.x);
	// cudaFree(agents_d.y);
	// cudaFree(agents_d.desiredPositionX);
	// cudaFree(agents_d.desiredPositionY);
	// cudaFree(agents_d.destination_idx);
	// for (size_t i = 0; i < agents_d.size; i++) {
	// 	cudaFree(agents_d.waypoints.x[i]);
	// 	cudaFree(agents_d.waypoints.y[i]);
	// 	cudaFree(agents_d.waypoints.r[i]);
	// }
	// cudaFree(agents_d.waypoints.x);
	// cudaFree(agents_d.waypoints.y);
	// cudaFree(agents_d.waypoints.r);
	// cudaFree(agents_d.waypoints.sz);
	cudaDeviceReset();

	// -- host --
	cudaFreeHost(agents_s.x);
	cudaFreeHost(agents_s.y);
	cudaFreeHost(agents_s.desiredPositionX);
	cudaFreeHost(agents_s.desiredPositionY);
	cudaFreeHost(agents_s.destination_idx);
	for (size_t i = 0; i < agents_s.size; i++) {
		cudaFreeHost(agents_s.waypoints.x[i]);
		cudaFreeHost(agents_s.waypoints.y[i]);
		cudaFreeHost(agents_s.waypoints.r[i]);
	}
	cudaFreeHost(agents_s.waypoints.x);
	cudaFreeHost(agents_s.waypoints.y);
	cudaFreeHost(agents_s.waypoints.r);
	cudaFreeHost(agents_s.waypoints.sz);
}

Ped::Model::~Model() {
	std::for_each(agents.begin(), agents.end(), [](Ped::Tagent *agent) { delete agent; });
	std::for_each(
		destinations.begin(), destinations.end(), [](Ped::Twaypoint *destination) { delete destination; });
	switch (implementation) {
	case Ped::VECTOR:
		printf("Cleaning up data structures for SIMD...\n");
		simd_dinit();
		printf("Data structures for SIMD released.\n");
		break;
	case Ped::CUDA:
		printf("Cleaning up data structures for cuda...\n");
		cuda_dinit();
		printf("Data structures for cuda released.\n");
		break;
	default:
		printf("No extra cleanup needed for given implementation.\n");
	}
}
