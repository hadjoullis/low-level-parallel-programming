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
					   IMPLEMENTATION implementation, bool timing_mode) {
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
	this->timing_mode = timing_mode;

	// Set up heatmap (relevant for Assignment 4)
	setupHeatmapSeq();

	switch (implementation) {
	case Ped::VECTOR:
		printf("Setting up data structures for SIMD...\n");
		agents_s = {0};
		simd_init(agents, &agents_s);
		printf("Data structures set up for SIMD complete.\n");
		break;
	case Ped::CUDA:
		printf("Setting up data structures for cuda...\n");
		agents_s = {0};
		agents_d = {0};
		cuda_init(agents, &agents_s, &agents_d);
		printf("Data structures set up for cuda complete.\n");
		break;
	default:
		printf("No extra setup needed for given implementation\n");
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
		}
		for (; i < agents_s.size; i++) {
			single_computeNextDesiredPosition(&agents_s, i);
		}
		break;
	}
	case Ped::CUDA: {
		static dim3 threads_per_block(THREADS_PER_BLOCK, 1, 1);
		static dim3 blocks(((agents_s.size + threads_per_block.x - 1) / threads_per_block.x), 1, 1);
		static const size_t bytes = sizeof(int) * agents_s.size;

		kernel_launch(blocks, threads_per_block, &agents_d);

		if (timing_mode) {
			break;
		}
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

Ped::Model::~Model() {
	std::for_each(agents.begin(), agents.end(), [](Ped::Tagent *agent) { delete agent; });
	std::for_each(
		destinations.begin(), destinations.end(), [](Ped::Twaypoint *destination) { delete destination; });

	switch (implementation) {
	case Ped::VECTOR:
		printf("Cleaning up data structures for SIMD...\n");
		simd_dinit(&agents_s);
		printf("Data structures for SIMD released.\n");
		break;
	case Ped::CUDA:
		printf("Cleaning up data structures for cuda...\n");
		cuda_dinit(&agents_s);
		printf("Data structures for cuda released.\n");
		break;
	default:
		printf("No extra cleanup needed for given implementation.\n");
	}
}
