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

#ifndef NOCUDA
#include "cuda_testkernel.h"
#endif

#include <stdlib.h>

void Ped::Model::setup(std::vector<Ped::Tagent *> agentsInScenario,
					   std::vector<Twaypoint *> destinationsInScenario,
					   IMPLEMENTATION implementation,
					   bool timing_mode) {
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
	case Ped::OMP_MV:
		printf("Setting up data structures for OMP_MV...\n");
		regions_init();
		printf("Data structures set up for OMP_MV complete.\n");
		break;
#ifndef NOCUDA
	case Ped::CUDA:
		printf("Setting up data structures for cuda...\n");
		agents_s = {0};
		agents_d = {0};
		cuda_init(agents, &agents_s, &agents_d);
		printf("Data structures set up for cuda complete.\n");
		break;
#endif
	default:
		printf("No extra setup needed for given implementation\n");
	}
}

void Ped::Model::regions_init(void) {
	int x_start = 0;
	const int region_width = GRID_WIDTH / NUM_REGIONS;
	// make x_end inclusive for easier logistics
	for (size_t i = 0; i < NUM_REGIONS; i++) {
		regions[i].x_start = x_start;
		regions[i].x_end = x_start + region_width - 1;
		if (i == NUM_REGIONS - 1) {
			regions[i].x_end = GRID_WIDTH;
		}
		x_start += region_width;

		omp_init_lock(&regions[i].llock);
		omp_init_lock(&regions[i].rlock);

		// coordinates range is inclusive
		regions[i].llock_taken = (bool *)calloc(GRID_HEIGHT + 1, sizeof(bool));
		regions[i].rlock_taken = (bool *)calloc(GRID_HEIGHT + 1, sizeof(bool));

		for (auto *const agent : this->agents) {
			const int x = agent->getX();
			const int y = agent->getY();
			if (x == regions[i].x_start) {
				regions[i].llock_taken[y] = true;
			} else if (x == regions[i].x_end) {
				regions[i].rlock_taken[y] = true;
			}
		}
	}
}

void Ped::Model::regions_dinit(void) {
	for (size_t i = 0; i < NUM_REGIONS; i++) {
		omp_destroy_lock(&regions[i].llock);
		omp_destroy_lock(&regions[i].rlock);

		free(regions[i].llock_taken);
		free(regions[i].rlock_taken);
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
	case Ped::SEQ_MV: {
		for (auto *const agent : this->agents) {
			agent->computeNextDesiredPosition();
			move(agent);
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
	case Ped::OMP_MV: {
		auto &agents = this->agents;
		const int n = agents.size();

#pragma omp parallel default(none) shared(n, agents)
		{
#pragma omp for
			for (int i = 0; i < n; i++) {
				auto *agent = agents[i];
				agent->computeNextDesiredPosition();
			}

#pragma omp for
			for (int i = 0; i < NUM_REGIONS; i++) {
				get_agents_in_region(&regions[i]);
				const int size = regions[i].region_agents.size();
				for (int agent_idx = 0; agent_idx < size; agent_idx++) {
					move_parallel(&regions[i], agent_idx);
				}
				regions[i].region_agents.clear();
				regions[i].taken_positions.clear();
			}
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
#ifndef NOCUDA
	case Ped::CUDA: {
		static dim3 threads_per_block(THREADS_PER_BLOCK, 1, 1);
		static dim3 blocks(((agents_s.size + threads_per_block.x - 1) / threads_per_block.x), 1, 1);
		static const size_t bytes = sizeof(int) * agents_s.size;

		kernel_launch(blocks, threads_per_block, &agents_d);

		if (timing_mode) {
			cudaDeviceSynchronize();
			break;
		}
		cudaMemcpy(agents_s.x, agents_d.x, bytes, cudaMemcpyDeviceToHost);
		cudaMemcpy(agents_s.y, agents_d.y, bytes, cudaMemcpyDeviceToHost);
		break;
	}
#endif
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

void Ped::Model::get_agents_in_region(struct region_s *region) {
	for (auto *const agent : this->agents) {
		const int x = agent->getX();
		const int y = agent->getY();
		if (x >= region->x_start && x <= region->x_end) {
			region->region_agents.push_back(agent);
			struct pair_s pair = {.x = x, .y = y};
			region->taken_positions.push_back(pair);
		}
		if ((region->x_start == 0 && x < 0) || (region->x_end == GRID_WIDTH && x > GRID_WIDTH)) {
			region->region_agents.push_back(agent);
			struct pair_s pair = {.x = x, .y = y};
			region->taken_positions.push_back(pair);
		}
	}
}

bool Ped::Model::try_place_on_border(struct region_s *region, Ped::Tagent *agent, int x, int y) {
	// assume x == x_start
	omp_lock_t *lock = &region->llock;
	bool *lock_taken = region->llock_taken;
	if (x == region->x_end) {
		lock = &region->rlock;
		lock_taken = region->rlock_taken;
	}

	bool success = false;
	omp_set_lock(lock);
	if (!lock_taken[y]) {
		agent->setX(x);
		agent->setY(y);

		lock_taken[y] = true;
		success = true;
	}
	omp_unset_lock(lock);

	return success;
}

bool Ped::Model::try_migrate_outside_grid(struct region_s *region, Ped::Tagent *agent, int x, int y) {
	// since the same region is responsible for agents out of the
	// grid, no need to lock
	struct pair_s pair = {.x = x, .y = y};
	if (find_pair(region->taken_positions, pair)) {
		return false;
	}

	// assume x < 0
	omp_lock_t *prev_lock = &region->llock;
	bool *prev_lock_taken = region->llock_taken;
	if (x > region->x_end) {
		prev_lock = &region->rlock;
		prev_lock_taken = region->rlock_taken;
	}

	omp_set_lock(prev_lock);
	int prev_y = agent->getY();
	agent->setX(x);
	agent->setY(y);
	prev_lock_taken[prev_y] = false;
	omp_unset_lock(prev_lock);

	return true;
}

bool Ped::Model::try_migrate(struct region_s *region, Ped::Tagent *agent, int x, int y) {
	if (x < 0 || x > GRID_WIDTH) {
		return try_migrate_outside_grid(region, agent, x, y);
	}
	// assume x == x_start - 1
	struct region_s *adjacent_region = region + (x == region->x_start - 1 ? -1 : 1);

	omp_lock_t *lock = &adjacent_region->rlock;
	bool *lock_taken = adjacent_region->rlock_taken;

	omp_lock_t *prev_lock = &region->llock;
	bool *prev_lock_taken = region->llock_taken;
	if (x == region->x_end + 1) {
		lock = &adjacent_region->llock;
		lock_taken = adjacent_region->llock_taken;

		prev_lock = &region->rlock;
		prev_lock_taken = region->rlock_taken;
	}

	bool success = false;
try_again:
	omp_set_lock(lock);
	if (!lock_taken[y]) {
		int acquired = omp_test_lock(prev_lock);
		// int acquired = 1;
		if (!acquired) {
			omp_unset_lock(lock);
			goto try_again;
		}
		int prev_y = agent->getY();

		agent->setX(x);
		agent->setY(y);

		lock_taken[y] = true;
		success = true;
		prev_lock_taken[prev_y] = false;
		omp_unset_lock(prev_lock);
	}
	omp_unset_lock(lock);

	return success;
}

void Ped::Model::leave_border(struct region_s *region, Ped::Tagent *agent, int x, int y) {
	omp_lock_t *prev_lock = &region->llock;
	bool *prev_lock_taken = region->llock_taken;
	if (agent->getX() == region->x_end) {
		prev_lock = &region->rlock;
		prev_lock_taken = region->rlock_taken;
	}

	int prev_y = agent->getY();

	omp_set_lock(prev_lock);
	agent->setX(x);
	agent->setY(y);
	prev_lock_taken[prev_y] = false;
	omp_unset_lock(prev_lock);
}

bool Ped::Model::find_pair(std::vector<struct pair_s> taken_positions, struct pair_s pair) {
	const int size = taken_positions.size();
	for (size_t i = 0; i < size; i++) {
		if (pair.x == taken_positions[i].x && pair.y == taken_positions[i].y) {
			return true;
		}
	}
	return false;
}

void Ped::Model::move_parallel(struct region_s *region, int agent_idx) {
	// Compute the three alternative positions that would bring the agent
	// closer to his desiredPosition, starting with the desiredPosition itself
	Ped::Tagent *agent = region->region_agents[agent_idx];
	struct pair_s prioritizedAlternatives[NUM_ALTERNATIVES];
	struct pair_s pDesired = {.x = agent->getDesiredX(), .y = agent->getDesiredY()};
	size_t alternatives_cnt = 0;
	prioritizedAlternatives[alternatives_cnt++] = pDesired;

	int diffX = pDesired.x - agent->getX();
	int diffY = pDesired.y - agent->getY();
	struct pair_s p1, p2;
	if (diffX == 0 || diffY == 0) {
		// Agent wants to walk straight to North, South, West or East
		p1.x = pDesired.x + diffY;
		p1.y = pDesired.y + diffX;
		p2.x = pDesired.x - diffY;
		p2.y = pDesired.y - diffX;
	} else {
		// Agent wants to walk diagonally
		p1.x = pDesired.x;
		p1.y = agent->getY();
		p2.x = agent->getX();
		p2.y = pDesired.y;
	}
	prioritizedAlternatives[alternatives_cnt++] = p1;
	prioritizedAlternatives[alternatives_cnt++] = p2;

	// Find the first empty alternative position
	bool success = false;
	for (size_t i = 0; i < NUM_ALTERNATIVES; i++) {
		const int desired_x = prioritizedAlternatives[i].x;
		const int desired_y = prioritizedAlternatives[i].y;
		if (desired_x == region->x_start || desired_x == region->x_end) {
			success = try_place_on_border(region, agent, desired_x, desired_y);
		} else if (desired_x == region->x_start - 1 || desired_x == region->x_end + 1) {
			success = try_migrate(region, agent, desired_x, desired_y);
		} else if (!find_pair(region->taken_positions, prioritizedAlternatives[i])) {
			// If the current position is not yet taken by any neighbor
			// Set the agent's position
			if (agent->getX() == region->x_start || agent->getX() == region->x_end) {
				leave_border(region, agent, desired_x, desired_y);
			} else {
				agent->setX(desired_x);
				agent->setY(desired_y);
			}

			success = true;
		}

		if (success) {
			region->taken_positions[agent_idx] = prioritizedAlternatives[i];
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
#ifndef NOCUDA
	case Ped::CUDA:
		printf("Cleaning up data structures for cuda...\n");
		cuda_dinit(&agents_s);
		printf("Data structures for cuda released.\n");
		break;
#endif
	default:
		printf("No extra cleanup needed for given implementation.\n");
	}
}
