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
	regions.resize(MAX_NUM_REGIONS);
	agents_buckets = (size_t *)calloc(GRID_WIDTH + 1, sizeof(size_t));

	auto &agents = this->agents;
	const int n = agents.size();
	for (int i = 0; i < n; i++) {
		auto *agent = agents[i];
		if (agent->getX() < 0) {
			agents_buckets[0]++;
		} else if (agent->getX() > GRID_WIDTH) {
			agents_buckets[GRID_WIDTH]++;
		} else {
			agents_buckets[agent->getX()]++;
		}
	}
	// coordinates range is inclusive
	for (size_t i = 0; i < MAX_NUM_REGIONS; i++) {
		regions[i].lborder = std::vector<std::atomic<bool>>(GRID_HEIGHT + 1);
		regions[i].rborder = std::vector<std::atomic<bool>>(GRID_HEIGHT + 1);

		for (int y = 0; y <= GRID_HEIGHT; y++) {
			regions[i].lborder[y].store(false, std::memory_order_relaxed);
			regions[i].rborder[y].store(false, std::memory_order_relaxed);
		}
	}
}

void Ped::Model::regions_dinit(void) { free(agents_buckets); }

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

void Ped::Model::setup_regions(void) {
	static const size_t IDEAL_LOAD = agents.size() / MAX_NUM_REGIONS;
	static const size_t DIFF_TOLERANCE = IDEAL_LOAD / 10;
	int x_start = 0, x_cur, cur_region = 0;
	size_t agents_cnt = 0;
	for (x_cur = 0; x_cur <= GRID_WIDTH; x_cur++) {
		agents_cnt += agents_buckets[x_cur];
		// We want minimum one column for lborder, one for rborder and one buffer
		// We also need to make sure that the last region has at least 3 columns
		// This means that the last region, ALWAYS needs to be assigned after
		// the loop
		if (agents_cnt < IDEAL_LOAD + DIFF_TOLERANCE || x_cur - x_start < 2 || GRID_WIDTH - x_cur < 3 ||
			cur_region == MAX_NUM_REGIONS - 1) {
			continue;
		}
		// make x_end inclusive for easier logistics
		regions[cur_region].x_start = x_start;
		regions[cur_region].x_end = x_cur;

		x_start = x_cur + 1;
		agents_cnt = 0;
		cur_region++;
	}
	regions[cur_region].x_start = x_start;
	regions[cur_region].x_end = GRID_WIDTH;
	CUR_NUM_REGIONS = cur_region + 1;
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

#pragma omp parallel default(none) shared(n, agents, regions)
		{
#pragma omp single nowait
			{
				setup_regions();
			}
#pragma omp for
			for (int i = 0; i < n; i++) {
				auto *agent = agents[i];
				agent->computeNextDesiredPosition();
			} // implicit barrier
#pragma omp for
			for (int i = 0; i < CUR_NUM_REGIONS; i++) {
				get_agents_in_region(&regions[i]);
			} // implicit barrier
#pragma omp for
			for (int i = 0; i < CUR_NUM_REGIONS; i++) {
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
	// since region might have changed we need to reset
	for (int y = 0; y <= GRID_HEIGHT; y++) {
		region->lborder[y].store(false, std::memory_order_relaxed);
		region->rborder[y].store(false, std::memory_order_relaxed);
	}
	for (auto *const agent : this->agents) {
		const int x = agent->getX();
		const int y = agent->getY();
		if ((x >= region->x_start && x <= region->x_end) || (region->x_start == 0 && x < 0) ||
			(region->x_end == GRID_WIDTH && x > GRID_WIDTH)) {
			region->region_agents.push_back(agent);
			struct pair_s pair = {.x = x, .y = y};
			region->taken_positions.push_back(pair);
			if (x == region->x_start) {
				region->lborder[y].store(true, std::memory_order_relaxed);
			} else if (x == region->x_end) {
				region->rborder[y].store(true, std::memory_order_relaxed);
			}
		}
	}
}

bool Ped::Model::try_place_on_border(struct region_s *region, Ped::Tagent *agent, int x, int y) {
	std::vector<std::atomic<bool>> *border;
	if (x == region->x_start) {
		border = &region->lborder;
	} else { // if (x == region->x_end)
		border = &region->rborder;
	}
	const bool taken = (*border)[y].load(std::memory_order_acquire);
	bool expected = false;
	if (taken || !(*border)[y].compare_exchange_strong(expected, true, std::memory_order_release)) {
		return false;
	}

	const int prev_x = agent->getX();
	const int prev_y = agent->getY();
	if (prev_x == region->x_start || prev_x == region->x_end) {
		(*border)[prev_y].store(false, std::memory_order_release);
	}
	agent->setX(x);
	agent->setY(y);

	// agents outside borders will be handled by the same thread, regardless
	if (prev_x >= 0 && prev_x <= GRID_WIDTH) {
#pragma omp atomic update
		agents_buckets[prev_x]--;
#pragma omp atomic update
		agents_buckets[x]++;
	}

	return true;
}

bool Ped::Model::try_migrate_outside_grid(struct region_s *region, Ped::Tagent *agent, int x, int y) {
	// since the same region is responsible for agents out of the grid, no need
	// to care about data races
	struct pair_s pair = {.x = x, .y = y};
	if (find_pair(region->taken_positions, pair)) {
		return false;
	}

	std::vector<std::atomic<bool>> *prev_border;
	if (x < 0) {
		prev_border = &region->lborder;
	} else { // if (x > GRID_WIDTH)
		prev_border = &region->rborder;
	}

	const int prev_x = agent->getX();
	const int prev_y = agent->getY();
	agent->setX(x);
	agent->setY(y);
	(*prev_border)[prev_y].store(false, std::memory_order_release);
	// agents outside borders will be handled by the same thread, regardless
	// agents_buckets[BORDER]--;
	// agents_buckets[BORDER]++;

	return true;
}

bool Ped::Model::try_migrate(struct region_s *region, Ped::Tagent *agent, int x, int y) {
	if (x < 0 || x > GRID_WIDTH) {
		return try_migrate_outside_grid(region, agent, x, y);
	}
	struct region_s *adjacent_region;
	std::vector<std::atomic<bool>> *border;
	std::vector<std::atomic<bool>> *prev_border;
	if (x == region->x_start - 1) {
		adjacent_region = region - 1;
		border = &adjacent_region->rborder;
		prev_border = &region->lborder;
	} else { // if (x == region->x_end + 1)
		adjacent_region = region + 1;
		border = &adjacent_region->lborder;
		prev_border = &region->rborder;
	}
	const bool taken = (*border)[y].load(std::memory_order_acquire);
	bool expected = false;
	if (taken || !(*border)[y].compare_exchange_strong(expected, true, std::memory_order_release)) {
		return false;
	}

	const int prev_x = agent->getX();
	const int prev_y = agent->getY();

	(*prev_border)[prev_y].store(false, std::memory_order_release);

	agent->setX(x);
	agent->setY(y);

#pragma omp atomic update
	agents_buckets[prev_x]--;
#pragma omp atomic update
	agents_buckets[x]++;

	return true;
}

void Ped::Model::leave_border(struct region_s *region, Ped::Tagent *agent, int x, int y) {
	std::vector<std::atomic<bool>> *prev_border;
	if (agent->getX() == region->x_start) {
		prev_border = &region->lborder;
	} else { // if (agent->getX() == region->x_end)
		prev_border = &region->rborder;
	}

	const int prev_x = agent->getX();
	const int prev_y = agent->getY();

	(*prev_border)[prev_y].store(false, std::memory_order_release);
	agent->setX(x);
	agent->setY(y);
#pragma omp atomic update
	agents_buckets[prev_x]--;
#pragma omp atomic update
	agents_buckets[x]++;
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
	struct pair_s prioritizedAlternatives[NUM_ALTERNATIVES] = {0};
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
		} else if ((desired_x == region->x_start - 1 && agent->getX() == region->x_start) ||
				   (desired_x == region->x_end + 1 && agent->getX() == region->x_end)) {
			success = try_migrate(region, agent, desired_x, desired_y);
		} else if (!find_pair(region->taken_positions, prioritizedAlternatives[i])) {
			// If the current position is not yet taken by any neighbor
			// Set the agent's position
			if (agent->getX() == region->x_start || agent->getX() == region->x_end) {
				leave_border(region, agent, desired_x, desired_y);
			} else {
				const int prev_x = agent->getX();
				agent->setX(desired_x);
				agent->setY(desired_y);
				// check prev_x or desired_x, no need to check both
				// agents outside borders will be handled by the same thread, regardless
				if (prev_x >= 0 && prev_x <= GRID_WIDTH) {
#pragma omp atomic update
					agents_buckets[prev_x]--;
#pragma omp atomic update
					agents_buckets[desired_x]++;
				}
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
	case Ped::OMP_MV:
		printf("Cleaning up data structures for OMP_MV...\n");
		regions_dinit();
		printf("Data structures for OMP_MV released.\n");
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
