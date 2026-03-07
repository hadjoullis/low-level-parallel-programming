#include "ped_move_parallel.h"

size_t *agents_buckets;

void mv_parallel_regions_init(std::vector<struct region_s> &regions, const std::vector<Ped::Tagent *> &agents) {
	regions.resize(MAX_NUM_REGIONS);
	agents_buckets = (size_t *)calloc(GRID_WIDTH + 1, sizeof(size_t));

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

void mv_parallel_regions_dinit(void) { free(agents_buckets); }

int mv_parallel_setup_regions(std::vector<struct region_s> &regions, const std::vector<Ped::Tagent *> &agents) {
	static const size_t IDEAL_LOAD = agents.size() / MAX_NUM_REGIONS;
	static const size_t DIFF_TOLERANCE = IDEAL_LOAD / 4;
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
	return cur_region + 1;
}

void mv_parallel_get_agents_in_region(const std::vector<Ped::Tagent *> &agents, struct region_s *region) {
	// since region might have changed we need to reset
	for (int y = 0; y <= GRID_HEIGHT; y++) {
		region->lborder[y].store(false, std::memory_order_relaxed);
		region->rborder[y].store(false, std::memory_order_relaxed);
	}
	for (auto *const agent : agents) {
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

static bool find_pair(std::vector<struct pair_s> taken_positions, struct pair_s pair) {
	const int size = taken_positions.size();
	for (size_t i = 0; i < size; i++) {
		if (pair.x == taken_positions[i].x && pair.y == taken_positions[i].y) {
			return true;
		}
	}
	return false;
}

static bool try_place_on_border(struct region_s *region, Ped::Tagent *agent, int x, int y) {
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

static bool try_migrate_outside_grid(struct region_s *region, Ped::Tagent *agent, int x, int y) {
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

static bool try_migrate(struct region_s *region, Ped::Tagent *agent, int x, int y) {
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

static void leave_border(struct region_s *region, Ped::Tagent *agent, int x, int y) {
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

void move_parallel(struct region_s *region, int agent_idx) {
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
