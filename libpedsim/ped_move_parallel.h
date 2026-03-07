#ifndef _ped_move_parallel_h_
#define _ped_move_parallel_h_ 1

#include "ped_agent.h"
#include <atomic>
#include <vector>

#include <stdlib.h>

#define GRID_WIDTH 160
#define GRID_HEIGHT 120
#define MAX_NUM_REGIONS 16
#define NUM_ALTERNATIVES 3

struct pair_s {
	int x, y;
};

struct region_s {
	int x_start, x_end;
	std::vector<std::atomic<bool>> lborder, rborder;
	std::vector<Ped::Tagent *> region_agents;
	std::vector<struct pair_s> taken_positions;
};

void mv_parallel_regions_init(std::vector<struct region_s> &regions, const std::vector<Ped::Tagent *> &agents);
void mv_parallel_regions_dinit(void);
int mv_parallel_setup_regions(std::vector<struct region_s> &regions, const std::vector<Ped::Tagent *> &agents);
void mv_parallel_get_agents_in_region(const std::vector<Ped::Tagent *> &agents, struct region_s *region);

void move_parallel(struct region_s *region, int agent_idx);

#endif
