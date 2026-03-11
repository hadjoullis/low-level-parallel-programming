#ifndef _ped_move_parallel_struct_h_
#define _ped_move_parallel_struct_h_ 1

#include "ped_compute_desired_cuda.cuh"

#include <atomic>
#include <vector>

#include <stdlib.h>

#define GRID_WIDTH 160
#define GRID_HEIGHT 120
#define MAX_NUM_REGIONS 16
#define NUM_ALTERNATIVES 3

struct pair_bn_s {
	int x, y;
};

struct region_bn_s {
	int x_start, x_end;
	std::vector<std::atomic<bool>> lborder, rborder;
	std::vector<int> region_agents;
	std::vector<struct pair_bn_s> taken_positions;
};

void mv_parallel_struct_regions_init(std::vector<struct region_bn_s> &regions, const int n, const int *xs);
void mv_parallel_struct_regions_dinit(void);
int mv_parallel_struct_setup_regions(std::vector<struct region_bn_s> &regions, const int agents_size);
void mv_parallel_struct_get_agents_in_region(struct dscu_agents_s *agents, struct region_bn_s *region);

void move_parallel_struct(struct region_bn_s *region, struct dscu_agents_s *agents, int agent_idx);

#endif
